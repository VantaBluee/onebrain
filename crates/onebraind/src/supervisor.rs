//! The M5 job supervisor (docs/resilience.md "Failure lifecycle (head)").
//!
//! The supervisor task owns the lifecycle of every generation job: the
//! gateway's [`crate::engine_host::DaemonBackend`] enqueues jobs here, the
//! supervisor issues them to the engine host as supervised attempts, and on
//! a distributed decode failure it — instead of surfacing an error —
//!
//! 1. marks the epoch failed,
//! 2. tears it down (model freed first, over dead bridges the patched RPC
//!    client tolerates — patches/0002; bridges closed after, ADR 0004),
//! 3. re-plans from live Connected peers (dead/suspect/draining excluded),
//! 4. reloads (solo or distributed) and re-issues the generation with the
//!    prefix `prompt_tokens + generated_tokens`, streaming only NEW pieces
//!    into the SAME client stream — exactly ONE transparent retry per job;
//!    a second failure (or an infeasible re-plan) surfaces the typed error
//!    naming the lost node, and the model is marked unloaded.
//!
//! # M7 concurrency (docs/perf.md §6)
//!
//! Since M7 the engine host runs up to `[perf] max_concurrent_requests`
//! generations at once, so jobs are no longer single-flighted here: each
//! [`SupervisorMsg::Generate`] spawns its own lifecycle task. What stays
//! SERIALIZED — via the [`RetryLedger`] mutex on
//! [`crate::server::InternalState`] — is every piece of epoch surgery:
//! interruption handling, the no-job teardown, and the rejoin re-plan run
//! one at a time (contract rule: the retry path may serialize; correctness
//! first). A distributed decode failure interrupts EVERY active sequence
//! at once; the first interruption to take the mutex is the LEADER (full
//! failure lifecycle: teardown → re-plan → reload) and records the failed
//! epoch's outcome in the ledger, and the other interrupted jobs — the
//! followers — consult that record instead of tearing down the epoch the
//! leader just replaced: on a successful reload each re-issues its own
//! resume prefix (re-prefilling THE AFFECTED SEQUENCES), on a failed one
//! each surfaces the same typed error.
//!
//! The cluster task's peer-event consumer (crate::cluster) feeds two more
//! message kinds: [`SupervisorMsg::EpochFailed`] (death/drain detection
//! with no job in flight → teardown here, serialized via the same mutex)
//! and [`SupervisorMsg::PeerRejoined`] (lazy re-plan once the engine host
//! is idle — in-flight/queued work always finishes first, which every job
//! task guarantees by re-checking after it ends).

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Duration;

use onebrain_api::backend::{GenerateJob, TokenEvent};
use onebrain_api::ApiError;
use onebrain_mesh::{PeerState, PeerStatus};
use onebrain_proto::plan::{Assignment, Strategy};
use tokio::sync::{mpsc, oneshot};
use tokio::task::JoinHandle;

use crate::engine_host::{
    DraftRequest, GenOutcome, HostMsg, InterruptedGen, ResumeState, SupervisedGenerate,
};
use crate::server::{
    activate_distributed_plan, activate_solo_plan, plan_load, InternalState, LocalModel,
    PlanLoadError,
};

/// How long the supervisor waits for the host to confirm an unload during
/// epoch teardown. The host answers as soon as the (possibly torn) model's
/// frees complete; the patched RPC client never blocks on a dead bridge.
const UNLOAD_TIMEOUT: Duration = Duration::from_secs(30);

/// How many failed-epoch outcomes the retry ledger retains (bounded memory
/// over a long daemon life on a flaky cluster). Epochs are monotonic per
/// daemon (`next_epoch`), and a follower only ever looks up the epoch its
/// own in-flight attempt ran against — an attempt interrupted within the
/// last few epoch surgeries (bounded by `[perf] max_concurrent_requests`,
/// far below this window), so pruning older records can never break a
/// lookup.
const RETAIN_EPOCHS: u64 = 32;

/// Work for the supervisor task.
#[derive(Debug)]
pub enum SupervisorMsg {
    /// One generation job from the gateway (its whole lifecycle runs here).
    Generate(GenerateJob),
    /// The cluster task marked `epoch` failed (peer death or polite drain,
    /// docs/resilience.md step 1). With no job in flight the supervisor
    /// tears the epoch down; with one in flight the retry path — which this
    /// message queues behind — already owns the teardown, and the stale
    /// epoch check below makes this a no-op.
    EpochFailed { epoch: u64 },
    /// A peer returned to Connected while a model is loaded: run the lazy
    /// re-plan when idle (docs/resilience.md step 5). The request itself is
    /// latched on [`crate::cluster::ClusterState`]; this message is the
    /// wakeup.
    PeerRejoined,
}

/// Sender half of the supervisor queue.
pub type SupervisorTx = mpsc::UnboundedSender<SupervisorMsg>;

/// Create the supervisor queue. The receiver goes to [`spawn`]; senders go
/// to the gateway backend and the cluster task.
pub fn channel() -> (SupervisorTx, mpsc::UnboundedReceiver<SupervisorMsg>) {
    mpsc::unbounded_channel()
}

/// What happened to a failed epoch, recorded by the interruption LEADER so
/// the other sequences interrupted by the same decode failure resolve
/// consistently without a second teardown (docs/perf.md §6 + the module
/// docs).
#[derive(Debug, Clone)]
enum EpochOutcome {
    /// Teardown + re-plan + reload succeeded: followers re-issue their
    /// resume prefixes against the new epoch. `lost` is the blamed node's
    /// label — followers with a spent retry budget need it for the typed
    /// exhaustion error.
    Reloaded { lost: String },
    /// The epoch is gone and nothing was reloaded (infeasible re-plan,
    /// reload failure, or a spent-budget leader): followers surface
    /// `message` verbatim, budget or not.
    Failed { message: String },
}

/// Serialization point + memory for epoch surgery. Held (as
/// `tokio::sync::Mutex<RetryLedger>` on [`InternalState`]) across every
/// interruption's failure lifecycle, the no-job teardown, and the rejoin
/// re-plan — one at a time, correctness first.
#[derive(Debug, Default)]
pub struct RetryLedger {
    outcomes: HashMap<u64, EpochOutcome>,
}

/// Spawn the supervisor task. It ends when every sender is gone (daemon
/// teardown drops the backend and the cluster task). Generation jobs run
/// in their own tasks (M7: the engine host serves them concurrently);
/// epoch-level messages run inline, serialized by the retry ledger.
pub fn spawn(
    state: Arc<InternalState>,
    mut rx: mpsc::UnboundedReceiver<SupervisorMsg>,
) -> JoinHandle<()> {
    tokio::spawn(async move {
        while let Some(msg) = rx.recv().await {
            match msg {
                SupervisorMsg::Generate(job) => {
                    let state = state.clone();
                    tokio::spawn(async move {
                        run_supervised(&state, job).await;
                        // A rejoin that arrived mid-job waited for it
                        // (queued work always finishes first); run it now
                        // if we are the last job out.
                        maybe_rejoin_replan(&state).await;
                    });
                }
                SupervisorMsg::EpochFailed { epoch } => {
                    handle_epoch_failed(&state, epoch).await;
                }
                SupervisorMsg::PeerRejoined => {
                    maybe_rejoin_replan(&state).await;
                }
            }
        }
        tracing::debug!("supervisor task stopped");
    })
}

// ---------------------------------------------------------------------------
// Decision table (pure — unit-tested below)
// ---------------------------------------------------------------------------

/// What the supervisor does with an interrupted generation attempt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum InterruptedAction {
    /// Tear the epoch down, re-plan, reload, and re-issue with the carried
    /// prefix (docs/resilience.md step 4).
    RetryOnce,
    /// The single retry is spent: surface the typed error and mark the
    /// model unloaded.
    FailTyped,
}

/// The retry budget rule: exactly ONE transparent retry per job.
pub(crate) fn decide_interrupted(retries_used: u32) -> InterruptedAction {
    if retries_used == 0 {
        InterruptedAction::RetryOnce
    } else {
        InterruptedAction::FailTyped
    }
}

/// The contract's typed error for a lost node whose absence makes the model
/// infeasible (docs/resilience.md step 3 — wording is binding).
pub(crate) fn lost_node_error(lost: &str, needs_mb: u64, have_mb: u64) -> String {
    format!(
        "the node '{lost}' was lost mid-generation and the remaining nodes cannot hold the \
         model (needs {needs_mb} MB, have {have_mb} MB); reconnect the node or choose a \
         smaller model"
    )
}

/// The typed error when the single transparent retry is spent (the re-plan
/// was feasible, but the retried generation failed again).
pub(crate) fn retry_exhausted_error(lost: &str) -> String {
    format!(
        "the node '{lost}' was lost mid-generation and the automatic retry failed too; \
         check the cluster with `onebrain status` and retry the request"
    )
}

/// Name the node lost to a failed epoch, best evidence first: a peer whose
/// bridge stream tore or whose death/drain the mesh reported (`lost_ids`),
/// then any plan participant not currently Connected (or draining), then —
/// when everything still looks alive — the plan's first worker. Returns the
/// peer-store display name where known, a shortened id otherwise.
pub(crate) fn lost_node_label(
    assignments: &[Assignment],
    own_id: &str,
    lost_ids: &HashSet<String>,
    peers: &[PeerStatus],
) -> String {
    let name_of = |id: &str| {
        peers
            .iter()
            .find(|p| p.id == id)
            .map(|p| p.name.clone())
            .unwrap_or_else(|| id.chars().take(8).collect())
    };
    let workers = || assignments.iter().filter(|a| a.node.0 != own_id);
    if let Some(a) = workers().find(|a| lost_ids.contains(&a.node.0)) {
        return name_of(&a.node.0);
    }
    if let Some(a) = workers().find(|a| {
        let peer = peers.iter().find(|p| p.id == a.node.0);
        !peer.is_some_and(|p| p.state == PeerState::Connected && !p.draining)
    }) {
        return name_of(&a.node.0);
    }
    workers()
        .next()
        .map(|a| name_of(&a.node.0))
        .unwrap_or_else(|| "unknown".to_string())
}

// ---------------------------------------------------------------------------
// Job lifecycle
// ---------------------------------------------------------------------------

/// Result of sending one supervised attempt to the engine host.
enum Attempt {
    Outcome(GenOutcome),
    /// The host thread is gone; the job is returned when the send itself
    /// failed (so its stream can still be terminated).
    HostGone(Option<GenerateJob>),
}

async fn send_attempt(
    state: &Arc<InternalState>,
    job: GenerateJob,
    resume: Option<ResumeState>,
) -> Attempt {
    let (outcome_tx, outcome_rx) = oneshot::channel();
    let msg = HostMsg::Generate(SupervisedGenerate {
        job,
        resume,
        outcome: outcome_tx,
    });
    match state.host.send_or_return(msg) {
        Ok(()) => match outcome_rx.await {
            Ok(outcome) => Attempt::Outcome(outcome),
            // Host thread exited mid-attempt (daemon teardown): the job and
            // its stream went with it.
            Err(_) => Attempt::HostGone(None),
        },
        Err(msg) => match *msg {
            HostMsg::Generate(sup) => Attempt::HostGone(Some(sup.job)),
            _ => Attempt::HostGone(None),
        },
    }
}

/// Terminate a job's stream with an error (best effort — the client may
/// have gone away).
async fn fail_job(job: &GenerateJob, message: String) {
    let _ = job.tx.send(TokenEvent::Error(message)).await;
}

/// Drive one job to its terminal event, transparently retrying once on a
/// distributed mid-generation failure. Runs in its own task since M7; the
/// interruption path serializes internally on the retry ledger.
async fn run_supervised(state: &Arc<InternalState>, job: GenerateJob) {
    let mut retries_used: u32 = 0;
    let mut next: Option<(GenerateJob, Option<ResumeState>)> = Some((job, None));
    while let Some((job, resume)) = next.take() {
        // The epoch this attempt runs against, for leader/follower
        // resolution when it gets interrupted (module docs).
        let attempt_epoch = state.cluster.active().map(|a| a.plan.epoch.0);
        match send_attempt(state, job, resume).await {
            Attempt::Outcome(GenOutcome::Finished) => {}
            Attempt::HostGone(Some(job)) => {
                fail_job(&job, ApiError::ShuttingDown.to_string()).await;
            }
            Attempt::HostGone(None) => {}
            Attempt::Outcome(GenOutcome::Interrupted(interrupted)) => {
                if let Some((job, resume, counted)) =
                    handle_interrupted(state, *interrupted, retries_used, attempt_epoch).await
                {
                    // An UNCOUNTED re-issue is the stale-snapshot case: a
                    // rejoin epoch swap landed between the epoch snapshot
                    // and the attempt running, so this job never actually
                    // failed on the epoch it was blamed for — its single
                    // transparent retry stays available for a genuine
                    // failure of the re-issue.
                    if counted {
                        retries_used += 1;
                    }
                    next = Some((job, Some(resume)));
                }
            }
        }
    }
    state.host.job_finished();
}

/// The failure lifecycle for one interruption. Returns `Some((job, resume,
/// counted))` when the job should be re-issued (the caller loops; `counted`
/// says whether the re-issue consumes the single transparent retry), `None`
/// when it reached a terminal event here.
///
/// Serialized on the retry ledger: the first interrupted job of a failed
/// epoch (the LEADER — the epoch is still the active one when it takes the
/// mutex) runs the full lifecycle and records the outcome; jobs
/// interrupted by the same failure (FOLLOWERS — by the time they take the
/// mutex the active epoch has changed or is gone) resolve from the record
/// instead of tearing down whatever the leader just built.
async fn handle_interrupted(
    state: &Arc<InternalState>,
    interrupted: InterruptedGen,
    retries_used: u32,
    attempt_epoch: Option<u64>,
) -> Option<(GenerateJob, ResumeState, bool)> {
    let engine_error = interrupted.error.clone();
    let (job, resume) = interrupted.into_retry();

    let mut ledger = state.retry.lock().await;

    // Follower path: this attempt's epoch is no longer the active one —
    // another sequence's interruption already resolved it.
    let still_active = match attempt_epoch {
        Some(epoch) => state
            .cluster
            .active()
            .is_some_and(|a| a.plan.epoch.0 == epoch),
        // No epoch recorded at send time: cannot be matched against the
        // ledger; fall through to the leader path (its "no active plan"
        // handling covers this).
        None => false,
    };
    if let Some(epoch) = attempt_epoch {
        if !still_active {
            let outcome = ledger.outcomes.get(&epoch).cloned();
            drop(ledger);
            return match outcome {
                Some(EpochOutcome::Reloaded { lost }) => {
                    if decide_interrupted(retries_used) == InterruptedAction::FailTyped {
                        fail_job(&job, retry_exhausted_error(&lost)).await;
                        None
                    } else {
                        tracing::info!(
                            epoch,
                            generated = resume.generated_tokens.len(),
                            "re-issuing a sequence interrupted with the epoch another \
                             job already recovered"
                        );
                        Some((job, resume, true))
                    }
                }
                Some(EpochOutcome::Failed { message, .. }) => {
                    fail_job(&job, message).await;
                    None
                }
                None => {
                    // The epoch changed hands without a recorded failure
                    // (e.g. a rejoin swap raced the interruption — the
                    // attempt may even have RUN against the new epoch, its
                    // snapshot taken before the swap landed). With a model
                    // still loaded and budget left, re-issue against it —
                    // UNCOUNTED: this job never failed on an epoch anyone
                    // resolved, so burning its single retry here would let
                    // one genuine failure of the re-issue exhaust the
                    // budget after zero recovered re-plans. Bounded: once
                    // the re-issue is in flight the host is non-idle, so no
                    // new rejoin swap can start, and any later epoch change
                    // goes through a leader that records the ledger — this
                    // branch fires at most once per job.
                    if decide_interrupted(retries_used) == InterruptedAction::RetryOnce
                        && state.cluster.loaded_source().is_some()
                    {
                        Some((job, resume, false))
                    } else {
                        fail_job(
                            &job,
                            format!(
                                "the distributed generation failed ({engine_error}); \
                                 check the cluster with `onebrain status` and retry"
                            ),
                        )
                        .await;
                        None
                    }
                }
            };
        }
    }

    // Leader path: full failure lifecycle. State of the failed epoch,
    // captured before teardown clears it.
    let failed = state.cluster.active();
    let failed_epoch = failed.as_ref().map(|a| a.plan.epoch.0);
    let source = state.cluster.loaded_source();
    if let Some(epoch) = failed_epoch {
        state.cluster.mark_epoch_failed(epoch, None);
    }
    tracing::warn!(
        error = %engine_error,
        epoch = failed_epoch,
        retries_used,
        "distributed generation interrupted; epoch marked failed"
    );

    // Teardown (docs/resilience.md step 3): free the model while the
    // bridges still stand — the patched frees tolerate the dead ones
    // (patches/0002) — then close the bridges (ADR 0004 ordering).
    teardown_failed_epoch(state).await;

    // Who did we lose? Bridge-stream tears and mesh death/drain transitions
    // recorded against the failed epoch, refined against the live peer view.
    let peers = state.mesh.peers().await.unwrap_or_default();
    let lost_ids = failed_epoch
        .map(|e| state.cluster.epoch_lost_peers(e))
        .unwrap_or_default();
    let own_id = state.mesh.endpoint_id().to_string();
    let lost = failed
        .as_ref()
        .map(|a| lost_node_label(&a.plan.assignments, &own_id, &lost_ids, &peers))
        .unwrap_or_else(|| "unknown".to_string());

    // Record the epoch's outcome + terminate or re-issue this job. The
    // record closure keeps ledger writes next to the decision they encode.
    let record = |ledger: &mut RetryLedger, outcome: EpochOutcome| {
        if let Some(epoch) = failed_epoch {
            ledger.outcomes.insert(epoch, outcome);
            // Bounded retention (see RETAIN_EPOCHS): without it a
            // long-running daemon on a flaky cluster leaks one record per
            // failed epoch forever.
            ledger.outcomes.retain(|e, _| e + RETAIN_EPOCHS > epoch);
        }
    };

    if decide_interrupted(retries_used) == InterruptedAction::FailTyped {
        // The single transparent retry is spent (step 4: "a second failure
        // surfaces the typed error"). M5 posture preserved: no reload is
        // attempted, the model is marked unloaded — followers of this
        // epoch (fresh or spent) surface the same typed error.
        state.cluster.clear_loaded();
        let message = retry_exhausted_error(&lost);
        record(
            &mut ledger,
            EpochOutcome::Failed {
                message: message.clone(),
            },
        );
        drop(ledger);
        fail_job(&job, message).await;
        return None;
    }

    let Some(source) = source else {
        state.cluster.clear_loaded();
        let message = format!(
            "the distributed generation failed ({engine_error}) and the loaded model \
             could not be identified for a retry; reload it with `onebrain run`"
        );
        record(
            &mut ledger,
            EpochOutcome::Failed {
                message: message.clone(),
            },
        );
        drop(ledger);
        fail_job(&job, message).await;
        return None;
    };

    // Re-plan from live Connected peers, dead/suspect/draining excluded
    // (step 3). The lost set is excluded explicitly: the mesh may still
    // report the dead node Connected for a few seconds.
    let local = LocalModel::from(&source);
    let planned = match plan_load(
        state,
        &source.name,
        source.header_path(),
        source.size_bytes,
        None,
        &lost_ids,
        /* exclude_draining */ true,
    )
    .await
    {
        Ok(planned) => planned,
        Err(PlanLoadError::DoesNotFit {
            required_mb,
            available_mb,
            ..
        }) => {
            // Nothing fits pooled: the contract's typed error, model
            // unloaded (step 3).
            state.cluster.clear_loaded();
            let message = lost_node_error(&lost, required_mb, available_mb);
            record(
                &mut ledger,
                EpochOutcome::Failed {
                    message: message.clone(),
                },
            );
            drop(ledger);
            fail_job(&job, message).await;
            return None;
        }
        Err(other) => {
            state.cluster.clear_loaded();
            let message = other.into_message();
            record(
                &mut ledger,
                EpochOutcome::Failed {
                    message: message.clone(),
                },
            );
            drop(ledger);
            fail_job(&job, message).await;
            return None;
        }
    };

    // Reload on the new plan (step 4). A reload failure consumes the retry:
    // the job surfaces the reload's own error. The recorded draft rides
    // along (docs/perf.md §5: "retry re-prefills and continues
    // speculating").
    let new_epoch = planned.plan.epoch.0;
    let draft = source
        .draft_reference
        .clone()
        .map(|reference| DraftRequest {
            reference,
            cache_root: state.cache_root.clone(),
        });
    let reload = if planned.plan.strategy == Strategy::Solo {
        activate_solo_plan(state, &source.reference, &local, planned, draft).await
    } else {
        activate_distributed_plan(state, &source.reference, &local, planned, draft).await
    };
    if let Err(message) = reload {
        state.cluster.clear_loaded();
        record(
            &mut ledger,
            EpochOutcome::Failed {
                message: message.clone(),
            },
        );
        drop(ledger);
        fail_job(&job, message).await;
        return None;
    }
    record(&mut ledger, EpochOutcome::Reloaded { lost });
    drop(ledger);
    tracing::info!(
        epoch = new_epoch,
        generated = resume.generated_tokens.len(),
        pieces_sent = resume.pieces_sent,
        "reloaded after mid-generation failure; retrying transparently with the carried prefix"
    );
    Some((job, resume, true))
}

/// Tear down the failed epoch's local half: unload the model (its frees
/// cross the still-standing bridges; dead ones are tolerated by
/// patches/0002), then abort the bridges, then clear the active-plan view.
async fn teardown_failed_epoch(state: &Arc<InternalState>) {
    let (unload_tx, unload_rx) = oneshot::channel();
    if state.host.send(HostMsg::Unload { resp: unload_tx }).is_ok()
        && tokio::time::timeout(UNLOAD_TIMEOUT, unload_rx)
            .await
            .is_err()
    {
        tracing::warn!(
            "the engine host did not confirm the unload within {UNLOAD_TIMEOUT:?}; \
             closing the bridges anyway"
        );
    }
    state.cluster.teardown_head_bridges();
    state.cluster.set_active(None);
}

// ---------------------------------------------------------------------------
// Death/drain with no job in flight, and the lazy rejoin re-plan
// ---------------------------------------------------------------------------

/// Death or polite drain detected with (possibly) no job in flight
/// (docs/resilience.md step 1): when the failed epoch is still the active
/// one and the host is idle, tear it down so the next load re-plans. The
/// retry ledger serializes this against any in-flight interruption's
/// lifecycle; the stale-epoch and busy checks below fence the cases the
/// retry path owns.
async fn handle_epoch_failed(state: &Arc<InternalState>, epoch: u64) {
    let _ledger = state.retry.lock().await;
    let Some(active) = state.cluster.active() else {
        return;
    };
    if active.role != "head" || active.plan.epoch.0 != epoch {
        return;
    }
    if !state.host.is_idle() {
        // A job is in flight against the failed epoch: its decode failure
        // reaches this task as an Interrupted outcome, whose retry path
        // owns the teardown.
        tracing::debug!(
            epoch,
            "epoch failed with a job in flight; deferring to the retry path"
        );
        return;
    }
    tracing::warn!(
        epoch,
        "active epoch failed with no job in flight; tearing it down (the next load re-plans)"
    );
    teardown_failed_epoch(state).await;
    state.cluster.clear_loaded();
}

/// The lazy rejoin re-plan (docs/resilience.md step 5): with a model loaded
/// and the engine host idle, re-plan with the FULL peer set and swap to the
/// new epoch through the normal proposal/ack/load flow. When the planner
/// keeps the same assignments the active epoch stays (a pointless reload
/// serves nobody). A busy host re-arms the request; it runs after the last
/// queued job finishes. Serialized on the retry ledger so a swap can never
/// interleave with an interruption's teardown/reload.
async fn maybe_rejoin_replan(state: &Arc<InternalState>) {
    if !state.cluster.take_rejoin_request() {
        return;
    }
    let _ledger = state.retry.lock().await;
    let Some(source) = state.cluster.loaded_source() else {
        return; // nothing loaded (anymore); moot
    };
    let Some(active) = state.cluster.active() else {
        return;
    };
    if active.role != "head" {
        return;
    }
    if !state.host.is_idle() {
        // In-flight/queued work always finishes first; the post-job check
        // re-runs this.
        state.cluster.defer_rejoin_request();
        return;
    }
    let local = LocalModel::from(&source);
    let planned = match plan_load(
        state,
        &source.name,
        source.header_path(),
        source.size_bytes,
        None,
        &HashSet::new(),
        false,
    )
    .await
    {
        Ok(planned) => planned,
        Err(e) => {
            tracing::warn!(
                error = %e.into_message(),
                "rejoin re-plan failed; keeping the current epoch"
            );
            return;
        }
    };
    if planned.plan.strategy == active.plan.strategy
        && planned.plan.assignments == active.plan.assignments
    {
        tracing::info!(
            epoch = active.plan.epoch.0,
            "rejoin re-plan reproduced the active assignments; keeping the current epoch"
        );
        return;
    }
    tracing::info!(
        old_epoch = active.plan.epoch.0,
        new_epoch = planned.plan.epoch.0,
        strategy = ?planned.plan.strategy,
        "rejoin re-plan changes the placement; swapping epochs"
    );
    let draft = source
        .draft_reference
        .clone()
        .map(|reference| DraftRequest {
            reference,
            cache_root: state.cache_root.clone(),
        });
    let result = if planned.plan.strategy == Strategy::Solo {
        activate_solo_plan(state, &source.reference, &local, planned, draft).await
    } else {
        activate_distributed_plan(state, &source.reference, &local, planned, draft).await
    };
    match result {
        Ok(model) => {
            tracing::info!(model = %model.name, "rejoin re-plan activated a new epoch");
        }
        Err(message) => {
            // The swap dropped the old model; the daemon stays healthy but
            // the model is unloaded (same posture as a failed user load).
            state.cluster.clear_loaded();
            tracing::error!(error = %message, "rejoin re-plan reload failed; model unloaded");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use onebrain_proto::plan::{LayerRange, NodeId};

    fn assignment(id: &str) -> Assignment {
        Assignment {
            node: NodeId(id.to_string()),
            layers: LayerRange { start: 0, end: 1 },
            stage: 0,
        }
    }

    fn peer(id: &str, name: &str, state: PeerState, draining: bool) -> PeerStatus {
        PeerStatus {
            name: name.to_string(),
            id: id.to_string(),
            state,
            rtt_ms: None,
            bandwidth_mbps: None,
            loss: None,
            last_seen_unix: None,
            usable_memory_bytes: Some(1 << 30),
            prefill_tps: None,
            decode_tps: None,
            disk_mbps: None,
            product_version: None,
            engine_build: None,
            draining,
        }
    }

    #[test]
    fn decision_table_grants_exactly_one_retry() {
        // The contract's decision table: interrupted → retry once → typed
        // error (docs/resilience.md step 4).
        assert_eq!(decide_interrupted(0), InterruptedAction::RetryOnce);
        assert_eq!(decide_interrupted(1), InterruptedAction::FailTyped);
        assert_eq!(decide_interrupted(2), InterruptedAction::FailTyped);
    }

    #[test]
    fn lost_node_error_is_the_contract_wording() {
        let msg = lost_node_error("gaming-pc", 9000, 4096);
        assert_eq!(
            msg,
            "the node 'gaming-pc' was lost mid-generation and the remaining nodes cannot \
             hold the model (needs 9000 MB, have 4096 MB); reconnect the node or choose a \
             smaller model"
        );
    }

    #[test]
    fn retry_exhausted_error_names_the_node_and_a_remedy() {
        let msg = retry_exhausted_error("gaming-pc");
        assert!(msg.contains("'gaming-pc'"), "{msg}");
        assert!(msg.contains("onebrain status"), "remedy missing: {msg}");
    }

    #[test]
    fn lost_label_prefers_recorded_losses() {
        let assignments = vec![assignment("w1"), assignment("w2"), assignment("head")];
        let peers = vec![
            peer("w1", "alpha", PeerState::Connected, false),
            peer("w2", "bravo", PeerState::Connected, false),
        ];
        // w2's bridge tore (recorded loss) even though the mesh still says
        // Connected — the recorded evidence wins.
        let lost: HashSet<String> = ["w2".to_string()].into_iter().collect();
        assert_eq!(
            lost_node_label(&assignments, "head", &lost, &peers),
            "bravo"
        );
    }

    #[test]
    fn lost_label_falls_back_to_non_connected_participants() {
        let assignments = vec![assignment("w1"), assignment("w2"), assignment("head")];
        let peers = vec![
            peer("w1", "alpha", PeerState::Connected, false),
            peer("w2", "bravo", PeerState::Down, false),
        ];
        let lost = HashSet::new();
        assert_eq!(
            lost_node_label(&assignments, "head", &lost, &peers),
            "bravo"
        );
    }

    #[test]
    fn lost_label_counts_draining_as_lost() {
        let assignments = vec![assignment("w1"), assignment("head")];
        let peers = vec![peer("w1", "alpha", PeerState::Connected, true)];
        let lost = HashSet::new();
        assert_eq!(
            lost_node_label(&assignments, "head", &lost, &peers),
            "alpha"
        );
    }

    #[test]
    fn lost_label_last_resort_is_the_first_worker_never_the_head() {
        let assignments = vec![assignment("head"), assignment("w1")];
        let peers = vec![peer("w1", "alpha", PeerState::Connected, false)];
        let lost = HashSet::new();
        // Everything looks alive: blame the first WORKER (the head cannot
        // have lost itself).
        assert_eq!(
            lost_node_label(&assignments, "head", &lost, &peers),
            "alpha"
        );
    }

    #[test]
    fn lost_label_uses_short_id_for_unknown_peers() {
        let assignments = vec![assignment("deadbeefdeadbeef"), assignment("head")];
        let lost = HashSet::new();
        assert_eq!(
            lost_node_label(&assignments, "head", &lost, &[]),
            "deadbeef"
        );
    }
}
