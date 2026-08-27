//! Scheduler v1 proper (M4, binding contract in `docs/scheduler-v1.md`):
//! real KV budgeting, memory-and-compute scores, boundary-on-fastest-link
//! stage ordering, and the "a node joins only when it helps" rule.
//!
//! # Budget and capacity model
//!
//! A node's budget is its usable memory minus a fixed
//! [`OVERHEAD_RESERVE_BYTES`] compute/graph reserve (clamped at zero) — this
//! replaces M3's flat 85% ceiling. Layer capacity uses the documented
//! *uniform-layer approximation*: every layer is costed at the model's mean
//! weight bytes per layer plus the KV bytes one layer accrues at the
//! requested context —
//!
//! ```text
//! capacity_layers = budget / (mean_weight_per_layer + kv_per_layer_at_ctx)
//! ```
//!
//! The choice is deliberate: contiguous ranges make exact per-layer packing
//! order-dependent (which range a node gets depends on stage order, which
//! depends on who participates, which depends on capacity…), while the mean
//! over the *real* total tensor bytes is stable, cheap, and within one
//! embedding/output amortization of exact. The per-layer weight vector is
//! still real bytes and is used for the utilization figures in the
//! explanation.
//!
//! # Score
//!
//! Layer shares are proportional to
//! `capacity_layers × (0.5 + 0.5 × decode_tps / max_decode_tps)`: memory is
//! the hard cap, compute tilts the split (a node half as fast as the fastest
//! takes 0.75× its memory share). Nodes without a [`ComputeProfile`] take
//! the neutral factor 1.0, so a cluster with no profiles at all degrades to
//! exactly M3's memory-only weighting.
//!
//! # Predicted time-per-token (units)
//!
//! The plan comparison metric is
//! `max_stage(layers_stage / decode_tps_node) × 1000 + Σ boundary RTT ms`.
//! `decode_tps` is the microbench throughput on the tiny registry model, so
//! the compute term is a *relative* cost in pseudo-milliseconds — consistent
//! across candidate plans for the same model, but not a latency promise and
//! never surfaced as one (honest-UX rule §1.6). Nodes without a profile are
//! costed at the slowest profiled node's rate (conservative); if no node is
//! profiled the compute term is zero and only boundary RTTs differ, which
//! makes "more nodes never help unless memory demands it" fall out
//! naturally.

use onebrain_proto::plan::{Assignment, Epoch, LayerRange, NodeId, Plan, Strategy};

use crate::dims::ModelDims;
use crate::profile::ComputeProfile;
use crate::{mb_ceil, mb_floor, PlanInput, PlannedPlacement, ScheduleError};

/// Fixed per-node compute/graph reserve subtracted from usable memory
/// (docs/scheduler-v1.md: 512 MiB; replaces M3's 85% ceiling).
pub const OVERHEAD_RESERVE_BYTES: u64 = 512 * 1024 * 1024;

/// RTT assumed for a node pair absent from the link table (milliseconds) —
/// a mid-range wired-LAN figure so unprobed links neither dominate nor
/// vanish from the boundary cost.
pub const DEFAULT_LINK_RTT_MS: f64 = 1.0;

/// Minimum relative predicted time-per-token improvement an additional node
/// must deliver to join a plan that is already memory-feasible (the
/// third-node rule, ≥ 5%).
pub const ADDED_NODE_TPT_GAIN: f64 = 0.05;

/// Stage order is found by exact permutation search up to this many
/// participating nodes; beyond it a greedy nearest-neighbor chain is built.
pub const EXACT_ORDER_MAX_NODES: usize = 8;

/// One node as the v1 planner sees it: identity, budgetable memory, and the
/// optional compute microbench result (None ⇒ memory-only weighting).
#[derive(Debug, Clone, PartialEq)]
pub struct NodeCaps {
    pub node: NodeId,
    /// Measured free memory minus the OS reserve — never total RAM.
    pub usable_memory_bytes: u64,
    /// From the node's `NodeStatus` / persisted profile; `None` until the
    /// node has benched.
    pub compute: Option<ComputeProfile>,
}

/// Measured round-trip time between two nodes, from the mesh prober.
/// Undirected: `(a, b)` covers `(b, a)`.
#[derive(Debug, Clone, PartialEq)]
pub struct LinkRtt {
    pub a: NodeId,
    pub b: NodeId,
    pub rtt_ms: f64,
}

/// Everything [`plan_v1`] needs to place one model load.
#[derive(Debug, Clone, PartialEq)]
pub struct PlanRequest {
    /// The node the user is on. Pinned to the last pipeline stage so it owns
    /// the output head and sampling stays local.
    pub head: NodeCaps,
    /// Paired workers eligible for this plan, in a stable order.
    pub workers: Vec<NodeCaps>,
    /// Model memory shape from the GGUF header ([`crate::model_dims`]).
    pub dims: ModelDims,
    /// Requested context length; drives the KV part of every budget.
    pub ctx_len: u32,
    /// `--nodes N` semantics, unchanged from M3: `None`/`Some(0)` =
    /// automatic, `Some(1)` forces solo, `Some(n ≥ 2)` forces exactly `n`.
    pub forced_nodes: Option<u32>,
    /// Measured per-link RTTs. Missing pairs default to
    /// [`DEFAULT_LINK_RTT_MS`].
    pub links: Vec<LinkRtt>,
}

/// M3 compatibility: an M3-shaped [`PlanInput`] carries only a total byte
/// size and a layer count, so the conversion synthesizes uniform dims with
/// an *unknown (zero) KV rate*, no compute profiles, and no link table.
/// Note the budget rule still changes (512 MiB reserve instead of the 85%
/// ceiling): sims that cap nodes below ~512 MiB must raise their `[debug]`
/// override before switching to [`plan_v1`].
impl From<&PlanInput> for PlanRequest {
    fn from(input: &PlanInput) -> PlanRequest {
        let to_caps = |n: &crate::NodeBudget| NodeCaps {
            node: n.node.clone(),
            usable_memory_bytes: n.usable_memory_bytes,
            compute: None,
        };
        PlanRequest {
            head: to_caps(&input.head),
            workers: input.workers.iter().map(to_caps).collect(),
            dims: ModelDims::uniform(input.model_bytes, input.n_layers),
            ctx_len: input.ctx_len,
            forced_nodes: input.forced_nodes,
            links: Vec::new(),
        }
    }
}

/// The bytes the v1 planner may fill on a node: usable memory minus the
/// fixed overhead reserve, clamped at zero.
pub fn node_budget(usable_memory_bytes: u64) -> u64 {
    usable_memory_bytes.saturating_sub(OVERHEAD_RESERVE_BYTES)
}

/// Whole layers a node can hold at `ctx_len` under the uniform-layer
/// approximation (module docs). Zero when the budget cannot cover even one
/// mean layer plus its KV.
pub fn node_layer_capacity(usable_memory_bytes: u64, dims: &ModelDims, ctx_len: u32) -> u64 {
    let cost = per_layer_cost(dims, ctx_len);
    (node_budget(usable_memory_bytes) as f64 / cost).floor() as u64
}

/// Mean weight + KV cost of one layer at `ctx_len`, as f64, never below 1.0
/// (division guard for degenerate synthetic models).
fn per_layer_cost(dims: &ModelDims, ctx_len: u32) -> f64 {
    ((dims.mean_weight_bytes_per_layer() + dims.kv_bytes_per_layer(ctx_len)) as f64).max(1.0)
}

/// One candidate node with its derived planning numbers.
struct Cand<'a> {
    caps: &'a NodeCaps,
    is_head: bool,
    budget: u64,
    /// Fractional layer capacity (budget / per-layer cost).
    cap_f: f64,
    /// Whole-layer capacity (floor of `cap_f`).
    cap_layers: u64,
    /// Compute tilt factor: `0.5 + 0.5 × decode/max_decode`, or 1.0 when
    /// unprofiled (module docs).
    factor: f64,
    /// decode_tps used in the tpt prediction: own, or the slowest profiled
    /// node's (conservative), or `None` when no node is profiled.
    tpt_decode: Option<f64>,
}

/// A fully evaluated candidate participant set.
struct Evaluated {
    /// Layers per candidate index (0 = does not participate).
    counts: Vec<u32>,
    /// Stage order: candidate indices of nodes with layers, stage 0 first.
    order: Vec<usize>,
    /// RTT of each pipeline boundary, in `order` (len = order.len() - 1).
    boundary_rtts: Vec<f64>,
    /// Predicted time-per-token comparison metric (module docs).
    tpt: f64,
}

/// Compute a placement per the M4 v1 rules. Same output contract as
/// [`crate::plan`] (epoch/model left for the caller to stamp).
pub fn plan_v1(input: &PlanRequest) -> Result<PlannedPlacement, ScheduleError> {
    let available = input.workers.len() as u32 + 1;
    let forced = match input.forced_nodes {
        Some(0) | None => None,
        Some(n) => Some(n),
    };
    if let Some(requested) = forced {
        if requested > available {
            return Err(ScheduleError::NotEnoughNodes {
                requested,
                available,
            });
        }
    }

    let dims = &input.dims;
    let kv_layer = dims.kv_bytes_per_layer(input.ctx_len);
    let required = dims.total_required_bytes(input.ctx_len);
    let head_budget = node_budget(input.head.usable_memory_bytes);

    // Auto-solo, now against the real KV budget: weights + KV for every
    // layer at the requested context must fit the head's budget.
    let solo_fits = required <= head_budget;
    match forced {
        None if solo_fits => return Ok(solo_placement(input, head_budget, required, false)),
        Some(1) => {
            if solo_fits {
                return Ok(solo_placement(input, head_budget, required, true));
            }
            return Err(ScheduleError::DoesNotFit {
                required_mb: mb_ceil(required),
                available_mb: mb_floor(head_budget),
            });
        }
        _ => {}
    }

    distributed_v1(input, forced, head_budget, required, kv_layer, solo_fits)
}

fn solo_placement(
    input: &PlanRequest,
    head_budget: u64,
    required: u64,
    forced: bool,
) -> PlannedPlacement {
    let dims = &input.dims;
    let kv_total = dims.n_layers as u64 * dims.kv_bytes_per_layer(input.ctx_len);
    let why = if forced {
        "--nodes 1 forced solo".to_string()
    } else {
        "auto-solo (§1.4): the model fits on this node, so distribution is not engaged".to_string()
    };
    let explanation = format!(
        "Solo on head '{}': weights {} MB + KV cache {} MB at ctx {} ({} MB together) fit \
         within its {} MB budget ({} MB usable minus the {} MB overhead reserve). {}.",
        input.head.node.0,
        mb_ceil(dims.total_weight_bytes),
        mb_ceil(kv_total),
        input.ctx_len,
        mb_ceil(required),
        mb_floor(head_budget),
        mb_floor(input.head.usable_memory_bytes),
        mb_ceil(OVERHEAD_RESERVE_BYTES),
        why
    );
    tracing::debug!(
        node = %input.head.node.0,
        required_mb = mb_ceil(required),
        budget_mb = mb_floor(head_budget),
        forced,
        "planned solo placement (v1)"
    );
    PlannedPlacement {
        plan: Plan {
            epoch: Epoch(0),
            model: String::new(),
            strategy: Strategy::Solo,
            assignments: vec![Assignment {
                node: input.head.node.clone(),
                layers: LayerRange {
                    start: 0,
                    end: dims.n_layers,
                },
                stage: 0,
            }],
            ctx_len: input.ctx_len,
        },
        tensor_split: vec![1.0],
        explanation,
    }
}

/// Undirected RTT lookup with the documented default for unprobed pairs.
fn link_rtt(links: &[LinkRtt], a: &NodeId, b: &NodeId) -> f64 {
    links
        .iter()
        .find(|l| (&l.a == a && &l.b == b) || (&l.a == b && &l.b == a))
        .map(|l| l.rtt_ms)
        .unwrap_or(DEFAULT_LINK_RTT_MS)
}

/// Largest-remainder apportionment of `n_layers` over the candidates,
/// proportional to `cap_f × factor`, clamped to each node's whole-layer
/// capacity. `None` when the set cannot hold the model at all.
fn apportion(cands: &[&Cand], n_layers: u32) -> Option<Vec<u32>> {
    let scores: Vec<f64> = cands.iter().map(|c| c.cap_f * c.factor).collect();
    let total_score: f64 = scores.iter().sum();
    let total_cap: u64 = cands.iter().map(|c| c.cap_layers).sum();
    if total_score <= 0.0 || total_cap < n_layers as u64 {
        return None;
    }
    let quotas: Vec<f64> = scores
        .iter()
        .map(|s| n_layers as f64 * s / total_score)
        .collect();
    let mut counts: Vec<u32> = quotas
        .iter()
        .zip(cands)
        .map(|(q, c)| (q.floor() as u64).min(c.cap_layers) as u32)
        .collect();
    let mut leftover = n_layers - counts.iter().sum::<u32>();
    // Hand out the rest one layer at a time to the node with the largest
    // unmet quota that still has capacity; ties break on score, then index.
    while leftover > 0 {
        let mut best: Option<usize> = None;
        for i in 0..cands.len() {
            if (counts[i] as u64) >= cands[i].cap_layers {
                continue;
            }
            let better = match best {
                None => true,
                Some(b) => {
                    let key_i = quotas[i] - counts[i] as f64;
                    let key_b = quotas[b] - counts[b] as f64;
                    key_i > key_b || (key_i == key_b && scores[i] > scores[b])
                }
            };
            if better {
                best = Some(i);
            }
        }
        let i = best.expect("total capacity >= n_layers guarantees a node with slack");
        counts[i] += 1;
        leftover -= 1;
    }
    Some(counts)
}

/// Choose the stage order: nodes holding layers, head pinned to the last
/// stage (sampling locality), arranged to minimize the summed RTT across
/// consecutive stages. Exact permutation search up to
/// [`EXACT_ORDER_MAX_NODES`] participants, greedy nearest-neighbor beyond.
fn order_stages(cands: &[&Cand], counts: &[u32], links: &[LinkRtt]) -> (Vec<usize>, Vec<f64>) {
    let active: Vec<usize> = (0..cands.len()).filter(|&i| counts[i] > 0).collect();
    if active.len() <= 1 {
        return (active, Vec::new());
    }
    let tail: Option<usize> = active.iter().copied().find(|&i| cands[i].is_head);
    let movable: Vec<usize> = active
        .iter()
        .copied()
        .filter(|&i| Some(i) != tail)
        .collect();

    let chain_cost = |chain: &[usize]| -> f64 {
        chain
            .windows(2)
            .map(|w| link_rtt(links, &cands[w[0]].caps.node, &cands[w[1]].caps.node))
            .sum()
    };
    let finish = |chain: Vec<usize>| -> (Vec<usize>, Vec<f64>) {
        let rtts = chain
            .windows(2)
            .map(|w| link_rtt(links, &cands[w[0]].caps.node, &cands[w[1]].caps.node))
            .collect();
        (chain, rtts)
    };

    if active.len() <= EXACT_ORDER_MAX_NODES {
        let mut best: Option<(f64, Vec<usize>)> = None;
        for perm in permutations(&movable) {
            let mut chain = perm;
            if let Some(t) = tail {
                chain.push(t);
            }
            let cost = chain_cost(&chain);
            // Strict `<` keeps the first (lexicographic) minimum:
            // deterministic output for equal-cost arrangements.
            let improves = match &best {
                None => true,
                Some((c, _)) => cost < *c,
            };
            if improves {
                best = Some((cost, chain));
            }
        }
        let (_, chain) = best.expect("at least one permutation exists");
        return finish(chain);
    }

    // Greedy: grow the chain from the tail end, always attaching the
    // remaining node with the cheapest link to the current front.
    let mut chain: Vec<usize> = Vec::with_capacity(active.len());
    let mut remaining = movable;
    let anchor = tail.unwrap_or_else(|| remaining.pop().expect("active.len() > 1"));
    chain.push(anchor);
    while !remaining.is_empty() {
        let front = chain[0];
        let (pos, _) = remaining
            .iter()
            .enumerate()
            .map(|(p, &i)| {
                (
                    p,
                    link_rtt(links, &cands[i].caps.node, &cands[front].caps.node),
                )
            })
            .min_by(|a, b| a.1.partial_cmp(&b.1).expect("rtt is never NaN"))
            .expect("remaining is non-empty");
        let node = remaining.remove(pos);
        chain.insert(0, node);
    }
    finish(chain)
}

/// All permutations of `items`, in lexicographic order over the input.
fn permutations(items: &[usize]) -> Vec<Vec<usize>> {
    if items.is_empty() {
        return vec![Vec::new()];
    }
    let mut out = Vec::new();
    for (i, &x) in items.iter().enumerate() {
        let mut rest: Vec<usize> = items.to_vec();
        rest.remove(i);
        for mut p in permutations(&rest) {
            p.insert(0, x);
            out.push(p);
        }
    }
    out
}

/// Apportion + order + predict for one candidate participant set.
/// `None` when the set cannot hold the model.
fn evaluate(cands: &[&Cand], n_layers: u32, links: &[LinkRtt]) -> Option<Evaluated> {
    let counts = apportion(cands, n_layers)?;
    let (order, boundary_rtts) = order_stages(cands, &counts, links);
    let compute_ms = order
        .iter()
        .map(|&i| match cands[i].tpt_decode {
            Some(tps) => counts[i] as f64 / tps * 1000.0,
            None => 0.0,
        })
        .fold(0.0f64, f64::max);
    let tpt = compute_ms + boundary_rtts.iter().sum::<f64>();
    Some(Evaluated {
        counts,
        order,
        boundary_rtts,
        tpt,
    })
}

fn distributed_v1(
    input: &PlanRequest,
    forced: Option<u32>,
    head_budget: u64,
    required: u64,
    kv_layer: u64,
    solo_fits: bool,
) -> Result<PlannedPlacement, ScheduleError> {
    let dims = &input.dims;
    let n_layers = dims.n_layers;
    let cost = per_layer_cost(dims, input.ctx_len);

    // Compute factors over every candidate (head + workers).
    let all_caps: Vec<&NodeCaps> = input
        .workers
        .iter()
        .chain(std::iter::once(&input.head))
        .collect();
    let profiled: Vec<f64> = all_caps
        .iter()
        .filter_map(|c| c.compute.map(|p| p.decode_tps))
        .filter(|d| *d > 0.0)
        .collect();
    let max_decode = profiled.iter().copied().fold(f64::NAN, f64::max);
    let min_decode = profiled.iter().copied().fold(f64::NAN, f64::min);

    // Candidates: workers in their stable order, head last.
    let cands: Vec<Cand> = all_caps
        .iter()
        .enumerate()
        .map(|(i, caps)| {
            let budget = node_budget(caps.usable_memory_bytes);
            let cap_f = budget as f64 / cost;
            let decode = caps.compute.map(|p| p.decode_tps).filter(|d| *d > 0.0);
            let factor = match decode {
                Some(d) if max_decode.is_finite() => 0.5 + 0.5 * d / max_decode,
                // Unprofiled: neutral factor — no tilt for or against. With
                // no profiles anywhere this is memory-only weighting (M3).
                _ => 1.0,
            };
            let tpt_decode = decode.or(if min_decode.is_finite() {
                Some(min_decode)
            } else {
                None
            });
            Cand {
                caps,
                is_head: i == all_caps.len() - 1,
                budget,
                cap_f,
                cap_layers: (cap_f.floor().max(0.0) as u64).min(u64::from(u32::MAX)),
                factor,
                tpt_decode,
            }
        })
        .collect();
    let head_idx = cands.len() - 1;
    let pooled_budget: u64 = cands.iter().map(|c| c.budget).sum();

    // ---- Participant selection ------------------------------------------
    // Worker candidate order: by memory-and-compute score, descending
    // (ties: earlier stable order).
    let mut ranked_workers: Vec<usize> = (0..head_idx).collect();
    ranked_workers.sort_by(|&a, &b| {
        let (sa, sb) = (
            cands[a].cap_f * cands[a].factor,
            cands[b].cap_f * cands[b].factor,
        );
        sb.partial_cmp(&sa)
            .expect("scores are never NaN")
            .then(a.cmp(&b))
    });

    let mut selection_notes: Vec<String> = Vec::new();
    let mut selected: Vec<usize> = Vec::new(); // worker candidate indices

    let participants = |selected: &[usize]| -> Vec<&Cand> {
        let mut set: Vec<usize> = selected.to_vec();
        set.sort_unstable();
        set.push(head_idx);
        set.iter().map(|&i| &cands[i]).collect()
    };

    match forced {
        Some(n) => {
            // Forced: exactly the n-1 best-scoring workers; no 5% rule.
            selected = ranked_workers
                .iter()
                .copied()
                .take(n as usize - 1)
                .collect();
            selection_notes.push(format!(
                "--nodes {n} forced: the {} highest-scoring worker(s) joined without the \
                 time-per-token test",
                n - 1
            ));
        }
        None => {
            for &wi in &ranked_workers {
                let cand = &cands[wi];
                if cand.cap_layers == 0 {
                    selection_notes.push(format!(
                        "Node '{}': excluded — it cannot hold even one layer at ctx {} \
                         ({} MB budget vs ~{} MB per layer with KV)",
                        cand.caps.node.0,
                        input.ctx_len,
                        mb_floor(cand.budget),
                        mb_ceil(cost as u64),
                    ));
                    continue;
                }
                let current = evaluate(&participants(&selected), n_layers, &input.links);
                match current {
                    None => {
                        let have: u64 = participants(&selected).iter().map(|c| c.cap_layers).sum();
                        selection_notes.push(format!(
                            "Node '{}': included — needed for memory (capacity so far {} of \
                             {} layers)",
                            cand.caps.node.0, have, n_layers
                        ));
                        selected.push(wi);
                    }
                    Some(cur) => {
                        let mut with_set = selected.clone();
                        with_set.push(wi);
                        let Some(with) = evaluate(&participants(&with_set), n_layers, &input.links)
                        else {
                            continue;
                        };
                        let gain = (cur.tpt - with.tpt) / cur.tpt.max(1e-9);
                        if gain >= ADDED_NODE_TPT_GAIN {
                            selection_notes.push(format!(
                                "Node '{}': included — predicted time-per-token improves \
                                 {:.1}% ({:.1} -> {:.1}, ≥ {:.0}% threshold)",
                                cand.caps.node.0,
                                gain * 100.0,
                                cur.tpt,
                                with.tpt,
                                ADDED_NODE_TPT_GAIN * 100.0
                            ));
                            selected.push(wi);
                        } else if gain > 0.0 {
                            selection_notes.push(format!(
                                "Node '{}': excluded — predicted time-per-token improves only \
                                 {:.1}% ({:.1} -> {:.1}), below the {:.0}% threshold",
                                cand.caps.node.0,
                                gain * 100.0,
                                cur.tpt,
                                with.tpt,
                                ADDED_NODE_TPT_GAIN * 100.0
                            ));
                        } else {
                            selection_notes.push(format!(
                                "Node '{}': excluded — adding it would not improve predicted \
                                 time-per-token ({:.1} -> {:.1}; every extra boundary costs \
                                 one RTT per token)",
                                cand.caps.node.0, cur.tpt, with.tpt
                            ));
                        }
                    }
                }
            }
        }
    }

    let final_set = participants(&selected);
    let Some(eval) = evaluate(&final_set, n_layers, &input.links) else {
        return Err(ScheduleError::DoesNotFit {
            required_mb: mb_ceil(required),
            available_mb: mb_floor(pooled_budget),
        });
    };

    // ---- Assemble assignments in stage order ----------------------------
    let mut assignments = Vec::new();
    let mut tensor_split = Vec::new();
    let mut per_node_prose = Vec::new();
    let mut binding: Option<(usize, u64)> = None;
    let mut cursor: u32 = 0;
    for (stage, &pi) in eval.order.iter().enumerate() {
        let cand = final_set[pi];
        let layers = eval.counts[pi];
        let range = LayerRange {
            start: cursor,
            end: cursor + layers,
        };
        cursor += layers;
        // Real bytes of the assigned range (not the uniform approximation).
        let weight_bytes: u64 = dims.weight_bytes_per_layer
            [range.start as usize..range.end as usize]
            .iter()
            .sum();
        let kv_bytes = layers as u64 * kv_layer;
        let util_pct = if cand.budget == 0 {
            u64::MAX
        } else {
            ((weight_bytes + kv_bytes) as u128 * 100 / cand.budget as u128) as u64
        };
        let role = if cand.is_head { "head" } else { "worker" };
        per_node_prose.push(format!(
            "  stage {} — {} '{}': layers {}..{} ({} layers, ~{} MB weights + {} MB KV, \
             {}% of its {} MB budget)",
            stage,
            role,
            cand.caps.node.0,
            range.start,
            range.end,
            layers,
            mb_ceil(weight_bytes),
            mb_ceil(kv_bytes),
            util_pct,
            mb_floor(cand.budget),
        ));
        let is_new_binding = match binding {
            None => true,
            Some((_, best)) => util_pct > best,
        };
        if is_new_binding {
            binding = Some((assignments.len(), util_pct));
        }
        assignments.push(Assignment {
            node: cand.caps.node.clone(),
            layers: range,
            stage: stage as u32,
        });
        tensor_split.push(layers as f32 / n_layers as f32);
    }

    // Participants that rounded to zero layers.
    let dropped: Vec<&Cand> = (0..final_set.len())
        .filter(|&i| eval.counts[i] == 0)
        .map(|i| final_set[i])
        .collect();

    // ---- Explanation ----------------------------------------------------
    let why = match forced {
        Some(n) if solo_fits => {
            format!("distribution forced by --nodes {n} (the model would fit the head alone)")
        }
        Some(n) => format!(
            "distribution engaged: --nodes {n}, and weights + KV {} MB exceed the head's \
             {} MB budget",
            mb_ceil(required),
            mb_floor(head_budget)
        ),
        None => format!(
            "distribution engaged: weights + KV at ctx {} need {} MB, exceeding the head's \
             {} MB budget ({} MB usable minus the {} MB overhead reserve)",
            input.ctx_len,
            mb_ceil(required),
            mb_floor(head_budget),
            mb_floor(input.head.usable_memory_bytes),
            mb_ceil(OVERHEAD_RESERVE_BYTES)
        ),
    };
    let mut explanation = format!(
        "Pipeline-parallel across {} nodes ({why}):\n{}",
        assignments.len(),
        per_node_prose.join("\n")
    );
    explanation.push_str(&format!(
        "\nKV budget: {} MB per layer at ctx {} ({} bytes/token/layer); overhead reserve \
         {} MB per node.",
        mb_ceil(kv_layer),
        input.ctx_len,
        dims.kv_bytes_per_layer_per_ctx_token,
        mb_ceil(OVERHEAD_RESERVE_BYTES),
    ));
    if !eval.boundary_rtts.is_empty() {
        let hops: Vec<String> = eval
            .order
            .windows(2)
            .zip(&eval.boundary_rtts)
            .map(|(pair, rtt)| {
                format!(
                    "'{}' -> '{}' {:.1} ms",
                    final_set[pair[0]].caps.node.0, final_set[pair[1]].caps.node.0, rtt
                )
            })
            .collect();
        explanation.push_str(&format!(
            "\nBoundaries (stage order minimizes summed RTT; head pinned last): {} \
             (total {:.1} ms/token).",
            hops.join(", "),
            eval.boundary_rtts.iter().sum::<f64>()
        ));
    }
    explanation.push_str(&format!(
        "\nPredicted decode cost: {:.1} (relative: max stage layers/decode_tps × 1000 + \
         boundary RTTs; used only to compare plans, not a latency promise).",
        eval.tpt
    ));
    if let Some((idx, pct)) = binding {
        explanation.push_str(&format!(
            "\nBinding constraint: node '{}' at {}% of its memory budget.",
            assignments[idx].node.0, pct
        ));
    }
    for note in &selection_notes {
        explanation.push('\n');
        explanation.push_str(note);
    }
    for d in &dropped {
        explanation.push_str(&format!(
            "\nDropped: node '{}' ({} MB budget) rounded to 0 layers and left the plan.",
            d.caps.node.0,
            mb_floor(d.budget)
        ));
    }

    tracing::debug!(
        nodes = assignments.len(),
        dropped = dropped.len(),
        required_mb = mb_ceil(required),
        pooled_mb = mb_floor(pooled_budget),
        tpt = eval.tpt,
        "planned pipeline-parallel placement (v1)"
    );

    Ok(PlannedPlacement {
        plan: Plan {
            epoch: Epoch(0),
            model: String::new(),
            strategy: Strategy::PipelineParallel,
            assignments,
            ctx_len: input.ctx_len,
        },
        tensor_split,
        explanation,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::NodeBudget;

    const MIB: u64 = 1 << 20;
    const GIB: u64 = 1 << 30;

    /// Uniform-weight dims: `n_layers` layers of `weight_mib` each, KV
    /// growing at `kv_rate` bytes per layer per context token.
    fn dims(n_layers: u32, weight_mib: u64, kv_rate: u64) -> ModelDims {
        ModelDims {
            n_layers,
            kv_bytes_per_layer_per_ctx_token: kv_rate,
            weight_bytes_per_layer: vec![weight_mib * MIB; n_layers as usize],
            total_weight_bytes: weight_mib * MIB * n_layers as u64,
        }
    }

    fn caps(id: &str, usable_mib: u64, decode_tps: Option<f64>) -> NodeCaps {
        NodeCaps {
            node: NodeId(id.into()),
            usable_memory_bytes: usable_mib * MIB,
            compute: decode_tps.map(|d| ComputeProfile {
                prefill_tps: d * 8.0,
                decode_tps: d,
            }),
        }
    }

    fn link(a: &str, b: &str, rtt_ms: f64) -> LinkRtt {
        LinkRtt {
            a: NodeId(a.into()),
            b: NodeId(b.into()),
            rtt_ms,
        }
    }

    fn assert_contiguous(placed: &PlannedPlacement, n_layers: u32) {
        let a = &placed.plan.assignments;
        assert_eq!(a[0].layers.start, 0);
        for pair in a.windows(2) {
            assert_eq!(pair[0].layers.end, pair[1].layers.start);
            assert!(!pair[0].layers.is_empty());
        }
        assert_eq!(a.last().unwrap().layers.end, n_layers);
        let sum: f32 = placed.tensor_split.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn auto_solo_uses_the_real_kv_budget() {
        // 32 × 100 MiB weights; KV rate 16384 B/token/layer. Head usable
        // 4600 MiB → budget 4088 MiB. At ctx 512 (KV 8 MiB/layer) the model
        // needs 3456 MiB → solo. At ctx 2048 (KV 32 MiB/layer) it needs
        // 4224 MiB → the same head must distribute: ctx moved the plan.
        let d = dims(32, 100, 16384);
        let solo = plan_v1(&PlanRequest {
            head: caps("head", 4600, None),
            workers: vec![caps("w1", 4600, None)],
            dims: d.clone(),
            ctx_len: 512,
            forced_nodes: None,
            links: vec![],
        })
        .unwrap();
        assert_eq!(solo.plan.strategy, Strategy::Solo);
        assert!(solo.explanation.contains("auto-solo"));
        assert!(solo.explanation.contains("KV cache"));

        let split = plan_v1(&PlanRequest {
            head: caps("head", 4600, None),
            workers: vec![caps("w1", 4600, None)],
            dims: d,
            ctx_len: 2048,
            forced_nodes: None,
            links: vec![],
        })
        .unwrap();
        assert_eq!(split.plan.strategy, Strategy::PipelineParallel);
        assert_eq!(split.plan.assignments.len(), 2);
    }

    #[test]
    fn compute_tilt_shifts_layers_within_one_of_prediction() {
        // Equal memory, worker decodes 2× the head: shares ∝
        // 0.5 + 0.5 × decode/max ⇒ fast 1.0 vs slow 0.75. Prediction:
        // slow = 32 × 0.75 / 1.75 ≈ 13.71 layers, fast ≈ 18.29.
        let placed = plan_v1(&PlanRequest {
            head: caps("slow", 3072, Some(10.0)),
            workers: vec![caps("fast", 3072, Some(20.0))],
            dims: dims(32, 100, 4096),
            ctx_len: 4096,
            forced_nodes: None,
            links: vec![],
        })
        .unwrap();
        assert_eq!(placed.plan.strategy, Strategy::PipelineParallel);
        let a = &placed.plan.assignments;
        assert_eq!(a.len(), 2);
        let fast = a.iter().find(|x| x.node.0 == "fast").unwrap();
        let slow = a.iter().find(|x| x.node.0 == "slow").unwrap();
        assert!(
            fast.layers.len() > slow.layers.len(),
            "the 2× node must take more layers: {a:?}"
        );
        let predicted_slow = 32.0 * 0.75 / 1.75;
        assert!(
            (slow.layers.len() as f64 - predicted_slow).abs() <= 1.0,
            "slow node got {} layers, predicted {predicted_slow:.2}",
            slow.layers.len()
        );
        assert_contiguous(&placed, 32);
    }

    #[test]
    fn stage_order_minimizes_boundary_rtts() {
        // Three forced nodes, distinct RTTs: w1↔head 0.2 ms, w2↔head 5 ms,
        // w1↔w2 1 ms. Head is pinned last, so the chains on offer are
        // [w1, w2, head] = 6.0 ms and [w2, w1, head] = 1.2 ms.
        let placed = plan_v1(&PlanRequest {
            head: caps("head", 3072, None),
            workers: vec![caps("w1", 3072, None), caps("w2", 3072, None)],
            dims: dims(32, 100, 16384),
            ctx_len: 2048,
            forced_nodes: Some(3),
            links: vec![
                link("w1", "head", 0.2),
                link("w2", "head", 5.0),
                link("w1", "w2", 1.0),
            ],
        })
        .unwrap();
        let a = &placed.plan.assignments;
        assert_eq!(a.len(), 3);
        assert_eq!(a[0].node.0, "w2");
        assert_eq!(a[1].node.0, "w1");
        assert_eq!(a[2].node.0, "head");
        assert_eq!(a[2].stage, 2);
        assert_contiguous(&placed, 32);
        // The explanation names the per-boundary RTTs of the chosen chain.
        assert!(
            placed.explanation.contains("'w2' -> 'w1' 1.0 ms"),
            "{}",
            placed.explanation
        );
        assert!(
            placed.explanation.contains("'w1' -> 'head' 0.2 ms"),
            "{}",
            placed.explanation
        );
    }

    #[test]
    fn third_node_excluded_when_gain_below_threshold() {
        // Two 3 GiB nodes fit the model; every link costs 45 ms. Adding w2
        // trims the max stage from 16 to 11 layers (160 → 110 relative ms
        // at 100 tok/s) but adds a 45 ms boundary: 205 → 200, a 2.4% gain,
        // below the 5% bar ⇒ stay 2-node and say why.
        let placed = plan_v1(&PlanRequest {
            head: caps("head", 3072, Some(100.0)),
            workers: vec![caps("w1", 3072, Some(100.0)), caps("w2", 3072, Some(100.0))],
            dims: dims(32, 100, 16384),
            ctx_len: 2048,
            forced_nodes: None,
            links: vec![
                link("w1", "head", 45.0),
                link("w2", "head", 45.0),
                link("w1", "w2", 45.0),
            ],
        })
        .unwrap();
        let a = &placed.plan.assignments;
        assert_eq!(a.len(), 2, "{}", placed.explanation);
        assert!(a.iter().all(|x| x.node.0 != "w2"));
        assert!(
            placed.explanation.contains("Node 'w2': excluded"),
            "{}",
            placed.explanation
        );
        assert!(
            placed.explanation.contains("below the 5% threshold"),
            "{}",
            placed.explanation
        );
        // And the winning boundary is named with its RTT.
        assert!(
            placed.explanation.contains("45.0 ms"),
            "{}",
            placed.explanation
        );
    }

    #[test]
    fn third_node_included_when_two_cannot_hold_the_model() {
        // 2200 MiB usable → 1688 MiB budget → 12-layer capacity each; two
        // nodes hold 24 < 32 layers, three hold 36 ⇒ the third joins for
        // memory and the explanation says so.
        let placed = plan_v1(&PlanRequest {
            head: caps("head", 2200, Some(100.0)),
            workers: vec![caps("w1", 2200, Some(100.0)), caps("w2", 2200, Some(100.0))],
            dims: dims(32, 100, 16384),
            ctx_len: 2048,
            forced_nodes: None,
            links: vec![],
        })
        .unwrap();
        assert_eq!(placed.plan.assignments.len(), 3, "{}", placed.explanation);
        assert!(
            placed.explanation.contains("needed for memory"),
            "{}",
            placed.explanation
        );
        assert_contiguous(&placed, 32);
    }

    #[test]
    fn ctx_growth_shrinks_capacity_and_forces_a_wider_split() {
        // Same caps at 2k vs 16k ctx: KV per layer grows 32 → 256 MiB, so
        // per-node capacity drops 30 → 11 layers; the 2-node plan becomes
        // infeasible and a third node must join, leaving fewer layers per
        // node (11 max vs 16).
        let d = dims(32, 100, 16384);
        assert_eq!(node_layer_capacity(4600 * MIB, &d, 2048), 30);
        assert_eq!(node_layer_capacity(4600 * MIB, &d, 16384), 11);

        let req = |ctx| PlanRequest {
            head: caps("head", 4600, None),
            workers: vec![caps("w1", 4600, None), caps("w2", 4600, None)],
            dims: d.clone(),
            ctx_len: ctx,
            forced_nodes: None,
            links: vec![],
        };
        let small = plan_v1(&req(2048)).unwrap();
        assert_eq!(small.plan.assignments.len(), 2, "{}", small.explanation);
        let max_small = small
            .plan
            .assignments
            .iter()
            .map(|a| a.layers.len())
            .max()
            .unwrap();
        assert_eq!(max_small, 16);

        let big = plan_v1(&req(16384)).unwrap();
        assert_eq!(big.plan.assignments.len(), 3, "{}", big.explanation);
        let max_big = big
            .plan
            .assignments
            .iter()
            .map(|a| a.layers.len())
            .max()
            .unwrap();
        assert!(
            max_big < max_small,
            "16k ctx must leave fewer layers per node ({max_big} vs {max_small})"
        );
        assert_contiguous(&big, 32);
    }

    #[test]
    fn node_below_one_layer_is_excluded_with_a_reason() {
        // 600 MiB usable → 88 MiB budget < one 132 MiB layer ⇒ the node
        // cannot participate at all (distinct from rounding to 0 layers).
        let placed = plan_v1(&PlanRequest {
            head: caps("head", 3072, None),
            workers: vec![caps("w1", 3072, None), caps("tiny", 600, None)],
            dims: dims(32, 100, 16384),
            ctx_len: 2048,
            forced_nodes: None,
            links: vec![],
        })
        .unwrap();
        assert_eq!(placed.plan.assignments.len(), 2);
        assert!(placed.plan.assignments.iter().all(|a| a.node.0 != "tiny"));
        assert!(
            placed.explanation.contains("cannot hold even one layer"),
            "{}",
            placed.explanation
        );
    }

    #[test]
    fn m3_shaped_input_converts_and_plans_proportionally() {
        // The M3 compatibility conversion: uniform dims, zero KV rate, no
        // profiles, no links. The forced 2-node split lands on the same
        // 10/20 proportions as M3's memory-proportional rule.
        let input = crate::PlanInput {
            head: NodeBudget {
                node: NodeId("head".into()),
                usable_memory_bytes: 12 * GIB,
            },
            workers: vec![NodeBudget {
                node: NodeId("w1".into()),
                usable_memory_bytes: 6 * GIB,
            }],
            model_bytes: 4 * GIB,
            n_layers: 30,
            ctx_len: 4096,
            forced_nodes: Some(2),
        };
        let req = PlanRequest::from(&input);
        assert_eq!(req.dims.n_layers, 30);
        assert_eq!(req.dims.total_weight_bytes, 4 * GIB);
        assert_eq!(req.dims.kv_bytes_per_layer_per_ctx_token, 0);

        let placed = plan_v1(&req).unwrap();
        assert_eq!(placed.plan.strategy, Strategy::PipelineParallel);
        let a = &placed.plan.assignments;
        assert_eq!(a.len(), 2);
        assert_eq!(a[0].node.0, "w1");
        assert_eq!(a[0].layers, LayerRange { start: 0, end: 10 });
        assert_eq!(a[1].node.0, "head");
        assert_eq!(a[1].layers, LayerRange { start: 10, end: 30 });
        assert!(placed.explanation.contains("forced by --nodes 2"));
        assert_contiguous(&placed, 30);
    }

    #[test]
    fn v1_errors_carry_the_numbers() {
        // Pooled budgets too small: 1 GiB usable each → 512 MiB budgets,
        // 3200 MiB of weights.
        let too_small = plan_v1(&PlanRequest {
            head: caps("head", 1024, None),
            workers: vec![caps("w1", 1024, None)],
            dims: dims(32, 100, 0),
            ctx_len: 2048,
            forced_nodes: None,
            links: vec![],
        });
        match too_small.unwrap_err() {
            ScheduleError::DoesNotFit {
                required_mb,
                available_mb,
            } => {
                assert_eq!(required_mb, 3200);
                assert_eq!(available_mb, 1024);
            }
            other => panic!("expected DoesNotFit, got {other:?}"),
        }

        let not_enough = plan_v1(&PlanRequest {
            head: caps("head", 4096, None),
            workers: vec![caps("w1", 4096, None)],
            dims: dims(32, 100, 0),
            ctx_len: 2048,
            forced_nodes: Some(4),
            links: vec![],
        });
        match not_enough.unwrap_err() {
            ScheduleError::NotEnoughNodes {
                requested,
                available,
            } => {
                assert_eq!(requested, 4);
                assert_eq!(available, 2);
            }
            other => panic!("expected NotEnoughNodes, got {other:?}"),
        }
    }
}
