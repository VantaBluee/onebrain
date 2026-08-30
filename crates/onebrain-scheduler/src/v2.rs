//! Scheduler v2-lite (M7, binding contract `docs/perf.md` §7 + §8):
//! cost-model candidate search in the prima.cpp spirit, kept enumerable.
//!
//! What changes relative to [`plan_v1`](crate::plan_v1) — and, deliberately,
//! what does not:
//!
//! - **Inclusion rules are unchanged** (contract §7): the same greedy
//!   worker admission — draining nodes last, "needed for memory" beats
//!   everything, otherwise the ≥5% predicted time-per-token gain — decides
//!   *who* participates. v2 only searches harder over *how* the selected
//!   set is used.
//! - **Candidate search**: for the participant set, every layer-share tilt
//!   in the family {memory-proportional, the current 0.5+0.5 compute tilt,
//!   full compute-proportional, each with the slowest node underweighted
//!   one layer-quantum} × every exact stage order (≤
//!   [`EXACT_ORDER_MAX_NODES`] participants, as today; greedy chain
//!   beyond) is evaluated; minimum predicted tpt wins. `--explain` reports
//!   the number of candidates considered and the winner's rationale.
//! - **The decode tpt metric stays primary**, exactly as documented in
//!   [`v1`](crate::v1)'s module docs
//!   (`max_stage(layer_units / decode_tps) × 1000 + Σ boundary RTT ms`).
//!   The new prefill transfer term — per stage boundary
//!   `RTT + (4·n_embd·n_ubatch)/measured_bandwidth` — is a SECONDARY key:
//!   it breaks exact ties between candidates the decode metric cannot
//!   separate (typically stage orders over symmetric nodes). Links without
//!   a measured [`LinkRtt::bandwidth_mbps`] contribute no transfer time, so
//!   an RTT-only request reproduces today's decisions bit-for-bit.
//! - **MoE compute scaling** (§8): stage compute divides
//!   [`ModelDims::active_compute_units`] — not the raw layer count — by
//!   `decode_tps`, so a MoE layer costs only the weight fraction a token
//!   actually touches. Dense models reduce to the exact v1 numbers; memory
//!   math is untouched (all experts are resident).
//! - **Pipeline reserve** (§3): distributed participants budget
//!   `usable − (OVERHEAD_RESERVE_BYTES + 4·(4·n_embd·n_ubatch))` — pipeline
//!   parallelism (patch 0003) keeps up to 4 in-flight compute-buffer
//!   copies per backend. Solo plans have nothing to overlap and keep the
//!   plain v1 budget.
//!
//! # Determinism
//!
//! The search is fully deterministic: the tilt family is enumerated in a
//! fixed order (base tilts, then underweighted variants), stage orders in
//! lexicographic permutation order, and a candidate replaces the incumbent
//! only when strictly better on `(decode tpt, prefill cost)` — equal-cost
//! candidates keep the earliest, so equal inputs give equal plans.

use onebrain_proto::plan::{Assignment, Epoch, LayerRange, Plan, Strategy};

use crate::dims::ModelDims;
use crate::v1::{
    apportion, build_cands, link_bandwidth, link_rtt, node_budget, order_stages, participants,
    per_layer_cost, permutations, select_workers, solo_placement, Cand, LinkRtt, NodeCaps,
    PlanRequest, DEFAULT_N_UBATCH, EXACT_ORDER_MAX_NODES, OVERHEAD_RESERVE_BYTES,
};
use crate::{mb_ceil, mb_floor, PlannedPlacement, ScheduleError};

/// The effective microbatch size: [`PlanRequest::n_ubatch`], with 0 mapped
/// to [`DEFAULT_N_UBATCH`] (mirrors the engine's zero-means-default
/// convention).
pub fn effective_n_ubatch(n_ubatch: u32) -> u32 {
    if n_ubatch == 0 {
        DEFAULT_N_UBATCH
    } else {
        n_ubatch
    }
}

/// Wall-clock milliseconds to move one pipeline-boundary activation copy
/// (`4·n_embd·n_ubatch` bytes — f32 activations for one microbatch,
/// docs/perf.md §7) over a link of `bandwidth_mbps` megabits/second.
/// Returns 0 for an unmeasured/nonsense bandwidth or an unknown embedding
/// width (`n_embd == 0`), which keeps RTT-only inputs on today's behavior.
/// `n_ubatch == 0` means [`DEFAULT_N_UBATCH`].
pub fn boundary_transfer_ms(n_embd: u64, n_ubatch: u32, bandwidth_mbps: f64) -> f64 {
    if bandwidth_mbps <= 0.0 {
        return 0.0;
    }
    let bytes = 4.0 * n_embd as f64 * effective_n_ubatch(n_ubatch) as f64;
    // 1 Mbps = 10^6 bits/s, so ms = bits / (mbps × 1000).
    bytes * 8.0 / (bandwidth_mbps * 1000.0)
}

/// The bytes the v2 planner may fill on a *distributed* participant:
/// usable memory minus the fixed overhead reserve minus the
/// pipeline-parallel compute-buffer copies patch 0003 enables — up to 4
/// in-flight copies of the `4·n_embd·n_ubatch` boundary activation per
/// backend (contract §3; folded into the reserve math rather than a second
/// named constant so [`OVERHEAD_RESERVE_BYTES`] stays the single tunable).
/// With an unknown `n_embd` (synthetic dims) this is exactly
/// [`node_budget`]. Solo plans keep the plain v1 budget — nothing overlaps.
pub fn node_budget_v2(usable_memory_bytes: u64, n_embd: u64, n_ubatch: u32) -> u64 {
    let pipeline_copies = 16u64
        .saturating_mul(n_embd)
        .saturating_mul(effective_n_ubatch(n_ubatch) as u64);
    usable_memory_bytes.saturating_sub(OVERHEAD_RESERVE_BYTES.saturating_add(pipeline_copies))
}

/// Whole layers a node can hold at `ctx_len` under the v2 budget
/// ([`node_budget_v2`]) and the uniform-layer approximation.
pub fn node_layer_capacity_v2(
    usable_memory_bytes: u64,
    dims: &ModelDims,
    ctx_len: u32,
    n_ubatch: u32,
) -> u64 {
    let cost = per_layer_cost(dims, ctx_len);
    (node_budget_v2(usable_memory_bytes, dims.n_embd, n_ubatch) as f64 / cost).floor() as u64
}

/// The layer-share tilt family (contract §7), enumerated in this fixed
/// order for deterministic tie-breaks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Tilt {
    /// Shares ∝ fractional layer capacity (memory only — M3's rule).
    Memory,
    /// Shares ∝ `cap_f × (0.5 + 0.5·decode/max_decode)` — v1's rule.
    Hybrid,
    /// Shares ∝ `decode/max_decode` alone; memory still caps via the
    /// whole-layer capacity clamp in [`apportion`].
    Compute,
}

impl Tilt {
    const ALL: [Tilt; 3] = [Tilt::Memory, Tilt::Hybrid, Tilt::Compute];

    fn label(self) -> &'static str {
        match self {
            Tilt::Memory => "memory-proportional",
            Tilt::Hybrid => "0.5+0.5 compute tilt",
            Tilt::Compute => "compute-proportional",
        }
    }

    fn score(self, c: &Cand) -> f64 {
        match self {
            Tilt::Memory => c.cap_f,
            Tilt::Hybrid => c.cap_f * c.factor,
            // factor = 0.5 + 0.5·(decode/max_decode), so the pure ratio is
            // 2·factor − 1. Unprofiled nodes keep the neutral 1.0 (no tilt
            // for or against — v1's philosophy), which also makes a
            // profile-less cluster tie every tilt and fall back to
            // memory-proportional, i.e. today's behavior.
            Tilt::Compute => (2.0 * c.factor - 1.0).max(0.0),
        }
    }
}

/// One member of the tilt family with its apportioned layer counts.
struct FamilyCandidate {
    tilt: Tilt,
    underweighted: bool,
    counts: Vec<u32>,
}

/// Enumerate the tilt family for a participant set: the three base tilts,
/// then each with the slowest profiled node underweighted one
/// layer-quantum. Variants that are infeasible, have no identifiable
/// slowest node, or would evict a node entirely (subset membership belongs
/// to the inclusion rules) are skipped.
fn family_counts(cands: &[&Cand], n_layers: u32) -> Vec<FamilyCandidate> {
    let caps: Vec<u64> = cands.iter().map(|c| c.cap_layers).collect();
    // The slowest participant by its own profiled decode rate; unprofiled
    // nodes are never "the slowest" (their speed is unknown). First index
    // wins ties — deterministic.
    let slowest: Option<usize> = cands
        .iter()
        .enumerate()
        .filter_map(|(i, c)| c.decode.map(|d| (i, d)))
        .min_by(|a, b| a.1.partial_cmp(&b.1).expect("decode is never NaN"))
        .map(|(i, _)| i);

    let mut out = Vec::new();
    for underweighted in [false, true] {
        for tilt in Tilt::ALL {
            let scores: Vec<f64> = cands.iter().map(|c| tilt.score(c)).collect();
            let Some((mut counts, quotas)) = apportion(&scores, &caps, n_layers) else {
                continue;
            };
            if underweighted {
                let Some(s) = slowest else { continue };
                if counts[s] <= 1 {
                    // One quantum off a 1-layer node would evict it.
                    continue;
                }
                // Recipient: the best other node with capacity slack, by
                // the same key apportion's leftover rule uses (largest
                // unmet quota, then score, then index).
                let mut recipient: Option<usize> = None;
                for i in 0..cands.len() {
                    if i == s || (counts[i] as u64) >= caps[i] {
                        continue;
                    }
                    let better = match recipient {
                        None => true,
                        Some(r) => {
                            let key_i = quotas[i] - counts[i] as f64;
                            let key_r = quotas[r] - counts[r] as f64;
                            key_i > key_r || (key_i == key_r && scores[i] > scores[r])
                        }
                    };
                    if better {
                        recipient = Some(i);
                    }
                }
                let Some(r) = recipient else { continue };
                counts[s] -= 1;
                counts[r] += 1;
            }
            out.push(FamilyCandidate {
                tilt,
                underweighted,
                counts,
            });
        }
    }
    out
}

/// All stage orders to evaluate for one set of layer counts: every
/// permutation of the active non-head nodes with the head pinned last
/// (sampling locality), or v1's greedy nearest-neighbor chain when the set
/// is too large for exhaustive search.
fn enumerate_orders(cands: &[&Cand], counts: &[u32], links: &[LinkRtt]) -> Vec<Vec<usize>> {
    let active: Vec<usize> = (0..cands.len()).filter(|&i| counts[i] > 0).collect();
    if active.len() <= 1 {
        return vec![active];
    }
    if active.len() > EXACT_ORDER_MAX_NODES {
        return vec![order_stages(cands, counts, links).0];
    }
    let tail: Option<usize> = active.iter().copied().find(|&i| cands[i].is_head);
    let movable: Vec<usize> = active
        .iter()
        .copied()
        .filter(|&i| Some(i) != tail)
        .collect();
    permutations(&movable)
        .into_iter()
        .map(|mut chain| {
            if let Some(t) = tail {
                chain.push(t);
            }
            chain
        })
        .collect()
}

/// The winning candidate of a v2 search, plus how many were considered.
struct V2Eval {
    /// Layers per candidate index (0 = does not participate).
    counts: Vec<u32>,
    /// Stage order: candidate indices of nodes with layers, stage 0 first.
    order: Vec<usize>,
    /// RTT of each pipeline boundary, in `order` (len = order.len() - 1).
    boundary_rtts: Vec<f64>,
    /// Transfer time of the per-ubatch activation copy over each boundary
    /// (0 for unmeasured bandwidth), parallel to `boundary_rtts`.
    boundary_transfer_ms: Vec<f64>,
    /// PRIMARY comparison key: the v1 decode metric with MoE active-unit
    /// scaling (module docs).
    tpt: f64,
    /// SECONDARY key: prefill boundary cost `Σ (RTT + transfer)` ms.
    prefill_cost: f64,
    tilt: Tilt,
    underweighted: bool,
    /// Total (tilt-variant × stage-order) candidates evaluated.
    considered: usize,
    /// Tilt-family variants that produced a feasible apportionment.
    variants: usize,
}

/// Evaluate the whole candidate family for one participant set and return
/// the minimum-predicted-tpt winner (`None` when the set cannot hold the
/// model at all).
fn search_v2(
    cands: &[&Cand],
    n_layers: u32,
    links: &[LinkRtt],
    dims: &ModelDims,
    n_ubatch: u32,
) -> Option<V2Eval> {
    let family = family_counts(cands, n_layers);
    let variants = family.len();
    let mut considered = 0usize;
    let mut best: Option<V2Eval> = None;

    for fc in family {
        for order in enumerate_orders(cands, &fc.counts, links) {
            considered += 1;
            let mut boundary_rtts = Vec::with_capacity(order.len().saturating_sub(1));
            let mut boundary_transfer = Vec::with_capacity(order.len().saturating_sub(1));
            for w in order.windows(2) {
                let (a, b) = (&cands[w[0]].caps.node, &cands[w[1]].caps.node);
                boundary_rtts.push(link_rtt(links, a, b));
                boundary_transfer.push(
                    link_bandwidth(links, a, b)
                        .map(|bw| boundary_transfer_ms(dims.n_embd, n_ubatch, bw))
                        .unwrap_or(0.0),
                );
            }
            // Stage compute walks the order because MoE active units are
            // per-layer: which contiguous range a stage owns matters. For
            // dense models units == layer count and this reduces to v1.
            let mut cursor = 0u32;
            let mut compute_ms = 0.0f64;
            for &i in &order {
                let units = dims.active_compute_units(cursor, cursor + fc.counts[i]);
                cursor += fc.counts[i];
                if let Some(tps) = cands[i].tpt_decode {
                    compute_ms = compute_ms.max(units / tps * 1000.0);
                }
            }
            let rtt_sum: f64 = boundary_rtts.iter().sum();
            let tpt = compute_ms + rtt_sum;
            let prefill_cost = rtt_sum + boundary_transfer.iter().sum::<f64>();
            let better = match &best {
                None => true,
                // Strictly better on the primary key, or an exact primary
                // tie broken by the prefill term; equal-on-both keeps the
                // earliest candidate (deterministic).
                Some(b) => tpt < b.tpt || (tpt == b.tpt && prefill_cost < b.prefill_cost),
            };
            if better {
                best = Some(V2Eval {
                    counts: fc.counts.clone(),
                    order,
                    boundary_rtts,
                    boundary_transfer_ms: boundary_transfer,
                    tpt,
                    prefill_cost,
                    tilt: fc.tilt,
                    underweighted: fc.underweighted,
                    considered: 0,
                    variants,
                });
            }
        }
    }

    best.map(|mut b| {
        b.considered = considered;
        b
    })
}

/// Compute a placement per the M7 v2-lite rules (module docs). Same output
/// contract as [`crate::plan_v1`]: epoch and model are left for the caller
/// to stamp.
pub fn plan_v2(input: &PlanRequest) -> Result<PlannedPlacement, ScheduleError> {
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

    let n_ubatch = effective_n_ubatch(input.n_ubatch);
    let dims = &input.dims;
    let kv_layer = dims.kv_bytes_per_layer(input.ctx_len);
    let required = dims.total_required_bytes(input.ctx_len);
    // The solo decision keeps the plain v1 budget: a solo plan runs no RPC
    // pipeline, so the extra compute-buffer copies never materialize
    // (contract §3 charges them per node of a pipeline-parallel plan).
    let head_budget = node_budget(input.head.usable_memory_bytes);
    let solo_fits = required <= head_budget;
    match forced {
        None if solo_fits => {
            let mut placed = solo_placement(input, head_budget, required, false);
            placed
                .explanation
                .push_str(&format!("\nEffective n_ubatch {n_ubatch}."));
            return Ok(placed);
        }
        Some(1) => {
            if solo_fits {
                let mut placed = solo_placement(input, head_budget, required, true);
                placed
                    .explanation
                    .push_str(&format!("\nEffective n_ubatch {n_ubatch}."));
                return Ok(placed);
            }
            return Err(ScheduleError::DoesNotFit {
                required_mb: mb_ceil(required),
                available_mb: mb_floor(head_budget),
            });
        }
        _ => {}
    }

    distributed_v2(
        input,
        forced,
        head_budget,
        required,
        kv_layer,
        solo_fits,
        n_ubatch,
    )
}

#[allow(clippy::too_many_arguments)]
fn distributed_v2(
    input: &PlanRequest,
    forced: Option<u32>,
    head_budget: u64,
    required: u64,
    kv_layer: u64,
    solo_fits: bool,
    n_ubatch: u32,
) -> Result<PlannedPlacement, ScheduleError> {
    let dims = &input.dims;
    let n_layers = dims.n_layers;
    let cost = per_layer_cost(dims, input.ctx_len);

    // Candidates: workers in their stable order, head last, budgeted per
    // the v2 rule (overhead reserve + pipeline-parallel copies).
    let all_caps: Vec<&NodeCaps> = input
        .workers
        .iter()
        .chain(std::iter::once(&input.head))
        .collect();
    let budget_of = |usable: u64| node_budget_v2(usable, dims.n_embd, n_ubatch);
    let cands = build_cands(&all_caps, cost, &budget_of);
    let pooled_budget: u64 = cands.iter().map(|c| c.budget).sum();

    // Inclusion rules unchanged (contract §7); only the evaluator differs —
    // each subset is judged by the best of its candidate family.
    let mut eval_tpt =
        |set: &[&Cand]| search_v2(set, n_layers, &input.links, dims, n_ubatch).map(|e| e.tpt);
    let (selected, selection_notes) =
        select_workers(&cands, forced, n_layers, input.ctx_len, cost, &mut eval_tpt);

    let final_set = participants(&cands, &selected);
    let Some(eval) = search_v2(&final_set, n_layers, &input.links, dims, n_ubatch) else {
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
    let pipeline_copies_bytes = 16u64
        .saturating_mul(dims.n_embd)
        .saturating_mul(n_ubatch as u64);
    explanation.push_str(&format!(
        "\nKV budget: {} MB per layer at ctx {} ({} bytes/token/layer); overhead reserve \
         {} MB per node ({} MB base + {} MB pipeline-parallel copy buffers at \
         4·(4·n_embd·n_ubatch), effective n_ubatch {}).",
        mb_ceil(kv_layer),
        input.ctx_len,
        dims.kv_bytes_per_layer_per_ctx_token,
        mb_ceil(OVERHEAD_RESERVE_BYTES + pipeline_copies_bytes),
        mb_ceil(OVERHEAD_RESERVE_BYTES),
        mb_ceil(pipeline_copies_bytes),
        n_ubatch,
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
            "\nBoundaries (stage order from the candidate search; head pinned last): {} \
             (total {:.1} ms/token).",
            hops.join(", "),
            eval.boundary_rtts.iter().sum::<f64>()
        ));
    }
    let transfer_total: f64 = eval.boundary_transfer_ms.iter().sum();
    if transfer_total > 0.0 {
        let priced: Vec<String> = eval
            .order
            .windows(2)
            .zip(&eval.boundary_transfer_ms)
            .filter(|(_, t)| **t > 0.0)
            .map(|(pair, t)| {
                let (a, b) = (&final_set[pair[0]].caps.node, &final_set[pair[1]].caps.node);
                let bw = link_bandwidth(&input.links, a, b).unwrap_or(0.0);
                format!("'{}' -> '{}' at {bw:.0} Mbps -> {t:.1} ms", a.0, b.0)
            })
            .collect();
        explanation.push_str(&format!(
            "\nPrefill transfer term: 4·n_embd·n_ubatch = {} KiB per ubatch boundary; {} \
             (total {:.1} ms per ubatch; unmeasured links are costed RTT-only).",
            (4 * dims.n_embd * n_ubatch as u64).div_ceil(1024),
            priced.join(", "),
            transfer_total,
        ));
    }
    explanation.push_str(&format!(
        "\nPredicted decode cost: {:.1} (relative: max stage active-layer-units/decode_tps \
         × 1000 + boundary RTTs; used only to compare plans, not a latency promise); \
         prefill boundary cost {:.1} ms (secondary tie-break).",
        eval.tpt, eval.prefill_cost
    ));
    let winner = if eval.underweighted {
        format!(
            "{}, slowest node underweighted one layer",
            eval.tilt.label()
        )
    } else {
        eval.tilt.label().to_string()
    };
    explanation.push_str(&format!(
        "\nCandidate search: evaluated {} candidate placement(s) ({} layer-share tilt \
         variant(s) × stage orders; docs/perf.md §7); winner: {winner}.",
        eval.considered, eval.variants,
    ));
    if dims.n_expert > 0 {
        if dims.n_expert_used > 0 && dims.n_expert_used < dims.n_expert {
            explanation.push_str(&format!(
                "\nMoE: {} experts, {} active per token — compute predictions scaled to \
                 the active-expert fraction; memory still budgets all resident expert \
                 weights (KV is attention-side and unaffected).",
                dims.n_expert, dims.n_expert_used,
            ));
        } else {
            explanation.push_str(&format!(
                "\nMoE: {} experts declared, {} active per token — compute costed dense \
                 (conservative; scaling engages only when fewer experts run than are \
                 resident).",
                dims.n_expert, dims.n_expert_used,
            ));
        }
    }
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
        prefill_cost = eval.prefill_cost,
        candidates = eval.considered,
        winner = %winner,
        "planned pipeline-parallel placement (v2)"
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
    use crate::plan_v1;
    use crate::profile::ComputeProfile;
    use onebrain_proto::plan::NodeId;

    const MIB: u64 = 1 << 20;

    /// Uniform-weight dims with a configurable embedding width (v2's
    /// transfer term and pipeline reserve need `n_embd`).
    fn dims(n_layers: u32, weight_mib: u64, kv_rate: u64, n_embd: u64) -> ModelDims {
        ModelDims {
            n_layers,
            kv_bytes_per_layer_per_ctx_token: kv_rate,
            weight_bytes_per_layer: vec![weight_mib * MIB; n_layers as usize],
            total_weight_bytes: weight_mib * MIB * n_layers as u64,
            n_embd,
            n_expert: 0,
            n_expert_used: 0,
            expert_weight_bytes_per_layer: vec![0; n_layers as usize],
        }
    }

    /// [`dims`] with every layer split into dense + routed-expert bytes.
    fn moe_dims(
        n_layers: u32,
        weight_mib: u64,
        expert_mib: u64,
        n_expert: u32,
        n_expert_used: u32,
    ) -> ModelDims {
        ModelDims {
            n_expert,
            n_expert_used,
            expert_weight_bytes_per_layer: vec![expert_mib * MIB; n_layers as usize],
            ..dims(n_layers, weight_mib, 0, 0)
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
            draining: false,
        }
    }

    fn link(a: &str, b: &str, rtt_ms: f64, bandwidth_mbps: Option<f64>) -> LinkRtt {
        LinkRtt {
            a: NodeId(a.into()),
            b: NodeId(b.into()),
            rtt_ms,
            bandwidth_mbps,
        }
    }

    #[test]
    fn transfer_term_math_vectors() {
        // 4 × 4096 × 512 = 8 MiB of f32 activations; over 1 Gbit/s that is
        // 8_388_608 × 8 / 10^9 s = 67.108864 ms.
        assert_eq!(boundary_transfer_ms(4096, 512, 1000.0), 67.108864);
        // Tenth the bandwidth, ten times the wait.
        assert_eq!(boundary_transfer_ms(4096, 512, 100.0), 671.08864);
        // n_ubatch 0 means the 512 default — same figure.
        assert_eq!(boundary_transfer_ms(4096, 0, 1000.0), 67.108864);
        // Unknown embedding width or unmeasured/nonsense bandwidth: no
        // transfer term (RTT-only, today's behavior).
        assert_eq!(boundary_transfer_ms(0, 512, 1000.0), 0.0);
        assert_eq!(boundary_transfer_ms(4096, 512, 0.0), 0.0);
        assert_eq!(boundary_transfer_ms(4096, 512, -5.0), 0.0);
    }

    #[test]
    fn v2_budget_folds_pipeline_copies_into_the_reserve() {
        const GIB: u64 = 1 << 30;
        // 4 copies × 4 × 4096 × 512 bytes = 32 MiB on top of the 512 MiB
        // base reserve.
        assert_eq!(
            node_budget_v2(3 * GIB, 4096, 512),
            3 * GIB - 512 * MIB - 32 * MIB
        );
        // Unknown n_embd: exactly the v1 budget.
        assert_eq!(node_budget_v2(3 * GIB, 0, 512), node_budget(3 * GIB));
        // n_ubatch 0 = the 512 default.
        assert_eq!(
            node_budget_v2(3 * GIB, 4096, 0),
            node_budget_v2(3 * GIB, 4096, 512)
        );

        // The extra reserve can cost a marginal node its only layer: 110
        // MiB of post-reserve headroom holds one 100 MiB layer under v1
        // but not under v2 with a 32 MiB pipeline share.
        let d = dims(32, 100, 0, 4096);
        let usable = 512 * MIB + 110 * MIB;
        assert_eq!(crate::node_layer_capacity(usable, &d, 4096), 1);
        assert_eq!(node_layer_capacity_v2(usable, &d, 4096, 512), 0);
    }

    #[test]
    fn v2_matches_v1_without_new_inputs() {
        // No profiles, no bandwidth, unknown n_embd: every tilt ties, the
        // transfer term is inert, and the pipeline reserve is zero — the
        // v2 plan must be identical to v1's (the contract's "absent ⇒
        // today's behavior" degradation).
        let req = PlanRequest {
            head: caps("head", 4600, None),
            workers: vec![caps("w1", 4600, None), caps("w2", 4600, None)],
            dims: dims(32, 100, 16384, 0),
            ctx_len: 2048,
            forced_nodes: None,
            links: vec![link("w1", "head", 0.4, None), link("w2", "head", 3.0, None)],
            n_ubatch: DEFAULT_N_UBATCH,
        };
        let v1 = plan_v1(&req).unwrap();
        let v2 = plan_v2(&req).unwrap();
        assert_eq!(v1.plan, v2.plan);
        assert_eq!(v1.tensor_split, v2.tensor_split);

        // Solo degradation too, with the effective n_ubatch mentioned.
        let solo_req = PlanRequest {
            head: caps("head", 8000, None),
            workers: vec![],
            dims: dims(4, 100, 0, 0),
            ctx_len: 2048,
            forced_nodes: None,
            links: vec![],
            n_ubatch: 0,
        };
        let solo = plan_v2(&solo_req).unwrap();
        assert_eq!(solo.plan.strategy, Strategy::Solo);
        assert_eq!(solo.plan, plan_v1(&solo_req).unwrap().plan);
        assert!(
            solo.explanation.contains("Effective n_ubatch 512."),
            "{}",
            solo.explanation
        );
    }

    #[test]
    fn candidate_search_is_deterministic_and_reported() {
        let req = PlanRequest {
            head: caps("head", 3072, Some(90.0)),
            workers: vec![caps("w1", 3072, Some(120.0)), caps("w2", 2600, Some(60.0))],
            dims: dims(32, 100, 16384, 2048),
            ctx_len: 2048,
            forced_nodes: Some(3),
            links: vec![
                link("w1", "head", 0.3, Some(940.0)),
                link("w2", "head", 1.7, Some(320.0)),
                link("w1", "w2", 0.9, None),
            ],
            n_ubatch: DEFAULT_N_UBATCH,
        };
        let a = plan_v2(&req).unwrap();
        let b = plan_v2(&req).unwrap();
        assert_eq!(a, b, "equal inputs must give equal plans");
        assert!(
            a.explanation.contains("Candidate search: evaluated"),
            "{}",
            a.explanation
        );
        assert!(a.explanation.contains("winner:"), "{}", a.explanation);
    }

    #[test]
    fn bandwidth_flips_the_stage_order_versus_rtt_only() {
        // Three symmetric nodes, identical RTT everywhere: the decode
        // metric ties both stage orders, so v1-style RTT-only inputs keep
        // the first (lexicographic) chain w1 -> w2 -> head. Measured
        // bandwidth breaks the tie: w2's link to the head is 10× slower,
        // so the transfer term prefers ... -> w1 -> head.
        let req = |with_bandwidth: bool| {
            let bw = |mbps: f64| with_bandwidth.then_some(mbps);
            PlanRequest {
                head: caps("head", 3072, None),
                workers: vec![caps("w1", 3072, None), caps("w2", 3072, None)],
                dims: dims(32, 100, 0, 4096),
                ctx_len: 2048,
                forced_nodes: Some(3),
                links: vec![
                    link("w1", "head", 1.0, bw(1000.0)),
                    link("w2", "head", 1.0, bw(100.0)),
                    link("w1", "w2", 1.0, bw(1000.0)),
                ],
                n_ubatch: DEFAULT_N_UBATCH,
            }
        };

        let rtt_only = plan_v2(&req(false)).unwrap();
        let order: Vec<&str> = rtt_only
            .plan
            .assignments
            .iter()
            .map(|a| a.node.0.as_str())
            .collect();
        assert_eq!(order, ["w1", "w2", "head"], "{}", rtt_only.explanation);

        let with_bw = plan_v2(&req(true)).unwrap();
        let order: Vec<&str> = with_bw
            .plan
            .assignments
            .iter()
            .map(|a| a.node.0.as_str())
            .collect();
        assert_eq!(order, ["w2", "w1", "head"], "{}", with_bw.explanation);
        assert!(
            with_bw.explanation.contains("Prefill transfer term"),
            "{}",
            with_bw.explanation
        );
        // 4·4096·512 B over 100 Mbps = 671.1 ms priced on the slow hop —
        // now sitting on the w2 -> w1 boundary is impossible (that link is
        // fast); the surviving slow figure is gone from the winning chain,
        // whose boundaries are w2->w1 (1 Gbit) and w1->head (1 Gbit).
        assert!(
            with_bw.explanation.contains("at 1000 Mbps -> 67.1 ms"),
            "{}",
            with_bw.explanation
        );
    }

    #[test]
    fn underweight_candidate_wins_under_a_constructed_profile() {
        // Head is fast but memory-tight (10-layer capacity); w1 is medium,
        // w2 slow. Compute-proportional wants 18+ layers on the head, the
        // capacity clamp spills them onto the slow node, and the
        // one-quantum underweight of w2 relieves the resulting bottleneck:
        //   memory 262.0 > mem-uw 242.0 > hybrid 222.0 > hybrid-uw 202.0
        //   > compute 182.0 > compute-uw 162.0  (all + 2 × 1.0 ms RTT).
        let placed = plan_v2(&PlanRequest {
            head: caps("head", 1562, Some(200.0)),
            workers: vec![caps("w1", 3072, Some(100.0)), caps("w2", 3072, Some(50.0))],
            dims: dims(32, 100, 0, 0),
            ctx_len: 2048,
            forced_nodes: Some(3),
            links: vec![],
            n_ubatch: DEFAULT_N_UBATCH,
        })
        .unwrap();
        let a = &placed.plan.assignments;
        assert_eq!(a.len(), 3, "{}", placed.explanation);
        // Winner: compute-proportional with w2 underweighted — w1 14, w2 8,
        // head 10 (base compute-proportional would leave w2 at 9).
        assert_eq!(a[0].node.0, "w1");
        assert_eq!(a[0].layers, LayerRange { start: 0, end: 14 });
        assert_eq!(a[1].node.0, "w2");
        assert_eq!(a[1].layers, LayerRange { start: 14, end: 22 });
        assert_eq!(a[2].node.0, "head");
        assert_eq!(a[2].layers, LayerRange { start: 22, end: 32 });
        assert!(
            placed
                .explanation
                .contains("winner: compute-proportional, slowest node underweighted one layer"),
            "{}",
            placed.explanation
        );
        assert!(
            placed.explanation.contains("Predicted decode cost: 162.0"),
            "{}",
            placed.explanation
        );
    }

    #[test]
    fn moe_active_expert_scaling_lowers_predicted_compute() {
        // Two equal nodes, 16 layers each. Dense: 16 layers / 100 tok/s
        // × 1000 + 1 ms RTT = 161.0. MoE with half of each layer in routed
        // experts and 2 of 8 active: fraction (50 + 50 × 0.25)/100 =
        // 0.625 → 16 × 0.625 / 100 × 1000 + 1 = 101.0.
        let req = |d: ModelDims| PlanRequest {
            head: caps("head", 3072, Some(100.0)),
            workers: vec![caps("w1", 3072, Some(100.0))],
            dims: d,
            ctx_len: 2048,
            forced_nodes: None,
            links: vec![],
            n_ubatch: DEFAULT_N_UBATCH,
        };
        let dense = plan_v2(&req(dims(32, 100, 0, 0))).unwrap();
        assert!(
            dense.explanation.contains("Predicted decode cost: 161.0"),
            "{}",
            dense.explanation
        );
        assert!(!dense.explanation.contains("MoE:"), "{}", dense.explanation);

        let moe = plan_v2(&req(moe_dims(32, 100, 50, 8, 2))).unwrap();
        assert_eq!(moe.plan.assignments.len(), 2);
        // Same split — memory math never scales by active experts.
        assert_eq!(
            moe.plan.assignments[0].layers,
            LayerRange { start: 0, end: 16 }
        );
        assert!(
            moe.explanation.contains("Predicted decode cost: 101.0"),
            "{}",
            moe.explanation
        );
        assert!(
            moe.explanation
                .contains("MoE: 8 experts, 2 active per token"),
            "{}",
            moe.explanation
        );
    }

    #[test]
    fn m4_style_inclusion_rules_still_hold_in_v2() {
        // The 5% third-node rule (M4) must survive the family search: same
        // scenario as v1's third_node_excluded_when_gain_below_threshold —
        // equal nodes and 45 ms links, where the third node's 2.4% gain
        // stays under the bar for every tilt in the family.
        let placed = plan_v2(&PlanRequest {
            head: caps("head", 3104, Some(100.0)),
            workers: vec![caps("w1", 3104, Some(100.0)), caps("w2", 3104, Some(100.0))],
            dims: dims(32, 100, 16384, 0),
            ctx_len: 2048,
            forced_nodes: None,
            links: vec![
                link("w1", "head", 45.0, None),
                link("w2", "head", 45.0, None),
                link("w1", "w2", 45.0, None),
            ],
            n_ubatch: DEFAULT_N_UBATCH,
        })
        .unwrap();
        assert_eq!(placed.plan.assignments.len(), 2, "{}", placed.explanation);
        assert!(
            placed.explanation.contains("below the 5% threshold"),
            "{}",
            placed.explanation
        );
    }
}
