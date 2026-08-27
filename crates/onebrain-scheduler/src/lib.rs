//! Placement: decide which layers run on which nodes.
//!
//! M3 ships scheduler v1-lite per `docs/distributed.md`: the auto-solo
//! short-circuit (§1.4 — if the model fits one node's usable memory, never
//! distribute), and otherwise pipeline-parallel contiguous layer ranges
//! proportional to usable memory over `[workers..., head]` — head last, so
//! the head owns the tail layers and sampling stays local. Full v1 (compute
//! scores, real KV accounting, link-aware boundaries) is M4; the prima.cpp
//! style cost-model search is M7.
//!
//! # Memory budgeting (M3 rule)
//!
//! KV-cache and runtime overhead are modeled by a flat utilization ceiling:
//! a node's *budget* is 85% of its reported `usable_memory_bytes`; the 15%
//! reserve stands in for KV at the requested context length plus engine
//! overhead. Real per-range KV accounting at `ctx_len` replaces this in M4.
//! `ctx_len` is therefore recorded in the plan but does not yet influence
//! placement.

use serde::{Deserialize, Serialize};

use onebrain_proto::plan::{Assignment, Epoch, LayerRange, NodeId, Plan, Strategy};

/// Percentage of a node's usable memory the planner may budget against
/// (the flat M3 stand-in for KV + overhead; see the module docs).
pub const UTILIZATION_CEILING_PCT: u64 = 85;

const MIB: u64 = 1024 * 1024;

#[derive(Debug, thiserror::Error)]
pub enum ScheduleError {
    #[error(
        "model needs {required_mb} MB pooled but the cluster has {available_mb} MB usable; \
         add a node, choose a smaller quant, or lower the context length"
    )]
    DoesNotFit { required_mb: u64, available_mb: u64 },
    #[error("no nodes available to plan on")]
    NoNodes,
    #[error(
        "--nodes {requested} requested but only {available} nodes are available (paired workers \
         plus this one); pair more devices with `onebrain pair` or lower --nodes"
    )]
    NotEnoughNodes { requested: u32, available: u32 },
}

/// A node's measured capabilities, filled by pairing-time profiling and
/// `onebrain bench`. Usable memory is measured free memory minus OS reserve
/// (on Macs, Metal's recommendedMaxWorkingSetSize) — never total RAM.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DeviceProfile {
    pub node: NodeId,
    pub usable_memory_bytes: u64,
    /// Prefill throughput from the compute microbench (tokens/sec).
    pub prefill_tps: f64,
    /// Decode throughput from the compute microbench (tokens/sec).
    pub decode_tps: f64,
    pub disk_read_mbps: f64,
}

/// The auto-solo rule: given the model's total memory need (weights + KV at
/// the requested context) and a candidate node, decide if it runs alone.
pub fn fits_solo(profile: &DeviceProfile, required_bytes: u64) -> bool {
    profile.usable_memory_bytes >= required_bytes
}

/// What the planner knows about one node: identity and budgetable memory
/// (from the worker's `NodeStatus`, or measured locally for the head).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NodeBudget {
    pub node: NodeId,
    /// Measured free memory of the chosen device minus a fixed OS reserve —
    /// never total RAM. The planner budgets against 85% of this value
    /// ([`UTILIZATION_CEILING_PCT`]).
    pub usable_memory_bytes: u64,
}

/// Everything `plan` needs to place one model load.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlanInput {
    /// The node the user is on (scheduler, API gateway). Always participates
    /// last in stage order so it owns the tail layers and the output head.
    pub head: NodeBudget,
    /// Paired workers eligible for this plan, in stage order (stage 0 first).
    pub workers: Vec<NodeBudget>,
    /// Total weight bytes of the model (from the GGUF header).
    pub model_bytes: u64,
    /// Transformer layer count (embedding/output are handled by the engine).
    pub n_layers: u32,
    /// Requested context length. Recorded in the plan; KV is budgeted via
    /// the flat 85% ceiling in M3 (module docs), so this does not yet move
    /// placement.
    pub ctx_len: u32,
    /// `--nodes N`: `None` = automatic (auto-solo when it fits), `Some(1)`
    /// forces solo, `Some(n >= 2)` forces distribution across exactly `n`
    /// nodes. Callers pass `N >= 1`; `Some(0)` is treated as `None`.
    pub forced_nodes: Option<u32>,
}

/// A computed placement, ready for the epoch machinery.
#[derive(Debug, Clone, PartialEq)]
pub struct PlannedPlacement {
    /// The plan with `epoch` set to the placeholder `Epoch(0)` and `model`
    /// left empty — the caller stamps the real epoch and the model's
    /// manifest hash before broadcasting.
    pub plan: Plan,
    /// One fraction per participating device, in devices order = workers in
    /// stage order, then the head. Fractions are exact layer shares
    /// (`layers / n_layers`), so the engine's layer split reproduces the
    /// plan's ranges precisely; they sum to ~1.0.
    pub tensor_split: Vec<f32>,
    /// Human prose for `--explain`: per node — layers, weight MB — plus the
    /// binding constraint and why distribution did or did not engage.
    pub explanation: String,
}

/// The budget the planner may fill on a node: 85% of usable memory.
fn budget(usable_memory_bytes: u64) -> u64 {
    (usable_memory_bytes as u128 * UTILIZATION_CEILING_PCT as u128 / 100) as u64
}

fn mb_ceil(bytes: u64) -> u64 {
    bytes.div_ceil(MIB)
}

fn mb_floor(bytes: u64) -> u64 {
    bytes / MIB
}

/// Compute a placement per the M3 v1-lite rules (see module docs and
/// `docs/distributed.md` "Placement").
pub fn plan(input: &PlanInput) -> Result<PlannedPlacement, ScheduleError> {
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

    let head_budget = budget(input.head.usable_memory_bytes);

    // Auto-solo short-circuit (§1.4), or `--nodes 1` forcing solo. The
    // "margin" is the 15% ceiling itself: solo engages when the weights fit
    // inside the head's 85% budget.
    let solo_fits = input.model_bytes <= head_budget;
    match forced {
        None if solo_fits => return Ok(solo_placement(input, head_budget, false)),
        Some(1) => {
            if solo_fits {
                return Ok(solo_placement(input, head_budget, true));
            }
            return Err(ScheduleError::DoesNotFit {
                required_mb: mb_ceil(input.model_bytes),
                available_mb: mb_floor(head_budget),
            });
        }
        _ => {}
    }

    distributed_placement(input, forced, head_budget, solo_fits)
}

fn solo_placement(input: &PlanInput, head_budget: u64, forced: bool) -> PlannedPlacement {
    let why = if forced {
        "--nodes 1 forced solo".to_string()
    } else {
        "auto-solo (§1.4): the model fits on this node, so distribution is not engaged".to_string()
    };
    let explanation = format!(
        "Solo on head '{}': model weights {} MB fit within its {} MB budget ({}% of {} MB \
         usable; the reserve stands in for KV cache at ctx {} and engine overhead in M3). {}.",
        input.head.node.0,
        mb_ceil(input.model_bytes),
        mb_floor(head_budget),
        UTILIZATION_CEILING_PCT,
        mb_floor(input.head.usable_memory_bytes),
        input.ctx_len,
        why
    );
    tracing::debug!(
        node = %input.head.node.0,
        model_mb = mb_ceil(input.model_bytes),
        budget_mb = mb_floor(head_budget),
        forced,
        "planned solo placement"
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
                    end: input.n_layers,
                },
                stage: 0,
            }],
            ctx_len: input.ctx_len,
        },
        tensor_split: vec![1.0],
        explanation,
    }
}

fn distributed_placement(
    input: &PlanInput,
    forced: Option<u32>,
    head_budget: u64,
    solo_fits: bool,
) -> Result<PlannedPlacement, ScheduleError> {
    // Select workers. Automatic mode uses all of them (v1-lite; M4 prunes by
    // cost model). `--nodes n` keeps the n-1 largest-budget workers, in
    // their original stage order.
    let selected_workers: Vec<&NodeBudget> = match forced {
        Some(n) => {
            let keep = n as usize - 1;
            let mut order: Vec<usize> = (0..input.workers.len()).collect();
            order.sort_by_key(|&i| (std::cmp::Reverse(input.workers[i].usable_memory_bytes), i));
            let mut chosen: Vec<usize> = order.into_iter().take(keep).collect();
            chosen.sort_unstable();
            chosen.iter().map(|&i| &input.workers[i]).collect()
        }
        None => input.workers.iter().collect(),
    };

    // Participants in stage order: workers first, head last (head owns the
    // tail layers; sampling stays local).
    let mut participants: Vec<&NodeBudget> = selected_workers;
    participants.push(&input.head);

    let budgets: Vec<u64> = participants
        .iter()
        .map(|p| budget(p.usable_memory_bytes))
        .collect();
    let pooled: u64 = budgets.iter().sum();

    if pooled < input.model_bytes {
        return Err(ScheduleError::DoesNotFit {
            required_mb: mb_ceil(input.model_bytes),
            available_mb: mb_floor(pooled),
        });
    }

    // Proportional layer apportionment with largest-remainder rounding.
    // Exact rational quotas over the common denominator `pooled` avoid any
    // float drift; ties break on larger budget, then earlier stage.
    let n_layers = input.n_layers as u128;
    let pooled128 = pooled as u128;
    let mut layer_counts: Vec<u32> = Vec::with_capacity(participants.len());
    let mut remainders: Vec<(u128, u64, usize)> = Vec::with_capacity(participants.len());
    let mut assigned: u32 = 0;
    for (i, &b) in budgets.iter().enumerate() {
        let exact = n_layers * b as u128;
        let floor = (exact / pooled128) as u32;
        layer_counts.push(floor);
        assigned += floor;
        remainders.push((exact % pooled128, b, i));
    }
    let mut leftover = input.n_layers - assigned;
    remainders.sort_by(|a, b| b.0.cmp(&a.0).then(b.1.cmp(&a.1)).then(a.2.cmp(&b.2)));
    for &(_, _, i) in remainders.iter() {
        if leftover == 0 {
            break;
        }
        layer_counts[i] += 1;
        leftover -= 1;
    }

    // Nodes that round to zero layers drop out of the plan (contract:
    // minimum 1 layer per participating node). Largest-remainder rounding
    // already sums to n_layers, so coverage is unaffected.
    let mut assignments = Vec::new();
    let mut tensor_split = Vec::new();
    let mut dropped: Vec<&NodeBudget> = Vec::new();
    let mut per_node_prose = Vec::new();
    let mut binding: Option<(usize, u64)> = None; // (assignments index, utilization pct)
    let mut cursor: u32 = 0;
    let mut stage: u32 = 0;
    for (i, part) in participants.iter().enumerate() {
        let layers = layer_counts[i];
        if layers == 0 {
            dropped.push(*part);
            continue;
        }
        let range = LayerRange {
            start: cursor,
            end: cursor + layers,
        };
        cursor += layers;
        // Estimated weight share of this range and how hard it presses on
        // the node's budget — the node with the highest utilization is the
        // binding constraint of the plan.
        let weight_bytes = (layers as u128 * input.model_bytes as u128 / n_layers.max(1)) as u64;
        let util_pct = if budgets[i] == 0 {
            u64::MAX
        } else {
            (weight_bytes as u128 * 100 / budgets[i] as u128) as u64
        };
        let role = if i == participants.len() - 1 {
            "head"
        } else {
            "worker"
        };
        per_node_prose.push(format!(
            "  stage {} — {} '{}': layers {}..{} ({} layers, ~{} MB weights, {}% of its {} MB \
             budget)",
            stage,
            role,
            part.node.0,
            range.start,
            range.end,
            layers,
            mb_ceil(weight_bytes),
            util_pct,
            mb_floor(budgets[i]),
        ));
        let is_new_binding = match binding {
            None => true,
            Some((_, best)) => util_pct > best,
        };
        if is_new_binding {
            binding = Some((assignments.len(), util_pct));
        }
        assignments.push(Assignment {
            node: part.node.clone(),
            layers: range,
            stage,
        });
        tensor_split.push(layers as f32 / input.n_layers as f32);
        stage += 1;
    }

    let why = match forced {
        Some(n) if solo_fits => {
            format!("distribution forced by --nodes {n} (the model would fit the head alone)")
        }
        Some(n) => format!(
            "distribution engaged: --nodes {n}, and model weights {} MB exceed the head's {} MB \
             budget",
            mb_ceil(input.model_bytes),
            mb_floor(head_budget)
        ),
        None => format!(
            "distribution engaged: model weights {} MB exceed the head's {} MB budget ({}% of \
             {} MB usable)",
            mb_ceil(input.model_bytes),
            mb_floor(head_budget),
            UTILIZATION_CEILING_PCT,
            mb_floor(input.head.usable_memory_bytes)
        ),
    };
    let mut explanation = format!(
        "Pipeline-parallel across {} nodes ({why}):\n{}",
        assignments.len(),
        per_node_prose.join("\n")
    );
    if let Some((idx, pct)) = binding {
        explanation.push_str(&format!(
            "\nBinding constraint: node '{}' at {}% of its memory budget.",
            assignments[idx].node.0, pct
        ));
    }
    for d in &dropped {
        explanation.push_str(&format!(
            "\nDropped: node '{}' ({} MB budget) rounded to 0 layers and left the plan.",
            d.node.0,
            mb_floor(budget(d.usable_memory_bytes))
        ));
    }

    tracing::debug!(
        nodes = assignments.len(),
        dropped = dropped.len(),
        model_mb = mb_ceil(input.model_bytes),
        pooled_mb = mb_floor(pooled),
        "planned pipeline-parallel placement"
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

    fn node(mem: u64) -> DeviceProfile {
        DeviceProfile {
            node: NodeId("n".into()),
            usable_memory_bytes: mem,
            prefill_tps: 100.0,
            decode_tps: 20.0,
            disk_read_mbps: 1000.0,
        }
    }

    #[test]
    fn solo_when_it_fits() {
        assert!(fits_solo(&node(16 << 30), 10 << 30));
    }

    #[test]
    fn distribute_when_it_does_not() {
        assert!(!fits_solo(&node(8 << 30), 10 << 30));
    }

    fn nb(id: &str, mem: u64) -> NodeBudget {
        NodeBudget {
            node: NodeId(id.into()),
            usable_memory_bytes: mem,
        }
    }

    const GIB: u64 = 1 << 30;

    #[test]
    fn auto_solo_short_circuits() {
        let input = PlanInput {
            head: nb("head", 16 * GIB),
            workers: vec![nb("w1", 32 * GIB)],
            model_bytes: 10 * GIB,
            n_layers: 32,
            ctx_len: 4096,
            forced_nodes: None,
        };
        let placed = plan(&input).unwrap();
        assert_eq!(placed.plan.strategy, Strategy::Solo);
        assert_eq!(placed.plan.epoch, Epoch(0));
        assert_eq!(placed.plan.assignments.len(), 1);
        assert_eq!(placed.plan.assignments[0].node, NodeId("head".into()));
        assert_eq!(
            placed.plan.assignments[0].layers,
            LayerRange { start: 0, end: 32 }
        );
        assert_eq!(placed.tensor_split, vec![1.0]);
        assert!(placed.explanation.contains("Solo on head 'head'"));
        assert!(placed.explanation.contains("auto-solo"));
    }

    #[test]
    fn forced_distribution_splits_two_nodes_proportionally() {
        // Model fits the head alone (budget 10.2 GiB) — --nodes 2 forces the
        // split anyway. Budgets: worker 5.1 GiB, head 10.2 GiB → 10 + 20 of
        // 30 layers.
        let input = PlanInput {
            head: nb("head", 12 * GIB),
            workers: vec![nb("w1", 6 * GIB)],
            model_bytes: 4 * GIB,
            n_layers: 30,
            ctx_len: 4096,
            forced_nodes: Some(2),
        };
        let placed = plan(&input).unwrap();
        assert_eq!(placed.plan.strategy, Strategy::PipelineParallel);
        let a = &placed.plan.assignments;
        assert_eq!(a.len(), 2);
        // Workers-then-head stage order; head owns the tail.
        assert_eq!(a[0].node, NodeId("w1".into()));
        assert_eq!(a[0].stage, 0);
        assert_eq!(a[0].layers, LayerRange { start: 0, end: 10 });
        assert_eq!(a[1].node, NodeId("head".into()));
        assert_eq!(a[1].stage, 1);
        assert_eq!(a[1].layers, LayerRange { start: 10, end: 30 });
        // tensor_split mirrors the ranges in the same order and sums to ~1.
        assert_eq!(placed.tensor_split.len(), 2);
        assert!((placed.tensor_split[0] - 10.0 / 30.0).abs() < 1e-6);
        assert!((placed.tensor_split[1] - 20.0 / 30.0).abs() < 1e-6);
        let sum: f32 = placed.tensor_split.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        assert!(placed.explanation.contains("forced by --nodes 2"));
    }

    #[test]
    fn asymmetric_budgets_give_asymmetric_contiguous_ranges() {
        let input = PlanInput {
            head: nb("head", 8 * GIB),
            workers: vec![nb("big", 16 * GIB), nb("small", 4 * GIB)],
            model_bytes: 20 * GIB,
            n_layers: 32,
            ctx_len: 8192,
            forced_nodes: None,
        };
        let placed = plan(&input).unwrap();
        let a = &placed.plan.assignments;
        assert_eq!(a.len(), 3);
        // Contiguous, non-overlapping, covering exactly [0, 32).
        assert_eq!(a[0].layers.start, 0);
        for pair in a.windows(2) {
            assert_eq!(pair[0].layers.end, pair[1].layers.start);
            assert!(!pair[0].layers.is_empty() && !pair[1].layers.is_empty());
        }
        assert_eq!(a.last().unwrap().layers.end, 32);
        // Asymmetric: the 16 GiB worker holds more than the 4 GiB one, and
        // the head (8 GiB) sits in between.
        assert_eq!(a[0].node, NodeId("big".into()));
        assert_eq!(a[1].node, NodeId("small".into()));
        assert_eq!(a[2].node, NodeId("head".into()));
        assert!(a[0].layers.len() > a[2].layers.len());
        assert!(a[2].layers.len() > a[1].layers.len());
        let sum: f32 = placed.tensor_split.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn pooled_too_small_errors_with_both_numbers() {
        let input = PlanInput {
            head: nb("head", 4 * GIB),
            workers: vec![nb("w1", 4 * GIB)],
            model_bytes: 10 * GIB,
            n_layers: 32,
            ctx_len: 4096,
            forced_nodes: None,
        };
        match plan(&input).unwrap_err() {
            ScheduleError::DoesNotFit {
                required_mb,
                available_mb,
            } => {
                assert_eq!(required_mb, 10 * 1024);
                // Pooled budget: 2 × 85% of 4 GiB = 6963.2 MB, floored.
                assert_eq!(available_mb, 6963);
            }
            other => panic!("expected DoesNotFit, got {other:?}"),
        }
    }

    #[test]
    fn forced_nodes_beyond_available_errors() {
        let input = PlanInput {
            head: nb("head", 8 * GIB),
            workers: vec![nb("w1", 8 * GIB)],
            model_bytes: 4 * GIB,
            n_layers: 32,
            ctx_len: 4096,
            forced_nodes: Some(4),
        };
        match plan(&input).unwrap_err() {
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

    #[test]
    fn forced_solo_that_does_not_fit_errors_with_head_numbers() {
        let input = PlanInput {
            head: nb("head", 4 * GIB),
            workers: vec![nb("w1", 32 * GIB)],
            model_bytes: 10 * GIB,
            n_layers: 32,
            ctx_len: 4096,
            forced_nodes: Some(1),
        };
        match plan(&input).unwrap_err() {
            ScheduleError::DoesNotFit {
                required_mb,
                available_mb,
            } => {
                assert_eq!(required_mb, 10 * 1024);
                assert_eq!(available_mb, 3481); // 85% of 4 GiB, floored MB
            }
            other => panic!("expected DoesNotFit, got {other:?}"),
        }
    }

    #[test]
    fn tiny_node_rounds_to_zero_layers_and_drops_out() {
        // 200 MiB worker against two 10 GiB nodes: its quota rounds to 0
        // layers, it leaves the plan, and the survivors still cover [0, 10).
        let input = PlanInput {
            head: nb("head", 10 * GIB),
            workers: vec![nb("w1", 10 * GIB), nb("tiny", 200 * MIB)],
            model_bytes: 12 * GIB,
            n_layers: 10,
            ctx_len: 4096,
            forced_nodes: None,
        };
        let placed = plan(&input).unwrap();
        let a = &placed.plan.assignments;
        assert_eq!(a.len(), 2);
        assert!(a.iter().all(|x| x.node != NodeId("tiny".into())));
        assert_eq!(a[0].node, NodeId("w1".into()));
        assert_eq!(a[1].node, NodeId("head".into()));
        // Stages renumber contiguously after the drop; coverage is exact.
        assert_eq!(a[0].stage, 0);
        assert_eq!(a[1].stage, 1);
        assert_eq!(a[0].layers.start, 0);
        assert_eq!(a[0].layers.end, a[1].layers.start);
        assert_eq!(a[1].layers.end, 10);
        assert_eq!(placed.tensor_split.len(), 2);
        assert!(placed.explanation.contains("Dropped: node 'tiny'"));
    }

    #[test]
    fn explanation_mentions_binding_node() {
        // "small" runs closest to its budget, so it is the binding
        // constraint (see asymmetric case: 92% vs 82% on the others).
        let input = PlanInput {
            head: nb("head", 8 * GIB),
            workers: vec![nb("big", 16 * GIB), nb("small", 4 * GIB)],
            model_bytes: 20 * GIB,
            n_layers: 32,
            ctx_len: 8192,
            forced_nodes: None,
        };
        let placed = plan(&input).unwrap();
        assert!(placed
            .explanation
            .contains("Binding constraint: node 'small'"));
        // Per-node prose lists layers and MB for every participant.
        for id in ["big", "small", "head"] {
            assert!(placed.explanation.contains(&format!("'{id}'")));
        }
        assert!(placed.explanation.contains("MB"));
    }
}
