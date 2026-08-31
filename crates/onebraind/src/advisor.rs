//! The metrics endpoint's bottleneck advisor (M8, docs/product.md §1):
//! one-line findings computed SERVER-SIDE by pure functions over already-
//! measured data. The honesty rule is structural: every rule consumes only
//! measurements (link probes, NodeStatus reports, battery verdicts, stored
//! Hello data) or the scheduler's own recorded selection notes — no rule
//! speculates, and no advice exists without a measurement behind it
//! (§1.6: the story is capacity, never speed multiplication).
//!
//! Each rule is its own function so its firing condition can be unit-tested
//! on constructed inputs; [`advise`] runs them all in a stable order.

use onebrain_mesh::{PeerState, PeerStatus};
use onebrain_proto::plan::{Plan, Strategy};
use serde::Serialize;

use crate::cluster::ActivePlanView;

/// Below this measured link bandwidth the slow-link rule fires. Wired
/// gigabit reliably measures well above it (~940 Mbps on the mesh probe);
/// Wi-Fi links and shaped/netem links land below. The margin keeps the
/// advice honest: a link this slow genuinely pays for a wire on the
/// pipeline boundary transfer (docs/perf.md §7's 4·n_embd·n_ubatch per
/// ubatch).
pub const SLOW_LINK_MBPS: f64 = 400.0;

/// KV-and-overhead headroom assumed on top of a node's weight share when
/// judging memory starvation (matches the scheduler's overhead-reserve
/// order of magnitude, docs/perf.md §7).
pub const MEMORY_HEADROOM_BYTES: u64 = 1 << 30;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Severity {
    Info,
    Warn,
}

/// One advisor line: `{severity, text}` exactly as the contract's schema.
#[derive(Debug, Clone, Serialize)]
pub struct Finding {
    pub severity: Severity,
    pub text: String,
}

/// Per-node measured facts the plan-shaped rules consume — the head's
/// local probes and the peers' last `NodeStatus`, flattened so tests can
/// construct them without a mesh.
#[derive(Debug, Clone)]
pub struct NodeFacts {
    /// Mesh endpoint id (matches `Plan` assignment node ids).
    pub id: String,
    /// Human-readable name for the finding text.
    pub name: String,
    /// Last measured schedulable memory; `None` = never reported (the
    /// memory rule then stays silent for this node — no measurement, no
    /// advice).
    pub usable_memory_bytes: Option<u64>,
    /// Battery-drain verdict from the last measurement.
    pub draining: bool,
}

/// Everything [`advise`] consumes, assembled by the metrics endpoint.
#[derive(Debug)]
pub struct AdvisorInput<'a> {
    pub node_name: &'a str,
    /// This node's mesh endpoint id (to recognize itself in assignments).
    pub own_id: &'a str,
    pub product_version: &'a str,
    pub engine_build: &'a str,
    /// This node's freshly measured usable memory.
    pub usable_memory_bytes: u64,
    /// This node's battery verdict.
    pub draining: bool,
    pub peers: &'a [PeerStatus],
    pub plan: Option<&'a ActivePlanView>,
    /// Size on disk of the loaded model, when one is loaded (head side) —
    /// the basis for a node's plan-share estimate.
    pub loaded_size_bytes: Option<u64>,
}

/// Run every rule in a stable order (plan-endangering warnings first).
pub fn advise(input: &AdvisorInput) -> Vec<Finding> {
    let mut nodes: Vec<NodeFacts> = vec![NodeFacts {
        id: input.own_id.to_string(),
        name: input.node_name.to_string(),
        usable_memory_bytes: Some(input.usable_memory_bytes),
        draining: input.draining,
    }];
    nodes.extend(input.peers.iter().map(|p| NodeFacts {
        id: p.id.clone(),
        name: p.name.clone(),
        usable_memory_bytes: p.usable_memory_bytes,
        draining: p.draining,
    }));
    let plan = input.plan.map(|view| &view.plan);
    let explanation = input.plan.and_then(|view| view.explanation.as_deref());

    let mut findings = Vec::new();
    if let Some(plan) = plan {
        findings.extend(draining_in_plan(plan, &nodes));
        if let Some(size) = input.loaded_size_bytes {
            findings.extend(memory_starved(plan, size, &nodes));
        }
    }
    findings.extend(slow_link(input.node_name, input.peers, plan));
    findings.extend(version_skew(
        input.product_version,
        input.engine_build,
        input.peers,
    ));
    findings.extend(solo_because_infeasible(plan, explanation));
    findings.extend(battery_draining_worker(input.peers, plan));
    findings
}

/// Whether `plan` (distributed) assigns layers to node `id`.
fn in_plan(plan: Option<&Plan>, id: &str) -> bool {
    plan.is_some_and(|p| {
        p.strategy == Strategy::PipelineParallel && p.assignments.iter().any(|a| a.node.0 == id)
    })
}

/// Slow-link rule: a Connected peer whose PROBED bandwidth measures below
/// [`SLOW_LINK_MBPS`]. Fires only on a real probe result (`None` = never
/// measured = silent). A link inside the active distributed plan is a
/// warning — every token crosses it — otherwise informational.
pub fn slow_link(local_name: &str, peers: &[PeerStatus], plan: Option<&Plan>) -> Vec<Finding> {
    peers
        .iter()
        .filter(|p| p.state == PeerState::Connected)
        .filter_map(|p| {
            let mbps = p
                .bandwidth_mbps
                .filter(|b| *b > 0.0 && *b < SLOW_LINK_MBPS)?;
            let severity = if in_plan(plan, &p.id) {
                Severity::Warn
            } else {
                Severity::Info
            };
            Some(Finding {
                severity,
                text: format!(
                    "link {local_name}\u{2194}{} measures ~{mbps:.0} Mbps — a wired connection \
                     would lift the pipeline's boundary transfer",
                    p.name
                ),
            })
        })
        .collect()
}

/// Memory-starved rule: a node of the active distributed plan whose last
/// MEASURED usable memory is far below its share of the model plus KV
/// headroom (usable < (share + headroom) / 2 — the contract's "≪"). The
/// scheduler only ever packs a node to 85% of its usable memory, so a
/// fresh healthy plan cannot fire this; it fires when memory was eaten
/// since planning (or a later NodeStatus reported the drop).
pub fn memory_starved(plan: &Plan, model_size_bytes: u64, nodes: &[NodeFacts]) -> Vec<Finding> {
    if plan.strategy != Strategy::PipelineParallel {
        return Vec::new();
    }
    let total_layers: u64 = plan
        .assignments
        .iter()
        .map(|a| u64::from(a.layers.end.saturating_sub(a.layers.start)))
        .sum();
    if total_layers == 0 {
        return Vec::new();
    }
    plan.assignments
        .iter()
        .filter_map(|a| {
            let node = nodes.iter().find(|n| n.id == a.node.0)?;
            let usable = node.usable_memory_bytes?;
            let layers = u64::from(a.layers.end.saturating_sub(a.layers.start));
            let share = (model_size_bytes as u128 * layers as u128 / total_layers as u128) as u64;
            let need = share.saturating_add(MEMORY_HEADROOM_BYTES);
            (usable < need / 2).then(|| Finding {
                severity: Severity::Warn,
                text: format!(
                    "node '{}' reports {} MB usable but its share of the active plan needs \
                     ~{} MB plus KV headroom — free memory there, or reload the model to \
                     re-plan around it",
                    node.name,
                    usable / (1 << 20),
                    share.div_ceil(1 << 20),
                ),
            })
        })
        .collect()
}

/// Draining-in-plan rule: a node of the active distributed plan whose last
/// measured battery verdict says draining. New plans avoid such nodes
/// (docs/resilience.md); one already inside a plan is a standing risk.
pub fn draining_in_plan(plan: &Plan, nodes: &[NodeFacts]) -> Vec<Finding> {
    if plan.strategy != Strategy::PipelineParallel {
        return Vec::new();
    }
    plan.assignments
        .iter()
        .filter_map(|a| {
            let node = nodes.iter().find(|n| n.id == a.node.0)?;
            node.draining.then(|| Finding {
                severity: Severity::Warn,
                text: format!(
                    "node '{}' is draining its battery while serving the active plan — plug \
                     it in, or reload the model to re-plan without it",
                    node.name
                ),
            })
        })
        .collect()
}

/// Version/engine-skew rule, from STORED Hello data (retained by the mesh
/// at handshake time — including handshakes judged incompatible). A peer
/// that never exchanged a Hello has nothing recorded and stays silent. A
/// product-version difference is reported once per peer; an engine-build
/// difference is reported only when the versions match (same release,
/// different build flags — otherwise the version line already covers it).
pub fn version_skew(own_version: &str, own_engine: &str, peers: &[PeerStatus]) -> Vec<Finding> {
    peers
        .iter()
        .filter_map(
            |p| match (p.product_version.as_deref(), p.engine_build.as_deref()) {
                (Some(theirs), _) if theirs != own_version => Some(Finding {
                    severity: Severity::Warn,
                    text: format!(
                        "node '{}' runs OneBrain {theirs} while this node runs {own_version} — \
                         mismatched builds refuse to cooperate; run `onebrain self-update` on \
                         the node with the older version",
                        p.name
                    ),
                }),
                (_, Some(engine)) if engine != own_engine => Some(Finding {
                    severity: Severity::Warn,
                    text: format!(
                        "node '{}' reports engine build '{engine}' but this node runs \
                         '{own_engine}' — distributed sessions are refused across engine \
                         builds; reinstall both from the same release",
                        p.name
                    ),
                }),
                _ => None,
            },
        )
        .collect()
}

/// Solo-because-infeasible rule: the active plan runs on a single node AND
/// the scheduler's own selection notes record that it excluded peers
/// (cannot hold one layer, draining, …). The finding quotes the first
/// note verbatim — the measurement-backed reason, not a guess. A plan that
/// went solo via the auto-solo short-circuit carries no exclusion notes
/// and stays silent (solo-because-it-fits is not a problem).
pub fn solo_because_infeasible(plan: Option<&Plan>, explanation: Option<&str>) -> Vec<Finding> {
    let Some(plan) = plan else { return Vec::new() };
    let mut nodes: Vec<&str> = plan.assignments.iter().map(|a| a.node.0.as_str()).collect();
    nodes.sort_unstable();
    nodes.dedup();
    if nodes.len() != 1 {
        return Vec::new();
    }
    let Some(note) = explanation.and_then(|text| {
        text.lines()
            .find(|line| line.trim_start().starts_with("Node '") && line.contains(": excluded"))
    }) else {
        return Vec::new();
    };
    vec![Finding {
        severity: Severity::Info,
        text: format!(
            "the model runs on one node although peers are paired — the planner noted: {}",
            note.trim()
        ),
    }]
}

/// Battery-draining-worker rule: a Connected peer measured as draining
/// that is NOT in the active plan. Its capacity is out of the pool until
/// it charges (the scheduler admits draining nodes only when nothing fits
/// without them). Peers inside the plan are covered by the sharper
/// [`draining_in_plan`] warning instead.
pub fn battery_draining_worker(peers: &[PeerStatus], plan: Option<&Plan>) -> Vec<Finding> {
    peers
        .iter()
        .filter(|p| p.state == PeerState::Connected && p.draining && !in_plan(plan, &p.id))
        .map(|p| Finding {
            severity: Severity::Info,
            text: format!(
                "node '{}' is discharging on battery — new plans avoid it while it drains; \
                 plug it in to return its capacity to the pool",
                p.name
            ),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use onebrain_proto::plan::{Assignment, Epoch, LayerRange, NodeId};

    fn peer(name: &str, id: &str) -> PeerStatus {
        PeerStatus {
            name: name.to_string(),
            id: id.to_string(),
            state: PeerState::Connected,
            rtt_ms: Some(2.0),
            bandwidth_mbps: None,
            loss: None,
            last_seen_unix: Some(1),
            usable_memory_bytes: Some(8 << 30),
            prefill_tps: None,
            decode_tps: None,
            disk_mbps: None,
            product_version: None,
            engine_build: None,
            draining: false,
        }
    }

    fn plan(nodes: &[&str]) -> Plan {
        let assignments = nodes
            .iter()
            .enumerate()
            .map(|(i, id)| Assignment {
                node: NodeId(id.to_string()),
                layers: LayerRange {
                    start: i as u32 * 4,
                    end: i as u32 * 4 + 4,
                },
                stage: i as u32,
            })
            .collect();
        Plan {
            epoch: Epoch(1),
            model: "m".into(),
            strategy: if nodes.len() > 1 {
                Strategy::PipelineParallel
            } else {
                Strategy::Solo
            },
            assignments,
            ctx_len: 4096,
        }
    }

    fn facts(id: &str, usable: u64, draining: bool) -> NodeFacts {
        NodeFacts {
            id: id.to_string(),
            name: format!("name-{id}"),
            usable_memory_bytes: Some(usable),
            draining,
        }
    }

    #[test]
    fn slow_link_fires_only_on_measured_slow_bandwidth() {
        let mut fast = peer("fast", "f");
        fast.bandwidth_mbps = Some(940.0);
        let mut slow = peer("slow", "s");
        slow.bandwidth_mbps = Some(80.0);
        let unmeasured = peer("unknown", "u"); // no probe: no advice
        let mut down_slow = peer("gone", "g"); // not connected: no advice
        down_slow.bandwidth_mbps = Some(10.0);
        down_slow.state = PeerState::Down;

        let peers = vec![fast, slow, unmeasured, down_slow];
        let found = slow_link("head", &peers, None);
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].severity, Severity::Info, "not in a plan");
        assert!(
            found[0].text.contains("head\u{2194}slow"),
            "{}",
            found[0].text
        );
        assert!(found[0].text.contains("~80 Mbps"), "{}", found[0].text);
        assert!(
            found[0].text.contains("wired connection"),
            "{}",
            found[0].text
        );

        // The same slow link inside the active plan escalates to a warning.
        let active = plan(&["s", "own"]);
        let found = slow_link("head", &peers, Some(&active));
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].severity, Severity::Warn);
    }

    #[test]
    fn memory_starved_fires_on_a_starved_share_and_holds_otherwise() {
        // 8 layers over two nodes, 4 GiB model → 2 GiB share each.
        let active = plan(&["w", "h"]);
        let model = 4u64 << 30;
        // Healthy: 8 GiB usable ≫ (2 GiB share + 1 GiB headroom) / 2.
        let healthy = [facts("w", 8 << 30, false), facts("h", 8 << 30, false)];
        assert!(memory_starved(&active, model, &healthy).is_empty());
        // Starved: 1 GiB usable < 1.5 GiB — fires, naming node and MBs.
        let starved = [facts("w", 1 << 30, false), facts("h", 8 << 30, false)];
        let found = memory_starved(&active, model, &starved);
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].severity, Severity::Warn);
        assert!(found[0].text.contains("name-w"), "{}", found[0].text);
        assert!(
            found[0].text.contains("1024 MB usable"),
            "{}",
            found[0].text
        );
        assert!(found[0].text.contains("~2048 MB"), "{}", found[0].text);
        // Unmeasured memory: silent (no measurement, no advice).
        let unknown = [
            NodeFacts {
                usable_memory_bytes: None,
                ..facts("w", 0, false)
            },
            facts("h", 8 << 30, false),
        ];
        assert!(memory_starved(&active, model, &unknown).is_empty());
        // Solo plans have no shares to starve.
        assert!(memory_starved(&plan(&["h"]), model, &starved).is_empty());
    }

    #[test]
    fn draining_in_plan_names_exactly_the_draining_participant() {
        let active = plan(&["w1", "w2", "h"]);
        let nodes = [
            facts("w1", 8 << 30, true),
            facts("w2", 8 << 30, false),
            facts("h", 8 << 30, false),
        ];
        let found = draining_in_plan(&active, &nodes);
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].severity, Severity::Warn);
        assert!(found[0].text.contains("name-w1"), "{}", found[0].text);
        assert!(found[0].text.contains("active plan"), "{}", found[0].text);
        // A draining bystander (not in the plan) does not fire this rule.
        let bystander = [facts("w1", 8 << 30, false), facts("h", 8 << 30, false)];
        assert!(draining_in_plan(&active, &bystander).is_empty());
        // Solo plans never fire it.
        assert!(draining_in_plan(&plan(&["h"]), &nodes).is_empty());
    }

    #[test]
    fn version_skew_reports_version_first_then_engine_only_cases() {
        let mut older = peer("older", "o");
        older.product_version = Some("0.0.9".into());
        older.engine_build = Some("llama.cpp-x/cpu/p2".into());
        let mut engine_only = peer("engine", "e");
        engine_only.product_version = Some("0.1.0".into());
        engine_only.engine_build = Some("llama.cpp-x/cuda/p3".into());
        let mut same = peer("same", "s");
        same.product_version = Some("0.1.0".into());
        same.engine_build = Some("llama.cpp-x/cpu/p3".into());
        let silent = peer("nohello", "n"); // nothing retained: silent

        let peers = vec![older, engine_only, same, silent];
        let found = version_skew("0.1.0", "llama.cpp-x/cpu/p3", &peers);
        assert_eq!(found.len(), 2);
        assert!(found[0].text.contains("older") && found[0].text.contains("0.0.9"));
        assert!(
            found[0].text.contains("onebrain self-update"),
            "remedy missing: {}",
            found[0].text
        );
        assert!(
            found[1].text.contains("engine build") && found[1].text.contains("cuda"),
            "{}",
            found[1].text
        );
        assert!(found.iter().all(|f| f.severity == Severity::Warn));
    }

    #[test]
    fn solo_because_infeasible_needs_one_node_and_an_exclusion_note() {
        let solo = plan(&["h"]);
        let notes = "Pipeline-parallel across 1 nodes (…):\n  stage 0 …\n\
                     Node 'tiny': excluded — it cannot hold even one layer at ctx 4096 \
                     (512 MB budget vs ~900 MB per layer with KV)";
        let found = solo_because_infeasible(Some(&solo), Some(notes));
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].severity, Severity::Info);
        assert!(
            found[0].text.contains("cannot hold even one layer"),
            "must quote the planner's note: {}",
            found[0].text
        );
        // Auto-solo (no exclusion notes) is not a finding.
        let auto = "Solo on head 'h': model weights fit. auto-solo (§1.4).";
        assert!(solo_because_infeasible(Some(&solo), Some(auto)).is_empty());
        // A real multi-node plan never fires it, notes or not.
        let multi = plan(&["w", "h"]);
        assert!(solo_because_infeasible(Some(&multi), Some(notes)).is_empty());
        // No plan, no finding.
        assert!(solo_because_infeasible(None, Some(notes)).is_empty());
    }

    #[test]
    fn battery_draining_worker_skips_plan_members_and_disconnected_peers() {
        let mut idle_drainer = peer("idle", "i");
        idle_drainer.draining = true;
        let mut in_plan_drainer = peer("busy", "b");
        in_plan_drainer.draining = true;
        let mut gone_drainer = peer("gone", "g");
        gone_drainer.draining = true;
        gone_drainer.state = PeerState::Down;
        let charged = peer("charged", "c");

        let active = plan(&["b", "h"]);
        let peers = vec![idle_drainer, in_plan_drainer, gone_drainer, charged];
        let found = battery_draining_worker(&peers, Some(&active));
        assert_eq!(found.len(), 1, "only the connected, out-of-plan drainer");
        assert_eq!(found[0].severity, Severity::Info);
        assert!(found[0].text.contains("idle"), "{}", found[0].text);
        assert!(found[0].text.contains("plug it in"), "{}", found[0].text);
    }

    /// End-to-end shape check: `advise` merges the rules and serializes to
    /// the contract's `{severity, text}` schema.
    #[test]
    fn advise_assembles_findings_with_the_contract_schema() {
        let mut slow = peer("slow", "s");
        slow.bandwidth_mbps = Some(50.0);
        slow.draining = true;
        let peers = vec![slow];
        let input = AdvisorInput {
            node_name: "head",
            own_id: "h",
            product_version: "0.1.0",
            engine_build: "build",
            usable_memory_bytes: 8 << 30,
            draining: false,
            peers: &peers,
            plan: None,
            loaded_size_bytes: None,
        };
        let findings = advise(&input);
        assert_eq!(findings.len(), 2, "slow link + draining worker");
        let json = serde_json::to_value(&findings).unwrap();
        assert!(json[0]["severity"].is_string());
        assert!(json[0]["text"].is_string());
        assert_eq!(json[1]["severity"], "info");
    }
}
