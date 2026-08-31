//! Tolerant client-side views of `GET /api/internal/metrics`
//! (docs/product.md §1).
//!
//! The endpoint lands daemon-side in parallel with this consumer and the
//! document is contractually additive-stable — so unknown fields are
//! ignored, every field defaults, and the plausible spellings of the
//! Hello-derived fields are accepted as aliases. Only what doctor's skew
//! check reads is modeled; grow these types as consumers appear (an unread
//! field would only earn a dead-code warning today).

use serde::Deserialize;

/// The whole metrics document. `plan`, `requests` and `advisor` exist in
/// the document but have no CLI reader yet.
#[derive(Debug, Default, Clone, Deserialize)]
#[serde(default)]
pub struct MetricsDoc {
    pub node: MetricsNode,
    pub peers: Vec<MetricsPeer>,
}

/// This node's identity as the daemon reports it.
#[derive(Debug, Default, Clone, Deserialize)]
#[serde(default)]
pub struct MetricsNode {
    #[serde(alias = "product_version")]
    pub version: String,
    #[serde(alias = "engine_build_id", alias = "engine_build_hash")]
    pub engine_build: String,
}

/// One paired peer, with the version + engine build its Hello carried
/// (onebrain-proto's `Hello { product_version, engine_build }`).
#[derive(Debug, Default, Clone, Deserialize)]
#[serde(default)]
pub struct MetricsPeer {
    pub name: String,
    #[serde(alias = "product_version")]
    pub version: String,
    #[serde(alias = "engine_build_id", alias = "engine_build_hash")]
    pub engine_build: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_a_full_document_ignoring_everything_unmodeled() {
        // Shaped like docs/product.md §1: node/peers plus the sections the
        // CLI does not read yet, and deliberate unknown fields throughout.
        let doc: MetricsDoc = serde_json::from_str(
            r#"{
                "node": {
                    "name": "desk", "platform": "windows", "version": "0.1.0",
                    "engine_build": "llama.cpp-abc+cpu",
                    "memory": { "usable": 1, "total": 2 },
                    "battery": null, "sleep_inhibited": true
                },
                "peers": [
                    { "name": "laptop", "id_prefix": "ab12cd34",
                      "state": "Connected", "rtt_ms": 1.5,
                      "bandwidth_mbps": 940.0, "loss": 0.0,
                      "version": "0.2.0", "engine_build": "llama.cpp-def+cpu",
                      "profile": { "decode_tps": 12.0 } }
                ],
                "plan": { "epoch": 3 },
                "requests": [ { "id": "r1", "prefill_ms": 10 } ],
                "advisor": [ { "severity": "warn", "text": "slow link" } ]
            }"#,
        )
        .unwrap();
        assert_eq!(doc.node.version, "0.1.0");
        assert_eq!(doc.node.engine_build, "llama.cpp-abc+cpu");
        assert_eq!(doc.peers.len(), 1);
        assert_eq!(doc.peers[0].name, "laptop");
        assert_eq!(doc.peers[0].version, "0.2.0");
        assert_eq!(doc.peers[0].engine_build, "llama.cpp-def+cpu");
    }

    #[test]
    fn missing_sections_default_instead_of_failing() {
        let doc: MetricsDoc = serde_json::from_str("{}").unwrap();
        assert!(doc.peers.is_empty());
        assert_eq!(doc.node.version, "");

        // The Hello spelling of the fields is accepted as an alias.
        let doc: MetricsDoc = serde_json::from_str(
            r#"{ "peers": [ { "name": "p", "product_version": "0.3.0",
                              "engine_build_id": "b" } ] }"#,
        )
        .unwrap();
        assert_eq!(doc.peers[0].version, "0.3.0");
        assert_eq!(doc.peers[0].engine_build, "b");
    }
}
