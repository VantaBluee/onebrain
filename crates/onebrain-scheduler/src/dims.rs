//! Model dimensions for placement: per-layer weight bytes and the KV-cache
//! growth rate, both derived from the GGUF header (docs/scheduler-v1.md
//! "Placement algorithm" §1).
//!
//! # KV math
//!
//! Per layer, per context token, the cache holds one K and one V vector of
//! `n_embd_kv` elements each, stored f16:
//!
//! ```text
//! kv_bytes_per_layer_per_ctx_token = 2 (K+V) × n_embd_kv × 2 bytes (f16)
//! n_embd_kv = n_head_kv × head_dim,   head_dim = n_embd / n_head
//! ```
//!
//! Metadata keys (with documented fallbacks):
//! - `{arch}.embedding_length` (`n_embd`) — required; without it no KV
//!   estimate is possible at all.
//! - `{arch}.attention.head_count` (`n_head`) — missing ⇒ conservative
//!   `n_embd_kv = n_embd` (no GQA discount).
//! - `{arch}.attention.head_count_kv` (`n_head_kv`) — missing ⇒
//!   `n_head_kv = n_head` (multi-head attention, again no GQA discount).
//!
//! Both fallbacks only ever over-estimate the cache, so a plan computed with
//! them cannot OOM a node for KV reasons.
//!
//! # Weight bytes per layer
//!
//! Real bytes from the header's tensor ranges (each range extends to the next
//! tensor's start, so alignment padding is included — exactly the bytes a
//! shard download fetches and the mmap touches). Tensors named `blk.<i>.*`
//! belong to transformer layer `i`; `token_embd*` is amortized onto the
//! first layer and every other non-block tensor (`output*`, `output_norm*`,
//! rope tables, …) onto the last, matching the contract's "embedding/output
//! counted on their host" (stage 0 embeds, the tail stage projects logits).
//!
//! # MoE awareness (M7, docs/perf.md §8)
//!
//! `{arch}.expert_count` / `{arch}.expert_used_count` are read when present
//! (0 = dense). Routed-expert tensors (`blk.<i>.*_exps*` — the llama.cpp
//! naming for stacked per-expert FFN weights) are tracked per layer so
//! *compute* predictions can scale a layer's cost by the fraction of its
//! weights a token actually touches
//! ([`ModelDims::active_compute_fraction`]): dense math over-costs MoE
//! compute because only `expert_used_count` of `expert_count` routed
//! experts run per token. Shared experts (`*_shexp`) and the router
//! (`ffn_gate_inp`) run for every token and stay in the dense share. Two
//! things deliberately do NOT change: weight/memory math stays
//! tensor-range-driven (every expert is resident), and the KV rate is
//! untouched — the KV cache is attention-side (`n_head_kv × head_dim`) and
//! independent of expert count.

use onebrain_models::gguf::GgufHeader;

use crate::ScheduleError;

/// Everything the placement algorithm needs to know about a model's memory
/// shape, derived once per load from the GGUF header.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelDims {
    /// Transformer block count (`{arch}.block_count`).
    pub n_layers: u32,
    /// KV-cache bytes one layer accrues per context token (module docs).
    pub kv_bytes_per_layer_per_ctx_token: u64,
    /// Real weight bytes per layer, `len == n_layers`; embedding amortized
    /// onto `[0]`, output head onto `[n_layers - 1]` (module docs).
    pub weight_bytes_per_layer: Vec<u64>,
    /// Sum of `weight_bytes_per_layer` — the whole tensor-data section.
    pub total_weight_bytes: u64,
    /// Embedding width (`{arch}.embedding_length`); 0 = unknown (synthetic
    /// [`ModelDims::uniform`] dims). Sizes the per-ubatch pipeline boundary
    /// activation copy (`4·n_embd·n_ubatch` bytes) in scheduler v2's
    /// transfer term and budget reserve; 0 disables both, reproducing the
    /// pre-M7 math.
    pub n_embd: u64,
    /// MoE routed-expert count (`{arch}.expert_count`); 0 = dense.
    pub n_expert: u32,
    /// Routed experts active per token (`{arch}.expert_used_count`); 0 =
    /// dense or undeclared — compute is then costed dense (conservative:
    /// only ever over-estimates).
    pub n_expert_used: u32,
    /// Routed-expert weight bytes per layer (tensors named
    /// `blk.<i>.*_exps*`); always `len == n_layers`, all zeros for dense
    /// models. A subset of [`ModelDims::weight_bytes_per_layer`]; shared
    /// experts (`*_shexp`) and the router stay in the dense share because
    /// they run for every token (module docs).
    pub expert_weight_bytes_per_layer: Vec<u64>,
}

impl ModelDims {
    /// KV bytes one layer needs at the requested context length.
    pub fn kv_bytes_per_layer(&self, ctx_len: u32) -> u64 {
        self.kv_bytes_per_layer_per_ctx_token * ctx_len as u64
    }

    /// Total memory the model needs at `ctx_len`: all weights plus KV for
    /// every layer (the auto-solo requirement).
    pub fn total_required_bytes(&self, ctx_len: u32) -> u64 {
        self.total_weight_bytes + self.n_layers as u64 * self.kv_bytes_per_layer(ctx_len)
    }

    /// Mean weight bytes per layer (rounded up), the uniform-layer
    /// approximation the capacity model uses (see `plan_v1` docs).
    pub fn mean_weight_bytes_per_layer(&self) -> u64 {
        self.total_weight_bytes
            .div_ceil(self.n_layers.max(1) as u64)
    }

    /// Whether active-expert compute scaling applies (docs/perf.md §8):
    /// experts are declared AND fewer are active per token than resident.
    /// `expert_used_count >= expert_count` (or either missing) means every
    /// resident expert runs, i.e. dense compute.
    fn moe_scaled(&self) -> bool {
        self.n_expert > 0 && self.n_expert_used > 0 && self.n_expert_used < self.n_expert
    }

    /// Fraction of one layer's weight bytes a single token actually touches:
    /// 1.0 for dense layers; for MoE layers the routed-expert bytes are
    /// scaled by `expert_used_count / expert_count` while the dense share
    /// (attention, norms, shared experts, router) counts in full. Compute
    /// predictions use this; memory math never does (all experts are
    /// resident).
    pub fn active_compute_fraction(&self, layer: usize) -> f64 {
        if !self.moe_scaled() {
            return 1.0;
        }
        let total = self.weight_bytes_per_layer.get(layer).copied().unwrap_or(0);
        let expert = self
            .expert_weight_bytes_per_layer
            .get(layer)
            .copied()
            .unwrap_or(0)
            .min(total);
        if total == 0 {
            // A layer with no recorded bytes has nothing to scale; neutral
            // (conservative) rather than dividing by zero.
            return 1.0;
        }
        let used = self.n_expert_used as f64 / self.n_expert as f64;
        ((total - expert) as f64 + expert as f64 * used) / total as f64
    }

    /// Compute-relative "layer units" of the contiguous range
    /// `[start, end)`: the sum of each layer's
    /// [`active_compute_fraction`](Self::active_compute_fraction). Exactly
    /// `end - start` for dense models, less for MoE — scheduler v2's
    /// predicted-cost term divides this (not the raw layer count) by a
    /// node's `decode_tps` so MoE compute stops being over-costed.
    pub fn active_compute_units(&self, start: u32, end: u32) -> f64 {
        let span = end.saturating_sub(start);
        if !self.moe_scaled() {
            // Dense shortcut keeps the value bit-exact with the layer count
            // (no float summation of 1.0s).
            return span as f64;
        }
        (start..end)
            .map(|l| self.active_compute_fraction(l as usize))
            .sum()
    }

    /// Synthetic dims for callers that only know a total byte size and a
    /// layer count (the M3-shaped [`crate::PlanInput`] compatibility path):
    /// weights spread uniformly, KV growth rate unknown and therefore zero —
    /// such plans budget only the fixed overhead reserve, like M3's flat
    /// ceiling stood in for both.
    pub fn uniform(total_weight_bytes: u64, n_layers: u32) -> ModelDims {
        let n = n_layers.max(1);
        let per = total_weight_bytes / n as u64;
        let mut weights = vec![per; n as usize];
        // Keep the exact total: the remainder rides on the last layer.
        let assigned = per * n as u64;
        if let Some(last) = weights.last_mut() {
            *last += total_weight_bytes - assigned;
        }
        ModelDims {
            n_layers: n,
            kv_bytes_per_layer_per_ctx_token: 0,
            weight_bytes_per_layer: weights,
            total_weight_bytes,
            // Unknown embedding width and no expert data: v2's transfer
            // term and MoE scaling both degrade to today's dense math.
            n_embd: 0,
            n_expert: 0,
            n_expert_used: 0,
            expert_weight_bytes_per_layer: vec![0; n as usize],
        }
    }
}

/// Read a required `{arch}.<suffix>` u64 metadata value.
fn meta_u64(header: &GgufHeader, key: &str) -> Option<u64> {
    header.metadata.get(key).and_then(|v| v.as_u64())
}

/// Derive [`ModelDims`] from a parsed GGUF header and the file's total size
/// (needed to close the last tensor's byte range).
pub fn model_dims(header: &GgufHeader, file_size: u64) -> Result<ModelDims, ScheduleError> {
    let arch = header
        .architecture()
        .ok_or_else(|| ScheduleError::MissingMetadata {
            key: "general.architecture".to_string(),
        })?
        .to_string();

    let n_layers_u64 = meta_u64(header, &format!("{arch}.block_count"))
        .filter(|n| *n > 0)
        .ok_or_else(|| ScheduleError::MissingMetadata {
            key: format!("{arch}.block_count"),
        })?;
    let n_layers = u32::try_from(n_layers_u64).map_err(|_| ScheduleError::BadModel {
        detail: format!("absurd layer count {n_layers_u64}"),
    })?;

    let n_embd = meta_u64(header, &format!("{arch}.embedding_length"))
        .filter(|n| *n > 0)
        .ok_or_else(|| ScheduleError::MissingMetadata {
            key: format!("{arch}.embedding_length"),
        })?;

    // GQA-aware KV width with the conservative fallbacks from the module
    // docs: anything missing degrades toward n_embd (over-estimating KV).
    let n_head = meta_u64(header, &format!("{arch}.attention.head_count")).filter(|n| *n > 0);
    let n_embd_kv = match n_head {
        Some(n_head) => {
            let head_dim = n_embd / n_head;
            let n_head_kv = meta_u64(header, &format!("{arch}.attention.head_count_kv"))
                .filter(|n| *n > 0)
                .unwrap_or(n_head);
            (n_head_kv * head_dim).min(n_embd).max(1)
        }
        None => n_embd,
    };
    let kv_bytes_per_layer_per_ctx_token = 2 * 2 * n_embd_kv; // K+V, f16

    // MoE metadata (docs/perf.md §8). Optional: absent or zero means dense.
    // The KV rate above is deliberately NOT touched — the cache is
    // attention-side and independent of expert count (module docs).
    let expert_u32 = |suffix: &str| -> Result<u32, ScheduleError> {
        let key = format!("{arch}.{suffix}");
        match meta_u64(header, &key).filter(|n| *n > 0) {
            None => Ok(0),
            Some(v) => u32::try_from(v).map_err(|_| ScheduleError::BadModel {
                detail: format!("absurd {key} {v}"),
            }),
        }
    };
    let n_expert = expert_u32("expert_count")?;
    let n_expert_used = expert_u32("expert_used_count")?;
    if n_expert > 0 && n_expert_used > n_expert {
        return Err(ScheduleError::BadModel {
            detail: format!(
                "the header declares {n_expert_used} active experts per token but only \
                 {n_expert} experts"
            ),
        });
    }

    let ranges = header
        .tensor_ranges(file_size)
        .map_err(|e| ScheduleError::BadModel {
            detail: e.to_string(),
        })?;

    let mut weight_bytes_per_layer = vec![0u64; n_layers as usize];
    let mut expert_weight_bytes_per_layer = vec![0u64; n_layers as usize];
    let mut total_weight_bytes = 0u64;
    for range in &ranges {
        let size = range.end - range.start;
        total_weight_bytes += size;
        let slot = if let Some(rest) = range.name.strip_prefix("blk.") {
            let idx: usize = rest
                .split('.')
                .next()
                .and_then(|s| s.parse().ok())
                .ok_or_else(|| ScheduleError::BadModel {
                    detail: format!("tensor '{}' has a malformed block index", range.name),
                })?;
            if idx >= n_layers as usize {
                return Err(ScheduleError::BadModel {
                    detail: format!(
                        "tensor '{}' names layer {idx} but the model declares {n_layers} layers",
                        range.name
                    ),
                });
            }
            // Routed-expert stacks carry the `_exps` name part
            // (`ffn_gate_exps`, `ffn_up_exps`, `ffn_down_exps`); shared
            // experts are `_shexp` (no trailing `s`) and stay dense —
            // they run for every token (module docs, MoE section).
            if rest.contains("_exps") {
                expert_weight_bytes_per_layer[idx] += size;
            }
            idx
        } else if range.name.starts_with("token_embd") {
            0
        } else {
            n_layers as usize - 1
        };
        weight_bytes_per_layer[slot] += size;
    }

    Ok(ModelDims {
        n_layers,
        kv_bytes_per_layer_per_ctx_token,
        weight_bytes_per_layer,
        total_weight_bytes,
        n_embd,
        n_expert,
        n_expert_used,
        expert_weight_bytes_per_layer,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ScheduleError;

    /// Minimal synthetic GGUF header builder (the models crate's test
    /// builder is `#[cfg(test)]`-private, so this crate keeps its own).
    /// Layout per spec v3: magic, version, tensor_count, kv_count, KVs,
    /// tensor infos.
    pub(crate) struct Gguf {
        kv_count: u64,
        tensor_count: u64,
        kvs: Vec<u8>,
        tensors: Vec<u8>,
    }

    const T_U32: u32 = 4;
    const T_STRING: u32 = 8;

    impl Gguf {
        pub(crate) fn new() -> Self {
            Gguf {
                kv_count: 0,
                tensor_count: 0,
                kvs: Vec::new(),
                tensors: Vec::new(),
            }
        }

        fn string_into(out: &mut Vec<u8>, s: &str) {
            out.extend((s.len() as u64).to_le_bytes());
            out.extend(s.as_bytes());
        }

        pub(crate) fn kv_str(mut self, key: &str, val: &str) -> Self {
            Self::string_into(&mut self.kvs, key);
            self.kvs.extend(T_STRING.to_le_bytes());
            Self::string_into(&mut self.kvs, val);
            self.kv_count += 1;
            self
        }

        pub(crate) fn kv_u32(mut self, key: &str, val: u32) -> Self {
            Self::string_into(&mut self.kvs, key);
            self.kvs.extend(T_U32.to_le_bytes());
            self.kvs.extend(val.to_le_bytes());
            self.kv_count += 1;
            self
        }

        pub(crate) fn tensor(mut self, name: &str, offset: u64) -> Self {
            Self::string_into(&mut self.tensors, name);
            self.tensors.extend(1u32.to_le_bytes()); // n_dims
            self.tensors.extend(1u64.to_le_bytes()); // dim[0]
            self.tensors.extend(0u32.to_le_bytes()); // ggml type (f32)
            self.tensors.extend(offset.to_le_bytes());
            self.tensor_count += 1;
            self
        }

        pub(crate) fn parse(self) -> GgufHeader {
            let mut buf = Vec::new();
            buf.extend(0x4655_4747u32.to_le_bytes()); // "GGUF"
            buf.extend(3u32.to_le_bytes());
            buf.extend(self.tensor_count.to_le_bytes());
            buf.extend(self.kv_count.to_le_bytes());
            buf.extend(&self.kvs);
            buf.extend(&self.tensors);
            GgufHeader::parse(&buf).expect("synthetic header must parse")
        }
    }

    /// A 2-layer llama-flavored header: token_embd, blk.0, blk.1, output —
    /// 4096 bytes each in the data section.
    fn two_layer_header() -> GgufHeader {
        Gguf::new()
            .kv_str("general.architecture", "llama")
            .kv_u32("llama.block_count", 2)
            .kv_u32("llama.embedding_length", 4096)
            .kv_u32("llama.attention.head_count", 32)
            .kv_u32("llama.attention.head_count_kv", 8)
            .tensor("token_embd.weight", 0)
            .tensor("blk.0.attn_q.weight", 4096)
            .tensor("blk.1.attn_q.weight", 8192)
            .tensor("output.weight", 12288)
            .parse()
    }

    #[test]
    fn kv_rate_matches_hand_computed_gqa_example() {
        // n_embd 4096, n_head 32 → head_dim 128; n_head_kv 8 →
        // n_embd_kv = 8 × 128 = 1024; rate = 2 (K+V) × 1024 × 2 (f16) =
        // 4096 bytes per layer per context token.
        let h = two_layer_header();
        let file_size = h.data_offset + 16384;
        let dims = model_dims(&h, file_size).unwrap();
        assert_eq!(dims.kv_bytes_per_layer_per_ctx_token, 4096);
        // At ctx 2048 one layer's KV is 8 MiB; both layers 16 MiB.
        assert_eq!(dims.kv_bytes_per_layer(2048), 8 << 20);
        assert_eq!(
            dims.total_required_bytes(2048),
            dims.total_weight_bytes + (16 << 20)
        );
    }

    #[test]
    fn kv_fallbacks_are_conservative() {
        // Missing head_count_kv → n_head_kv = n_head (no GQA discount):
        // n_embd_kv = 32 × 128 = 4096 → rate 16384.
        let h = Gguf::new()
            .kv_str("general.architecture", "llama")
            .kv_u32("llama.block_count", 1)
            .kv_u32("llama.embedding_length", 4096)
            .kv_u32("llama.attention.head_count", 32)
            .tensor("blk.0.w", 0)
            .parse();
        let dims = model_dims(&h, h.data_offset + 64).unwrap();
        assert_eq!(dims.kv_bytes_per_layer_per_ctx_token, 16384);

        // Missing head_count entirely → conservative n_embd_kv = n_embd.
        let h = Gguf::new()
            .kv_str("general.architecture", "llama")
            .kv_u32("llama.block_count", 1)
            .kv_u32("llama.embedding_length", 4096)
            .tensor("blk.0.w", 0)
            .parse();
        let dims = model_dims(&h, h.data_offset + 64).unwrap();
        assert_eq!(dims.kv_bytes_per_layer_per_ctx_token, 16384);
    }

    #[test]
    fn missing_embedding_length_names_the_key() {
        let h = Gguf::new()
            .kv_str("general.architecture", "llama")
            .kv_u32("llama.block_count", 1)
            .tensor("blk.0.w", 0)
            .parse();
        match model_dims(&h, h.data_offset + 64).unwrap_err() {
            ScheduleError::MissingMetadata { key } => {
                assert_eq!(key, "llama.embedding_length");
            }
            other => panic!("expected MissingMetadata, got {other:?}"),
        }
    }

    #[test]
    fn weights_amortize_embedding_first_and_output_last() {
        let h = two_layer_header();
        let file_size = h.data_offset + 16384;
        let dims = model_dims(&h, file_size).unwrap();
        assert_eq!(dims.n_layers, 2);
        // token_embd (4096) rides on layer 0 with blk.0 (4096); output
        // (4096, extends to file end) rides on layer 1 with blk.1 (4096).
        assert_eq!(dims.weight_bytes_per_layer, vec![8192, 8192]);
        assert_eq!(dims.total_weight_bytes, 16384);
        assert_eq!(dims.mean_weight_bytes_per_layer(), 8192);
    }

    #[test]
    fn out_of_range_block_index_is_rejected() {
        let h = Gguf::new()
            .kv_str("general.architecture", "llama")
            .kv_u32("llama.block_count", 1)
            .kv_u32("llama.embedding_length", 64)
            .tensor("blk.7.w", 0)
            .parse();
        match model_dims(&h, h.data_offset + 64).unwrap_err() {
            ScheduleError::BadModel { detail } => {
                assert!(detail.contains("blk.7.w"), "{detail}");
            }
            other => panic!("expected BadModel, got {other:?}"),
        }
    }

    #[test]
    fn uniform_dims_keep_the_exact_total() {
        let dims = ModelDims::uniform(1000, 3);
        assert_eq!(dims.n_layers, 3);
        assert_eq!(dims.kv_bytes_per_layer_per_ctx_token, 0);
        assert_eq!(dims.weight_bytes_per_layer, vec![333, 333, 334]);
        assert_eq!(dims.total_weight_bytes, 1000);
        // Synthetic dims declare no embedding width and no experts: v2's
        // transfer term and MoE scaling both stay disabled.
        assert_eq!(dims.n_embd, 0);
        assert_eq!(dims.n_expert, 0);
        assert_eq!(dims.n_expert_used, 0);
        assert_eq!(dims.expert_weight_bytes_per_layer, vec![0, 0, 0]);
        assert_eq!(dims.active_compute_units(0, 3), 3.0);
    }

    #[test]
    fn n_embd_is_captured_from_the_header() {
        let h = two_layer_header();
        let dims = model_dims(&h, h.data_offset + 16384).unwrap();
        assert_eq!(dims.n_embd, 4096);
    }

    #[test]
    fn dense_models_report_no_experts_and_unit_fractions() {
        let h = two_layer_header();
        let dims = model_dims(&h, h.data_offset + 16384).unwrap();
        assert_eq!(dims.n_expert, 0);
        assert_eq!(dims.n_expert_used, 0);
        assert_eq!(dims.expert_weight_bytes_per_layer, vec![0, 0]);
        assert_eq!(dims.active_compute_fraction(0), 1.0);
        assert_eq!(dims.active_compute_units(0, 2), 2.0);
    }

    /// A 1-layer MoE header: 4096 B of dense attention weights, 4096 B of
    /// routed experts (`_exps`), 4096 B of shared expert (`_shexp`, dense),
    /// 4096 B of router — 16384 B in the data section.
    fn moe_header(expert_count: u32, expert_used: u32) -> GgufHeader {
        Gguf::new()
            .kv_str("general.architecture", "llama")
            .kv_u32("llama.block_count", 1)
            .kv_u32("llama.embedding_length", 4096)
            .kv_u32("llama.expert_count", expert_count)
            .kv_u32("llama.expert_used_count", expert_used)
            .tensor("blk.0.attn_q.weight", 0)
            .tensor("blk.0.ffn_gate_exps.weight", 4096)
            .tensor("blk.0.ffn_down_shexp.weight", 8192)
            .tensor("blk.0.ffn_gate_inp.weight", 12288)
            .parse()
    }

    #[test]
    fn moe_metadata_and_expert_tensor_bytes_parse() {
        let h = moe_header(8, 2);
        let dims = model_dims(&h, h.data_offset + 16384).unwrap();
        assert_eq!(dims.n_expert, 8);
        assert_eq!(dims.n_expert_used, 2);
        // Only the routed `_exps` stack counts as expert bytes; the shared
        // expert (`_shexp`) and the router stay in the dense share.
        assert_eq!(dims.expert_weight_bytes_per_layer, vec![4096]);
        assert_eq!(dims.weight_bytes_per_layer, vec![16384]);
        // Active fraction: (16384 - 4096 + 4096 × 2/8) / 16384 = 0.8125.
        assert_eq!(dims.active_compute_fraction(0), 0.8125);
        assert_eq!(dims.active_compute_units(0, 1), 0.8125);
        // KV math is expert-independent: same GQA-less rate as any dense
        // header with these attention dims would produce.
        assert_eq!(dims.kv_bytes_per_layer_per_ctx_token, 16384);
    }

    #[test]
    fn moe_with_all_experts_active_costs_dense() {
        // used == count means every resident expert runs: no scaling.
        let h = moe_header(4, 4);
        let dims = model_dims(&h, h.data_offset + 16384).unwrap();
        assert_eq!(dims.active_compute_fraction(0), 1.0);
        assert_eq!(dims.active_compute_units(0, 1), 1.0);
    }

    #[test]
    fn moe_used_beyond_count_is_rejected() {
        let h = moe_header(2, 4);
        match model_dims(&h, h.data_offset + 16384).unwrap_err() {
            ScheduleError::BadModel { detail } => {
                assert!(detail.contains("4 active experts"), "{detail}");
                assert!(detail.contains("2 experts"), "{detail}");
            }
            other => panic!("expected BadModel, got {other:?}"),
        }
    }
}
