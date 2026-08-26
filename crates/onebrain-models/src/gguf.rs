//! GGUF header parsing (spec versions 2 and 3, little-endian).
//!
//! The header (magic, metadata KVs, tensor infos) lives at the front of the
//! file and is small relative to the weights, so a node can fetch just the
//! header over HTTP, then compute the exact byte range of every tensor and
//! download only the layers a plan assigns to it (§6 of the product spec).
//!
//! Tensor byte sizes are derived from the offset of the *next* tensor (and
//! the file size for the last one) rather than from a ggml type-size table:
//! that is exactly how range requests slice the file, and it stays correct
//! when upstream adds new quantization types.
//!
//! Inputs are untrusted (downloaded files): every length is bounds-checked
//! against the available bytes before allocation.

use std::collections::BTreeMap;

pub const GGUF_MAGIC: u32 = 0x4655_4747; // "GGUF" little-endian
pub const GGUF_DEFAULT_ALIGNMENT: u64 = 32;

#[derive(Debug, thiserror::Error)]
pub enum GgufError {
    #[error("not a GGUF file (bad magic)")]
    BadMagic,
    #[error("unsupported GGUF version {0} (supported: 2, 3)")]
    UnsupportedVersion(u32),
    #[error(
        "header is longer than the provided bytes; fetch at least {need_hint} bytes and retry"
    )]
    NeedMoreData { need_hint: u64 },
    #[error("malformed header: {0}")]
    Malformed(&'static str),
}

/// A GGUF metadata value. Arrays are kept homogeneous per spec.
#[derive(Debug, Clone, PartialEq)]
pub enum MetaValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    Str(String),
    U64(u64),
    I64(i64),
    F64(f64),
    Array(Vec<MetaValue>),
}

impl MetaValue {
    pub fn as_u64(&self) -> Option<u64> {
        match *self {
            MetaValue::U8(v) => Some(v as u64),
            MetaValue::U16(v) => Some(v as u64),
            MetaValue::U32(v) => Some(v as u64),
            MetaValue::U64(v) => Some(v),
            MetaValue::I8(v) if v >= 0 => Some(v as u64),
            MetaValue::I16(v) if v >= 0 => Some(v as u64),
            MetaValue::I32(v) if v >= 0 => Some(v as u64),
            MetaValue::I64(v) if v >= 0 => Some(v as u64),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            MetaValue::Str(s) => Some(s),
            _ => None,
        }
    }
}

/// One tensor's descriptor from the header.
#[derive(Debug, Clone, PartialEq)]
pub struct TensorInfo {
    pub name: String,
    pub dims: Vec<u64>,
    /// Raw ggml type id; interpreting it is the engine's business.
    pub ggml_type: u32,
    /// Offset relative to the start of the tensor-data section.
    pub offset: u64,
}

/// Parsed GGUF header.
#[derive(Debug, Clone)]
pub struct GgufHeader {
    pub version: u32,
    pub metadata: BTreeMap<String, MetaValue>,
    /// In file order (not offset order).
    pub tensors: Vec<TensorInfo>,
    /// Byte length of the header itself (magic through last tensor info).
    pub header_len: u64,
    pub alignment: u64,
    /// Absolute file offset where the tensor-data section begins.
    pub data_offset: u64,
}

/// A tensor's absolute byte range within the file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorRange {
    pub name: String,
    pub start: u64,
    /// Exclusive; includes any alignment padding up to the next tensor.
    pub end: u64,
}

impl GgufHeader {
    /// Parse a header from the leading bytes of a GGUF file. If the slice is
    /// too short, returns [`GgufError::NeedMoreData`] with a fetch hint.
    pub fn parse(bytes: &[u8]) -> Result<GgufHeader, GgufError> {
        let mut cur = Cursor { bytes, pos: 0 };

        if cur.u32()? != GGUF_MAGIC {
            return Err(GgufError::BadMagic);
        }
        let version = cur.u32()?;
        if version != 2 && version != 3 {
            return Err(GgufError::UnsupportedVersion(version));
        }
        let tensor_count = cur.u64()?;
        let kv_count = cur.u64()?;

        let mut metadata = BTreeMap::new();
        for _ in 0..kv_count {
            let key = cur.string()?;
            let value = cur.value()?;
            metadata.insert(key, value);
        }

        let mut tensors = Vec::new();
        for _ in 0..tensor_count {
            let name = cur.string()?;
            let n_dims = cur.u32()?;
            if n_dims > 8 {
                return Err(GgufError::Malformed("tensor has more than 8 dimensions"));
            }
            let mut dims = Vec::with_capacity(n_dims as usize);
            for _ in 0..n_dims {
                dims.push(cur.u64()?);
            }
            let ggml_type = cur.u32()?;
            let offset = cur.u64()?;
            tensors.push(TensorInfo {
                name,
                dims,
                ggml_type,
                offset,
            });
        }

        let header_len = cur.pos as u64;
        let alignment = metadata
            .get("general.alignment")
            .and_then(MetaValue::as_u64)
            .filter(|a| *a > 0 && a.is_power_of_two())
            .unwrap_or(GGUF_DEFAULT_ALIGNMENT);
        let data_offset = header_len.div_ceil(alignment) * alignment;

        Ok(GgufHeader {
            version,
            metadata,
            tensors,
            header_len,
            alignment,
            data_offset,
        })
    }

    /// The model architecture (e.g. "llama", "qwen3"), if declared.
    pub fn architecture(&self) -> Option<&str> {
        self.metadata.get("general.architecture")?.as_str()
    }

    /// Number of transformer blocks, resolved via the architecture prefix.
    pub fn block_count(&self) -> Option<u64> {
        let arch = self.architecture()?;
        self.metadata.get(&format!("{arch}.block_count"))?.as_u64()
    }

    /// Absolute byte ranges for every tensor, given the total file size.
    /// Each range extends to the next tensor's start (or file end), so
    /// alignment padding travels with the preceding tensor.
    pub fn tensor_ranges(&self, file_size: u64) -> Result<Vec<TensorRange>, GgufError> {
        let mut sorted: Vec<&TensorInfo> = self.tensors.iter().collect();
        sorted.sort_by_key(|t| t.offset);

        let mut out = Vec::with_capacity(sorted.len());
        for (i, t) in sorted.iter().enumerate() {
            let start = self.data_offset + t.offset;
            let end = match sorted.get(i + 1) {
                Some(next) => self.data_offset + next.offset,
                None => file_size,
            };
            if start > end || end > file_size {
                return Err(GgufError::Malformed(
                    "tensor offsets exceed file size or overlap",
                ));
            }
            out.push(TensorRange {
                name: t.name.clone(),
                start,
                end,
            });
        }
        Ok(out)
    }
}

// GGUF metadata value type ids per spec.
const T_U8: u32 = 0;
const T_I8: u32 = 1;
const T_U16: u32 = 2;
const T_I16: u32 = 3;
const T_U32: u32 = 4;
const T_I32: u32 = 5;
const T_F32: u32 = 6;
const T_BOOL: u32 = 7;
const T_STRING: u32 = 8;
const T_ARRAY: u32 = 9;
const T_U64: u32 = 10;
const T_I64: u32 = 11;
const T_F64: u32 = 12;

struct Cursor<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl Cursor<'_> {
    fn take(&mut self, n: usize) -> Result<&[u8], GgufError> {
        let end = self
            .pos
            .checked_add(n)
            .ok_or(GgufError::Malformed("length overflow"))?;
        if end > self.bytes.len() {
            return Err(GgufError::NeedMoreData {
                need_hint: (end as u64).max(self.bytes.len() as u64 * 2),
            });
        }
        let s = &self.bytes[self.pos..end];
        self.pos = end;
        Ok(s)
    }

    fn u32(&mut self) -> Result<u32, GgufError> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    fn u64(&mut self) -> Result<u64, GgufError> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }

    fn string(&mut self) -> Result<String, GgufError> {
        let len = self.u64()?;
        if len > 1 << 24 {
            return Err(GgufError::Malformed("unreasonably long string"));
        }
        let raw = self.take(len as usize)?;
        String::from_utf8(raw.to_vec()).map_err(|_| GgufError::Malformed("string is not UTF-8"))
    }

    fn value(&mut self) -> Result<MetaValue, GgufError> {
        let ty = self.u32()?;
        self.typed_value(ty)
    }

    fn typed_value(&mut self, ty: u32) -> Result<MetaValue, GgufError> {
        Ok(match ty {
            T_U8 => MetaValue::U8(self.take(1)?[0]),
            T_I8 => MetaValue::I8(self.take(1)?[0] as i8),
            T_U16 => MetaValue::U16(u16::from_le_bytes(self.take(2)?.try_into().unwrap())),
            T_I16 => MetaValue::I16(i16::from_le_bytes(self.take(2)?.try_into().unwrap())),
            T_U32 => MetaValue::U32(self.u32()?),
            T_I32 => MetaValue::I32(self.u32()? as i32),
            T_F32 => MetaValue::F32(f32::from_le_bytes(self.take(4)?.try_into().unwrap())),
            T_BOOL => MetaValue::Bool(self.take(1)?[0] != 0),
            T_STRING => MetaValue::Str(self.string()?),
            T_U64 => MetaValue::U64(self.u64()?),
            T_I64 => MetaValue::I64(self.u64()? as i64),
            T_F64 => MetaValue::F64(f64::from_le_bytes(self.take(8)?.try_into().unwrap())),
            T_ARRAY => {
                let elem_ty = self.u32()?;
                if elem_ty == T_ARRAY {
                    return Err(GgufError::Malformed("nested arrays are not allowed"));
                }
                let count = self.u64()?;
                // Each element is at least one byte on the wire; reject
                // counts the remaining input cannot possibly satisfy before
                // allocating.
                if count > (self.bytes.len() - self.pos) as u64 {
                    return Err(GgufError::NeedMoreData {
                        need_hint: self.pos as u64 + count,
                    });
                }
                let mut items = Vec::with_capacity(count as usize);
                for _ in 0..count {
                    items.push(self.typed_value(elem_ty)?);
                }
                MetaValue::Array(items)
            }
            _ => return Err(GgufError::Malformed("unknown metadata value type")),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a tiny synthetic GGUF header for tests.
    struct Builder {
        buf: Vec<u8>,
        kv_count: u64,
        tensor_count: u64,
        kvs: Vec<u8>,
        tensors: Vec<u8>,
    }

    impl Builder {
        fn new() -> Self {
            Builder {
                buf: Vec::new(),
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

        fn kv_str(mut self, key: &str, val: &str) -> Self {
            Self::string_into(&mut self.kvs, key);
            self.kvs.extend(T_STRING.to_le_bytes());
            Self::string_into(&mut self.kvs, val);
            self.kv_count += 1;
            self
        }

        fn kv_u32(mut self, key: &str, val: u32) -> Self {
            Self::string_into(&mut self.kvs, key);
            self.kvs.extend(T_U32.to_le_bytes());
            self.kvs.extend(val.to_le_bytes());
            self.kv_count += 1;
            self
        }

        fn tensor(mut self, name: &str, dims: &[u64], ggml_type: u32, offset: u64) -> Self {
            Self::string_into(&mut self.tensors, name);
            self.tensors.extend((dims.len() as u32).to_le_bytes());
            for d in dims {
                self.tensors.extend(d.to_le_bytes());
            }
            self.tensors.extend(ggml_type.to_le_bytes());
            self.tensors.extend(offset.to_le_bytes());
            self.tensor_count += 1;
            self
        }

        fn build(mut self) -> Vec<u8> {
            self.buf.extend(GGUF_MAGIC.to_le_bytes());
            self.buf.extend(3u32.to_le_bytes());
            self.buf.extend(self.tensor_count.to_le_bytes());
            self.buf.extend(self.kv_count.to_le_bytes());
            self.buf.extend(&self.kvs);
            self.buf.extend(&self.tensors);
            self.buf
        }
    }

    #[test]
    fn rejects_bad_magic() {
        assert!(matches!(
            GgufHeader::parse(&[0u8; 32]),
            Err(GgufError::BadMagic)
        ));
    }

    #[test]
    fn short_input_asks_for_more() {
        let full = Builder::new()
            .kv_str("general.architecture", "llama")
            .build();
        let short = &full[..full.len() - 3];
        assert!(matches!(
            GgufHeader::parse(short),
            Err(GgufError::NeedMoreData { .. })
        ));
        assert!(GgufHeader::parse(&full).is_ok());
    }

    #[test]
    fn parses_metadata_and_tensors() {
        let bytes = Builder::new()
            .kv_str("general.architecture", "llama")
            .kv_u32("llama.block_count", 22)
            .tensor("token_embd.weight", &[64, 1000], 0, 0)
            .tensor("blk.0.attn_q.weight", &[64, 64], 8, 4096)
            .build();

        let h = GgufHeader::parse(&bytes).unwrap();
        assert_eq!(h.version, 3);
        assert_eq!(h.architecture(), Some("llama"));
        assert_eq!(h.block_count(), Some(22));
        assert_eq!(h.tensors.len(), 2);
        assert_eq!(h.alignment, GGUF_DEFAULT_ALIGNMENT);
        assert_eq!(h.data_offset % h.alignment, 0);
        assert!(h.data_offset >= h.header_len);
    }

    #[test]
    fn tensor_ranges_cover_data_section() {
        let bytes = Builder::new()
            .kv_str("general.architecture", "llama")
            .tensor("a", &[10], 0, 0)
            .tensor("b", &[10], 0, 4096)
            .build();
        let h = GgufHeader::parse(&bytes).unwrap();
        let file_size = h.data_offset + 8192;
        let ranges = h.tensor_ranges(file_size).unwrap();
        assert_eq!(ranges.len(), 2);
        assert_eq!(ranges[0].start, h.data_offset);
        assert_eq!(ranges[0].end, h.data_offset + 4096);
        assert_eq!(ranges[1].start, h.data_offset + 4096);
        assert_eq!(ranges[1].end, file_size);
    }

    #[test]
    fn custom_alignment_respected() {
        let bytes = Builder::new()
            .kv_u32("general.alignment", 64)
            .tensor("a", &[1], 0, 0)
            .build();
        let h = GgufHeader::parse(&bytes).unwrap();
        assert_eq!(h.alignment, 64);
        assert_eq!(h.data_offset % 64, 0);
    }

    #[test]
    fn rejects_offsets_beyond_file() {
        let bytes = Builder::new().tensor("a", &[1], 0, 1 << 40).build();
        let h = GgufHeader::parse(&bytes).unwrap();
        assert!(matches!(
            h.tensor_ranges(4096),
            Err(GgufError::Malformed(_))
        ));
    }
}
