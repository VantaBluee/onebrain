//! Split-GGUF part naming (the `llama_split_path` convention).
//!
//! llama.cpp names multi-part models `<prefix>-%05d-of-%05d.gguf`. A model
//! reference naming *any* one part implies the whole set, so the downloader
//! derives every sibling file name (and URL) from the single name it was
//! given and fetches each part as its own cache download — full-file or
//! ranged. The parsing here is deliberately strict (exactly five digits on
//! both sides, 1-based index no greater than the count) so ordinary file
//! names that merely contain digits never get misread as split sets.

/// A parsed `<prefix>-%05d-of-%05d.gguf` file name.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SplitName {
    /// Everything before the `-%05d-of-%05d.gguf` suffix.
    pub prefix: String,
    /// 1-based index of the part this name refers to.
    pub index: u32,
    /// Total number of parts in the set.
    pub count: u32,
}

impl SplitName {
    /// File name of part `index` (1-based, `1..=count`).
    pub fn part_file_name(&self, index: u32) -> String {
        format!("{}-{:05}-of-{:05}.gguf", self.prefix, index, self.count)
    }

    /// All part file names, in load order (part 1 first).
    pub fn part_file_names(&self) -> Vec<String> {
        (1..=self.count).map(|i| self.part_file_name(i)).collect()
    }
}

/// Parse a file name like `model-00001-of-00003.gguf`. Returns `None` for
/// anything that is not split-named (including a bare `model.gguf`).
pub fn parse_split_name(file_name: &str) -> Option<SplitName> {
    // `.gguf` is matched case-insensitively (model refs allow either), but
    // the emitted names always use the lowercase suffix llama.cpp writes.
    if !file_name.to_ascii_lowercase().ends_with(".gguf") {
        return None;
    }
    let stem = &file_name[..file_name.len() - ".gguf".len()];
    // The stem must end `-NNNNN-of-NNNNN` (15 bytes) with a non-empty prefix.
    const TAIL: usize = 15;
    if stem.len() <= TAIL || !stem.is_char_boundary(stem.len() - TAIL) {
        return None;
    }
    let (prefix, tail) = stem.split_at(stem.len() - TAIL);
    let bytes = tail.as_bytes();
    let digits = |range: std::ops::Range<usize>| -> Option<u32> {
        if !bytes[range.clone()].iter().all(u8::is_ascii_digit) {
            return None;
        }
        tail[range].parse().ok()
    };
    // Compare as bytes: `tail` may end in multi-byte characters, and byte
    // comparisons can never panic on a char boundary.
    if bytes[0] != b'-' || &bytes[6..10] != b"-of-" {
        return None;
    }
    let index = digits(1..6)?;
    let count = digits(10..15)?;
    if index == 0 || count == 0 || index > count {
        return None;
    }
    Some(SplitName {
        prefix: prefix.to_string(),
        index,
        count,
    })
}

/// Does this model reference (or plain file name) name one part of a split
/// set? Only the final path component is considered.
pub fn is_split_ref(model_ref: &str) -> bool {
    let name = model_ref.rsplit(['/', '\\']).next().unwrap_or(model_ref);
    parse_split_name(name).is_some()
}

/// Derive a sibling part's URL from any part's URL by swapping the final
/// path segment (query string and fragment, if any, are preserved). This is
/// how `resolve/main/<file>` URLs address every part of the same repo.
pub fn sibling_url(url: &str, part_file_name: &str) -> String {
    let (path, suffix) = match url.find(['?', '#']) {
        Some(i) => url.split_at(i),
        None => (url, ""),
    };
    match path.rfind('/') {
        Some(i) => format!("{}/{part_file_name}{suffix}", &path[..i]),
        None => format!("{part_file_name}{suffix}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_and_enumerates_parts() {
        let s = parse_split_name("Qwen3-32B-Q4_K_M-00001-of-00003.gguf").unwrap();
        assert_eq!(s.prefix, "Qwen3-32B-Q4_K_M");
        assert_eq!(s.index, 1);
        assert_eq!(s.count, 3);
        assert_eq!(
            s.part_file_names(),
            vec![
                "Qwen3-32B-Q4_K_M-00001-of-00003.gguf",
                "Qwen3-32B-Q4_K_M-00002-of-00003.gguf",
                "Qwen3-32B-Q4_K_M-00003-of-00003.gguf",
            ]
        );
    }

    #[test]
    fn two_digit_part_counts_enumerate_in_order() {
        // NN > 9: the %05d padding keeps lexicographic order == load order.
        let s = parse_split_name("glm-4.5-air-00007-of-00012.gguf").unwrap();
        assert_eq!((s.index, s.count), (7, 12));
        let names = s.part_file_names();
        assert_eq!(names.len(), 12);
        assert_eq!(names[0], "glm-4.5-air-00001-of-00012.gguf");
        assert_eq!(names[9], "glm-4.5-air-00010-of-00012.gguf");
        assert_eq!(names[11], "glm-4.5-air-00012-of-00012.gguf");
        let mut sorted = names.clone();
        sorted.sort();
        assert_eq!(sorted, names, "%05d names must sort in load order");
    }

    #[test]
    fn non_split_names_are_rejected() {
        for name in [
            "model.gguf",
            "model-1-of-3.gguf",         // not zero-padded
            "model-00001-of-00003.bin",  // wrong extension
            "model-00000-of-00003.gguf", // index 0 (parts are 1-based)
            "model-00004-of-00003.gguf", // index > count
            "model-00001-of-00000.gguf", // zero parts
            "-00001-of-00003.gguf",      // empty prefix
            "model-0000x-of-00003.gguf", // non-digit
            "00001-of-00003.gguf",       // no `-` before the index
        ] {
            assert!(parse_split_name(name).is_none(), "{name} must not parse");
        }
    }

    #[test]
    fn is_split_ref_looks_at_the_last_component() {
        assert!(is_split_ref("hf:org/repo/subdir/model-00001-of-00002.gguf"));
        assert!(is_split_ref("model-00002-of-00002.gguf"));
        assert!(!is_split_ref("hf:org/repo/model.gguf"));
        assert!(!is_split_ref("C:\\models\\plain.gguf"));
        assert!(is_split_ref("C:\\models\\m-00001-of-00002.gguf"));
    }

    #[test]
    fn sibling_url_swaps_the_last_segment() {
        assert_eq!(
            sibling_url(
                "https://huggingface.co/o/r/resolve/main/m-00001-of-00002.gguf",
                "m-00002-of-00002.gguf"
            ),
            "https://huggingface.co/o/r/resolve/main/m-00002-of-00002.gguf"
        );
        assert_eq!(
            sibling_url(
                "https://host/path/m-00001-of-00002.gguf?download=true",
                "m-00002-of-00002.gguf"
            ),
            "https://host/path/m-00002-of-00002.gguf?download=true"
        );
    }
}
