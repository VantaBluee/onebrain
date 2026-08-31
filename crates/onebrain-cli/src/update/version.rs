//! Semver-ish version ordering for self-update decisions and doctor's
//! skew findings.
//!
//! Hand-rolled instead of pulling the `semver` crate: release tags only
//! need core + pre-release precedence (build metadata is ignored, SemVer
//! §10), and keeping the dependency tree flat is an explicit product goal.
//! The precedence rules implemented are SemVer 2.0.0 §11.

use std::fmt;

/// One dot-separated pre-release identifier. The derived `Ord` is exactly
/// the SemVer rule: numeric identifiers (the first variant) sort below
/// alphanumeric ones, numerics compare by value, alphanumerics by ASCII.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
enum PreId {
    Num(u64),
    Alpha(String),
}

impl fmt::Display for PreId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PreId::Num(n) => write!(f, "{n}"),
            PreId::Alpha(a) => f.write_str(a),
        }
    }
}

/// A parsed release version: `MAJOR.MINOR.PATCH[-PRE]`, build metadata
/// discarded. Total order per SemVer §11 (`0.1.0-rc.1 < 0.1.0`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Version {
    major: u64,
    minor: u64,
    patch: u64,
    pre: Vec<PreId>,
}

impl Version {
    /// Parse `[vV]MAJOR.MINOR[.PATCH][-PRE][+BUILD]`. Tolerant where real
    /// release tags are sloppy (leading `v`, missing patch, build
    /// metadata), strict where sloppiness would corrupt an ordering
    /// decision (non-numeric core, empty identifiers).
    pub fn parse(text: &str) -> Result<Version, String> {
        let trimmed = text.trim().trim_start_matches(['v', 'V']);
        let no_build = trimmed.split('+').next().unwrap_or_default();
        let (core, pre_text) = match no_build.split_once('-') {
            Some((core, pre)) => (core, Some(pre)),
            None => (no_build, None),
        };

        let mut nums = core.split('.');
        let major = parse_num(nums.next(), text)?;
        let minor = parse_num(nums.next(), text)?;
        let patch = match nums.next() {
            Some(part) => parse_num(Some(part), text)?,
            None => 0,
        };
        if nums.next().is_some() {
            return Err(format!("`{text}` has more than three version components"));
        }

        let mut pre = Vec::new();
        if let Some(pre_text) = pre_text {
            for id in pre_text.split('.') {
                if id.is_empty() {
                    return Err(format!("`{text}` has an empty pre-release identifier"));
                }
                pre.push(if id.bytes().all(|b| b.is_ascii_digit()) {
                    PreId::Num(
                        id.parse()
                            .map_err(|_| format!("`{text}`: pre-release number out of range"))?,
                    )
                } else {
                    PreId::Alpha(id.to_string())
                });
            }
        }

        Ok(Version {
            major,
            minor,
            patch,
            pre,
        })
    }
}

fn parse_num(part: Option<&str>, whole: &str) -> Result<u64, String> {
    let part = part
        .filter(|p| !p.is_empty())
        .ok_or_else(|| format!("`{whole}` is not a version (want MAJOR.MINOR.PATCH)"))?;
    part.parse()
        .map_err(|_| format!("`{whole}`: `{part}` is not a number"))
}

impl Ord for Version {
    fn cmp(&self, other: &Version) -> std::cmp::Ordering {
        use std::cmp::Ordering;
        (self.major, self.minor, self.patch)
            .cmp(&(other.major, other.minor, other.patch))
            .then_with(|| match (self.pre.is_empty(), other.pre.is_empty()) {
                (true, true) => Ordering::Equal,
                // A release outranks any of its own pre-releases (§11.3).
                (true, false) => Ordering::Greater,
                (false, true) => Ordering::Less,
                // Vec's lexicographic order is the §11.4 identifier walk,
                // including "shorter prefix sorts lower".
                (false, false) => self.pre.cmp(&other.pre),
            })
    }
}

impl PartialOrd for Version {
    fn partial_cmp(&self, other: &Version) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl fmt::Display for Version {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)?;
        for (i, id) in self.pre.iter().enumerate() {
            f.write_str(if i == 0 { "-" } else { "." })?;
            write!(f, "{id}")?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(text: &str) -> Version {
        Version::parse(text).unwrap_or_else(|e| panic!("parsing {text}: {e}"))
    }

    #[test]
    fn core_versions_order_numerically() {
        assert!(v("0.1.0") < v("0.2.0"));
        assert!(v("0.2.0") < v("0.10.0"));
        assert!(v("0.10.0") < v("1.0.0"));
        assert!(v("1.0.0") < v("1.0.1"));
        assert_eq!(v("v0.1.0"), v("0.1.0"));
        // A missing patch component defaults to zero.
        assert_eq!(v("0.1"), v("0.1.0"));
    }

    #[test]
    fn prerelease_chain_matches_semver_spec_example() {
        // SemVer 2.0.0 §11's canonical precedence chain.
        let chain = [
            "1.0.0-alpha",
            "1.0.0-alpha.1",
            "1.0.0-alpha.beta",
            "1.0.0-beta",
            "1.0.0-beta.2",
            "1.0.0-beta.11",
            "1.0.0-rc.1",
            "1.0.0",
        ];
        for pair in chain.windows(2) {
            assert!(v(pair[0]) < v(pair[1]), "{} !< {}", pair[0], pair[1]);
        }
    }

    #[test]
    fn rc_tags_order_below_their_release() {
        // The exact shape release.yml validates with (v0.1.0-rc.1 → v0.1.0).
        assert!(v("v0.1.0-rc.1") < v("v0.1.0"));
        assert!(v("v0.1.0-rc.1") < v("v0.1.0-rc.2"));
        assert!(v("v0.1.0-rc.2") < v("v0.1.0-rc.10"));
    }

    #[test]
    fn build_metadata_is_ignored() {
        assert_eq!(v("1.0.0+deadbeef"), v("1.0.0"));
        assert_eq!(v("1.0.0-rc.1+42"), v("1.0.0-rc.1"));
    }

    #[test]
    fn garbage_is_a_parse_error_not_a_bad_decision() {
        for bad in ["", "abc", "1", "1.x.0", "1.2.3.4", "1.0.0-", "1..0"] {
            assert!(Version::parse(bad).is_err(), "`{bad}` should not parse");
        }
    }

    #[test]
    fn display_round_trips() {
        assert_eq!(v("v1.2.3").to_string(), "1.2.3");
        assert_eq!(v("1.2.3-rc.1").to_string(), "1.2.3-rc.1");
        assert_eq!(v("1.2.3-rc.1+build").to_string(), "1.2.3-rc.1");
    }
}
