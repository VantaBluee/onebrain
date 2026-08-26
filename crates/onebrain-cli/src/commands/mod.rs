pub mod doctor;
pub mod version;

use std::fmt;

#[derive(Debug)]
pub struct CliError(pub String);

impl fmt::Display for CliError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for CliError {}

/// Uniform message for commands whose milestone hasn't landed yet.
pub fn not_yet(command: &str, milestone: &str) -> Result<(), CliError> {
    Err(CliError(format!(
        "`onebrain {command}` is not implemented yet; it arrives in milestone {milestone}. \
         Track progress in STATUS.md."
    )))
}
