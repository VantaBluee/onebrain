//! `onebrain pair`: host a pairing window (no argument) or join one
//! (ticket or 6-digit code argument). Contract in `docs/mesh.md` §CLI.
//!
//! Host mode streams `POST /api/internal/pair/start` NDJSON: the `window`
//! event carries the code + ticket (printed with a QR of
//! `onebrain:<ticket>`), then the window runs until a terminal event.
//! Joiner mode classifies the argument — exactly 6 ASCII digits is a code,
//! anything else is a ticket — and calls `POST /api/internal/pair/join`.

use std::io::Write;

use onebraind::paths::AppPaths;

use super::{up, CliError};
use crate::client::DaemonClient;

/// What the positional `onebrain pair <target>` argument turned out to be.
#[derive(Debug, Clone, PartialEq, Eq)]
enum JoinTarget {
    /// Exactly 6 ASCII digits: discover the host on the LAN by code.
    Code(String),
    /// Anything else: an endpoint ticket (dialing works cross-network).
    Ticket(String),
}

/// Classify the joiner argument per the contract: exactly 6 ASCII digits is
/// a code, everything else is a ticket. A QR payload prefix (`onebrain:`)
/// is stripped first and always denotes a ticket — the QR only ever wraps
/// tickets.
fn classify(arg: &str) -> JoinTarget {
    let arg = arg.trim();
    if let Some(ticket) = arg.strip_prefix("onebrain:") {
        return JoinTarget::Ticket(ticket.to_string());
    }
    if is_six_digits(arg) {
        JoinTarget::Code(arg.to_string())
    } else {
        JoinTarget::Ticket(arg.to_string())
    }
}

fn is_six_digits(s: &str) -> bool {
    s.len() == 6 && s.bytes().all(|b| b.is_ascii_digit())
}

pub fn run(target: Option<&str>, code: Option<&str>, json: bool) -> Result<(), CliError> {
    let paths = AppPaths::resolve()?;
    let outcome = up::ensure_up(&paths)?;
    let client = outcome.client;

    match target {
        None => host(&client, code, json),
        Some(arg) => join(&client, arg, code, json),
    }
}

/// Host mode: open a window, print code + ticket + QR, stream events.
fn host(client: &DaemonClient, code: Option<&str>, json: bool) -> Result<(), CliError> {
    if code.is_some() {
        return Err(CliError(
            "--code only makes sense when joining (`onebrain pair <ticket> --code <code>`); \
             hosting generates its own code"
                .to_string(),
        ));
    }

    let terminal = client.pair_start(|event| {
        if json {
            // NDJSON pass-through: scripts see exactly what the daemon sent.
            println!("{event}");
            return;
        }
        match event.get("status").and_then(|s| s.as_str()) {
            Some("window") => print_window(event),
            Some("attempt") => println!("a device is attempting to pair..."),
            _ => {}
        }
    })?;

    if json {
        println!("{terminal}");
    }
    match terminal.get("status").and_then(|s| s.as_str()) {
        Some("paired") => {
            if !json {
                let name = terminal
                    .pointer("/peer/name")
                    .and_then(|n| n.as_str())
                    .unwrap_or("?");
                let id = terminal
                    .pointer("/peer/id")
                    .and_then(|i| i.as_str())
                    .unwrap_or("");
                println!("paired with {name} ({})", short_id(id));
            }
            Ok(())
        }
        Some("expired") => Err(CliError(
            "the pairing window expired with no device paired; \
             run `onebrain pair` again to open a new one"
                .to_string(),
        )),
        Some("failed") => {
            let message = terminal
                .get("message")
                .and_then(|m| m.as_str())
                .unwrap_or("unknown error");
            Err(CliError(format!("pairing failed: {message}")))
        }
        _ => Err(CliError(
            "the daemon sent an unrecognized terminal pairing status; \
             update both onebrain and retry"
                .to_string(),
        )),
    }
}

/// Print the `window` event: the 6-digit code big and clear, the ticket on
/// its own line, and a QR of `onebrain:<ticket>`.
fn print_window(event: &serde_json::Value) {
    let code = event.get("code").and_then(|c| c.as_str()).unwrap_or("?");
    let ticket = event.get("ticket").and_then(|t| t.as_str()).unwrap_or("");

    println!("pairing window open (120 s, up to 3 attempts)");
    println!();
    println!("    code:   {}", spaced_code(code));
    println!();
    println!("ticket (works across networks):");
    println!("{ticket}");
    println!();
    match qr_unicode(&format!("onebrain:{ticket}")) {
        Ok(qr) => {
            println!("scan to pair:");
            println!("{qr}");
        }
        Err(e) => println!("(could not render a QR code: {e}; use the ticket text instead)"),
    }
    println!("on the other device:");
    println!("  onebrain pair <ticket>   any network (asks for the code)");
    println!("  onebrain pair {code}     same LAN");
    println!();
    println!("waiting for the other device (Ctrl+C to cancel)...");
}

/// `"123456"` → `"1 2 3 4 5 6"` so the code reads at a glance.
fn spaced_code(code: &str) -> String {
    let mut out = String::with_capacity(code.len() * 2);
    for (i, ch) in code.chars().enumerate() {
        if i > 0 {
            out.push(' ');
        }
        out.push(ch);
    }
    out
}

/// Render a QR to terminal text with Unicode half-blocks. Inverted colors
/// (light modules on the terminal's dark background) per the `qrcode`
/// crate's terminal example; scanners handle inverted codes.
fn qr_unicode(data: &str) -> Result<String, String> {
    use qrcode::render::unicode;
    let code = qrcode::QrCode::new(data.as_bytes()).map_err(|e| e.to_string())?;
    Ok(code
        .render::<unicode::Dense1x2>()
        .dark_color(unicode::Dense1x2::Light)
        .light_color(unicode::Dense1x2::Dark)
        .build())
}

/// Joiner mode: classify the argument, get a code alongside a ticket
/// (flag or stdin prompt), POST the join, report the peer.
fn join(
    client: &DaemonClient,
    arg: &str,
    code_flag: Option<&str>,
    json: bool,
) -> Result<(), CliError> {
    let (target, code) = match classify(arg) {
        JoinTarget::Code(code) => {
            if code_flag.is_some() {
                return Err(CliError(
                    "the argument is already a 6-digit code; --code is only for tickets \
                     (`onebrain pair <ticket> --code <code>`)"
                        .to_string(),
                ));
            }
            (code, None)
        }
        JoinTarget::Ticket(ticket) => {
            let code = match code_flag {
                Some(code) => code.trim().to_string(),
                None if json => {
                    return Err(CliError(
                        "a ticket needs the 6-digit code: pass --code <code> \
                         (no interactive prompt with --json)"
                            .to_string(),
                    ));
                }
                None => prompt_code()?,
            };
            if !is_six_digits(&code) {
                return Err(CliError(format!(
                    "the pairing code is 6 digits, got {code:?}; \
                     read it from `onebrain pair` on the host"
                )));
            }
            (ticket, Some(code))
        }
    };

    if !json {
        println!("pairing...");
    }
    let response = client.pair_join(&target, code.as_deref())?;

    if json {
        println!("{response}");
        return Ok(());
    }
    let name = response
        .pointer("/peer/name")
        .and_then(|n| n.as_str())
        .unwrap_or("?");
    let id = response
        .pointer("/peer/id")
        .and_then(|i| i.as_str())
        .unwrap_or("");
    println!("paired with {name} ({})", short_id(id));
    println!("`onebrain status` shows the link once it connects.");
    Ok(())
}

/// Prompt for the 6-digit code on stdin (plain read, trimmed).
fn prompt_code() -> Result<String, CliError> {
    print!("6-digit code shown on the host: ");
    std::io::stdout()
        .flush()
        .map_err(|e| CliError(format!("could not flush stdout ({e}); retry")))?;
    let mut line = String::new();
    std::io::stdin().read_line(&mut line).map_err(|e| {
        CliError(format!(
            "could not read the code from stdin ({e}); pass it with --code instead"
        ))
    })?;
    Ok(line.trim().to_string())
}

/// First 8 chars of an endpoint id, or `"?"` when unknown.
fn short_id(id: &str) -> String {
    if id.is_empty() {
        return "?".to_string();
    }
    id.chars().take(8).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn six_ascii_digits_classify_as_code() {
        assert_eq!(classify("123456"), JoinTarget::Code("123456".into()));
        assert_eq!(classify("000000"), JoinTarget::Code("000000".into()));
        assert_eq!(classify("  042042  "), JoinTarget::Code("042042".into()));
    }

    #[test]
    fn everything_else_classifies_as_ticket() {
        // Too short / too long / non-digit → ticket, even if digit-heavy.
        assert_eq!(classify("12345"), JoinTarget::Ticket("12345".into()));
        assert_eq!(classify("1234567"), JoinTarget::Ticket("1234567".into()));
        assert_eq!(classify("12345a"), JoinTarget::Ticket("12345a".into()));
        assert_eq!(
            classify("endpointxyzabc123"),
            JoinTarget::Ticket("endpointxyzabc123".into())
        );
    }

    #[test]
    fn non_ascii_digits_are_not_a_code() {
        // Six digits, but not ASCII (fullwidth) — must be treated as ticket.
        assert_eq!(
            classify("１２３４５６"),
            JoinTarget::Ticket("１２３４５６".into())
        );
    }

    #[test]
    fn qr_scheme_prefix_always_means_ticket() {
        assert_eq!(
            classify("onebrain:someticketpayload"),
            JoinTarget::Ticket("someticketpayload".into())
        );
        // Even a digit-only payload: the QR only ever wraps tickets.
        assert_eq!(
            classify("onebrain:123456"),
            JoinTarget::Ticket("123456".into())
        );
    }

    #[test]
    fn spaced_code_spreads_digits() {
        assert_eq!(spaced_code("123456"), "1 2 3 4 5 6");
        assert_eq!(spaced_code(""), "");
    }

    #[test]
    fn short_id_truncates_to_eight() {
        assert_eq!(short_id("abcdef1234567890"), "abcdef12");
        assert_eq!(short_id("abc"), "abc");
        assert_eq!(short_id(""), "?");
    }

    #[test]
    fn qr_renders_for_a_ticket_sized_payload() {
        let payload = format!("onebrain:{}", "x".repeat(300));
        let qr = qr_unicode(&payload).unwrap();
        assert!(!qr.is_empty());
        // Half-block rendering uses the block glyphs.
        assert!(qr.chars().any(|c| "█▀▄".contains(c)));
    }
}
