//! Bounds a child report before it enters the main thread.
//!
//! A report that fits the budget passes through unchanged. A larger one is
//! replaced by a digest of at most the budget, or, when the digest is
//! unavailable or does not fit, by a clip of its own prefix. Either way the
//! last line names the archived full text, which `wait` returns.

use std::time::Duration;

use serde_json::{Value, json};

/// Default byte budget for one report in the main thread.
pub const DEFAULT_MAIN_REPORT_BYTES: usize = 1500;
/// Smallest accepted budget. Below it the pointer tail crowds out the text.
pub const MIN_MAIN_REPORT_BYTES: usize = 512;
/// Largest report sent to the digest model. Longer reports are clipped without a call.
pub const DIGEST_INPUT_LIMIT: usize = 128 * 1024;
/// Upper bound on one digest call. A slower digest falls back to a clip.
pub const DIGEST_TIMEOUT: Duration = Duration::from_secs(60);
/// Longest failure reason copied into a clip marker.
const MAX_REASON_BYTES: usize = 160;

/// One report to bound. `key` is the pointer id that `wait` resolves to the full text.
pub struct Report<'a> {
    pub body: &'a str,
    pub data: Option<&'a Value>,
    pub key: &'a str,
}

/// Bound `report` to `budget` bytes. Returns `None` when the report already fits
/// and must pass through unchanged. Otherwise returns the bounded body, which is
/// at most `budget` bytes and ends with the pointer line. `digest(limit)` yields
/// text of at most `limit` bytes; it runs at most twice, the second time with a
/// smaller limit.
pub async fn bound<F, Fut>(
    report: Report<'_>,
    budget: usize,
    timeout: Duration,
    mut digest: F,
) -> Option<String>
where
    F: FnMut(usize) -> Fut,
    Fut: Future<Output = Result<String, String>>,
{
    assert!(
        budget >= MIN_MAIN_REPORT_BYTES,
        "main report budget below {MIN_MAIN_REPORT_BYTES}"
    );
    if report.body.len() <= budget && report.data.is_none() {
        return None;
    }
    let tail = tail(&report, budget);
    if report.body.len() + tail.len() < budget {
        return Some(format!("{}\n{tail}", report.body));
    }
    if report.body.len() > DIGEST_INPUT_LIMIT {
        let reason = format!(
            "digest unavailable: report of {} bytes exceeds the digest input limit",
            report.body.len()
        );
        return Some(clip(&report, budget, &reason));
    }
    let text_limit = budget - tail.len() - 1;
    let mut limit = text_limit;
    for attempt in 0..2 {
        let text = match tokio::time::timeout(timeout, digest(limit)).await {
            Ok(Ok(text)) => text,
            Ok(Err(reason)) => {
                return Some(clip(
                    &report,
                    budget,
                    &format!("digest unavailable: {reason}"),
                ));
            }
            Err(_) => return Some(clip(&report, budget, "digest unavailable: timed out")),
        };
        let text = text.trim();
        if !text.is_empty() && text.len() <= text_limit {
            return Some(format!("{text}\n{tail}"));
        }
        if attempt == 0 {
            limit = limit * 3 / 4;
        }
    }
    Some(clip(&report, budget, "digest exceeded budget"))
}

/// Longest prefix of `text` of at most `max_bytes` bytes that ends on a char boundary.
pub fn prefix_at_char_boundary(text: &str, max_bytes: usize) -> &str {
    if text.len() <= max_bytes {
        return text;
    }
    let mut end = max_bytes;
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    &text[..end]
}

/// Closing lines of every bounded body: the data value when it is small
/// enough, then the pointer that returns the full report through `wait`.
fn tail(report: &Report<'_>, budget: usize) -> String {
    let pointer = format!("[full report: wait {}]", json!({"ids": [report.key]}));
    match report.data {
        None => pointer,
        Some(value) => {
            let json = value.to_string();
            let data = if json.len() <= budget / 3 {
                format!("data: {json}")
            } else {
                format!("data: {} bytes omitted; full value via wait", json.len())
            };
            format!("{data}\n{pointer}")
        }
    }
}

/// The body's prefix, then a marker naming why the digest is missing, then the tail.
fn clip(report: &Report<'_>, budget: usize, reason: &str) -> String {
    let reason: String = reason
        .chars()
        .map(|c| if c == '\n' || c == '\r' { ' ' } else { c })
        .collect();
    let marker = format!(
        "[report clipped: {}]",
        prefix_at_char_boundary(&reason, MAX_REASON_BYTES)
    );
    let tail = tail(report, budget);
    let room = budget.saturating_sub(marker.len() + tail.len() + 2);
    let head = prefix_at_char_boundary(report.body, room);
    [head, marker.as_str(), tail.as_str()]
        .into_iter()
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>()
        .join("\n")
}

#[cfg(test)]
mod tests {
    use std::cell::{Cell, RefCell};

    use super::*;

    const KEY: &str = "report-key";
    const LONG: Duration = Duration::from_secs(5);

    fn pointer() -> String {
        format!("[full report: wait {{\"ids\":[\"{KEY}\"]}}]")
    }

    fn report(body: &str) -> Report<'_> {
        Report {
            body,
            data: None,
            key: KEY,
        }
    }

    #[tokio::test]
    async fn report_at_budget_passes_through_without_a_digest_call() {
        let calls = Cell::new(0);
        let body = "a".repeat(600);
        let out = bound(report(&body), 600, LONG, |_| {
            calls.set(calls.get() + 1);
            async { Ok(String::new()) }
        })
        .await;
        assert_eq!(out, None);
        assert_eq!(calls.get(), 0);
    }

    #[tokio::test]
    async fn oversized_report_is_digested_within_budget_with_pointer() {
        let body = "x".repeat(20_000);
        let out = bound(report(&body), 1500, LONG, |_| async {
            Ok("status: done".to_string())
        })
        .await
        .expect("over budget");
        assert!(out.len() <= 1500, "len {}", out.len());
        assert!(out.starts_with("status: done\n"));
        assert!(out.ends_with(&pointer()), "tail: {out}");
    }

    #[tokio::test]
    async fn overlong_digest_is_reasked_shorter_then_clipped() {
        let body = "x".repeat(20_000);
        let limits = RefCell::new(Vec::new());
        let out = bound(report(&body), 1500, LONG, |limit| {
            limits.borrow_mut().push(limit);
            async { Ok("y".repeat(10_000)) }
        })
        .await
        .expect("over budget");
        let limits = limits.into_inner();
        assert_eq!(limits.len(), 2);
        assert!(limits[1] < limits[0]);
        assert!(out.len() <= 1500);
        assert!(out.contains("[report clipped: digest exceeded budget]"));
        assert!(out.ends_with(&pointer()));
        assert!(body.starts_with(out.split('\n').next().unwrap()));
    }

    #[tokio::test]
    async fn reasked_digest_that_fits_is_used() {
        let body = "x".repeat(20_000);
        let calls = Cell::new(0);
        let out = bound(report(&body), 1500, LONG, |_| {
            let n = calls.get();
            calls.set(n + 1);
            async move {
                Ok(if n == 0 {
                    "z".repeat(5_000)
                } else {
                    "short".to_string()
                })
            }
        })
        .await
        .expect("over budget");
        assert_eq!(calls.get(), 2);
        assert!(out.starts_with("short\n"));
        assert!(!out.contains("clipped"));
    }

    #[tokio::test]
    async fn digest_failure_is_a_marked_clip_with_pointer() {
        let body = "x".repeat(20_000);
        let calls = Cell::new(0);
        let out = bound(report(&body), 1500, LONG, |_| {
            calls.set(calls.get() + 1);
            async { Err("model unavailable".to_string()) }
        })
        .await
        .expect("over budget");
        assert_eq!(calls.get(), 1, "a failed digest is not re-asked");
        assert!(out.len() <= 1500);
        assert!(out.contains("[report clipped: digest unavailable: model unavailable]"));
        assert!(out.ends_with(&pointer()));
    }

    #[tokio::test]
    async fn stuck_digest_times_out_into_a_clip() {
        let body = "x".repeat(20_000);
        let out = bound(report(&body), 1500, Duration::from_millis(10), |_| async {
            std::future::pending::<Result<String, String>>().await
        })
        .await
        .expect("over budget");
        assert!(out.contains("[report clipped: digest unavailable: timed out]"));
        assert!(out.ends_with(&pointer()));
    }

    #[tokio::test]
    async fn report_beyond_digest_input_limit_is_clipped_without_a_call() {
        let body = "x".repeat(DIGEST_INPUT_LIMIT + 1);
        let calls = Cell::new(0);
        let out = bound(report(&body), 1500, LONG, |_| {
            calls.set(calls.get() + 1);
            async { Ok(String::new()) }
        })
        .await
        .expect("over budget");
        assert_eq!(calls.get(), 0);
        assert!(out.contains("exceeds the digest input limit"));
        assert!(out.len() <= 1500);
    }

    #[tokio::test]
    async fn clip_never_splits_a_multibyte_character() {
        let body = "é".repeat(4_000);
        let out = bound(report(&body), 1501, LONG, |_| async {
            Err("down".to_string())
        })
        .await
        .expect("over budget");
        assert!(out.len() <= 1501);
        let head = out.split("\n[report clipped").next().unwrap();
        assert!(!head.is_empty());
        assert!(head.chars().all(|c| c == 'é'));
    }

    #[test]
    fn prefix_stops_on_char_boundary() {
        assert_eq!(prefix_at_char_boundary("aé", 2), "a");
        assert_eq!(prefix_at_char_boundary("aé", 3), "aé");
        assert_eq!(prefix_at_char_boundary("aé", 0), "");
    }

    #[tokio::test]
    async fn small_data_is_inlined_and_large_data_is_deferred_to_wait() {
        let body = "x".repeat(20_000);
        let small = json!({"n": 1});
        let with_small = Report {
            body: &body,
            data: Some(&small),
            key: KEY,
        };
        let out = bound(with_small, 1500, LONG, |_| async { Ok("ok".to_string()) })
            .await
            .expect("over budget");
        assert!(out.contains("data: {\"n\":1}\n"));

        let large = json!({"blob": "z".repeat(2_000)});
        let with_large = Report {
            body: &body,
            data: Some(&large),
            key: KEY,
        };
        let out = bound(with_large, 1500, LONG, |_| async { Ok("ok".to_string()) })
            .await
            .expect("over budget");
        assert!(out.len() <= 1500);
        assert!(out.contains("bytes omitted; full value via wait"));
        assert!(!out.contains("zzzz"));
    }
}
