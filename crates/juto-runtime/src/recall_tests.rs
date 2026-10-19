use crate::session::EntryKind;
use juto_ai::{AssistantMessage, StopReason, ToolCall};
use serde_json::json;

use super::*;

fn user(text: &str) -> EntryKind {
    EntryKind::Message {
        message: Message::user(text),
    }
}

fn assistant(content: Vec<ContentBlock>) -> Message {
    Message::Assistant(AssistantMessage {
        content,
        api: "test".into(),
        provider: "test".into(),
        model: "m".into(),
        usage: Default::default(),
        stop_reason: StopReason::ToolUse,
        timestamp: 0,
        response_id: None,
        error: None,
    })
}

fn text_of(output: &ToolOutput) -> String {
    output
        .content
        .iter()
        .filter_map(|block| match block {
            ContentBlock::Text { text } => Some(text.as_str()),
            _ => None,
        })
        .collect()
}

/// Indices of the list lines in a page, in the order they appear.
fn listed_indices(text: &str) -> Vec<usize> {
    text.lines()
        .filter_map(|line| line.split_once('|')?.0.parse().ok())
        .collect()
}

/// The `from` a page says to continue with, if it has more.
fn more_from(text: &str) -> Option<usize> {
    text.lines()
        .find_map(|line| line.strip_prefix("more: recall {\"from\": "))
        .and_then(|rest| rest.trim_end_matches('}').parse().ok())
}

#[test]
fn list_pages_visit_every_index_once_in_order() {
    let dir = tempfile::tempdir().unwrap();
    let mut session = Session::create(dir.path(), Path::new("/work")).unwrap();
    for i in 0..300 {
        session
            .append(user(&format!("message {i} {}", "x".repeat(200))))
            .unwrap();
    }
    let mut seen = Vec::new();
    let mut pages = 0;
    let mut from = Some(0);
    while let Some(start) = from {
        let args = Args {
            from: Some(start),
            ..Args::default()
        };
        let text = text_of(&read(session.path(), args).unwrap());
        assert!(
            text.len() <= PAGE_BYTES + 512,
            "page is {} bytes",
            text.len()
        );
        seen.extend(listed_indices(&text));
        from = more_from(&text);
        pages += 1;
    }
    assert!(pages > 1, "300 long messages must not fit in one page");
    assert_eq!(seen, (0..300).collect::<Vec<_>>());
}

#[test]
fn body_pages_rebuild_the_original_text_across_multibyte_characters() {
    let body = "héllo wörld ✓ ".repeat(3000);
    let mut at = 0;
    let mut rebuilt = String::new();
    while let Some((start, end)) = page_bounds(&body, at) {
        rebuilt.push_str(&body[start..end]);
        if end == body.len() {
            break;
        }
        at = end;
    }
    assert_eq!(rebuilt, body);
}

#[test]
fn page_start_inside_a_character_floors_to_its_boundary() {
    // "é" is two bytes; offset 1 falls inside it.
    assert_eq!(page_bounds("é!", 1), Some((0, 3)));
    assert_eq!(page_bounds("ab", 3), None);
}

#[test]
fn tool_calls_and_errors_render_on_one_line() {
    let call = assistant(vec![
        ContentBlock::Text {
            text: "checking".into(),
        },
        ContentBlock::ToolCall(ToolCall {
            id: "c1".into(),
            name: "echo".into(),
            arguments: json!({"text": "hi"}),
            thought_signature: None,
        }),
    ]);
    let result = Message::ToolResult {
        tool_call_id: "c1".into(),
        tool_name: "echo".into(),
        content: vec![ContentBlock::Text {
            text: "boom".into(),
        }],
        is_error: true,
        details: Value::Null,
        timestamp: 0,
    };
    assert_eq!(
        index_line(&PathMessage {
            index: 3,
            id: "a",
            message: &call,
        }),
        r#"3|assistant: checking tool:echo({"text":"hi"})"#
    );
    assert_eq!(
        index_line(&PathMessage {
            index: 4,
            id: "b",
            message: &result,
        }),
        "4|result(error): echo: boom"
    );
}

#[test]
fn compaction_marker_sits_before_the_first_message_after_it() {
    let dir = tempfile::tempdir().unwrap();
    let mut session = Session::create(dir.path(), Path::new("/work")).unwrap();
    for i in 0..5 {
        session.append(user(&format!("m{i}"))).unwrap();
    }
    session.compact("summary".into(), None, 2).unwrap();
    session.append(user("m5")).unwrap();
    let text = text_of(&read(session.path(), Args::default()).unwrap());
    assert!(
        text.contains("-- compaction: 0..3 summarized; 3..5 kept --\n5|user: m5\n"),
        "{text}"
    );
}

#[test]
fn hidden_messages_stay_readable_by_index_after_compaction() {
    let dir = tempfile::tempdir().unwrap();
    let mut session = Session::create(dir.path(), Path::new("/work")).unwrap();
    for i in 0..5 {
        session.append(user(&format!("m{i}"))).unwrap();
    }
    session.compact("summary".into(), None, 2).unwrap();
    let args = Args {
        index: Some(0),
        ..Args::default()
    };
    let text = text_of(&read(session.path(), args).unwrap());
    assert!(text.starts_with("message 0 (user"), "{text}");
    assert!(text.lines().any(|line| line == "m0"), "{text}");
}

#[test]
fn misuse_is_a_tool_error_not_an_empty_page() {
    let dir = tempfile::tempdir().unwrap();
    let mut session = Session::create(dir.path(), Path::new("/work")).unwrap();
    assert!(read(session.path(), Args::default()).unwrap().is_error);
    session.append(user("only")).unwrap();
    let path = session.path().to_path_buf();
    let call = |args: Args| read(&path, args).unwrap();
    assert!(
        call(Args {
            from: Some(1),
            ..Args::default()
        })
        .is_error
    );
    assert!(
        call(Args {
            index: Some(1),
            ..Args::default()
        })
        .is_error
    );
    assert!(
        call(Args {
            at: Some(0),
            ..Args::default()
        })
        .is_error
    );
    assert!(
        call(Args {
            index: Some(0),
            from: Some(0),
            ..Args::default()
        })
        .is_error
    );
    assert!(
        call(Args {
            count: Some(0),
            ..Args::default()
        })
        .is_error
    );
    assert!(
        call(Args {
            index: Some(0),
            at: Some(999),
            ..Args::default()
        })
        .is_error
    );
    assert!(
        !call(Args {
            index: Some(0),
            at: Some(0),
            ..Args::default()
        })
        .is_error
    );
}

#[test]
fn snapshot_caps_summary_and_keeps_newest_visible_messages() {
    let dir = tempfile::tempdir().unwrap();
    let mut session = Session::create(dir.path(), Path::new("/work")).unwrap();
    session.append(user("hidden-original")).unwrap();
    session.compact("summary ".repeat(2000), None, 0).unwrap();
    for i in 0..200 {
        session
            .append(user(&format!("visible-{i} {}", "é".repeat(200))))
            .unwrap();
    }
    let text = snapshot(session.path()).unwrap();
    assert!(text.len() <= 12 * 1024);
    assert!(text.contains("[Earlier conversation summary]\nsummary"));
    assert!(!text.contains("hidden-original"));
    assert!(!text.contains("|user: visible-0 "));
    assert!(text.contains("200|user: visible-199 "));
}

#[test]
fn detail_continuation_hints_round_trip_through_public_arguments() {
    let dir = tempfile::tempdir().unwrap();
    let mut session = Session::create(dir.path(), Path::new("/work")).unwrap();
    let body = "é".repeat(PAGE_BYTES);
    session.append(user(&body)).unwrap();
    let mut args = json!({"id": 0});
    let mut rebuilt = String::new();
    loop {
        let text = text_of(&read(session.path(), serde_json::from_value(args).unwrap()).unwrap());
        let (_, body_page) = text.split_once('\n').unwrap();
        let (page, footer) = body_page.rsplit_once("\n[bytes ").unwrap();
        rebuilt.push_str(page);
        let Some((_, hint)) = footer.split_once("more: recall ") else {
            break;
        };
        args = serde_json::from_str(hint).unwrap();
    }
    assert_eq!(rebuilt, body);
}
