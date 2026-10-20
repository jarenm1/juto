use crate::session::EntryKind;
use juto_ai::{ContentBlock, Message};
use serde_json::json;

use super::*;

fn user(text: &str) -> EntryKind {
    EntryKind::Message {
        message: Message::user(text),
    }
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

fn zoom_call(path: &Path, args: Value) -> ToolOutput {
    read(path, serde_json::from_value(args).unwrap()).unwrap()
}

/// Expansion tokens `zz-NNN` in the order they appear in a page.
fn tokens(text: &str) -> Vec<String> {
    text.match_indices("zz-")
        .map(|(at, _)| text[at..at + 6].to_owned())
        .collect()
}

/// The newest entry's id and its span's message texts, read back from the journal.
fn latest_compaction(session: &Session) -> (String, Vec<String>) {
    let entries = session.active_entries().unwrap();
    let compaction = entries.last().unwrap();
    let EntryKind::Compaction {
        span_start_id: Some(start),
        span_end_id: Some(end),
        ..
    } = &compaction.kind
    else {
        panic!("latest entry is not a compaction with a span");
    };
    let texts = session
        .entries_in_span(start, end)
        .unwrap()
        .iter()
        .map(|entry| match &entry.kind {
            EntryKind::Message { message } => message.text(),
            other => panic!("span returned a non-message entry: {other:?}"),
        })
        .collect();
    (compaction.id.clone(), texts)
}

/// The offset a `more: zoom` hint continues from, if the page has one.
fn more_offset(text: &str) -> Option<usize> {
    let hint = text
        .lines()
        .find_map(|line| line.strip_prefix("more: zoom "))?;
    let args: Value = serde_json::from_str(hint).unwrap();
    Some(usize::try_from(args["offset"].as_u64().unwrap()).unwrap())
}

#[test]
fn zoom_returns_the_span_messages_in_order() {
    let dir = tempfile::tempdir().unwrap();
    let mut session = Session::create(dir.path(), Path::new("/work")).unwrap();
    for i in 0..6 {
        session.append(user(&format!("zz-{i:03}"))).unwrap();
    }
    session.compact("summary".into(), None, 2).unwrap();

    let (id, expected) = latest_compaction(&session);
    assert_eq!(expected, ["zz-000", "zz-001", "zz-002", "zz-003"]);

    let out = zoom_call(session.path(), json!({ "id": id }));
    assert!(!out.is_error, "{}", text_of(&out));
    assert_eq!(
        tokens(&text_of(&out)),
        expected
            .iter()
            .map(|t| t[..6].to_owned())
            .collect::<Vec<_>>()
    );
}

#[test]
fn unknown_and_non_compaction_ids_are_tool_errors() {
    let dir = tempfile::tempdir().unwrap();
    let mut session = Session::create(dir.path(), Path::new("/work")).unwrap();
    session.append(user("zz-000")).unwrap();
    session.append(user("zz-001")).unwrap();
    let first = session.active_entries().unwrap()[0].id.clone();
    session.compact("summary".into(), None, 2).unwrap();
    let no_span = session.active_entries().unwrap().last().unwrap().id.clone();

    let unknown = zoom_call(session.path(), json!({ "id": "no-such-id" }));
    assert!(unknown.is_error);
    let message = zoom_call(session.path(), json!({ "id": first }));
    assert!(message.is_error);
    let spanless = zoom_call(session.path(), json!({ "id": no_span }));
    assert!(spanless.is_error);
}

#[test]
fn pages_cover_the_span_once_in_order() {
    let dir = tempfile::tempdir().unwrap();
    let mut session = Session::create(dir.path(), Path::new("/work")).unwrap();
    for i in 0..300 {
        session
            .append(user(&format!("zz-{i:03} {}", "x".repeat(400))))
            .unwrap();
    }
    session.compact("summary".into(), None, 0).unwrap();

    let (id, expected) = latest_compaction(&session);
    assert_eq!(expected.len(), 300);
    let expected_tokens: Vec<String> = expected.iter().map(|t| t[..6].to_owned()).collect();

    for limit in [None, Some(40)] {
        let mut seen = Vec::new();
        let mut pages = 0;
        let mut offset = Some(0);
        while let Some(from) = offset {
            let mut args = json!({ "id": id, "offset": from });
            if let Some(limit) = limit {
                args["limit"] = json!(limit);
            }
            let out = zoom_call(session.path(), args);
            let text = text_of(&out);
            assert!(!out.is_error, "{text}");
            assert!(
                text.len() <= OUTPUT_MAX_BYTES,
                "page is {} bytes",
                text.len()
            );
            seen.extend(tokens(&text));
            offset = more_offset(&text);
            pages += 1;
        }
        assert_eq!(seen, expected_tokens);
        assert!(pages > 1);
    }
}

#[test]
fn oversized_message_is_cut_and_points_at_recall() {
    let dir = tempfile::tempdir().unwrap();
    let mut session = Session::create(dir.path(), Path::new("/work")).unwrap();
    session.append(user(&"é".repeat(60_000))).unwrap();
    session.compact("summary".into(), None, 0).unwrap();

    let (id, _) = latest_compaction(&session);
    let out = zoom_call(session.path(), json!({ "id": id }));
    let text = text_of(&out);
    assert!(!out.is_error, "{text}");
    assert!(
        text.len() <= OUTPUT_MAX_BYTES,
        "page is {} bytes",
        text.len()
    );
    assert!(text.contains("more: recall {\"id\": 0"), "{text}");
}
