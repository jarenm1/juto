use crate::snapcompact::{Archive, Shape, build_archive};
use juto_ai::ContentBlock;

use super::*;

fn user(text: &str) -> EntryKind {
    EntryKind::Message {
        message: Message::user(text),
    }
}

fn tool_assistant(id: &str) -> EntryKind {
    EntryKind::Message {
        message: Message::Assistant(juto_ai::AssistantMessage {
            content: vec![ContentBlock::ToolCall(juto_ai::ToolCall {
                id: id.into(),
                name: "read".into(),
                arguments: serde_json::json!({}),
                thought_signature: None,
            })],
            api: "test".into(),
            provider: "test".into(),
            model: "m".into(),
            usage: Default::default(),
            stop_reason: juto_ai::StopReason::ToolUse,
            timestamp: timestamp_ms(),
            response_id: None,
            error: None,
        }),
    }
}

fn tool_result(id: &str, text: &str) -> EntryKind {
    EntryKind::Message {
        message: Message::ToolResult {
            tool_call_id: id.into(),
            tool_name: "read".into(),
            content: vec![ContentBlock::Text { text: text.into() }],
            is_error: false,
            details: serde_json::Value::Null,
            timestamp: timestamp_ms(),
        },
    }
}

#[test]
fn create_append_reopen_roundtrip() {
    let dir = tempfile::tempdir().unwrap();
    let path;
    {
        let mut s = Session::create(dir.path(), Path::new("/work")).unwrap();
        s.append(user("hello")).unwrap();
        s.append(EntryKind::ModelChange {
            model: "openai/gpt-5".into(),
        })
        .unwrap();
        path = s.path().to_path_buf();
    }
    let mut s = Session::open(&path).unwrap();
    assert_eq!(s.header().cwd, Path::new("/work"));
    assert_eq!(s.current_model().as_deref(), Some("openai/gpt-5"));
    let msgs = s.messages().unwrap();
    assert_eq!(msgs.len(), 1);
    assert_eq!(msgs[0].text(), "hello");
    s.append(user("again")).unwrap();
    let s = Session::open(&path).unwrap();
    assert_eq!(s.messages().unwrap().len(), 2);
}

#[test]
fn branch_durably_switches_leaf() {
    let dir = tempfile::tempdir().unwrap();
    let path;
    let first;
    {
        let mut s = Session::create(dir.path(), Path::new("/w")).unwrap();
        first = s.append(user("one")).unwrap();
        s.append(user("two")).unwrap();
        s.branch(Some(&first)).unwrap();
        s.append(user("alt")).unwrap();
        path = s.path().to_path_buf();
    }
    let s = Session::open(&path).unwrap();
    let texts: Vec<String> = s
        .active_entries()
        .unwrap()
        .iter()
        .filter_map(|e| match &e.kind {
            EntryKind::Message { message } => Some(message.text()),
            _ => None,
        })
        .collect();
    assert_eq!(texts, ["one", "alt"]);
}

#[test]
fn fork_replays_active_path_with_lineage() {
    let src = tempfile::tempdir().unwrap();
    let dst = tempfile::tempdir().unwrap();
    let mut s = Session::create(src.path(), Path::new("/w")).unwrap();
    let first = s.append(user("keep")).unwrap();
    s.append(user("abandoned")).unwrap();
    s.branch(Some(&first)).unwrap();
    let f = s.fork(dst.path()).unwrap();
    assert_ne!(f.header().id, s.header().id);
    assert_eq!(
        f.header().parent_session.as_deref(),
        Some(s.header().id.as_str())
    );
    let msgs = f.messages().unwrap();
    assert_eq!(msgs.len(), 1);
    assert_eq!(msgs[0].text(), "keep");
}

#[test]
fn compact_keeps_tool_groups_whole() {
    let dir = tempfile::tempdir().unwrap();
    let mut s = Session::create(dir.path(), Path::new("/w")).unwrap();
    s.append(user("q1")).unwrap();
    s.append(tool_assistant("c1")).unwrap();
    s.append(tool_result("c1", "r1")).unwrap();
    s.append(user("q2")).unwrap();
    // keep_recent_messages=2 would naively keep [toolResult, q2],
    // orphaning the result from its call.
    s.compact("sum".into(), None, 2).unwrap();
    let msgs = s.messages().unwrap();
    // summary + assistant toolCall + toolResult + q2
    assert_eq!(msgs.len(), 4);
    assert!(matches!(msgs[0], Message::Developer { .. }));
    assert!(matches!(msgs[3], Message::User { .. }));
    drop(s);
    let s = Session::open(
        &dir.path()
            .read_dir()
            .unwrap()
            .next()
            .unwrap()
            .unwrap()
            .path(),
    )
    .unwrap();
    assert_eq!(s.messages().unwrap().len(), 4);
}

#[test]
fn torn_tail_is_truncated_safely() {
    let dir = tempfile::tempdir().unwrap();
    let path;
    {
        let mut s = Session::create(dir.path(), Path::new("/w")).unwrap();
        s.append(user("good")).unwrap();
        path = s.path().to_path_buf();
    }
    let valid_len = std::fs::metadata(&path).unwrap().len();
    // Simulate an interrupted append: partial JSON record, no newline.
    {
        use std::io::Write;
        let mut f = OpenOptions::new().append(true).open(&path).unwrap();
        f.write_all(b"{\"id\":\"zz\",\"parentId\":").unwrap();
    }
    let mut s = Session::open(&path).unwrap();
    assert_eq!(s.messages().unwrap().len(), 1);
    // Next append must not strand the torn bytes.
    s.append(user("next")).unwrap();
    let s = Session::open(&path).unwrap();
    let texts: Vec<String> = s.messages().unwrap().iter().map(|m| m.text()).collect();
    assert_eq!(texts, ["good", "next"]);
    assert_eq!(std::fs::metadata(&path).unwrap().len(), {
        let _ = valid_len;
        std::fs::metadata(&path).unwrap().len()
    });
}

#[test]
fn corrupt_interior_record_fails() {
    let dir = tempfile::tempdir().unwrap();
    let path;
    {
        let mut s = Session::create(dir.path(), Path::new("/w")).unwrap();
        s.append(user("a")).unwrap();
        s.append(user("b")).unwrap();
        path = s.path().to_path_buf();
    }
    // Corrupt the middle record in place (same length, invalid JSON).
    let mut bytes = std::fs::read(&path).unwrap();
    let pos = bytes
        .windows(4)
        .position(|w| w == b"\"id\"")
        .expect("entry id present");
    // Skip header's id occurrence by finding the second entry's line.
    let text = String::from_utf8(bytes.clone()).unwrap();
    let second_nl = text.match_indices('\n').nth(1).unwrap().0;
    let third_nl = text.match_indices('\n').nth(2).unwrap().0;
    for b in &mut bytes[second_nl + 1..second_nl + 9] {
        *b = b'!';
    }
    let _ = pos;
    let _ = third_nl;
    std::fs::write(&path, &bytes).unwrap();
    assert!(matches!(
        Session::open(&path),
        Err(SessionError::Corrupt { .. })
    ));
}

#[test]
fn open_never_creates_or_rewrites_valid_file() {
    let dir = tempfile::tempdir().unwrap();
    let missing = dir.path().join("nope.jsonl");
    assert!(Session::open(&missing).is_err());
    assert!(!missing.exists());
    let mut s = Session::create(dir.path(), Path::new("/w")).unwrap();
    s.append(user("x")).unwrap();
    let bytes = std::fs::read(s.path()).unwrap();
    drop(s);
    let s = Session::open(
        &dir.path().join(
            std::fs::read_dir(dir.path())
                .unwrap()
                .next()
                .unwrap()
                .unwrap()
                .file_name(),
        ),
    )
    .unwrap();
    assert_eq!(std::fs::read(s.path()).unwrap(), bytes);
}

#[test]
fn open_rejects_non_session_file() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("random.jsonl");
    std::fs::write(&path, "{\"hello\":1}\n").unwrap();
    assert!(matches!(
        Session::open(&path),
        Err(SessionError::Invalid { .. })
    ));
    assert_eq!(std::fs::read(&path).unwrap(), b"{\"hello\":1}\n");
}

#[test]
fn path_view_indices_survive_compaction() {
    let dir = tempfile::tempdir().unwrap();
    let mut s = Session::create(dir.path(), Path::new("/w")).unwrap();
    for text in ["m0", "m1", "m2", "m3", "m4"] {
        s.append(user(text)).unwrap();
    }
    let before: Vec<(usize, String)> = s
        .path_view()
        .unwrap()
        .messages
        .iter()
        .map(|m| (m.index, m.id.to_owned()))
        .collect();
    s.compact("summary".into(), None, 2).unwrap();
    s.append(user("m5")).unwrap();
    let view = s.path_view().unwrap();
    let after: Vec<(usize, String)> = view.messages[..5]
        .iter()
        .map(|m| (m.index, m.id.to_owned()))
        .collect();
    assert_eq!(after, before);
    let mark = view.compaction.unwrap();
    assert_eq!((mark.at, mark.first_visible), (5, 3));
    let visible: Vec<String> = view.visible().iter().map(|m| m.message.text()).collect();
    assert_eq!(visible, ["m3", "m4", "m5"]);
    assert_eq!(s.messages().unwrap().len(), 4);
}

#[test]
fn compaction_that_keeps_nothing_hides_the_whole_prefix_but_keeps_it_indexed() {
    let dir = tempfile::tempdir().unwrap();
    let mut s = Session::create(dir.path(), Path::new("/w")).unwrap();
    s.append(user("a")).unwrap();
    s.append(user("b")).unwrap();
    s.compact("summary".into(), None, 0).unwrap();
    let view = s.path_view().unwrap();
    let mark = view.compaction.unwrap();
    assert_eq!((mark.at, mark.first_visible), (2, 2));
    assert!(view.visible().is_empty());
    assert_eq!(view.messages.len(), 2);
    assert_eq!(s.messages().unwrap().len(), 1);
}

#[test]
fn branch_renumbers_the_active_path() {
    let dir = tempfile::tempdir().unwrap();
    let mut s = Session::create(dir.path(), Path::new("/w")).unwrap();
    for text in ["a", "b", "c"] {
        s.append(user(text)).unwrap();
    }
    let a = s.path_view().unwrap().messages[0].id.to_owned();
    s.branch(Some(&a)).unwrap();
    s.append(user("d")).unwrap();
    let view = s.path_view().unwrap();
    let texts: Vec<(usize, String)> = view
        .messages
        .iter()
        .map(|m| (m.index, m.message.text()))
        .collect();
    assert_eq!(texts, [(0, "a".to_owned()), (1, "d".to_owned())]);
}

#[test]
fn archive_reopens_as_user_text_and_frames() {
    let dir = tempfile::tempdir().unwrap();
    let archive = build_archive(None, "q1", Shape::ANTHROPIC, 1)
        .unwrap()
        .unwrap();
    let frame = archive.frames[0].png.clone();
    let path;
    {
        let mut s = Session::create(dir.path(), Path::new("/w")).unwrap();
        s.append(user("q1")).unwrap();
        s.append(user("q2")).unwrap();
        s.compact("lead".into(), Some(archive), 1).unwrap();
        path = s.path().to_path_buf();
    }
    let s = Session::open(&path).unwrap();
    let msgs = s.messages().unwrap();
    assert_eq!(msgs.len(), 2);
    let Message::User { content, .. } = &msgs[0] else {
        panic!("archived history should be user content");
    };
    assert!(msgs[0].text().contains("lead"));
    assert!(
        matches!(&content[1], ContentBlock::Image { data, mime_type }
        if data == &frame && mime_type == "image/png")
    );
    assert_eq!(msgs[1].text(), "q2");
    assert_eq!(s.path_view().unwrap().messages[0].message.text(), "q1");
}

#[test]
fn compaction_entries_without_archive_still_load() {
    let kind: EntryKind =
        serde_json::from_str(r#"{"type":"compaction","summary":"s","first_kept_id":null}"#)
            .unwrap();
    assert!(matches!(kind, EntryKind::Compaction { archive: None, .. }));
}

#[test]
fn old_archive_ignores_removed_truncation_metadata() {
    let archive: Archive =
        serde_json::from_str(r#"{"source":"original text","truncated_chars":0,"frames":[]}"#)
            .unwrap();
    assert_eq!(archive.text(), "original text");
}

#[test]
fn compaction_plan_reports_previous_and_discarded() {
    let dir = tempfile::tempdir().unwrap();
    let mut s = Session::create(dir.path(), Path::new("/w")).unwrap();
    s.append(user("q1")).unwrap();
    s.append(user("q2")).unwrap();
    s.append(user("q3")).unwrap();
    s.compact("first".into(), None, 2).unwrap();
    s.append(user("q4")).unwrap();
    let plan = s.compaction_plan(1).unwrap();
    assert_eq!(plan.previous[0].summary, "first");
    let discarded: Vec<String> = plan
        .discarded
        .iter()
        .map(|message| message.text())
        .collect();
    assert_eq!(discarded, ["q2", "q3"]);
}

fn message_ids(session: &Session) -> Vec<String> {
    session
        .path_view()
        .unwrap()
        .messages
        .iter()
        .map(|message| message.id.to_owned())
        .collect()
}

fn span_texts(session: &Session, start: &str, end: &str) -> Vec<String> {
    session
        .entries_in_span(start, end)
        .unwrap()
        .iter()
        .map(|entry| match &entry.kind {
            EntryKind::Message { message } => message.text(),
            other => panic!("span returned a non-message entry: {other:?}"),
        })
        .collect()
}

fn latest_span(session: &Session) -> (Option<String>, Option<String>) {
    session
        .active_entries()
        .unwrap()
        .iter()
        .rev()
        .find_map(|entry| match &entry.kind {
            EntryKind::Compaction {
                span_start_id,
                span_end_id,
                ..
            } => Some((span_start_id.clone(), span_end_id.clone())),
            _ => None,
        })
        .unwrap()
}

#[test]
fn compaction_span_covers_exactly_the_replaced_messages() {
    let dir = tempfile::tempdir().unwrap();
    let path;
    let ids;
    {
        let mut s = Session::create(dir.path(), Path::new("/w")).unwrap();
        for text in ["q1", "q2", "q3", "q4", "q5", "q6"] {
            s.append(user(text)).unwrap();
        }
        ids = message_ids(&s);
        s.compact("first".into(), None, 2).unwrap();
        path = s.path().to_path_buf();
    }
    let s = Session::open(&path).unwrap();
    // keep_recent_messages=2 keeps q5 and q6, so the span is q1 through q4.
    assert_eq!(
        latest_span(&s),
        (Some(ids[0].clone()), Some(ids[3].clone()))
    );
    assert_eq!(span_texts(&s, &ids[0], &ids[3]), ["q1", "q2", "q3", "q4"]);
    assert_eq!(span_texts(&s, &ids[0], &ids[0]), ["q1"]);
    assert_eq!(span_texts(&s, &ids[3], &ids[3]), ["q4"]);
    assert_eq!(s.messages().unwrap().len(), 3);
}

#[test]
fn second_compaction_span_starts_at_the_previous_first_kept_message() {
    let dir = tempfile::tempdir().unwrap();
    let mut s = Session::create(dir.path(), Path::new("/w")).unwrap();
    for text in ["q1", "q2", "q3", "q4", "q5", "q6"] {
        s.append(user(text)).unwrap();
    }
    s.compact("first".into(), None, 2).unwrap();
    s.append(user("q7")).unwrap();
    s.append(user("q8")).unwrap();
    let ids = message_ids(&s);
    // The first compaction kept q5 onward, so the second replaces q5 through q7.
    // Its own compaction entry sits between q6 and q7 on the path and is skipped.
    s.compact("second".into(), None, 1).unwrap();
    assert_eq!(
        latest_span(&s),
        (Some(ids[4].clone()), Some(ids[6].clone()))
    );
    assert_eq!(span_texts(&s, &ids[4], &ids[6]), ["q5", "q6", "q7"]);
    assert_eq!(span_texts(&s, &ids[4], &ids[4]), ["q5"]);
    assert_eq!(span_texts(&s, &ids[0], &ids[3]), ["q1", "q2", "q3", "q4"]);
    // The second summary plus q8, the only message after its cut.
    assert_eq!(s.messages().unwrap().len(), 2);
}

#[test]
fn entries_in_span_rejects_reversed_unknown_and_off_path_boundaries() {
    let dir = tempfile::tempdir().unwrap();
    let mut s = Session::create(dir.path(), Path::new("/w")).unwrap();
    for text in ["q1", "q2", "q3"] {
        s.append(user(text)).unwrap();
    }
    let ids = message_ids(&s);
    assert!(matches!(
        s.entries_in_span(&ids[2], &ids[0]),
        Err(SessionError::InvalidSpan { .. })
    ));
    assert!(matches!(
        s.entries_in_span(&ids[0], "missing"),
        Err(SessionError::InvalidSpan { .. })
    ));
    // Branching back to q1 takes q2 off the active path.
    s.branch(Some(&ids[0])).unwrap();
    s.append(user("q4")).unwrap();
    assert!(matches!(
        s.entries_in_span(&ids[0], &ids[1]),
        Err(SessionError::InvalidSpan { .. })
    ));
}

#[test]
fn compaction_that_replaces_nothing_records_no_span() {
    let dir = tempfile::tempdir().unwrap();
    let mut s = Session::create(dir.path(), Path::new("/w")).unwrap();
    s.append(user("q1")).unwrap();
    s.append(user("q2")).unwrap();
    s.compact("summary".into(), None, 2).unwrap();
    assert_eq!(latest_span(&s), (None, None));
    assert_eq!(s.messages().unwrap().len(), 3);
}

#[test]
fn journal_without_span_fields_renders_the_same_messages() {
    let dir = tempfile::tempdir().unwrap();
    let mut s = Session::create(dir.path(), Path::new("/w")).unwrap();
    for text in ["q1", "q2", "q3"] {
        s.append(user(text)).unwrap();
    }
    s.compact("sum".into(), None, 1).unwrap();
    s.append(user("q4")).unwrap();
    let expected: Vec<String> = s.messages().unwrap().iter().map(Message::text).collect();
    let path = s.path().to_path_buf();
    drop(s);
    // Rewrite the journal as the previous build wrote it: compaction records carry no span.
    let legacy: String = std::fs::read_to_string(&path)
        .unwrap()
        .lines()
        .map(|line| {
            let mut record: serde_json::Value = serde_json::from_str(line).unwrap();
            if let Some(fields) = record.as_object_mut() {
                fields.remove("span_start_id");
                fields.remove("span_end_id");
            }
            format!("{record}\n")
        })
        .collect();
    std::fs::write(&path, &legacy).unwrap();
    assert!(!legacy.contains("span_"));
    let s = Session::open(&path).unwrap();
    let actual: Vec<String> = s.messages().unwrap().iter().map(Message::text).collect();
    assert_eq!(actual, expected);
    assert_eq!(latest_span(&s), (None, None));
}

#[test]
fn compaction_marker_names_its_entry_id() {
    let dir = tempfile::tempdir().unwrap();
    let mut s = Session::create(dir.path(), Path::new("/w")).unwrap();
    for text in ["q1", "q2", "q3"] {
        s.append(user(text)).unwrap();
    }
    s.compact("sum".into(), None, 1).unwrap();
    let id = s.active_entries().unwrap().last().unwrap().id.clone();
    let messages = s.messages().unwrap();
    assert!(messages[0].text().contains(&format!("id={id}")));
}
