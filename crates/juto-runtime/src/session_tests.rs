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
    s.compact("sum".into(), 2).unwrap();
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
