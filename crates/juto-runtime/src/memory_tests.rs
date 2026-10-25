use juto_ai::Message;

use super::*;

fn users(texts: &[&str]) -> Vec<Message> {
    texts.iter().map(|text| Message::user(*text)).collect()
}

/// View lines between `<chat>` and `</chat>`.
fn lines(memory: &Memory) -> Vec<String> {
    memory
        .view_text()
        .unwrap_or_default()
        .lines()
        .filter(|line| line.contains('|'))
        .map(str::to_owned)
        .collect()
}

/// Write every node the memory asks for with `line`, as the compactor would.
fn build_all(memory: &Memory, line: impl Fn(Part) -> String) {
    loop {
        let jobs = memory.take_jobs();
        if jobs.is_empty() {
            return;
        }
        for (part, _) in jobs {
            memory.store(part, line(part)).unwrap();
        }
    }
}

#[test]
fn short_messages_are_their_own_lines() {
    let dir = tempfile::tempdir().unwrap();
    let memory = Memory::open(dir.path()).unwrap();
    assert_eq!(memory.view_text(), None);
    memory
        .sync(&users(&["hello", "multi\n  line\nmessage"]), 100_000)
        .unwrap();
    assert_eq!(
        lines(&memory),
        ["0+1|user: hello", "1+1|user: multi line message"]
    );
    assert!(memory.take_jobs().is_empty(), "nothing needs the compactor");
    assert!(memory.settled().unwrap());
}

#[test]
fn a_long_message_waits_for_its_line() {
    let dir = tempfile::tempdir().unwrap();
    let memory = Memory::open(dir.path()).unwrap();
    let long = "x".repeat(NODE * 2);
    memory.sync(&users(&["short", &long]), 100_000).unwrap();
    assert_eq!(lines(&memory)[1], format!("1+1|{WAIT}"));
    assert!(!memory.settled().unwrap());

    let jobs = memory.take_jobs();
    assert_eq!(jobs.len(), 1);
    assert_eq!(jobs[0].0, Part::leaf(1));
    assert_eq!(jobs[0].1, format!("user: {long}"));
    // A job being written is not handed out twice.
    assert!(memory.take_jobs().is_empty());

    memory
        .store(Part::leaf(1), "a long run of x".into())
        .unwrap();
    assert_eq!(lines(&memory)[1], "1+1|a long run of x");
    assert!(memory.settled().unwrap());
}

#[test]
fn over_budget_the_oldest_pairs_merge_first_down_to_half() {
    let dir = tempfile::tempdir().unwrap();
    let memory = Memory::open(dir.path()).unwrap();
    // Sixteen 300-byte leaves (with "user: "), merged lines of 40 bytes.
    let texts: Vec<String> = (0..16)
        .map(|i| format!("{i:02}{}", "m".repeat(292)))
        .collect();
    let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
    let messages = users(&refs);
    let high = 3_000;
    memory.sync(&messages, high).unwrap();
    build_all(&memory, |part| format!("summary {}", part.reference()));
    assert!(memory.settled().unwrap());

    let view = lines(&memory);
    let refs: Vec<&str> = view
        .iter()
        .map(|line| line.split('|').next().unwrap())
        .collect();
    // Detail fades with age: old messages are covered by broad lines, the
    // newest stay whole, and the lines still cover every message in order.
    assert!(
        refs[0].ends_with("+8") || refs[0].ends_with("+4"),
        "{refs:?}"
    );
    assert_eq!(*refs.last().unwrap(), "15+1");
    let mut next = 0;
    for reference in &refs {
        let (id, n) = reference.split_once('+').unwrap();
        assert_eq!(id.parse::<usize>().unwrap(), next);
        next += n.parse::<usize>().unwrap();
    }
    assert_eq!(next, 16);
    let size: usize = view
        .iter()
        .map(|line| line.split_once('|').unwrap().1.len())
        .sum();
    assert!(size <= high / 2, "merged to {size} bytes");
}

#[test]
fn under_budget_the_view_only_grows_at_its_end() {
    let dir = tempfile::tempdir().unwrap();
    let memory = Memory::open(dir.path()).unwrap();
    let mut messages = Vec::new();
    let mut before: Vec<String> = Vec::new();
    for i in 0..20 {
        messages.push(Message::user(format!("message {i}")));
        memory.sync(&messages, 100_000).unwrap();
        let now = lines(&memory);
        assert_eq!(&now[..before.len()], &before[..], "an old line changed");
        before = now;
    }
}

#[test]
fn the_view_and_written_lines_survive_reopening() {
    let dir = tempfile::tempdir().unwrap();
    let texts: Vec<String> = (0..8).map(|i| format!("{i}{}", "y".repeat(600))).collect();
    let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
    let messages = users(&refs);
    let view = {
        let memory = Memory::open(dir.path()).unwrap();
        memory.sync(&messages, 2_048).unwrap();
        build_all(&memory, |part| format!("line {}", part.reference()));
        assert!(memory.settled().unwrap());
        lines(&memory)
    };
    let memory = Memory::open(dir.path()).unwrap();
    memory.sync(&messages, 2_048).unwrap();
    assert_eq!(lines(&memory), view);
    assert!(memory.take_jobs().is_empty(), "written lines are read back");
}

#[test]
fn summarized_until_is_the_end_of_the_last_merged_line() {
    let dir = tempfile::tempdir().unwrap();
    let main = dir.path().join("main.jsonl");
    assert_eq!(summarized_until(&main), 0, "no memory yet");

    let memory = Memory::open(&memory_dir(&main)).unwrap();
    memory.sync(&users(&["a", "b", "c"]), 100_000).unwrap();
    assert_eq!(summarized_until(&main), 0, "every message has its own line");

    let texts: Vec<String> = (0..16)
        .map(|i| format!("{i:02}{}", "m".repeat(292)))
        .collect();
    let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
    memory.sync(&users(&refs), 3_000).unwrap();
    build_all(&memory, |part| format!("summary {}", part.reference()));
    assert!(memory.settled().unwrap());
    let view = lines(&memory);
    // The first single-message line after the last merged one starts the edge.
    let edge = view
        .iter()
        .rev()
        .map(|line| line.split('|').next().unwrap())
        .find(|reference| !reference.ends_with("+1"))
        .map(|reference| {
            let (id, n) = reference.split_once('+').unwrap();
            id.parse::<usize>().unwrap() + n.parse::<usize>().unwrap()
        })
        .expect("over budget, old lines merged");
    assert!(edge > 0 && edge < 16, "{view:?}");
    assert_eq!(summarized_until(&main), edge);
}

#[test]
fn a_shorter_conversation_starts_the_view_over() {
    let dir = tempfile::tempdir().unwrap();
    let memory = Memory::open(dir.path()).unwrap();
    memory.sync(&users(&["a", "b", "c"]), 100_000).unwrap();
    memory.sync(&users(&["other"]), 100_000).unwrap();
    assert_eq!(lines(&memory), ["0+1|user: other"]);
}

#[test]
fn zoom_opens_a_line_into_its_halves_down_to_a_message() {
    let dir = tempfile::tempdir().unwrap();
    let memory = Memory::open(dir.path()).unwrap();
    memory.sync(&users(&["a", "b", "c", "d"]), 100_000).unwrap();
    let Ok(Zoomed::Lines(halves)) = memory.zoom(0, 4) else {
        panic!("4 messages make a node 0+4");
    };
    assert_eq!(halves, ["0+2|user: a user: b", "2+2|user: c user: d"]);
    assert!(matches!(memory.zoom(3, 1), Ok(Zoomed::Message(3))));
    for (id, n) in [(1, 2), (0, 3), (4, 1), (0, 8)] {
        assert!(memory.zoom(id, n).is_err(), "{id}+{n} is not a node");
    }
}

#[test]
fn compactor_input_keeps_the_head_and_tail_of_a_huge_source() {
    let source = format!(
        "{}{}{}",
        "h".repeat(INPUT_HEAD),
        "m".repeat(50_000),
        "t".repeat(INPUT_TAIL)
    );
    let input = compactor_input(&source);
    assert!(input.starts_with(&"h".repeat(INPUT_HEAD)));
    assert!(input.ends_with(&"t".repeat(INPUT_TAIL)));
    assert!(input.contains("[… 50000 bytes omitted …]"));
    assert_eq!(compactor_input("small"), "small");
}
