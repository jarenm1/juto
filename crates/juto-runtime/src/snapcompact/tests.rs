use juto_ai::{AssistantMessage, ContentBlock, Message, ToolCall};
use serde_json::json;

use super::{
    Shape, build_archive, frame_budget,
    render::{FRAME_WIDTH, render_page},
    text::{NEWLINE_GLYPH, normalize, serialize_conversation, truncate},
};

/// Decodes a PNG into (width, height, 8-bit grayscale pixels).
fn decode(png_bytes: &[u8]) -> (usize, usize, Vec<u8>) {
    let mut reader = png::Decoder::new(std::io::Cursor::new(png_bytes))
        .read_info()
        .unwrap();
    let (width, height) = (reader.info().width as usize, reader.info().height as usize);
    let mut pixels = vec![0; width * height];
    reader.next_frame(&mut pixels).unwrap();
    (width, height, pixels)
}

fn cells(text: &str) -> Vec<char> {
    text.chars().collect()
}

#[test]
fn shapes_fit_the_frame_width_and_share_capacity() {
    for shape in [Shape::ANTHROPIC, Shape::DEFAULT] {
        assert!(shape.cols() * shape.cell_width <= FRAME_WIDTH);
        assert!(shape.rows() * shape.cell_height <= FRAME_WIDTH);
    }
    assert_eq!(Shape::ANTHROPIC.capacity(), Shape::DEFAULT.capacity());
    assert_eq!(Shape::for_api("anthropic-messages"), Shape::ANTHROPIC);
    assert_eq!(Shape::for_api("openai-responses"), Shape::DEFAULT);
}

#[test]
fn provider_frames_have_fixed_width_and_minimum_height() {
    for (api, id, width) in [
        ("anthropic-messages", "claude-opus-4-8", 1932),
        ("google-generative-ai", "gemini-3", 2048),
        ("openai-responses", "gpt-5", 1568),
    ] {
        let shape = Shape::for_model(api, id);
        let (actual_width, height, _) = decode(&render_page(&cells("hello"), shape).unwrap());
        assert_eq!(actual_width, width);
        assert_eq!(height, 64);
    }
}

#[test]
fn render_is_deterministic_and_height_follows_used_rows() {
    let text = cells(&"lorem ipsum dolor ".repeat(60));
    let first = render_page(&text, Shape::ANTHROPIC).unwrap();
    let second = render_page(&text, Shape::ANTHROPIC).unwrap();
    assert_eq!(first, second);

    let (width, height, _) = decode(&first);
    assert_eq!(width, FRAME_WIDTH);
    // 1080 characters at 142 per row use 8 rows of 16 pixels.
    assert_eq!(height, 8 * Shape::ANTHROPIC.cell_height);

    let (_, short_height, _) = decode(&render_page(&cells("hi"), Shape::ANTHROPIC).unwrap());
    assert_eq!(
        short_height, 64,
        "short pages are padded to the minimum height"
    );
}

#[test]
fn glyphs_stay_inside_their_cell() {
    let shape = Shape::ANTHROPIC;
    let (_, _, pixels) = decode(&render_page(&cells("A"), shape).unwrap());
    let width = FRAME_WIDTH;
    let ink: Vec<(usize, usize)> = pixels
        .iter()
        .enumerate()
        .filter(|(_, value)| **value == 0)
        .map(|(index, _)| (index % width, index / width))
        .collect();
    assert!(!ink.is_empty(), "a visible glyph draws ink");
    assert!(
        ink.iter()
            .all(|(x, y)| *x < shape.cell_width && *y < shape.cell_height),
        "ink must stay inside the first cell"
    );
}

#[test]
fn newline_glyph_fills_exactly_one_cell() {
    let shape = Shape::ANTHROPIC;
    let (_, _, pixels) = decode(&render_page(&[NEWLINE_GLYPH], shape).unwrap());
    let width = FRAME_WIDTH;
    for y in 0..shape.cell_height {
        for x in 0..shape.cell_width {
            assert_eq!(pixels[y * width + x], 0);
        }
        assert_eq!(pixels[y * width + shape.cell_width], 255);
    }
}

#[test]
fn archive_rejects_budget_overflow_without_dropping_history() {
    let transcript = "word ".repeat(8_000);
    assert!(
        build_archive(None, &transcript, Shape::DEFAULT, 2)
            .unwrap()
            .is_none()
    );
    let text = "x".repeat(Shape::DEFAULT.capacity() + 1);
    let archive = build_archive(None, &text, Shape::DEFAULT, 2)
        .unwrap()
        .unwrap();
    assert_eq!(archive.frames.len(), 2);
    assert_eq!(archive.source, text);
}

#[test]
fn later_compactions_append_to_the_archived_text() {
    let first = build_archive(None, "first turn", Shape::DEFAULT, 4)
        .unwrap()
        .unwrap();
    let second = build_archive(Some(&first), "second turn", Shape::DEFAULT, 4)
        .unwrap()
        .unwrap();
    assert_eq!(
        second.source,
        format!("first turn{NEWLINE_GLYPH}second turn")
    );
}

#[test]
fn later_compaction_overflow_preserves_the_previous_archive() {
    let first = build_archive(
        None,
        &"a".repeat(Shape::DEFAULT.capacity()),
        Shape::DEFAULT,
        1,
    )
    .unwrap()
    .unwrap();
    assert!(
        build_archive(Some(&first), "tail", Shape::DEFAULT, 1)
            .unwrap()
            .is_none()
    );
    assert_eq!(first.source.len(), Shape::DEFAULT.capacity());
}

#[test]
fn frame_budget_respects_provider_window_and_floor() {
    assert_eq!(frame_budget("anthropic", Some(200_000)), 12);
    assert_eq!(frame_budget("anthropic", None), 80);
    // A quarter of a 1M window at 4096 tokens per frame: 61 frames, under the cap.
    assert_eq!(frame_budget("openai", Some(1_000_000)), 61);
    assert_eq!(frame_budget("openai", Some(2_000_000)), 80);
    assert_eq!(frame_budget("anthropic", Some(8_000)), 0);
    assert_eq!(frame_budget("some-local-server", None), 5);
}

#[test]
fn normalized_text_only_contains_renderable_characters() {
    let messy = "\u{1b}[31mred\u{1b}[0m \u{200b}\t漢字 🙂✅ “quoted” ┌─┐\n\n\ne\u{301} ＡＢ é";
    for ch in normalize(messy).chars() {
        assert!(
            ch == NEWLINE_GLYPH
                || ch == ' '
                || ch.is_ascii_graphic()
                || ('\u{a0}'..='\u{ff}').contains(&ch),
            "unexpected character {ch:?}"
        );
    }
}

#[test]
fn normalize_collapses_whitespace_and_folds_symbols() {
    assert_eq!(normalize("  a \n\n  b  "), format!("a{NEWLINE_GLYPH}b"));
    assert_eq!(normalize("a\u{200b}b"), "a\\u{200B}b");
    assert_eq!(normalize("e\u{301}"), "e\\u{301}");
    assert_eq!(normalize("✅ ok"), "[OK] ok");
    assert_eq!(normalize("┌─┐"), "+-+");
    assert_eq!(normalize("“q”"), "\"q\"");
    assert_eq!(normalize("ＡＢ"), "AB");
    assert_eq!(normalize("🙂"), "\\u{1F642}");
    assert_eq!(normalize("漢"), "\\u{6F22}");
}

fn tool_call(id: &str, arguments: serde_json::Value) -> ContentBlock {
    ContentBlock::ToolCall(ToolCall {
        id: id.into(),
        name: "bash".into(),
        arguments,
        thought_signature: None,
    })
}

fn assistant(content: Vec<ContentBlock>) -> Message {
    Message::Assistant(AssistantMessage {
        content,
        api: "test".into(),
        provider: "test".into(),
        model: "test".into(),
        usage: Default::default(),
        stop_reason: Default::default(),
        timestamp: 0,
        response_id: None,
        error: None,
    })
}

fn tool_result(id: &str, text: &str) -> Message {
    Message::ToolResult {
        tool_call_id: id.into(),
        tool_name: "bash".into(),
        content: vec![ContentBlock::Text { text: text.into() }],
        is_error: false,
        details: Default::default(),
        timestamp: 0,
    }
}

#[test]
fn serialization_omits_reasoning_and_attaches_output_to_its_call() {
    let messages = vec![
        Message::user("list the files"),
        assistant(vec![
            ContentBlock::Thinking {
                thinking: "SECRET-REASONING".into(),
                thinking_signature: None,
            },
            tool_call("c1", json!({"command": "ls", "intent": "list"})),
        ]),
        tool_result("c1", "a.txt"),
    ];
    let transcript = serialize_conversation(&messages);
    assert!(!transcript.contains("SECRET-REASONING"));
    assert_eq!(transcript.matches("a.txt").count(), 1);
    assert!(transcript.find("¶call:").unwrap() < transcript.find("a.txt").unwrap());
}

#[test]
fn serialization_caps_long_tool_output() {
    let messages = vec![
        assistant(vec![tool_call("c1", json!({"command": "cat big"}))]),
        tool_result("c1", &"z".repeat(50_000)),
    ];
    let transcript = serialize_conversation(&messages);
    assert!(transcript.contains("elided"));
    assert!(transcript.chars().count() < 2_500);
}

#[test]
fn truncate_keeps_the_start_and_the_end() {
    let text = format!("{}{}", "H".repeat(1_000), "T".repeat(1_000));
    let cut = truncate(&text, 100);
    assert!(cut.starts_with('H'));
    assert!(cut.ends_with('T'));
    assert!(cut.contains("elided"));
    assert_eq!(truncate("short", 100), "short");
}
