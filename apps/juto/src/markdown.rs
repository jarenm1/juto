//! Parses assistant Markdown into a small block tree. Rendering lives in
//! `markdown_element.rs`; this module has no GPUI dependency so its behavior
//! is testable on its own.
//!
//! Inline content comes out already flattened for layout (one shared string
//! plus span lengths), so a renderer can redraw a parsed document every frame
//! without building strings.

use std::iter::Peekable;
use std::sync::Arc;

use pulldown_cmark::{Alignment, CodeBlockKind, Event, Options, Parser, Tag, TagEnd};

/// One block-level element of a Markdown document.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Block {
    Paragraph(Inline),
    Heading {
        level: u8,
        text: Inline,
    },
    Code {
        language: Option<Arc<str>>,
        text: Arc<str>,
    },
    List {
        start: Option<u64>,
        items: Vec<Item>,
    },
    Quote(Vec<Block>),
    Table(Table),
    Rule,
}

/// One list item. `checked` is set for GFM task items.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Item {
    pub checked: Option<bool>,
    pub blocks: Vec<Block>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Table {
    pub align: Vec<Align>,
    pub header: Vec<Inline>,
    pub rows: Vec<Vec<Inline>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Align {
    None,
    Left,
    Center,
    Right,
}

/// Inline content: the text of every span joined, and the spans that tile it
/// in order. Adjacent spans never share a style.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Inline {
    pub text: Arc<str>,
    pub spans: Vec<Span>,
}

/// `len` bytes of an `Inline`'s text sharing one style.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Span {
    pub len: usize,
    pub style: Style,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Style {
    pub bold: bool,
    pub italic: bool,
    pub strike: bool,
    pub code: bool,
    pub link: Option<Arc<str>>,
}

/// Parses `source` as GitHub-flavored Markdown (tables, strikethrough, task
/// lists). Soft line breaks become `\n` so single newlines stay visible, as
/// they are in chat text. Raw HTML is shown as plain text.
pub fn parse(source: &str) -> Vec<Block> {
    let options =
        Options::ENABLE_TABLES | Options::ENABLE_STRIKETHROUGH | Options::ENABLE_TASKLISTS;
    let mut events = Parser::new_ext(source, options).peekable();
    blocks_until(&mut events, |_| false)
}

type Events<'a> = Peekable<Parser<'a>>;

fn blocks_until(events: &mut Events<'_>, stop: fn(&TagEnd) -> bool) -> Vec<Block> {
    let mut blocks = Vec::new();
    while let Some(event) = events.next() {
        match event {
            Event::End(end) if stop(&end) => break,
            Event::Start(Tag::Paragraph) => {
                let text = inlines_until(events, |end| matches!(end, TagEnd::Paragraph));
                blocks.push(Block::Paragraph(text));
            }
            Event::Start(Tag::Heading { level, .. }) => {
                let text = inlines_until(events, |end| matches!(end, TagEnd::Heading(_)));
                blocks.push(Block::Heading {
                    level: level as u8,
                    text,
                });
            }
            Event::Start(Tag::CodeBlock(kind)) => {
                let language = match kind {
                    CodeBlockKind::Fenced(info) => info.split_whitespace().next().map(Arc::from),
                    CodeBlockKind::Indented => None,
                };
                let mut text = String::new();
                for event in events.by_ref() {
                    match event {
                        Event::Text(t) => text.push_str(&t),
                        Event::End(TagEnd::CodeBlock) => break,
                        _ => {}
                    }
                }
                if text.ends_with('\n') {
                    text.pop();
                }
                blocks.push(Block::Code {
                    language,
                    text: text.into(),
                });
            }
            Event::Start(Tag::HtmlBlock) => {
                let mut text = String::new();
                for event in events.by_ref() {
                    match event {
                        Event::Html(t) => text.push_str(&t),
                        Event::End(TagEnd::HtmlBlock) => break,
                        _ => {}
                    }
                }
                let mut inline = InlineBuilder::default();
                inline.push(text.trim_end(), &Style::default());
                if !inline.text.is_empty() {
                    blocks.push(Block::Paragraph(inline.finish()));
                }
            }
            Event::Start(Tag::List(start)) => {
                let mut items = Vec::new();
                while let Some(event) = events.next() {
                    match event {
                        Event::Start(Tag::Item) => {
                            let checked = match events.peek() {
                                Some(Event::TaskListMarker(checked)) => {
                                    let checked = *checked;
                                    events.next();
                                    Some(checked)
                                }
                                _ => None,
                            };
                            let blocks = blocks_until(events, |end| matches!(end, TagEnd::Item));
                            items.push(Item { checked, blocks });
                        }
                        Event::End(TagEnd::List(_)) => break,
                        _ => {}
                    }
                }
                blocks.push(Block::List { start, items });
            }
            Event::Start(Tag::BlockQuote(_)) => {
                let inner = blocks_until(events, |end| matches!(end, TagEnd::BlockQuote(_)));
                blocks.push(Block::Quote(inner));
            }
            Event::Start(Tag::Table(alignments)) => {
                let align = alignments
                    .into_iter()
                    .map(|a| match a {
                        Alignment::None => Align::None,
                        Alignment::Left => Align::Left,
                        Alignment::Center => Align::Center,
                        Alignment::Right => Align::Right,
                    })
                    .collect();
                let mut header = Vec::new();
                let mut rows = Vec::new();
                while let Some(event) = events.next() {
                    match event {
                        Event::Start(Tag::TableHead) => {
                            header = table_row(events, TagEnd::TableHead)
                        }
                        Event::Start(Tag::TableRow) => {
                            rows.push(table_row(events, TagEnd::TableRow))
                        }
                        Event::End(TagEnd::Table) => break,
                        _ => {}
                    }
                }
                blocks.push(Block::Table(Table {
                    align,
                    header,
                    rows,
                }));
            }
            Event::Rule => blocks.push(Block::Rule),
            // A tight list item emits its text without a paragraph wrapper.
            // Collect that run of inline events as a paragraph.
            event if is_inline(&event) => {
                let mut inline = InlineBuilder::default();
                push_inline(events, event, &Style::default(), &mut inline);
                while events.peek().is_some_and(is_inline) {
                    if let Some(next) = events.next() {
                        push_inline(events, next, &Style::default(), &mut inline);
                    }
                }
                blocks.push(Block::Paragraph(inline.finish()));
            }
            _ => {}
        }
    }
    blocks
}

fn table_row(events: &mut Events<'_>, row_end: TagEnd) -> Vec<Inline> {
    let mut cells = Vec::new();
    while let Some(event) = events.next() {
        match event {
            Event::Start(Tag::TableCell) => {
                cells.push(inlines_until(events, |end| {
                    matches!(end, TagEnd::TableCell)
                }));
            }
            Event::End(end) if end == row_end => break,
            _ => {}
        }
    }
    cells
}

fn inlines_until(events: &mut Events<'_>, stop: fn(&TagEnd) -> bool) -> Inline {
    let mut inline = InlineBuilder::default();
    push_inlines(events, &Style::default(), &mut inline, stop);
    inline.finish()
}

fn push_inlines(
    events: &mut Events<'_>,
    style: &Style,
    inline: &mut InlineBuilder,
    stop: fn(&TagEnd) -> bool,
) {
    while let Some(event) = events.next() {
        match event {
            Event::End(end) if stop(&end) => break,
            event => push_inline(events, event, style, inline),
        }
    }
}

fn push_inline(
    events: &mut Events<'_>,
    event: Event<'_>,
    style: &Style,
    inline: &mut InlineBuilder,
) {
    match event {
        Event::Text(text) | Event::InlineHtml(text) => inline.push(&text, style),
        Event::Code(text) => inline.push(
            &text,
            &Style {
                code: true,
                ..style.clone()
            },
        ),
        Event::SoftBreak | Event::HardBreak => inline.push("\n", style),
        Event::FootnoteReference(label) => inline.push(&format!("[^{label}]"), style),
        Event::Start(Tag::Emphasis) => push_inlines(
            events,
            &Style {
                italic: true,
                ..style.clone()
            },
            inline,
            |end| matches!(end, TagEnd::Emphasis),
        ),
        Event::Start(Tag::Strong) => push_inlines(
            events,
            &Style {
                bold: true,
                ..style.clone()
            },
            inline,
            |end| matches!(end, TagEnd::Strong),
        ),
        Event::Start(Tag::Strikethrough) => push_inlines(
            events,
            &Style {
                strike: true,
                ..style.clone()
            },
            inline,
            |end| matches!(end, TagEnd::Strikethrough),
        ),
        Event::Start(Tag::Link { dest_url, .. }) => push_inlines(
            events,
            &Style {
                link: Some(Arc::from(dest_url.as_ref())),
                ..style.clone()
            },
            inline,
            |end| matches!(end, TagEnd::Link),
        ),
        Event::Start(Tag::Image { dest_url, .. }) => {
            // Show the alt text as a link; fall back to the URL when there is
            // no alt text, so the image is never silently dropped.
            let link_style = Style {
                link: Some(Arc::from(dest_url.as_ref())),
                ..style.clone()
            };
            let before = inline.text.len();
            push_inlines(events, &link_style, inline, |end| {
                matches!(end, TagEnd::Image)
            });
            if inline.text.len() == before {
                inline.push(&dest_url, &link_style);
            }
        }
        _ => {}
    }
}

fn is_inline(event: &Event<'_>) -> bool {
    matches!(
        event,
        Event::Text(_)
            | Event::Code(_)
            | Event::InlineHtml(_)
            | Event::SoftBreak
            | Event::HardBreak
            | Event::FootnoteReference(_)
            | Event::Start(Tag::Emphasis)
            | Event::Start(Tag::Strong)
            | Event::Start(Tag::Strikethrough)
            | Event::Start(Tag::Link { .. })
            | Event::Start(Tag::Image { .. })
    )
}

/// Accumulates one `Inline`, merging adjacent text of the same style.
#[derive(Default)]
struct InlineBuilder {
    text: String,
    spans: Vec<Span>,
}

impl InlineBuilder {
    fn push(&mut self, text: &str, style: &Style) {
        if text.is_empty() {
            return;
        }
        self.text.push_str(text);
        match self.spans.last_mut() {
            Some(last) if last.style == *style => last.len += text.len(),
            _ => self.spans.push(Span {
                len: text.len(),
                style: style.clone(),
            }),
        }
    }

    fn finish(self) -> Inline {
        Inline {
            text: self.text.into(),
            spans: self.spans,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Joins `parts` into one `Inline` with one span per part, as written.
    fn spans(parts: &[(&str, Style)]) -> Inline {
        Inline {
            text: parts
                .iter()
                .map(|(text, _)| *text)
                .collect::<String>()
                .into(),
            spans: parts
                .iter()
                .map(|(text, style)| Span {
                    len: text.len(),
                    style: style.clone(),
                })
                .collect(),
        }
    }

    fn plain(text: &str) -> Inline {
        spans(&[(text, Style::default())])
    }

    fn styled(edit: impl FnOnce(&mut Style)) -> Style {
        let mut style = Style::default();
        edit(&mut style);
        style
    }

    fn item(checked: Option<bool>, blocks: Vec<Block>) -> Item {
        Item { checked, blocks }
    }

    #[test]
    fn plain_text_is_one_paragraph() {
        assert_eq!(
            parse("hello world"),
            vec![Block::Paragraph(plain("hello world"))]
        );
    }

    #[test]
    fn soft_line_breaks_stay_as_newlines() {
        assert_eq!(
            parse("line one\nline two"),
            vec![Block::Paragraph(plain("line one\nline two"))]
        );
    }

    #[test]
    fn heading_keeps_level_and_text() {
        assert_eq!(
            parse("## Title"),
            vec![Block::Heading {
                level: 2,
                text: plain("Title"),
            }]
        );
    }

    #[test]
    fn inline_styles_split_runs() {
        let blocks = parse("a **b** *c* `d` ~~e~~ [f](https://x.test)");
        assert_eq!(
            blocks,
            vec![Block::Paragraph(spans(&[
                ("a ", Style::default()),
                ("b", styled(|s| s.bold = true)),
                (" ", Style::default()),
                ("c", styled(|s| s.italic = true)),
                (" ", Style::default()),
                ("d", styled(|s| s.code = true)),
                (" ", Style::default()),
                ("e", styled(|s| s.strike = true)),
                (" ", Style::default()),
                ("f", styled(|s| s.link = Some("https://x.test".into()))),
            ]))]
        );
    }

    #[test]
    fn tight_nested_list_items_hold_inline_text_and_sublists() {
        assert_eq!(
            parse("- one\n  - two\n- three"),
            vec![Block::List {
                start: None,
                items: vec![
                    item(
                        None,
                        vec![
                            Block::Paragraph(plain("one")),
                            Block::List {
                                start: None,
                                items: vec![item(None, vec![Block::Paragraph(plain("two"))])],
                            },
                        ],
                    ),
                    item(None, vec![Block::Paragraph(plain("three"))]),
                ],
            }]
        );
    }

    #[test]
    fn ordered_task_list_keeps_start_and_checkboxes() {
        assert_eq!(
            parse("2. [x] done\n3. [ ] todo"),
            vec![Block::List {
                start: Some(2),
                items: vec![
                    item(Some(true), vec![Block::Paragraph(plain("done"))]),
                    item(Some(false), vec![Block::Paragraph(plain("todo"))]),
                ],
            }]
        );
    }

    #[test]
    fn fenced_code_keeps_language_and_drops_final_newline() {
        assert_eq!(
            parse("```rust\nfn main() {}\n```\n"),
            vec![Block::Code {
                language: Some("rust".into()),
                text: "fn main() {}".into(),
            }]
        );
    }

    #[test]
    fn table_keeps_alignment_header_and_rows() {
        assert_eq!(
            parse("| a | b |\n|:-|-:|\n| 1 | 2 |"),
            vec![Block::Table(Table {
                align: vec![Align::Left, Align::Right],
                header: vec![plain("a"), plain("b")],
                rows: vec![vec![plain("1"), plain("2")]],
            })]
        );
    }

    #[test]
    fn blockquote_wraps_its_blocks() {
        assert_eq!(
            parse("> quoted **x**"),
            vec![Block::Quote(vec![Block::Paragraph(spans(&[
                ("quoted ", Style::default()),
                ("x", styled(|s| s.bold = true)),
            ]))])]
        );
    }
}
