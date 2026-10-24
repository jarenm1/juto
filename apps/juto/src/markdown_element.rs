//! GPUI rendering for the block tree from `markdown.rs`. Each block becomes a
//! flex child. Inline runs become one `StyledText` per block so a paragraph
//! wraps as a unit.
//!
//! GPUI resolves run font size and line height from the inherited text style
//! at layout time, while run colors and fonts come from the `TextRun`s. So
//! every run sets its color explicitly, and block sizes come from wrapping
//! `div`s.
//!
//! This runs every frame for every assistant message in the transcript, so
//! it only builds elements: text is shared with the parse, never copied or
//! re-parsed.

use std::sync::Arc;

use gpui::{
    AnyElement, FontStyle, FontWeight, HighlightStyle, InteractiveText, IntoElement, ParentElement,
    SharedString, StrikethroughStyle, Styled, StyledText, TextStyle, UnderlineStyle, Window, div,
    px, rgb,
};

use crate::markdown::{Align, Block, Inline, Item, Table};
use crate::theme;

/// Monospace face for code. Named explicitly: generic family names such as
/// `monospace` do not resolve in GPUI's Linux text system.
const MONO: &str = "DejaVu Sans Mono";

/// Renders `blocks` as a column. Link hit-testing ids are numbered within this
/// call, so the caller must place the result inside an element whose id is
/// unique in the window.
pub fn render(blocks: &[Block], window: &Window) -> AnyElement {
    let mut base = window.text_style();
    base.color = rgb(theme::TEXT).into();
    let mut links = 0;
    let mut column = div().flex().flex_col().gap_3().w_full().min_w_0();
    for block in blocks {
        column = column.child(block_element(block, &base, &mut links));
    }
    column.into_any_element()
}

fn block_element(block: &Block, base: &TextStyle, links: &mut usize) -> AnyElement {
    match block {
        Block::Paragraph(text) => div()
            .w_full()
            .min_w_0()
            .child(inline_element(text, base, links))
            .into_any_element(),
        Block::Heading { level, text } => {
            let mut bold = base.clone();
            bold.font_weight = FontWeight::BOLD;
            div()
                .w_full()
                .min_w_0()
                .text_size(px(heading_size(*level)))
                .child(inline_element(text, &bold, links))
                .into_any_element()
        }
        Block::Code { language, text } => code_block(language.as_ref(), text, base),
        Block::List { start, items } => list(*start, items, base, links),
        Block::Quote(blocks) => {
            let mut muted = base.clone();
            muted.color = rgb(theme::TEXT_MUTED).into();
            let mut body = div().flex_1().min_w_0().flex().flex_col().gap_2();
            for block in blocks {
                body = body.child(block_element(block, &muted, links));
            }
            div()
                .flex()
                .gap_3()
                .w_full()
                .min_w_0()
                .child(
                    div()
                        .flex_none()
                        .w(px(3.))
                        .rounded_sm()
                        .bg(rgb(theme::BORDER_STRONG)),
                )
                .child(body)
                .into_any_element()
        }
        Block::Table(table) => table_element(table, base, links),
        Block::Rule => div()
            .w_full()
            .h(px(1.))
            .bg(rgb(theme::BORDER))
            .into_any_element(),
    }
}

fn heading_size(level: u8) -> f32 {
    match level {
        1 => 24.,
        2 => 20.,
        3 => 18.,
        4 => 16.,
        _ => 15.,
    }
}

fn code_block(language: Option<&Arc<str>>, text: &Arc<str>, base: &TextStyle) -> AnyElement {
    let mut body = div()
        .w_full()
        .min_w_0()
        .flex()
        .flex_col()
        .gap_2()
        .p_3()
        .rounded_md()
        .border_1()
        .border_color(rgb(theme::BORDER))
        .bg(rgb(theme::BG_RAISED))
        .text_size(px(13.));
    if let Some(language) = language {
        body = body.child(
            div()
                .text_xs()
                .text_color(rgb(theme::TEXT_FAINT))
                .child(SharedString::from(language.clone())),
        );
    }
    if !text.is_empty() {
        let mut mono = base.clone();
        mono.font_family = MONO.into();
        let styled = StyledText::new(SharedString::from(text.clone()))
            .with_runs(vec![mono.to_run(text.len())]);
        body = body.child(styled);
    }
    body.into_any_element()
}

fn list(start: Option<u64>, items: &[Item], base: &TextStyle, links: &mut usize) -> AnyElement {
    let mut column = div().flex().flex_col().gap_1().w_full().min_w_0();
    for (index, item) in items.iter().enumerate() {
        let marker: SharedString = match (item.checked, start) {
            (Some(true), _) => "☑".into(),
            (Some(false), _) => "☐".into(),
            (None, Some(first)) => format!("{}.", first + index as u64).into(),
            (None, None) => "•".into(),
        };
        let mut body = div().flex_1().min_w_0().flex().flex_col().gap_2();
        for block in &item.blocks {
            body = body.child(block_element(block, base, links));
        }
        column = column.child(
            div()
                .flex()
                .gap_2()
                .w_full()
                .min_w_0()
                .child(div().flex_none().min_w(px(20.)).child(marker))
                .child(body),
        );
    }
    column.into_any_element()
}

fn table_element(table: &Table, base: &TextStyle, links: &mut usize) -> AnyElement {
    let mut bold = base.clone();
    bold.font_weight = FontWeight::BOLD;
    let mut column = div()
        .flex()
        .flex_col()
        .w_full()
        .min_w_0()
        .rounded_md()
        .border_1()
        .border_color(rgb(theme::BORDER))
        .overflow_hidden();
    column = column.child(table_row(&table.header, &table.align, &bold, false, links));
    for row in &table.rows {
        column = column.child(table_row(row, &table.align, base, true, links));
    }
    column.into_any_element()
}

fn table_row(
    cells: &[Inline],
    aligns: &[Align],
    base: &TextStyle,
    top_border: bool,
    links: &mut usize,
) -> AnyElement {
    let mut row = div().flex().w_full().min_w_0();
    if top_border {
        row = row.border_t_1().border_color(rgb(theme::BORDER));
    } else {
        row = row.bg(rgb(theme::BG_RAISED));
    }
    for (index, cell) in cells.iter().enumerate() {
        let align = aligns.get(index).copied().unwrap_or(Align::None);
        let mut cell_el = div().flex_1().min_w_0().px_3().py_1p5();
        if index > 0 {
            cell_el = cell_el.border_l_1().border_color(rgb(theme::BORDER));
        }
        // Block cell, not a flex row: a flex child sizes to its max-content width
        // and never wraps. Alignment is inherited text style, which StyledText paints with.
        cell_el = match align {
            Align::Center => cell_el.text_center(),
            Align::Right => cell_el.text_right(),
            Align::Left | Align::None => cell_el,
        };
        row = row.child(cell_el.child(inline_element(cell, base, links)));
    }
    row.into_any_element()
}

/// One `StyledText` for `inline`, wrapped for click handling when it holds an
/// openable link. The text is shared with the parse, not copied.
fn inline_element(inline: &Inline, base: &TextStyle, links: &mut usize) -> AnyElement {
    let mut runs = Vec::with_capacity(inline.spans.len());
    let mut ranges = Vec::new();
    let mut urls = Vec::new();
    let mut start = 0;
    for span in &inline.spans {
        let end = start + span.len;
        let mut highlight = HighlightStyle::default();
        if span.style.bold {
            highlight.font_weight = Some(FontWeight::BOLD);
        }
        if span.style.italic {
            highlight.font_style = Some(FontStyle::Italic);
        }
        if span.style.strike {
            highlight.strikethrough = Some(StrikethroughStyle {
                thickness: px(1.),
                color: None,
            });
        }
        if span.style.code {
            highlight.background_color = Some(rgb(theme::BG_RAISED).into());
        }
        if let Some(url) = span.style.link.as_ref().filter(|url| openable(url)) {
            highlight.color = Some(rgb(theme::ACCENT).into());
            highlight.underline = Some(UnderlineStyle {
                thickness: px(1.),
                color: Some(rgb(theme::ACCENT).into()),
                wavy: false,
            });
            ranges.push(start..end);
            urls.push(url.clone());
        }
        let run = if span.style.code {
            let mut mono = base.clone();
            mono.font_family = MONO.into();
            mono.highlight(highlight).to_run(span.len)
        } else if highlight == HighlightStyle::default() {
            base.to_run(span.len)
        } else {
            base.clone().highlight(highlight).to_run(span.len)
        };
        runs.push(run);
        start = end;
    }
    let styled = StyledText::new(SharedString::from(inline.text.clone())).with_runs(runs);
    if ranges.is_empty() {
        return styled.into_any_element();
    }
    let id = *links;
    *links += 1;
    InteractiveText::new(("markdown-link", id), styled)
        .on_click(ranges, move |index, _window, cx| {
            cx.open_url(&urls[index]);
        })
        .into_any_element()
}

/// Only web and mail links are made clickable. Model output must not launch
/// local files or custom URI handlers.
fn openable(url: &str) -> bool {
    url.starts_with("https://") || url.starts_with("http://") || url.starts_with("mailto:")
}
