// Adapted from GPUI 0.2.2 examples/input.rs:
// https://docs.rs/crate/gpui/0.2.2/source/examples/input.rs
// Copyright 2022 - 2025 Zed Industries, Inc.
// Licensed under Apache-2.0; see licenses/GPUI-APACHE-2.0.txt.
// Changes: local submission, dark styling, wrapped multi-line input with capped vertical scrolling, and IME range fixes.

use std::ops::Range;

use gpui::{
    App, AvailableSpace, Bounds, ClipboardItem, ContentMask, Context, CursorStyle, ElementId,
    ElementInputHandler, Entity, EntityInputHandler, EventEmitter, FocusHandle, Focusable,
    GlobalElementId, KeyBinding, LayoutId, MouseButton, MouseDownEvent, MouseMoveEvent,
    MouseUpEvent, PaintQuad, Pixels, Point, SharedString, Style, TextAlign, TextRun,
    UTF16Selection, UnderlineStyle, Window, WrappedLine, actions, div, fill, point, prelude::*, px,
    relative, rgb, rgba, size,
};
use unicode_segmentation::UnicodeSegmentation;

/// Line height of every composer row, including wrapped rows.
const LINE_HEIGHT_PX: f32 = 24.;
/// Rows shown before the composer stops growing and scrolls instead.
const MAX_VISIBLE_LINES: usize = 8;

actions!(
    chat_input,
    [
        Backspace,
        Delete,
        Left,
        Right,
        SelectLeft,
        SelectRight,
        SelectAll,
        Home,
        End,
        Paste,
        Copy,
        Cut,
        Submit
    ]
);

pub struct SubmitMessage;

pub struct ChatInput {
    focus_handle: FocusHandle,
    content: SharedString,
    selected_range: Range<usize>,
    selection_reversed: bool,
    marked_range: Option<Range<usize>>,
    last_layout: Option<WrappedLine>,
    last_bounds: Option<Bounds<Pixels>>,
    scroll_y: Pixels,
    is_selecting: bool,
}

impl EventEmitter<SubmitMessage> for ChatInput {}

pub fn init(cx: &mut App) {
    cx.bind_keys([
        KeyBinding::new("backspace", Backspace, Some("ChatInput")),
        KeyBinding::new("delete", Delete, Some("ChatInput")),
        KeyBinding::new("left", Left, Some("ChatInput")),
        KeyBinding::new("right", Right, Some("ChatInput")),
        KeyBinding::new("shift-left", SelectLeft, Some("ChatInput")),
        KeyBinding::new("shift-right", SelectRight, Some("ChatInput")),
        KeyBinding::new("ctrl-a", SelectAll, Some("ChatInput")),
        KeyBinding::new("home", Home, Some("ChatInput")),
        KeyBinding::new("end", End, Some("ChatInput")),
        KeyBinding::new("ctrl-v", Paste, Some("ChatInput")),
        KeyBinding::new("ctrl-c", Copy, Some("ChatInput")),
        KeyBinding::new("ctrl-x", Cut, Some("ChatInput")),
        KeyBinding::new("enter", Submit, Some("ChatInput")),
    ]);
}

impl ChatInput {
    pub fn new(cx: &mut Context<Self>) -> Self {
        Self {
            focus_handle: cx.focus_handle(),
            content: "".into(),
            selected_range: 0..0,
            selection_reversed: false,
            marked_range: None,
            last_layout: None,
            last_bounds: None,
            scroll_y: px(0.),
            is_selecting: false,
        }
    }

    pub fn has_text(&self) -> bool {
        !self.content.trim().is_empty()
    }

    pub fn is_composing(&self) -> bool {
        self.marked_range.is_some()
    }

    pub fn take_text(&mut self, cx: &mut Context<Self>) -> SharedString {
        let text = std::mem::replace(&mut self.content, "".into());
        self.selected_range = 0..0;
        self.selection_reversed = false;
        self.marked_range = None;
        self.scroll_y = px(0.);
        self.is_selecting = false;
        cx.notify();
        text
    }

    /// Replaces the composer text and moves the caret to its end.
    pub fn set_text(&mut self, text: &str, cx: &mut Context<Self>) {
        // Line breaks become spaces, as typed text does.
        let text = text.replace(['\r', '\n'], " ");
        self.selected_range = text.len()..text.len();
        self.selection_reversed = false;
        self.marked_range = None;
        self.content = text.into();
        cx.notify();
    }

    fn submit(&mut self, _: &Submit, _: &mut Window, cx: &mut Context<Self>) {
        if !self.is_composing() {
            cx.emit(SubmitMessage);
        }
    }

    fn cursor_offset(&self) -> usize {
        if self.selection_reversed {
            self.selected_range.start
        } else {
            self.selected_range.end
        }
    }

    fn previous_boundary(&self, offset: usize) -> usize {
        self.content
            .grapheme_indices(true)
            .rev()
            .find_map(|(index, _)| (index < offset).then_some(index))
            .unwrap_or(0)
    }

    fn next_boundary(&self, offset: usize) -> usize {
        self.content
            .grapheme_indices(true)
            .find_map(|(index, _)| (index > offset).then_some(index))
            .unwrap_or(self.content.len())
    }

    fn move_to(&mut self, offset: usize, cx: &mut Context<Self>) {
        self.selected_range = offset..offset;
        self.selection_reversed = false;
        cx.notify();
    }

    fn select_to(&mut self, offset: usize, cx: &mut Context<Self>) {
        if self.selection_reversed {
            self.selected_range.start = offset;
        } else {
            self.selected_range.end = offset;
        }
        if self.selected_range.end < self.selected_range.start {
            self.selection_reversed = !self.selection_reversed;
            self.selected_range = self.selected_range.end..self.selected_range.start;
        }
        cx.notify();
    }

    fn left(&mut self, _: &Left, _: &mut Window, cx: &mut Context<Self>) {
        let offset = if self.selected_range.is_empty() {
            self.previous_boundary(self.cursor_offset())
        } else {
            self.selected_range.start
        };
        self.move_to(offset, cx);
    }

    fn right(&mut self, _: &Right, _: &mut Window, cx: &mut Context<Self>) {
        let offset = if self.selected_range.is_empty() {
            self.next_boundary(self.cursor_offset())
        } else {
            self.selected_range.end
        };
        self.move_to(offset, cx);
    }

    fn select_left(&mut self, _: &SelectLeft, _: &mut Window, cx: &mut Context<Self>) {
        self.select_to(self.previous_boundary(self.cursor_offset()), cx);
    }

    fn select_right(&mut self, _: &SelectRight, _: &mut Window, cx: &mut Context<Self>) {
        self.select_to(self.next_boundary(self.cursor_offset()), cx);
    }

    fn select_all(&mut self, _: &SelectAll, _: &mut Window, cx: &mut Context<Self>) {
        self.move_to(0, cx);
        self.select_to(self.content.len(), cx);
    }

    fn home(&mut self, _: &Home, _: &mut Window, cx: &mut Context<Self>) {
        self.move_to(0, cx);
    }
    fn end(&mut self, _: &End, _: &mut Window, cx: &mut Context<Self>) {
        self.move_to(self.content.len(), cx);
    }

    fn backspace(&mut self, _: &Backspace, window: &mut Window, cx: &mut Context<Self>) {
        if self.selected_range.is_empty() {
            self.select_to(self.previous_boundary(self.cursor_offset()), cx);
        }
        self.replace_text_in_range(None, "", window, cx);
    }

    fn delete(&mut self, _: &Delete, window: &mut Window, cx: &mut Context<Self>) {
        if self.selected_range.is_empty() {
            self.select_to(self.next_boundary(self.cursor_offset()), cx);
        }
        self.replace_text_in_range(None, "", window, cx);
    }

    fn paste(&mut self, _: &Paste, window: &mut Window, cx: &mut Context<Self>) {
        if let Some(text) = cx.read_from_clipboard().and_then(|item| item.text()) {
            self.replace_text_in_range(None, &text, window, cx);
        }
    }

    fn copy(&mut self, _: &Copy, _: &mut Window, cx: &mut Context<Self>) {
        if !self.selected_range.is_empty() {
            cx.write_to_clipboard(ClipboardItem::new_string(
                self.content[self.selected_range.clone()].to_string(),
            ));
        }
    }

    fn cut(&mut self, _: &Cut, window: &mut Window, cx: &mut Context<Self>) {
        self.copy(&Copy, window, cx);
        if !self.selected_range.is_empty() {
            self.replace_text_in_range(None, "", window, cx);
        }
    }

    fn index_for_mouse_position(&self, position: Point<Pixels>) -> usize {
        let (Some(bounds), Some(line)) = (self.last_bounds, self.last_layout.as_ref()) else {
            return 0;
        };
        if self.content.is_empty() || position.y < bounds.top() {
            return 0;
        }
        if position.y > bounds.bottom() {
            return self.content.len();
        }
        let local = point(
            position.x - bounds.left(),
            position.y - bounds.top() + self.scroll_y,
        );
        match line.closest_index_for_position(local, px(LINE_HEIGHT_PX)) {
            Ok(index) | Err(index) => index,
        }
    }

    fn on_mouse_down(
        &mut self,
        event: &MouseDownEvent,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.focus_handle.focus(window);
        self.is_selecting = true;
        let index = self.index_for_mouse_position(event.position);
        if event.modifiers.shift {
            self.select_to(index, cx);
        } else {
            self.move_to(index, cx);
        }
    }

    fn on_mouse_up(&mut self, _: &MouseUpEvent, _: &mut Window, _: &mut Context<Self>) {
        self.is_selecting = false;
    }
    fn on_mouse_move(&mut self, event: &MouseMoveEvent, _: &mut Window, cx: &mut Context<Self>) {
        if self.is_selecting {
            self.select_to(self.index_for_mouse_position(event.position), cx);
        }
    }

    fn range_from_utf16(&self, range: &Range<usize>) -> Range<usize> {
        offset_from_utf16(&self.content, range.start)..offset_from_utf16(&self.content, range.end)
    }

    fn range_to_utf16(&self, range: &Range<usize>) -> Range<usize> {
        offset_to_utf16(&self.content, range.start)..offset_to_utf16(&self.content, range.end)
    }

    fn replace_range(&mut self, range: Range<usize>, new_text: &str) {
        // The composer is one paragraph, so line breaks become spaces. Each break maps to one
        // space of equal byte and UTF-16 length, so the caller's offsets stay valid.
        let new_text = new_text.replace(['\r', '\n'], " ");
        let mut content = String::with_capacity(self.content.len() - range.len() + new_text.len());
        content.push_str(&self.content[..range.start]);
        content.push_str(&new_text);
        content.push_str(&self.content[range.end..]);
        self.content = content.into();
        self.selection_reversed = false;
    }
}

fn offset_from_utf16(text: &str, offset: usize) -> usize {
    let mut utf8 = 0;
    let mut utf16 = 0;
    for character in text.chars() {
        if utf16 >= offset {
            break;
        }
        utf16 += character.len_utf16();
        utf8 += character.len_utf8();
    }
    utf8
}

fn offset_to_utf16(text: &str, offset: usize) -> usize {
    text.char_indices()
        .take_while(|(index, _)| *index < offset)
        .map(|(_, character)| character.len_utf16())
        .sum()
}

fn selection_after_replacement(
    start: usize,
    text: &str,
    selected: Option<Range<usize>>,
) -> Range<usize> {
    selected
        .map(|range| {
            start + offset_from_utf16(text, range.start)..start + offset_from_utf16(text, range.end)
        })
        .unwrap_or(start + text.len()..start + text.len())
}

/// Width available to wrapped text; the caret is drawn in the 2px to its right.
fn text_wrap_width(width: Pixels) -> Pixels {
    (width - px(2.)).max(px(0.))
}

/// Number of visual rows the wrapped composer text occupies.
fn visual_row_count(line: &WrappedLine) -> usize {
    line.wrap_boundaries().len() + 1
}

/// Composer height for `rows` visual rows, capped at `MAX_VISIBLE_LINES`.
fn composer_height(rows: usize) -> Pixels {
    px(LINE_HEIGHT_PX * rows.min(MAX_VISIBLE_LINES) as f32)
}

/// Shapes the composer text, or its placeholder, as one paragraph wrapped at `wrap_width`.
fn shape_composer_text(input: &ChatInput, window: &mut Window, wrap_width: Pixels) -> WrappedLine {
    let style = window.text_style();
    let display_text: SharedString = if input.content.is_empty() {
        "Message Juto…".into()
    } else {
        input.content.clone()
    };
    let base = TextRun {
        len: display_text.len(),
        font: style.font(),
        color: if input.content.is_empty() {
            rgb(0x909090).into()
        } else {
            style.color
        },
        background_color: None,
        underline: None,
        strikethrough: None,
    };
    let runs = match &input.marked_range {
        Some(marked) => vec![
            TextRun {
                len: marked.start,
                ..base.clone()
            },
            TextRun {
                len: marked.len(),
                underline: Some(UnderlineStyle {
                    color: Some(style.color),
                    thickness: px(1.),
                    wavy: false,
                }),
                ..base.clone()
            },
            TextRun {
                len: display_text.len() - marked.end,
                ..base
            },
        ],
        None => vec![base],
    };
    let font_size = style.font_size.to_pixels(window.rem_size());
    window
        .text_system()
        .shape_text(display_text, font_size, &runs, Some(wrap_width), None)
        .expect("composer text shaping failed")
        .into_iter()
        .next()
        .expect("single-paragraph composer text yields one wrapped line")
}

/// Scroll offset that keeps the caret row visible, clamped to the scrollable range.
fn scroll_y_for_caret(
    current: Pixels,
    caret_top: Pixels,
    caret_bottom: Pixels,
    viewport: Pixels,
    content: Pixels,
) -> Pixels {
    let max_scroll = (content - viewport).max(px(0.));
    let mut scroll = current.min(max_scroll);
    if caret_top < scroll {
        scroll = caret_top;
    }
    if caret_bottom > scroll + viewport {
        scroll = caret_bottom - viewport;
    }
    scroll.max(px(0.)).min(max_scroll)
}

/// One rectangle per visual row between `start` and `end`, both local to the wrapped text.
fn selection_row_bounds(
    origin: Point<Pixels>,
    start: Point<Pixels>,
    end: Point<Pixels>,
    wrap_width: Pixels,
    row_height: Pixels,
) -> Vec<Bounds<Pixels>> {
    let first_row = (start.y / row_height) as usize;
    let last_row = (end.y / row_height) as usize;
    (first_row..=last_row)
        .map(|row| {
            let left = if row == first_row { start.x } else { px(0.) };
            let right = if row == last_row { end.x } else { wrap_width };
            let y = origin.y + row_height * row as f32;
            Bounds::from_corners(
                point(origin.x + left, y),
                point(origin.x + right, y + row_height),
            )
        })
        .collect()
}

impl EntityInputHandler for ChatInput {
    fn text_for_range(
        &mut self,
        range_utf16: Range<usize>,
        actual_range: &mut Option<Range<usize>>,
        _: &mut Window,
        _: &mut Context<Self>,
    ) -> Option<String> {
        let range = self.range_from_utf16(&range_utf16);
        actual_range.replace(self.range_to_utf16(&range));
        Some(self.content[range].to_string())
    }

    fn selected_text_range(
        &mut self,
        _: bool,
        _: &mut Window,
        _: &mut Context<Self>,
    ) -> Option<UTF16Selection> {
        Some(UTF16Selection {
            range: self.range_to_utf16(&self.selected_range),
            reversed: self.selection_reversed,
        })
    }

    fn marked_text_range(&self, _: &mut Window, _: &mut Context<Self>) -> Option<Range<usize>> {
        self.marked_range
            .as_ref()
            .map(|range| self.range_to_utf16(range))
    }

    fn unmark_text(&mut self, _: &mut Window, cx: &mut Context<Self>) {
        self.marked_range = None;
        cx.notify();
    }

    fn replace_text_in_range(
        &mut self,
        range_utf16: Option<Range<usize>>,
        new_text: &str,
        _: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let range = range_utf16
            .as_ref()
            .map(|range| self.range_from_utf16(range))
            .or(self.marked_range.clone())
            .unwrap_or(self.selected_range.clone());
        let cursor = range.start + new_text.len();
        self.replace_range(range, new_text);
        self.selected_range = cursor..cursor;
        self.marked_range = None;
        cx.notify();
    }

    fn replace_and_mark_text_in_range(
        &mut self,
        range_utf16: Option<Range<usize>>,
        new_text: &str,
        new_selected_range_utf16: Option<Range<usize>>,
        _: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let range = range_utf16
            .as_ref()
            .map(|range| self.range_from_utf16(range))
            .or(self.marked_range.clone())
            .unwrap_or(self.selected_range.clone());
        let start = range.start;
        self.replace_range(range, new_text);
        self.marked_range = (!new_text.is_empty()).then_some(start..start + new_text.len());
        self.selected_range =
            selection_after_replacement(start, new_text, new_selected_range_utf16);
        cx.notify();
    }

    fn bounds_for_range(
        &mut self,
        range_utf16: Range<usize>,
        _: Bounds<Pixels>,
        _: &mut Window,
        _: &mut Context<Self>,
    ) -> Option<Bounds<Pixels>> {
        let bounds = self.last_bounds?;
        let line = self.last_layout.as_ref()?;
        let range = self.range_from_utf16(&range_utf16);
        let row_height = px(LINE_HEIGHT_PX);
        let start = line.position_for_index(range.start, row_height)?;
        let end = line.position_for_index(range.end, row_height)?;
        let top = bounds.top() + start.y - self.scroll_y;
        // Anchor on the start row; extend to the end x only when the range stays on that row.
        let right = if end.y == start.y { end.x } else { start.x };
        Some(Bounds::from_corners(
            point(bounds.left() + start.x, top),
            point(bounds.left() + right, top + row_height),
        ))
    }

    fn character_index_for_point(
        &mut self,
        position: Point<Pixels>,
        _: &mut Window,
        _: &mut Context<Self>,
    ) -> Option<usize> {
        self.last_bounds?.localize(&position)?;
        Some(offset_to_utf16(
            &self.content,
            self.index_for_mouse_position(position),
        ))
    }
}

struct TextElement {
    input: Entity<ChatInput>,
}
struct PrepaintState {
    line: Option<WrappedLine>,
    cursor: Option<PaintQuad>,
    selections: Vec<PaintQuad>,
    scroll_y: Pixels,
}

impl IntoElement for TextElement {
    type Element = Self;
    fn into_element(self) -> Self {
        self
    }
}

impl Element for TextElement {
    type RequestLayoutState = ();
    type PrepaintState = PrepaintState;

    fn id(&self) -> Option<ElementId> {
        None
    }
    fn source_location(&self) -> Option<&'static core::panic::Location<'static>> {
        None
    }

    fn request_layout(
        &mut self,
        _: Option<&GlobalElementId>,
        _: Option<&gpui::InspectorElementId>,
        window: &mut Window,
        _cx: &mut App,
    ) -> (LayoutId, ()) {
        let mut style = Style::default();
        style.size.width = relative(1.).into();
        let input = self.input.clone();
        let layout_id =
            window.request_measured_layout(style, move |known, available, window, cx| {
                let width = known.width.or(match available.width {
                    AvailableSpace::Definite(width) => Some(width),
                    _ => None,
                });
                let rows = match width {
                    Some(width) => {
                        let line =
                            shape_composer_text(input.read(cx), window, text_wrap_width(width));
                        visual_row_count(&line)
                    }
                    // Intrinsic-size queries have no definite width: report one row, and the
                    // definite-width pass wraps the text.
                    None => 1,
                };
                size(width.unwrap_or(px(0.)), composer_height(rows))
            });
        (layout_id, ())
    }

    fn prepaint(
        &mut self,
        _: Option<&GlobalElementId>,
        _: Option<&gpui::InspectorElementId>,
        bounds: Bounds<Pixels>,
        _: &mut (),
        window: &mut Window,
        cx: &mut App,
    ) -> PrepaintState {
        let input = self.input.read(cx);
        let row_height = px(LINE_HEIGHT_PX);
        let wrap_width = text_wrap_width(bounds.size.width);
        let line = shape_composer_text(input, window, wrap_width);
        let content_height = row_height * visual_row_count(&line) as f32;
        let caret = line
            .position_for_index(input.cursor_offset(), row_height)
            .expect("caret offset lies within composer text");
        let scroll_y = scroll_y_for_caret(
            input.scroll_y,
            caret.y,
            caret.y + row_height,
            bounds.size.height,
            content_height,
        );
        let origin = point(bounds.left(), bounds.top() - scroll_y);
        let (selections, cursor): (Vec<PaintQuad>, Option<PaintQuad>) =
            if input.selected_range.is_empty() {
                let cursor = fill(
                    Bounds::new(
                        point(origin.x + caret.x, origin.y + caret.y),
                        size(px(2.), row_height),
                    ),
                    rgb(0xd4d4d4),
                );
                (Vec::new(), Some(cursor))
            } else {
                let start = line
                    .position_for_index(input.selected_range.start, row_height)
                    .expect("selection start lies within composer text");
                let end = line
                    .position_for_index(input.selected_range.end, row_height)
                    .expect("selection end lies within composer text");
                let selections = selection_row_bounds(origin, start, end, wrap_width, row_height)
                    .into_iter()
                    .map(|rect| fill(rect, rgba(0xffffff30)))
                    .collect();
                (selections, None)
            };
        PrepaintState {
            line: Some(line),
            cursor,
            selections,
            scroll_y,
        }
    }

    fn paint(
        &mut self,
        _: Option<&GlobalElementId>,
        _: Option<&gpui::InspectorElementId>,
        bounds: Bounds<Pixels>,
        _: &mut (),
        prepaint: &mut PrepaintState,
        window: &mut Window,
        cx: &mut App,
    ) {
        let focus_handle = self.input.read(cx).focus_handle.clone();
        window.handle_input(
            &focus_handle,
            ElementInputHandler::new(bounds, self.input.clone()),
            cx,
        );
        let line = prepaint.line.take().unwrap();
        let origin = point(bounds.left(), bounds.top() - prepaint.scroll_y);
        window.with_content_mask(Some(ContentMask { bounds }), |window| {
            for selection in prepaint.selections.drain(..) {
                window.paint_quad(selection);
            }
            line.paint(
                origin,
                px(LINE_HEIGHT_PX),
                TextAlign::Left,
                Some(bounds),
                window,
                cx,
            )
            .unwrap();
            if focus_handle.is_focused(window)
                && let Some(cursor) = prepaint.cursor.take()
            {
                window.paint_quad(cursor);
            }
        });
        self.input.update(cx, |input, _| {
            input.scroll_y = prepaint.scroll_y;
            input.last_layout = Some(line);
            input.last_bounds = Some(bounds);
        });
    }
}

impl Render for ChatInput {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        div()
            .id("chat-input")
            .flex()
            .w_full()
            .min_w_0()
            .key_context("ChatInput")
            .track_focus(&self.focus_handle)
            .cursor(CursorStyle::IBeam)
            .px_2()
            .py_1()
            .line_height(px(LINE_HEIGHT_PX))
            .text_size(px(15.))
            .on_action(cx.listener(Self::backspace))
            .on_action(cx.listener(Self::delete))
            .on_action(cx.listener(Self::left))
            .on_action(cx.listener(Self::right))
            .on_action(cx.listener(Self::select_left))
            .on_action(cx.listener(Self::select_right))
            .on_action(cx.listener(Self::select_all))
            .on_action(cx.listener(Self::home))
            .on_action(cx.listener(Self::end))
            .on_action(cx.listener(Self::paste))
            .on_action(cx.listener(Self::copy))
            .on_action(cx.listener(Self::cut))
            .on_action(cx.listener(Self::submit))
            .on_mouse_down(MouseButton::Left, cx.listener(Self::on_mouse_down))
            .on_mouse_up(MouseButton::Left, cx.listener(Self::on_mouse_up))
            .on_mouse_up_out(MouseButton::Left, cx.listener(Self::on_mouse_up))
            .on_mouse_move(cx.listener(Self::on_mouse_move))
            .child(TextElement { input: cx.entity() })
    }
}

impl Focusable for ChatInput {
    fn focus_handle(&self, _: &App) -> FocusHandle {
        self.focus_handle.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn platform_offsets_preserve_non_ascii_text_boundaries() {
        let text = "a🙂e\u{301}界";
        assert_eq!(offset_from_utf16(text, 3), 5);
        assert_eq!(offset_from_utf16(text, 5), 8);
        assert_eq!(offset_to_utf16(text, 8), 5);
        assert_eq!(offset_from_utf16(text, 100), text.len());
    }

    #[test]
    fn ime_selection_is_relative_to_inserted_text_not_existing_content() {
        assert_eq!(selection_after_replacement(7, "🙂界", Some(2..3)), 11..14);
        assert_eq!(selection_after_replacement(7, "界", None), 10..10);
    }

    #[test]
    fn caret_scrolls_only_as_far_as_needed() {
        let viewport = px(192.);
        let content = px(240.);
        // Caret on the last row (y 216..240) of 10 rows, in an 8-row viewport.
        assert_eq!(
            scroll_y_for_caret(px(0.), px(216.), px(240.), viewport, content),
            px(48.)
        );
        // Caret moved above the viewport: scroll follows it up.
        assert_eq!(
            scroll_y_for_caret(px(48.), px(24.), px(48.), viewport, content),
            px(24.)
        );
        // Caret already visible: no movement.
        assert_eq!(
            scroll_y_for_caret(px(48.), px(100.), px(124.), viewport, content),
            px(48.)
        );
        // Content shrank below the old offset: clamp to the new maximum.
        assert_eq!(
            scroll_y_for_caret(px(48.), px(0.), px(24.), viewport, px(100.)),
            px(0.)
        );
    }

    #[test]
    fn selection_spans_each_visual_row() {
        let rects = selection_row_bounds(
            point(px(10.), px(20.)),
            point(px(30.), px(0.)),
            point(px(15.), px(48.)),
            px(100.),
            px(24.),
        );
        assert_eq!(
            rects,
            vec![
                Bounds::from_corners(point(px(40.), px(20.)), point(px(110.), px(44.))),
                Bounds::from_corners(point(px(10.), px(44.)), point(px(110.), px(68.))),
                Bounds::from_corners(point(px(10.), px(68.)), point(px(25.), px(92.))),
            ]
        );
    }
}
