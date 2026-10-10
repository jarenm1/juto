// Single-line text input component for GPUI.
// Adapted from GPUI 0.2.2 examples/input.rs (Apache-2.0, © Zed Industries, Inc.)

use std::ops::Range;

use gpui::{
    App, Bounds, ClipboardItem, ContentMask, Context, CursorStyle, ElementId, ElementInputHandler,
    Entity, EntityInputHandler, EventEmitter, FocusHandle, Focusable, GlobalElementId, KeyBinding,
    LayoutId, MouseButton, MouseDownEvent, MouseMoveEvent, MouseUpEvent, PaintQuad, Pixels, Point,
    ShapedLine, SharedString, Style, TextRun, UTF16Selection, UnderlineStyle, Window, actions, div,
    fill, point, prelude::*, px, relative, rgb, rgba, size,
};
use unicode_segmentation::UnicodeSegmentation;

actions!(
    text_input,
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

#[derive(Clone, Debug)]
#[allow(dead_code)]
pub struct SubmitText(pub SharedString);

pub struct TextInput {
    focus_handle: FocusHandle,
    content: SharedString,
    placeholder: SharedString,
    masked: bool,
    selected_range: Range<usize>,
    selection_reversed: bool,
    marked_range: Option<Range<usize>>,
    last_layout: Option<ShapedLine>,
    last_bounds: Option<Bounds<Pixels>>,
    scroll_x: Pixels,
    is_selecting: bool,
}

impl EventEmitter<SubmitText> for TextInput {}

pub fn init(cx: &mut App) {
    cx.bind_keys([
        KeyBinding::new("backspace", Backspace, Some("TextInput")),
        KeyBinding::new("delete", Delete, Some("TextInput")),
        KeyBinding::new("left", Left, Some("TextInput")),
        KeyBinding::new("right", Right, Some("TextInput")),
        KeyBinding::new("shift-left", SelectLeft, Some("TextInput")),
        KeyBinding::new("shift-right", SelectRight, Some("TextInput")),
        KeyBinding::new("ctrl-a", SelectAll, Some("TextInput")),
        KeyBinding::new("cmd-a", SelectAll, Some("TextInput")),
        KeyBinding::new("home", Home, Some("TextInput")),
        KeyBinding::new("end", End, Some("TextInput")),
        KeyBinding::new("ctrl-v", Paste, Some("TextInput")),
        KeyBinding::new("cmd-v", Paste, Some("TextInput")),
        KeyBinding::new("ctrl-c", Copy, Some("TextInput")),
        KeyBinding::new("cmd-c", Copy, Some("TextInput")),
        KeyBinding::new("ctrl-x", Cut, Some("TextInput")),
        KeyBinding::new("cmd-x", Cut, Some("TextInput")),
        KeyBinding::new("enter", Submit, Some("TextInput")),
    ]);
}

impl TextInput {
    pub fn new(cx: &mut Context<Self>) -> Self {
        Self {
            focus_handle: cx.focus_handle(),
            content: "".into(),
            placeholder: "".into(),
            masked: false,
            selected_range: 0..0,
            selection_reversed: false,
            marked_range: None,
            last_layout: None,
            last_bounds: None,
            scroll_x: px(0.),
            is_selecting: false,
        }
    }

    pub fn with_placeholder(mut self, placeholder: impl Into<SharedString>) -> Self {
        self.placeholder = placeholder.into();
        self
    }

    pub fn with_masked(mut self, masked: bool) -> Self {
        self.masked = masked;
        self
    }

    #[allow(dead_code)]
    pub fn set_placeholder(
        &mut self,
        placeholder: impl Into<SharedString>,
        cx: &mut Context<Self>,
    ) {
        self.placeholder = placeholder.into();
        cx.notify();
    }

    pub fn set_masked(&mut self, masked: bool, cx: &mut Context<Self>) {
        self.masked = masked;
        self.last_layout = None;
        cx.notify();
    }

    pub fn is_masked(&self) -> bool {
        self.masked
    }

    pub fn text(&self) -> &str {
        &self.content
    }

    #[allow(dead_code)]
    pub fn has_text(&self) -> bool {
        !self.content.is_empty()
    }

    pub fn is_composing(&self) -> bool {
        self.marked_range.is_some()
    }

    pub fn set_text(&mut self, text: impl Into<SharedString>, cx: &mut Context<Self>) {
        let text = text.into();
        let len = text.len();
        self.content = text;
        self.selected_range = len..len;
        self.selection_reversed = false;
        self.marked_range = None;
        self.scroll_x = px(0.);
        self.last_layout = None;
        cx.notify();
    }

    pub fn clear(&mut self, cx: &mut Context<Self>) {
        self.set_text("", cx);
    }

    #[allow(dead_code)]
    pub fn take_text(&mut self, cx: &mut Context<Self>) -> SharedString {
        let text = std::mem::replace(&mut self.content, "".into());
        self.selected_range = 0..0;
        self.selection_reversed = false;
        self.marked_range = None;
        self.scroll_x = px(0.);
        self.last_layout = None;
        cx.notify();
        text
    }

    fn submit(&mut self, _: &Submit, _: &mut Window, cx: &mut Context<Self>) {
        if !self.is_composing() {
            cx.emit(SubmitText(self.content.clone()));
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

    fn backspace(&mut self, _: &Backspace, _: &mut Window, cx: &mut Context<Self>) {
        if self.selected_range.is_empty() {
            let offset = self.cursor_offset();
            let previous = self.previous_boundary(offset);
            self.replace_range(previous..offset, "");
            self.selected_range = previous..previous;
        } else {
            let start = self.selected_range.start;
            self.replace_range(self.selected_range.clone(), "");
            self.selected_range = start..start;
        }
        cx.notify();
    }

    fn delete(&mut self, _: &Delete, _: &mut Window, cx: &mut Context<Self>) {
        if self.selected_range.is_empty() {
            let offset = self.cursor_offset();
            let next = self.next_boundary(offset);
            self.replace_range(offset..next, "");
            self.selected_range = offset..offset;
        } else {
            let start = self.selected_range.start;
            self.replace_range(self.selected_range.clone(), "");
            self.selected_range = start..start;
        }
        cx.notify();
    }

    fn left(&mut self, _: &Left, _: &mut Window, cx: &mut Context<Self>) {
        let cursor = if self.selected_range.is_empty() {
            self.previous_boundary(self.cursor_offset())
        } else {
            self.selected_range.start
        };
        self.selected_range = cursor..cursor;
        self.selection_reversed = false;
        cx.notify();
    }

    fn right(&mut self, _: &Right, _: &mut Window, cx: &mut Context<Self>) {
        let cursor = if self.selected_range.is_empty() {
            self.next_boundary(self.cursor_offset())
        } else {
            self.selected_range.end
        };
        self.selected_range = cursor..cursor;
        self.selection_reversed = false;
        cx.notify();
    }

    fn select_left(&mut self, _: &SelectLeft, _: &mut Window, cx: &mut Context<Self>) {
        let offset = self.cursor_offset();
        let next = self.previous_boundary(offset);
        if self.selected_range.is_empty() {
            self.selection_reversed = true;
            self.selected_range = next..offset;
        } else if self.selection_reversed {
            self.selected_range.start = next;
        } else {
            self.selected_range.end = next;
            if self.selected_range.is_empty() {
                self.selection_reversed = true;
                self.selected_range = next..offset;
            }
        }
        cx.notify();
    }

    fn select_right(&mut self, _: &SelectRight, _: &mut Window, cx: &mut Context<Self>) {
        let offset = self.cursor_offset();
        let next = self.next_boundary(offset);
        if self.selected_range.is_empty() {
            self.selection_reversed = false;
            self.selected_range = offset..next;
        } else if self.selection_reversed {
            self.selected_range.start = next;
            if self.selected_range.is_empty() {
                self.selection_reversed = false;
                self.selected_range = offset..next;
            }
        } else {
            self.selected_range.end = next;
        }
        cx.notify();
    }

    fn select_all(&mut self, _: &SelectAll, _: &mut Window, cx: &mut Context<Self>) {
        self.selected_range = 0..self.content.len();
        self.selection_reversed = false;
        cx.notify();
    }

    fn home(&mut self, _: &Home, _: &mut Window, cx: &mut Context<Self>) {
        self.selected_range = 0..0;
        self.selection_reversed = false;
        cx.notify();
    }

    fn end(&mut self, _: &End, _: &mut Window, cx: &mut Context<Self>) {
        let len = self.content.len();
        self.selected_range = len..len;
        self.selection_reversed = false;
        cx.notify();
    }

    fn copy(&mut self, _: &Copy, _: &mut Window, cx: &mut Context<Self>) {
        if !self.selected_range.is_empty() {
            cx.write_to_clipboard(ClipboardItem::new_string(
                self.content[self.selected_range.clone()].to_string(),
            ));
        }
    }

    fn cut(&mut self, _: &Cut, _: &mut Window, cx: &mut Context<Self>) {
        if !self.selected_range.is_empty() {
            cx.write_to_clipboard(ClipboardItem::new_string(
                self.content[self.selected_range.clone()].to_string(),
            ));
            let start = self.selected_range.start;
            self.replace_range(self.selected_range.clone(), "");
            self.selected_range = start..start;
            cx.notify();
        }
    }

    fn paste(&mut self, _: &Paste, _: &mut Window, cx: &mut Context<Self>) {
        if let Some(item) = cx.read_from_clipboard() {
            let text = item.text().unwrap_or_default().replace('\n', " ");
            let start = self.selected_range.start;
            self.replace_range(self.selected_range.clone(), &text);
            self.selected_range = start + text.len()..start + text.len();
            cx.notify();
        }
    }

    fn on_mouse_down(&mut self, event: &MouseDownEvent, _: &mut Window, cx: &mut Context<Self>) {
        self.is_selecting = true;
        if let Some(index) = self.index_for_mouse_position(event.position) {
            self.selected_range = index..index;
            self.selection_reversed = false;
            cx.notify();
        }
    }

    fn on_mouse_up(&mut self, _: &MouseUpEvent, _: &mut Window, _: &mut Context<Self>) {
        self.is_selecting = false;
    }

    fn on_mouse_move(&mut self, event: &MouseMoveEvent, _: &mut Window, cx: &mut Context<Self>) {
        if self.is_selecting
            && let Some(index) = self.index_for_mouse_position(event.position)
        {
            if self.selected_range.is_empty() {
                self.selection_reversed = index < self.selected_range.start;
            }
            if self.selection_reversed {
                self.selected_range.start = index;
            } else {
                self.selected_range.end = index;
            }
            cx.notify();
        }
    }

    fn index_for_mouse_position(&self, position: Point<Pixels>) -> Option<usize> {
        let layout = self.last_layout.as_ref()?;
        let bounds = self.last_bounds.as_ref()?;
        if bounds.size.width <= px(0.) {
            return None;
        }
        let local_x = position.x - bounds.left() + self.scroll_x;
        Some(layout.closest_index_for_x(local_x))
    }

    fn range_to_utf16(&self, range: &Range<usize>) -> Range<usize> {
        offset_to_utf16(&self.content, range.start)..offset_to_utf16(&self.content, range.end)
    }

    fn range_from_utf16(&self, range_utf16: &Range<usize>) -> Range<usize> {
        offset_from_utf16(&self.content, range_utf16.start)
            ..offset_from_utf16(&self.content, range_utf16.end)
    }

    fn replace_range(&mut self, range: Range<usize>, new_text: &str) {
        let mut content =
            String::with_capacity(self.content.len().saturating_sub(range.len()) + new_text.len());
        content.push_str(&self.content[..range.start]);
        content.push_str(new_text);
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

impl EntityInputHandler for TextInput {
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
        replacement_range: Option<Range<usize>>,
        text: &str,
        _: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let range = replacement_range
            .map(|range| self.range_from_utf16(&range))
            .or_else(|| self.marked_range.clone())
            .unwrap_or(self.selected_range.clone());
        self.replace_range(range.clone(), text);
        self.selected_range = range.start + text.len()..range.start + text.len();
        self.marked_range = None;
        cx.notify();
    }

    fn replace_and_mark_text_in_range(
        &mut self,
        replacement_range: Option<Range<usize>>,
        new_text: &str,
        new_selected_range: Option<Range<usize>>,
        _: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let range = replacement_range
            .map(|range| self.range_from_utf16(&range))
            .or_else(|| self.marked_range.clone())
            .unwrap_or(self.selected_range.clone());
        self.replace_range(range.clone(), new_text);
        self.marked_range = Some(range.start..range.start + new_text.len());
        self.selected_range =
            selection_after_replacement(range.start, new_text, new_selected_range);
        cx.notify();
    }

    fn bounds_for_range(
        &mut self,
        range_utf16: Range<usize>,
        bounds: Bounds<Pixels>,
        _: &mut Window,
        _: &mut Context<Self>,
    ) -> Option<Bounds<Pixels>> {
        let layout = self.last_layout.as_ref()?;
        let range = self.range_from_utf16(&range_utf16);
        Some(Bounds::from_corners(
            point(
                bounds.left() + layout.x_for_index(range.start) - self.scroll_x,
                bounds.top(),
            ),
            point(
                bounds.left() + layout.x_for_index(range.end) - self.scroll_x,
                bounds.bottom(),
            ),
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
            self.index_for_mouse_position(position)?,
        ))
    }
}

struct TextElement {
    input: Entity<TextInput>,
}

struct PrepaintState {
    line: Option<ShapedLine>,
    placeholder_line: Option<ShapedLine>,
    cursor: Option<PaintQuad>,
    selection: Option<PaintQuad>,
    scroll_x: Pixels,
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
        cx: &mut App,
    ) -> (LayoutId, ()) {
        let mut style = Style::default();
        style.size.width = relative(1.).into();
        style.size.height = window.line_height().into();
        (window.request_layout(style, [], cx), ())
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
        let text_style = window.text_style();
        let is_empty = input.content.is_empty();

        let placeholder_line = if is_empty && !input.placeholder.is_empty() {
            let mut ph_style = text_style.clone();
            ph_style.color = rgb(0x6e6e6e).into();
            Some(window.text_system().shape_line(
                input.placeholder.clone(),
                ph_style.font_size.to_pixels(window.rem_size()),
                &[TextRun {
                    len: input.placeholder.len(),
                    font: ph_style.font(),
                    color: ph_style.color,
                    background_color: None,
                    underline: None,
                    strikethrough: None,
                }],
                None,
            ))
        } else {
            None
        };

        let display_content: SharedString = if input.masked {
            "•".repeat(input.content.chars().count()).into()
        } else {
            input.content.clone()
        };

        let mut runs = Vec::new();
        if let Some(marked) = &input.marked_range {
            let marked_start = if input.masked {
                offset_to_utf16(&input.content, marked.start)
            } else {
                marked.start
            };
            let marked_end = if input.masked {
                offset_to_utf16(&input.content, marked.end)
            } else {
                marked.end
            };
            if marked_start > 0 {
                runs.push(TextRun {
                    len: marked_start,
                    font: text_style.font(),
                    color: text_style.color,
                    background_color: None,
                    underline: None,
                    strikethrough: None,
                });
            }
            runs.push(TextRun {
                len: marked_end.saturating_sub(marked_start),
                font: text_style.font(),
                color: text_style.color,
                background_color: None,
                underline: Some(UnderlineStyle {
                    color: Some(text_style.color),
                    thickness: px(1.),
                    wavy: false,
                }),
                strikethrough: None,
            });
            if marked_end < display_content.len() {
                runs.push(TextRun {
                    len: display_content.len() - marked_end,
                    font: text_style.font(),
                    color: text_style.color,
                    background_color: None,
                    underline: None,
                    strikethrough: None,
                });
            }
        } else {
            runs.push(TextRun {
                len: display_content.len(),
                font: text_style.font(),
                color: text_style.color,
                background_color: None,
                underline: None,
                strikethrough: None,
            });
        }

        let line = window.text_system().shape_line(
            display_content,
            text_style.font_size.to_pixels(window.rem_size()),
            &runs,
            None,
        );

        let cursor_index = if input.masked {
            offset_to_utf16(&input.content, input.cursor_offset())
        } else {
            input.cursor_offset()
        };
        let cursor_x = line.x_for_index(cursor_index);

        let mut scroll_x = input.scroll_x;
        let padding = px(8.);
        if cursor_x - scroll_x < padding {
            scroll_x = (cursor_x - padding).max(px(0.));
        } else if cursor_x - scroll_x > bounds.size.width - padding {
            scroll_x = cursor_x - bounds.size.width + padding;
        }
        let max_scroll = (line.width - bounds.size.width + padding).max(px(0.));
        scroll_x = scroll_x.min(max_scroll);

        let cursor = if input.focus_handle.is_focused(window) {
            Some(fill(
                Bounds::new(
                    point(bounds.left() + cursor_x - scroll_x, bounds.top()),
                    size(px(2.), bounds.size.height),
                ),
                rgb(0xe7e7e7),
            ))
        } else {
            None
        };

        let selection = if !input.selected_range.is_empty() {
            let sel_start = if input.masked {
                offset_to_utf16(&input.content, input.selected_range.start)
            } else {
                input.selected_range.start
            };
            let sel_end = if input.masked {
                offset_to_utf16(&input.content, input.selected_range.end)
            } else {
                input.selected_range.end
            };
            let start_x = line.x_for_index(sel_start);
            let end_x = line.x_for_index(sel_end);
            Some(fill(
                Bounds::from_corners(
                    point(bounds.left() + start_x - scroll_x, bounds.top()),
                    point(bounds.left() + end_x - scroll_x, bounds.bottom()),
                ),
                rgba(0x4080ff44),
            ))
        } else {
            None
        };

        PrepaintState {
            line: Some(line),
            placeholder_line,
            cursor,
            selection,
            scroll_x,
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
        window.with_content_mask(Some(ContentMask { bounds }), |window| {
            if let Some(placeholder) = prepaint.placeholder_line.take() {
                placeholder
                    .paint(
                        point(bounds.left(), bounds.top()),
                        window.line_height(),
                        window,
                        cx,
                    )
                    .unwrap();
            }
            if let Some(selection) = prepaint.selection.take() {
                window.paint_quad(selection);
            }
            line.paint(
                point(bounds.left() - prepaint.scroll_x, bounds.top()),
                window.line_height(),
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
            input.scroll_x = prepaint.scroll_x;
            input.last_layout = Some(line);
            input.last_bounds = Some(bounds);
        });
    }
}

impl Render for TextInput {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        div()
            .id("text-input")
            .flex()
            .w_full()
            .min_w_0()
            .key_context("TextInput")
            .track_focus(&self.focus_handle)
            .cursor(CursorStyle::IBeam)
            .px_2()
            .py_1()
            .line_height(px(20.))
            .text_size(px(14.))
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

impl Focusable for TextInput {
    fn focus_handle(&self, _: &App) -> FocusHandle {
        self.focus_handle.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_input_utf16_offset_mapping() {
        let text = "hello 🦀 world";
        assert_eq!(offset_from_utf16(text, 6), 6);
        // The crab emoji takes 2 utf-16 code units and 4 utf-8 bytes
        assert_eq!(offset_from_utf16(text, 8), 10);
        assert_eq!(offset_to_utf16(text, 10), 8);
    }

    #[test]
    fn text_input_selection_after_replacement() {
        let text = "inserted";
        let range = selection_after_replacement(5, text, None);
        assert_eq!(range, 13..13);
    }
}
