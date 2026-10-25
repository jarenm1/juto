//! A full-height pane right of a thread's transcript that hosts one built-in
//! app view, such as a desktop viewer.
//!
//! Each thread view owns its own pane, so an app opened beside one thread does
//! not appear beside another. The pane starts at half the area it shares with
//! the transcript; dragging its divider resizes both.

use gpui::{
    AnyView, App, ClickEvent, Context, DragMoveEvent, EmptyView, Entity, EntityId, IntoElement,
    Render, Window, div, prelude::*, px, relative, rgb,
};

use crate::theme;

/// Share of the row the pane takes when it opens.
const DEFAULT_FRACTION: f32 = 0.5;
/// Dragging the divider keeps the pane and the transcript at least this share.
const MIN_FRACTION: f32 = 0.2;

/// A view that can open in a [`SidePane`].
pub trait SidePaneItem: Render {
    /// Called with `true` when the item enters the pane and `false` when it
    /// leaves, by closing or by another item replacing it.
    fn set_docked(&mut self, docked: bool, cx: &mut Context<Self>);
}

/// Type-erased handle on a docked item.
trait ItemHandle {
    fn view(&self) -> AnyView;
    fn entity_id(&self) -> EntityId;
    fn set_docked(&self, docked: bool, cx: &mut App);
}

impl<T: SidePaneItem> ItemHandle for Entity<T> {
    fn view(&self) -> AnyView {
        self.clone().into()
    }

    fn entity_id(&self) -> EntityId {
        Entity::entity_id(self)
    }

    fn set_docked(&self, docked: bool, cx: &mut App) {
        self.update(cx, |item, cx| item.set_docked(docked, cx));
    }
}

/// Drag payload of the pane's divider.
struct Divider;

/// The pane beside one thread's transcript. Holds at most one item.
pub struct SidePane {
    item: Option<Box<dyn ItemHandle>>,
    /// Share of the row the pane takes.
    fraction: f32,
}

impl SidePane {
    pub fn new() -> Self {
        Self {
            item: None,
            fraction: DEFAULT_FRACTION,
        }
    }

    pub fn is_open(&self) -> bool {
        self.item.is_some()
    }

    /// Whether `entity` is the docked item.
    pub fn holds(&self, entity: EntityId) -> bool {
        self.item
            .as_ref()
            .is_some_and(|item| item.entity_id() == entity)
    }

    /// Docks `item`. An item already docked leaves the pane first.
    pub fn open<T: SidePaneItem>(&mut self, item: Entity<T>, cx: &mut Context<Self>) {
        if self.holds(item.entity_id()) {
            return;
        }
        self.close(cx);
        item.update(cx, |item, cx| item.set_docked(true, cx));
        self.item = Some(Box::new(item));
        cx.notify();
    }

    /// Undocks the item, if any, and gives the row back to the transcript.
    pub fn close(&mut self, cx: &mut Context<Self>) {
        if let Some(item) = self.item.take() {
            item.set_docked(false, cx);
            cx.notify();
        }
    }
}

/// The pane's share after its divider is dragged to `pointer_x`. The pane's
/// own width is `fraction` of the row, which gives the row's width.
fn dragged_fraction(pane_right: f32, pane_width: f32, fraction: f32, pointer_x: f32) -> f32 {
    if pane_width <= 0.0 || fraction <= 0.0 {
        return fraction;
    }
    let row_width = pane_width / fraction;
    ((pane_right - pointer_x) / row_width).clamp(MIN_FRACTION, 1.0 - MIN_FRACTION)
}

impl Render for SidePane {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let Some(item) = &self.item else {
            return div().into_any_element();
        };
        div()
            .id("side-pane")
            .flex_none()
            .w(relative(self.fraction))
            .h_full()
            .min_w_0()
            .flex()
            .on_drag_move(
                cx.listener(|this, event: &DragMoveEvent<Divider>, _window, cx| {
                    let pane = event.bounds;
                    this.fraction = dragged_fraction(
                        f32::from(pane.right()),
                        f32::from(pane.size.width),
                        this.fraction,
                        f32::from(event.event.position.x),
                    );
                    cx.notify();
                }),
            )
            .child(
                div()
                    .id("side-pane-divider")
                    .flex_none()
                    .w(px(6.))
                    .h_full()
                    .bg(rgb(theme::BORDER))
                    .hover(|style| style.bg(rgb(theme::ACCENT)))
                    .cursor_col_resize()
                    .on_drag(Divider, |_, _, _, cx| cx.new(|_| EmptyView)),
            )
            .child(
                div()
                    .flex_1()
                    .min_w_0()
                    .h_full()
                    .flex()
                    .flex_col()
                    .gap_2()
                    .p_3()
                    .bg(rgb(0x202124))
                    .child(
                        div().flex().justify_end().child(
                            div()
                                .id("side-pane-close")
                                .px_3()
                                .py_1()
                                .cursor_pointer()
                                .child("Close split")
                                .on_click(cx.listener(|this, _: &ClickEvent, _window, cx| {
                                    this.close(cx);
                                })),
                        ),
                    )
                    .child(div().flex_1().min_h_0().child(item.view())),
            )
            .into_any_element()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Dragging maps the pointer to the pane's new share of the row and keeps
    /// both sides at least the minimum share.
    #[test]
    fn divider_drag_resizes_within_limits() {
        // Pane is half of a 1000px row: 500px wide, right edge at 1000.
        assert_eq!(dragged_fraction(1000.0, 500.0, 0.5, 300.0), 0.7);
        assert_eq!(dragged_fraction(1000.0, 500.0, 0.5, 50.0), 0.8);
        assert_eq!(dragged_fraction(1000.0, 500.0, 0.5, 990.0), 0.2);
    }
}
