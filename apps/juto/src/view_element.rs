//! Maps `juto_agent::ViewNode` trees to GPUI elements. Each `tag` selects a host
//! component; an unknown tag renders a visible fallback that names the tag, so
//! a bad tree is never silently dropped.

use gpui::{AnyElement, IntoElement, ParentElement, SharedString, Styled, div, rgb};
use juto_agent::{ViewChild, ViewNode};

/// Renders one element and its subtree.
pub fn render_view(node: &ViewNode) -> AnyElement {
    match node.tag.as_str() {
        "Column" => div()
            .flex()
            .flex_col()
            .gap_2()
            .children(render_children(node))
            .into_any_element(),
        "Row" => div()
            .flex()
            .flex_row()
            .flex_wrap()
            .items_center()
            .gap_2()
            .children(render_children(node))
            .into_any_element(),
        "Text" => div()
            .text_sm()
            .text_color(rgb(0xe5e7eb))
            .children(render_children(node))
            .into_any_element(),
        "Badge" => badge(prop(node, "label"), prop(node, "tone")),
        "Subagent" => div()
            .flex()
            .flex_row()
            .items_center()
            .gap_2()
            .px_2p5()
            .py_1()
            .rounded_md()
            .border_1()
            .border_color(rgb(0x374151))
            .text_sm()
            .text_color(rgb(0xe5e7eb))
            .child(SharedString::from(prop(node, "name").to_owned()))
            .child(badge(prop(node, "status"), prop(node, "tone")))
            .into_any_element(),
        unknown => div()
            .px_2p5()
            .py_1()
            .rounded_md()
            .border_1()
            .border_color(rgb(0x7f1d1d))
            .text_xs()
            .text_color(rgb(0xfca5a5))
            .child(SharedString::from(format!("Unknown element <{unknown}>")))
            .into_any_element(),
    }
}

fn render_children(node: &ViewNode) -> Vec<AnyElement> {
    node.children
        .iter()
        .map(|child| match child {
            ViewChild::Text(text) => SharedString::from(text.clone()).into_any_element(),
            ViewChild::Node(node) => render_view(node),
        })
        .collect()
}

/// Background and foreground colors for a `Badge` tone. Unknown tones use neutral.
fn badge_colors(tone: &str) -> (u32, u32) {
    match tone {
        "success" => (0x14532d, 0xbbf7d0),
        "warning" => (0x78350f, 0xfde68a),
        "error" => (0x7f1d1d, 0xfecaca),
        "info" => (0x1e3a8a, 0xbfdbfe),
        _ => (0x374151, 0xe5e7eb),
    }
}

/// A string prop, or `""` when the key is absent or not a string.
fn prop<'a>(node: &'a ViewNode, key: &str) -> &'a str {
    node.props
        .get(key)
        .and_then(|value| value.as_str())
        .unwrap_or_default()
}

fn badge(label: &str, tone: &str) -> AnyElement {
    let (background, foreground) = badge_colors(tone);
    div()
        .px_2p5()
        .py_0p5()
        .rounded_md()
        .text_xs()
        .bg(rgb(background))
        .text_color(rgb(foreground))
        .child(SharedString::from(label.to_owned()))
        .into_any_element()
}
