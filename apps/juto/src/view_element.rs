//! Maps `juto_agent::ViewNode` trees to GPUI elements. Each `tag` selects a host
//! component; an unknown tag renders a visible fallback that names the tag, so
//! a bad tree is never silently dropped.
//!
//! The look follows shadcn/ui primitives (Card, destructive Alert) on the shared
//! palette in `theme.rs`; `Badge` and subagent rows are flat status marks. The tags
//! and props agents may use are fixed by `juto_agent::VIEW_TAGS`; this file only
//! decides how they look.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::rc::Rc;

use gpui::{
    AnyElement, AnyView, App, Bounds, Corners, Div, FontWeight, Hsla, PathBuilder, Pixels, Point,
    SharedString, StatefulInteractiveElement, Window, canvas, div, fill, point, prelude::*, px,
    relative, rgb, size,
};
use juto_agent::{MAX_TIMELINE_TICKS, ViewChild, ViewNode, parse_chart_labels, parse_chart_values};

use crate::theme;

/// Series colors in order of first appearance; later series reuse the list.
const SERIES_COLORS: [u32; 4] = [
    theme::ACCENT,
    theme::TEXT_FAINT,
    theme::WARNING,
    theme::DIFF_ADD,
];
/// Width of a timeline's lane-label column, in pixels.
const LANE_LABEL_WIDTH: f32 = 128.0;
/// Height of one bar, and the vertical step between stacked series bars.
const BAR_HEIGHT: f32 = 6.0;
const BAR_PITCH: f32 = 8.0;
/// Space above the first bar row inside a lane.
const LANE_PAD: f32 = 4.0;
/// Plot height of a `Chart`, in pixels.
const CHART_PLOT_HEIGHT: f32 = 180.0;
/// Narrowest a `Chart` may get before its `Row` wraps it onto a new line.
const CHART_MIN_WIDTH: f32 = 300.0;
/// Gridline intervals across a chart plot.
const CHART_GRID_STEPS: usize = 4;
/// Stroke width of a line series, in pixels.
const LINE_WIDTH: f32 = 2.0;
/// Diameter of a scatter dot and of a hover marker, in pixels.
const MARKER_SIZE: f32 = 7.0;
/// Opacity of an area chart's fill under its line.
const AREA_FILL_ALPHA: f32 = 0.2;
/// Opacity of the column tint under a hovered x position.
const HOVER_BAND_ALPHA: f32 = 0.05;

/// Host callbacks a rendered view can invoke. A `Button` click goes through
/// `on_submit`; without it every `Button` renders disabled.
#[derive(Clone, Default)]
pub struct ViewActions {
    on_submit: Option<Rc<SubmitFn>>,
}

/// Callback a `Button` click invokes with the button's `prompt`.
type SubmitFn = dyn Fn(String, &mut Window, &mut App);

impl ViewActions {
    /// No actions: buttons render disabled.
    pub fn none() -> Self {
        Self::default()
    }

    /// Clicking a button calls `on_submit` with that button's `prompt`.
    pub fn submit(on_submit: impl Fn(String, &mut Window, &mut App) + 'static) -> Self {
        Self {
            on_submit: Some(Rc::new(on_submit)),
        }
    }
}

/// Renders one element and its subtree.
pub fn render_view(node: &ViewNode, actions: &ViewActions) -> AnyElement {
    match node.tag.as_str() {
        "Column" => div()
            .flex()
            .flex_col()
            .gap_3()
            .children(column_children(node, actions))
            .into_any_element(),
        "Row" => div()
            .flex()
            .flex_row()
            .flex_wrap()
            .items_center()
            .gap_2()
            .children(render_children(node, actions))
            .into_any_element(),
        "Text" => div()
            .text_sm()
            .text_color(rgb(theme::TEXT))
            .children(render_children(node, actions))
            .into_any_element(),
        "Badge" => badge(prop(node, "label"), prop(node, "tone")),
        "Button" => button(node, actions),
        "Stats" => stats(node, actions),
        "Stat" => stat(prop(node, "label"), prop(node, "value")),
        "Timeline" => timeline(node),
        "Chart" => chart(node),
        "Subagent" => subagent_list(vec![node]),
        unknown => fallback(format!("Unknown element <{unknown}>")),
    }
}

fn render_children(node: &ViewNode, actions: &ViewActions) -> Vec<AnyElement> {
    node.children
        .iter()
        .map(|child| render_child(child, actions))
        .collect()
}

fn render_child(child: &ViewChild, actions: &ViewActions) -> AnyElement {
    match child {
        ViewChild::Text(text) => SharedString::from(text.clone()).into_any_element(),
        ViewChild::Node(node) => render_view(node, actions),
    }
}

/// Accent for a `Badge` tone, or `None` for an unknown tone (rendered neutral).
fn tone_color(tone: &str) -> Option<u32> {
    match tone {
        "success" => Some(theme::DIFF_ADD),
        "warning" => Some(theme::WARNING),
        "error" => Some(theme::DANGER),
        "info" => Some(theme::ACCENT),
        _ => None,
    }
}

/// A string prop, or `""` when the key is absent or not a string.
fn prop<'a>(node: &'a ViewNode, key: &str) -> &'a str {
    node.props
        .get(key)
        .and_then(|value| value.as_str())
        .unwrap_or_default()
}

/// A numeric prop, or `0.0` when it does not parse. The validator rejects
/// non-numbers before a tree reaches the host.
fn number(node: &ViewNode, key: &str) -> f64 {
    prop(node, key).parse().unwrap_or_default()
}

fn node_child(child: &ViewChild) -> Option<&ViewNode> {
    match child {
        ViewChild::Node(node) => Some(node),
        ViewChild::Text(_) => None,
    }
}

/// Red-ruled row for an element the host cannot draw.
fn fallback(message: String) -> AnyElement {
    div()
        .px_3()
        .py_2p5()
        .border_l_2()
        .border_color(rgb(theme::DANGER))
        .text_sm()
        .text_color(rgb(theme::DANGER))
        .child(SharedString::from(message))
        .into_any_element()
}

/// Flat status mark: a tone-colored dot and an uppercase label in the same color.
/// Shared by `Badge` and subagent rows.
fn status_mark(label: &str, color: u32) -> AnyElement {
    div()
        .flex()
        .flex_row()
        .items_center()
        .gap_2()
        .flex_none()
        .text_xs()
        .text_color(rgb(color))
        .child(div().size(px(6.0)).rounded_full().bg(rgb(color)))
        .child(SharedString::from(label.to_uppercase()))
        .into_any_element()
}

/// Status mark for a `Badge` tone; unknown tones use the muted text color.
fn badge(label: &str, tone: &str) -> AnyElement {
    status_mark(label, tone_color(tone).unwrap_or(theme::TEXT_MUTED))
}

/// Colors of a `Button` tone: background, text, border, and hover background.
/// Primary has no hover background and dims to 85% opacity instead.
struct ButtonTone {
    bg: u32,
    text: u32,
    border: u32,
    hover_bg: Option<u32>,
}

/// Palette for a `Button` tone. The validator admits only `primary`, `neutral`,
/// `danger`, or no tone, so everything else is neutral.
fn button_tone(tone: &str) -> ButtonTone {
    match tone {
        "primary" => ButtonTone {
            bg: theme::ACCENT,
            text: theme::ON_ACCENT,
            border: theme::ACCENT,
            hover_bg: None,
        },
        "danger" => ButtonTone {
            bg: 0x450a0a,
            text: 0xfca5a5,
            border: 0x7f1d1d,
            hover_bg: Some(0x7f1d1d),
        },
        _ => ButtonTone {
            bg: theme::BG_RAISED,
            text: theme::TEXT,
            border: theme::BORDER_STRONG,
            hover_bg: Some(theme::BG_HOVER),
        },
    }
}

/// Clickable button. A click submits the `prompt` prop as the next user turn.
/// Without a submit action the button renders disabled and inert.
fn button(node: &ViewNode, actions: &ViewActions) -> AnyElement {
    let label = prop(node, "label").to_owned();
    let base = div()
        .id(SharedString::from(format!(
            "view-button-{label}-{}",
            prop(node, "prompt")
        )))
        .flex_none()
        .px_3()
        .py_1()
        .rounded_sm()
        .border_1()
        .text_xs()
        .font_weight(FontWeight::SEMIBOLD);
    let Some(on_submit) = actions.on_submit.clone() else {
        return base
            .bg(rgb(theme::BG_RAISED))
            .border_color(rgb(theme::BORDER))
            .text_color(rgb(theme::TEXT_FAINT))
            .child(SharedString::from(label))
            .into_any_element();
    };
    let tone = button_tone(prop(node, "tone"));
    let prompt = prop(node, "prompt").to_owned();
    let styled = base
        .cursor_pointer()
        .bg(rgb(tone.bg))
        .border_color(rgb(tone.border))
        .text_color(rgb(tone.text));
    let styled = match tone.hover_bg {
        Some(hover) => styled.hover(move |s| s.bg(rgb(hover))),
        None => styled.hover(|s| s.opacity(0.85)),
    };
    styled
        .on_click(move |_, window, cx| on_submit(prompt.clone(), window, cx))
        .child(SharedString::from(label))
        .into_any_element()
}

/// Column children. Each run of adjacent `Subagent` nodes becomes one list, so
/// the rows sit flush with hairline dividers instead of gapped cards.
fn column_children(node: &ViewNode, actions: &ViewActions) -> Vec<AnyElement> {
    let mut out = Vec::new();
    let mut run: Vec<&ViewNode> = Vec::new();
    for child in &node.children {
        match node_child(child) {
            Some(subagent) if subagent.tag == "Subagent" => run.push(subagent),
            _ => {
                if !run.is_empty() {
                    out.push(subagent_list(std::mem::take(&mut run)));
                }
                out.push(render_child(child, actions));
            }
        }
    }
    if !run.is_empty() {
        out.push(subagent_list(run));
    }
    out
}

/// Top-ruled list of subagent rows; each row draws its own bottom divider.
fn subagent_list(rows: Vec<&ViewNode>) -> AnyElement {
    div()
        .flex()
        .flex_col()
        .border_t_1()
        .border_color(rgb(theme::BORDER))
        .children(rows.into_iter().map(subagent_row))
        .into_any_element()
}

/// Subagent row: name on the left; status dot and uppercase status on the
/// right, both in the tone's color. Hover tints the row.
fn subagent_row(node: &ViewNode) -> AnyElement {
    let tone = tone_color(prop(node, "tone")).unwrap_or(theme::TEXT_MUTED);
    div()
        .flex()
        .flex_row()
        .items_center()
        .justify_between()
        .gap_3()
        .px_3()
        .py_2p5()
        .border_b_1()
        .border_color(rgb(theme::BORDER))
        .hover(|row| row.bg(rgb(theme::BG_HOVER)))
        .text_sm()
        .text_color(rgb(theme::TEXT))
        .child(
            div()
                .min_w_0()
                .truncate()
                .child(SharedString::from(prop(node, "name").to_owned())),
        )
        .child(status_mark(prop(node, "status"), tone))
        .into_any_element()
}

/// Horizontal strip of `Stat` cells split by hairline dividers, closed by a rule.
fn stats(node: &ViewNode, actions: &ViewActions) -> AnyElement {
    let mut strip = div()
        .flex()
        .flex_row()
        .border_b_1()
        .border_color(rgb(theme::BORDER));
    let cells: Vec<&ViewNode> = node.children.iter().filter_map(node_child).collect();
    for (index, cell) in cells.iter().enumerate() {
        if index > 0 {
            strip = strip.child(div().w(px(1.0)).my_3().bg(rgb(theme::BORDER)));
        }
        strip = strip.child(div().flex_1().min_w_0().child(render_view(cell, actions)));
    }
    strip.into_any_element()
}

/// One stat cell: the value large, with a small uppercase label beneath it.
fn stat(label: &str, value: &str) -> AnyElement {
    div()
        .flex()
        .flex_col()
        .gap_0p5()
        .px_4()
        .py_3()
        .child(
            div()
                .text_lg()
                .font_weight(FontWeight::MEDIUM)
                .text_color(rgb(theme::TEXT))
                .child(SharedString::from(value.to_owned())),
        )
        .child(
            div()
                .text_xs()
                .text_color(rgb(theme::TEXT_MUTED))
                .child(SharedString::from(label.to_uppercase())),
        )
        .into_any_element()
}

/// A positive seconds prop, or `None` when it is missing, malformed, or not above zero.
fn seconds(node: &ViewNode, key: &str) -> Option<f64> {
    prop(node, key)
        .parse::<f64>()
        .ok()
        .filter(|value| value.is_finite() && *value > 0.0)
}

/// Series names in order of first appearance across all bars. A bar without a
/// `series` prop belongs to the unnamed series `""`.
fn series_names(lanes: &[&ViewNode]) -> Vec<String> {
    let mut names: Vec<String> = Vec::new();
    for bar in lanes
        .iter()
        .flat_map(|lane| lane.children.iter().filter_map(node_child))
    {
        let name = prop(bar, "series");
        if !names.iter().any(|known| known == name) {
            names.push(name.to_owned());
        }
    }
    names
}

fn series_index(series: &[String], name: &str) -> usize {
    series.iter().position(|known| known == name).unwrap_or(0)
}

fn series_color(index: usize) -> u32 {
    SERIES_COLORS[index % SERIES_COLORS.len()]
}

/// Gantt-style timeline: one row per `Lane`, bars on a shared axis from 0 to
/// `max` seconds, a dotted-free tick grid every `tick` seconds, and a legend.
fn timeline(node: &ViewNode) -> AnyElement {
    let (Some(max), Some(tick)) = (seconds(node, "max"), seconds(node, "tick")) else {
        return fallback("Invalid <Timeline>: max and tick must be positive numbers".into());
    };
    let lanes: Vec<&ViewNode> = node.children.iter().filter_map(node_child).collect();
    let series = series_names(&lanes);
    let ticks: Vec<f64> = (0..=MAX_TIMELINE_TICKS)
        .map(|step| step as f64 * tick)
        .take_while(|time| *time <= max)
        .collect();
    let lane_height = BAR_PITCH * series.len().max(1) as f32 + LANE_PAD * 2.0;

    let key = view_key(node);
    let mut body = div().flex().flex_col();
    for (lane_index, lane) in lanes.iter().enumerate() {
        body = body.child(lane_row(
            lane,
            lane_index,
            &key,
            &series,
            max,
            &ticks,
            lane_height,
        ));
    }
    div()
        .flex()
        .flex_col()
        .gap_2()
        .pt_3()
        .child(timeline_header(node, &series))
        .child(body)
        .child(axis_row(&ticks, max))
        .into_any_element()
}

/// Title on the left; the series legend on the right. The legend is omitted when
/// no bar names a series.
fn timeline_header(node: &ViewNode, series: &[String]) -> AnyElement {
    let mut legend = div().flex().flex_row().items_center().gap_3();
    if series.iter().any(|name| !name.is_empty()) {
        for (index, name) in series.iter().enumerate() {
            legend = legend.child(
                div()
                    .flex()
                    .flex_row()
                    .items_center()
                    .gap_1p5()
                    .text_xs()
                    .text_color(rgb(theme::TEXT_MUTED))
                    .child(div().size_2().rounded_full().bg(rgb(series_color(index))))
                    .child(SharedString::from(name.clone())),
            );
        }
    }
    div()
        .flex()
        .flex_row()
        .justify_between()
        .items_center()
        .gap_3()
        .child(
            div()
                .text_xs()
                .font_weight(FontWeight::SEMIBOLD)
                .text_color(rgb(theme::TEXT_MUTED))
                .child(SharedString::from(prop(node, "title").to_uppercase())),
        )
        .child(legend)
        .into_any_element()
}

/// One lane: its label, then a track holding the tick grid and each bar. Bars
/// stack by series, one row per series. Each bar shows its span on hover, and the
/// lane tints as a whole.
fn lane_row(
    lane: &ViewNode,
    lane_index: usize,
    key: &str,
    series: &[String],
    max: f64,
    ticks: &[f64],
    lane_height: f32,
) -> AnyElement {
    let mut track = div().relative().flex_1().min_w_0().h(px(lane_height));
    for time in ticks {
        track = track.child(
            div()
                .absolute()
                .top_0()
                .h_full()
                .left(relative((time / max) as f32))
                .w(px(1.0))
                .bg(rgb(theme::BORDER)),
        );
    }
    for (bar_index, bar) in lane.children.iter().filter_map(node_child).enumerate() {
        let start = number(bar, "start");
        let end = number(bar, "end");
        let name = prop(bar, "series");
        let index = series_index(series, name);
        let left = (start / max).clamp(0.0, 1.0);
        let width = ((end - start) / max).clamp(0.0, 1.0 - left);
        let label = if name.is_empty() { "Bar" } else { name };
        let span = format!(
            "{label}: {}s-{}s ({}s)",
            format_value(start),
            format_value(end),
            format_value(end - start)
        );
        track = track.child(
            div()
                .id(SharedString::from(format!(
                    "{key}-lane{lane_index}-bar{bar_index}"
                )))
                .absolute()
                .top(px(LANE_PAD + index as f32 * BAR_PITCH))
                .h(px(BAR_HEIGHT))
                .left(relative(left as f32))
                .w(relative(width as f32))
                .rounded_sm()
                .bg(rgb(series_color(index)))
                .hover(|bar| bar.border_1().border_color(rgb(theme::TEXT)))
                .tooltip(tip_builder(Tip {
                    header: span,
                    rows: Vec::new(),
                })),
        );
    }
    div()
        .flex()
        .flex_row()
        .items_center()
        .border_t_1()
        .border_color(rgb(theme::BORDER))
        .hover(|lane| lane.bg(rgb(theme::BG_HOVER)))
        .child(
            div()
                .w(px(LANE_LABEL_WIDTH))
                .flex_none()
                .text_sm()
                .text_color(rgb(theme::TEXT))
                .child(SharedString::from(prop(lane, "label").to_owned())),
        )
        .child(track)
        .into_any_element()
}

/// Tick labels under the tracks, aligned with the grid lines above them.
fn axis_row(ticks: &[f64], max: f64) -> AnyElement {
    let mut track = div().relative().flex_1().min_w_0().h(px(16.0));
    for time in ticks {
        let fraction = (time / max) as f32;
        let label = div()
            .absolute()
            .top_0()
            .text_xs()
            .text_color(rgb(theme::TEXT_MUTED))
            .child(SharedString::from(format_time(*time)));
        // A label at the far end would run past the track, so pin it to the right edge.
        let label = if fraction >= 1.0 {
            label.right_0()
        } else {
            label.left(relative(fraction))
        };
        track = track.child(label);
    }
    div()
        .flex()
        .flex_row()
        .child(div().w(px(LANE_LABEL_WIDTH)).flex_none())
        .child(track)
        .into_any_element()
}

/// Seconds as `m:ss`, for example `90` as `1:30`.
fn format_time(seconds: f64) -> String {
    let total = seconds.round() as u64;
    format!("{}:{:02}", total / 60, total % 60)
}

/// How a `Chart` draws its series.
#[derive(Clone, Copy, PartialEq, Eq)]
enum ChartKind {
    Bar,
    Line,
    Area,
    Scatter,
}

/// A `Chart` the host can draw: its kind, optional x labels, and its series.
struct ChartData {
    kind: ChartKind,
    labels: Vec<String>,
    series: Vec<(String, Vec<f64>)>,
}

/// Hover tooltip body: a header line and one row per series.
#[derive(Clone)]
struct Tip {
    header: String,
    rows: Vec<TipRow>,
}

/// One series row in a `Tip`: a colored dot, the series name when it has one, and
/// the value.
#[derive(Clone)]
struct TipRow {
    color: u32,
    name: String,
    value: String,
}

impl Render for Tip {
    fn render(&mut self, _window: &mut Window, _cx: &mut Context<Self>) -> impl IntoElement {
        let mut body = div()
            .flex()
            .flex_col()
            .gap_1()
            .px_2()
            .py_1p5()
            .rounded_md()
            .border_1()
            .border_color(rgb(theme::BORDER_STRONG))
            .bg(rgb(theme::BG_RAISED))
            .text_xs()
            .child(
                div()
                    .font_weight(FontWeight::SEMIBOLD)
                    .text_color(rgb(theme::TEXT))
                    .child(SharedString::from(self.header.clone())),
            );
        for row in &self.rows {
            body = body.child(
                div()
                    .flex()
                    .flex_row()
                    .items_center()
                    .gap_1p5()
                    .child(div().size(px(6.0)).rounded_full().bg(rgb(row.color)))
                    .when(!row.name.is_empty(), |line| {
                        line.child(
                            div()
                                .text_color(rgb(theme::TEXT_MUTED))
                                .child(SharedString::from(row.name.clone())),
                        )
                    })
                    .child(
                        div()
                            .text_color(rgb(theme::TEXT))
                            .child(SharedString::from(row.value.clone())),
                    ),
            );
        }
        body
    }
}

/// A chart or timeline value for display: rounded to two decimals, half away from
/// zero, with trailing zeros and a bare decimal point dropped. `3` shows as `3`,
/// `2.50` as `2.5`, `0.125` as `0.13`, and `-0.001` as `0`.
fn format_value(value: f64) -> String {
    let rounded = (value * 100.0).round() / 100.0;
    if rounded == 0.0 {
        return "0".to_owned();
    }
    format!("{rounded:.2}")
        .trim_end_matches('0')
        .trim_end_matches('.')
        .to_owned()
}

/// Centre of slot `index` out of `points`, as a fraction of the plot width. Lines,
/// markers, and hover slots all use it, so they line up.
fn slot_center(index: usize, points: usize) -> f32 {
    (index as f32 + 0.5) / points as f32
}

/// Parses a `Chart` and its `Series` children. The validator rejects these
/// trees before they reach the host; this guards persisted sessions that skip it.
fn chart_data(node: &ViewNode) -> Result<ChartData, String> {
    let kind = match prop(node, "kind") {
        "bar" => ChartKind::Bar,
        "line" => ChartKind::Line,
        "area" => ChartKind::Area,
        "scatter" => ChartKind::Scatter,
        other => {
            return Err(format!(
                "<Chart> kind `{other}` is not bar, line, scatter, or area"
            ));
        }
    };
    let series = node
        .children
        .iter()
        .filter_map(node_child)
        .filter(|child| child.tag == "Series")
        .map(|child| {
            let values = parse_chart_values(prop(child, "values"))
                .ok_or_else(|| "<Series> values must be comma-separated numbers".to_owned())?;
            Ok((prop(child, "name").to_owned(), values))
        })
        .collect::<Result<Vec<_>, String>>()?;
    let points = series.first().map(|(_, values)| values.len()).unwrap_or(0);
    if series.is_empty() || series.iter().any(|(_, values)| values.len() != points) {
        return Err(
            "<Chart> needs at least one <Series>, all with the same number of values".into(),
        );
    }
    let labels = if prop(node, "labels").is_empty() {
        Vec::new()
    } else {
        parse_chart_labels(prop(node, "labels"))
    };
    if !labels.is_empty() && labels.len() != points {
        return Err("<Chart> labels must match the number of values".into());
    }
    Ok(ChartData {
        kind,
        labels,
        series,
    })
}

/// Titled chart: a legend when there are several series, the plot, and x labels.
/// Each chart is `flex_1` with a minimum width, so a `Row` puts two side by side
/// when they fit and wraps them otherwise.
fn chart(node: &ViewNode) -> AnyElement {
    let data = match chart_data(node) {
        Ok(data) => data,
        Err(message) => return fallback(message),
    };
    let (lo, hi) = value_domain(&data.series);
    let key = view_key(node);
    let mut card = div()
        .flex_1()
        .min_w(px(CHART_MIN_WIDTH))
        .flex()
        .flex_col()
        .gap_2();
    let title = prop(node, "title");
    if !title.is_empty() {
        card = card.child(
            div()
                .text_sm()
                .font_weight(FontWeight::MEDIUM)
                .text_color(rgb(theme::TEXT))
                .child(SharedString::from(title.to_owned())),
        );
    }
    if data.series.len() > 1 {
        let legend = data
            .series
            .iter()
            .enumerate()
            .map(|(s, (name, _))| legend_item(name, series_color(s)));
        card = card.child(div().flex().flex_row().flex_wrap().gap_3().children(legend));
    }
    let plot = match data.kind {
        ChartKind::Bar => bar_plot(&data.series, lo, hi),
        kind => curve_plot(kind, &data.series, lo, hi),
    };
    let curve = data.kind != ChartKind::Bar;
    card = card.child(hover_layer(
        plot,
        &data.series,
        &data.labels,
        lo,
        hi,
        &key,
        curve,
    ));
    if !data.labels.is_empty() {
        card = card.child(x_labels(&data.labels));
    }
    card.into_any_element()
}

/// Shared y range covering every value and zero, so bars rise from a baseline.
fn value_domain(series: &[(String, Vec<f64>)]) -> (f64, f64) {
    let (mut lo, mut hi) = (0.0_f64, 0.0_f64);
    for value in series.iter().flat_map(|(_, values)| values) {
        lo = lo.min(*value);
        hi = hi.max(*value);
    }
    if hi == lo {
        hi = lo + 1.0;
    }
    (lo, hi)
}

/// Distance from the top of the plot, as a fraction of its height, for `value`.
fn value_fraction(value: f64, lo: f64, hi: f64) -> f32 {
    ((hi - value) / (hi - lo)) as f32
}

/// Plot box with horizontal gridlines and a stronger baseline at `zero`, the
/// fraction of the height where value 0 sits.
fn plot_box(zero: f32) -> Div {
    let mut plot = div().relative().w_full().h(px(CHART_PLOT_HEIGHT));
    for step in 0..=CHART_GRID_STEPS {
        plot = plot.child(
            div()
                .absolute()
                .left_0()
                .right_0()
                .top(relative(step as f32 / CHART_GRID_STEPS as f32))
                .h(px(1.0))
                .bg(rgb(theme::BORDER)),
        );
    }
    plot.child(
        div()
            .absolute()
            .left_0()
            .right_0()
            .top(relative(zero))
            .h(px(1.0))
            .bg(rgb(theme::BORDER_STRONG)),
    )
}

/// Grouped columns: each x position holds one bar per series, centered in its
/// slot with a 20% margin on each side.
fn bar_plot(series: &[(String, Vec<f64>)], lo: f64, hi: f64) -> Div {
    let points = series[0].1.len();
    let slot = 1.0 / points as f32;
    let group = 0.6 * slot;
    let bar_width = group / series.len() as f32;
    let zero = value_fraction(0.0, lo, hi);
    let mut plot = plot_box(zero);
    for (s, (_, values)) in series.iter().enumerate() {
        for (i, value) in values.iter().enumerate() {
            let edge = value_fraction(*value, lo, hi);
            plot = plot.child(
                div()
                    .absolute()
                    .top(relative(edge.min(zero)))
                    .h(relative((edge - zero).abs()))
                    .left(relative(
                        i as f32 * slot + 0.2 * slot + s as f32 * bar_width,
                    ))
                    .w(relative(bar_width))
                    .bg(rgb(series_color(s))),
            );
        }
    }
    plot
}

/// Line, area, or scatter series drawn on a canvas laid over the plot. Points sit
/// at the centre of their slot, matching the bar and label positions.
fn curve_plot(kind: ChartKind, series: &[(String, Vec<f64>)], lo: f64, hi: f64) -> Div {
    let points = series[0].1.len();
    let zero = value_fraction(0.0, lo, hi);
    let curves: Vec<(u32, Vec<f32>)> = series
        .iter()
        .enumerate()
        .map(|(s, (_, values))| {
            let fractions = values
                .iter()
                .map(|value| value_fraction(*value, lo, hi))
                .collect();
            (series_color(s), fractions)
        })
        .collect();
    let overlay = canvas(
        |_, _, _| (),
        move |bounds, (), window, _cx| {
            for (color, fractions) in &curves {
                let dots: Vec<Point<Pixels>> = fractions
                    .iter()
                    .enumerate()
                    .map(|(i, fraction)| {
                        point(
                            bounds.origin.x + bounds.size.width * slot_center(i, points),
                            bounds.origin.y + bounds.size.height * *fraction,
                        )
                    })
                    .collect();
                match kind {
                    ChartKind::Scatter => {
                        let radius = px(MARKER_SIZE / 2.0);
                        for dot in &dots {
                            let dot_box = Bounds::new(
                                point(dot.x - radius, dot.y - radius),
                                size(px(MARKER_SIZE), px(MARKER_SIZE)),
                            );
                            window.paint_quad(
                                fill(dot_box, rgb(*color)).corner_radii(Corners::all(radius)),
                            );
                        }
                    }
                    ChartKind::Area => {
                        let baseline = bounds.origin.y + bounds.size.height * zero;
                        fill_area(window, &dots, baseline, *color);
                        stroke_line(window, &dots, *color);
                    }
                    ChartKind::Line => stroke_line(window, &dots, *color),
                    // Columns are drawn by `bar_plot`, not on this canvas.
                    ChartKind::Bar => {}
                }
            }
        },
    );
    plot_box(zero).child(
        div()
            .absolute()
            .top_0()
            .left_0()
            .w_full()
            .h(px(CHART_PLOT_HEIGHT))
            .child(overlay.size_full()),
    )
}

/// Strokes a polyline through `dots`. A single point gets a short dash so it stays
/// visible.
fn stroke_line(window: &mut Window, dots: &[Point<Pixels>], color: u32) {
    let mut builder = PathBuilder::stroke(px(LINE_WIDTH));
    match dots {
        [] => return,
        [only] => {
            builder.move_to(*only);
            builder.line_to(*only + point(px(LINE_WIDTH * 2.0), px(0.0)));
        }
        [first, rest @ ..] => {
            builder.move_to(*first);
            for dot in rest {
                builder.line_to(*dot);
            }
        }
    }
    if let Ok(path) = builder.build() {
        window.paint_path(path, rgb(color));
    }
}

/// Fills the region between the curve through `dots` and the `baseline` row.
fn fill_area(window: &mut Window, dots: &[Point<Pixels>], baseline: Pixels, color: u32) {
    let (Some(first), Some(last)) = (dots.first(), dots.last()) else {
        return;
    };
    let mut builder = PathBuilder::fill();
    builder.move_to(point(first.x, baseline));
    for dot in dots {
        builder.line_to(*dot);
    }
    builder.line_to(point(last.x, baseline));
    builder.close();
    if let Ok(path) = builder.build() {
        window.paint_path(path, tint(color, AREA_FILL_ALPHA));
    }
}

/// Transparent hover slots, one per x position, laid over the plot. Hovering a slot
/// tints its column and opens a tooltip with each series' value there. Curve charts
/// also show a guide line and a marker on each series. Every slot is its own group,
/// so only its own guide and markers react.
fn hover_layer(
    mut plot: Div,
    series: &[(String, Vec<f64>)],
    labels: &[String],
    lo: f64,
    hi: f64,
    key: &str,
    curve: bool,
) -> Div {
    let points = series[0].1.len();
    for i in 0..points {
        let slot_name = format!("{key}-{i}");
        let tip = Tip {
            header: labels
                .get(i)
                .cloned()
                .unwrap_or_else(|| (i + 1).to_string()),
            rows: series
                .iter()
                .enumerate()
                .map(|(s, (name, values))| TipRow {
                    color: series_color(s),
                    name: name.clone(),
                    value: format_value(values[i]),
                })
                .collect(),
        };
        let mut slot = div()
            .id(SharedString::from(slot_name.clone()))
            .group(SharedString::from(slot_name.clone()))
            .absolute()
            .top_0()
            .h_full()
            .left(relative(i as f32 / points as f32))
            .w(relative(1.0 / points as f32))
            .hover(|slot| slot.bg(tint(theme::TEXT, HOVER_BAND_ALPHA)))
            .tooltip(tip_builder(tip));
        if curve {
            slot = slot.child(
                div()
                    .absolute()
                    .top_0()
                    .h_full()
                    .left(relative(0.5))
                    .w(px(1.0))
                    .group_hover(slot_name.clone(), |guide| guide.bg(rgb(theme::TEXT_FAINT))),
            );
            for (s, (_, values)) in series.iter().enumerate() {
                let top = value_fraction(values[i], lo, hi) * CHART_PLOT_HEIGHT - MARKER_SIZE / 2.0;
                slot = slot.child(
                    div()
                        .absolute()
                        .top(px(top))
                        .left(relative(0.5))
                        .ml(px(-MARKER_SIZE / 2.0))
                        .size(px(MARKER_SIZE))
                        .rounded_full()
                        .group_hover(slot_name.clone(), move |marker| {
                            marker.bg(rgb(series_color(s)))
                        }),
                );
            }
        }
        plot = plot.child(slot);
    }
    plot
}

/// `color` at `alpha`, for translucent fills over the plot.
fn tint(color: u32, alpha: f32) -> Hsla {
    Hsla::from(rgb(color)).alpha(alpha)
}

/// Tooltip builder that opens `tip` in a fresh view each time it is shown.
fn tip_builder(tip: Tip) -> impl Fn(&mut Window, &mut App) -> AnyView + 'static {
    move |_window, cx| cx.new(|_| tip.clone()).into()
}

/// Element-id prefix for one `Chart` or `Timeline`, derived from its contents so
/// hover state stays attached across frames. Two identical trees in one message
/// share it, so their hover state can overlap.
fn view_key(node: &ViewNode) -> String {
    let mut hasher = DefaultHasher::new();
    format!("{node:?}").hash(&mut hasher);
    format!("view-{:x}", hasher.finish())
}

/// Labels under the plot, one per x position and centered in its slot.
fn x_labels(labels: &[String]) -> AnyElement {
    let points = labels.len() as f32;
    let mut row = div().relative().w_full().h(px(16.0));
    for (i, label) in labels.iter().enumerate() {
        row = row.child(
            div()
                .absolute()
                .top_0()
                .left(relative(i as f32 / points))
                .w(relative(1.0 / points))
                .text_xs()
                .text_center()
                .truncate()
                .text_color(rgb(theme::TEXT_MUTED))
                .child(SharedString::from(label.clone())),
        );
    }
    row.into_any_element()
}

/// Legend entry: a series-colored dot and the series name.
fn legend_item(name: &str, color: u32) -> AnyElement {
    div()
        .flex()
        .flex_row()
        .items_center()
        .gap_1()
        .text_xs()
        .text_color(rgb(theme::TEXT_MUTED))
        .child(div().size(px(6.0)).rounded_full().bg(rgb(color)))
        .child(SharedString::from(name.to_owned()))
        .into_any_element()
}

#[cfg(test)]
mod tests {
    use juto_agent::MAX_CHART_POINTS;

    use super::*;

    #[test]
    fn format_value_rounds_to_two_decimals_and_trims() {
        assert_eq!(format_value(3.0), "3");
        assert_eq!(format_value(2.5), "2.5");
        assert_eq!(format_value(0.125), "0.13");
        assert_eq!(format_value(-2.0), "-2");
        assert_eq!(format_value(100.0), "100");
        assert_eq!(format_value(1234.5678), "1234.57");
    }

    #[test]
    fn format_value_never_shows_negative_zero() {
        assert_eq!(format_value(-0.001), "0");
        assert_eq!(format_value(-0.0), "0");
    }

    #[test]
    fn slot_centers_fall_inside_their_own_slot() {
        for points in 1..=MAX_CHART_POINTS {
            for index in 0..points {
                let center = slot_center(index, points);
                assert_eq!(
                    (center * points as f32).floor() as usize,
                    index,
                    "slot {index} of {points}"
                );
            }
        }
    }
}
