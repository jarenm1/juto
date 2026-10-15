//! Maps `juto_agent::ViewNode` trees to GPUI elements. Each `tag` selects a host
//! component; an unknown tag renders a visible fallback that names the tag, so
//! a bad tree is never silently dropped.
//!
//! The look follows shadcn/ui primitives (Card, destructive Alert) on the shared
//! palette in `theme.rs`; `Badge` and subagent rows are flat status marks. The tags
//! and props agents may use are fixed by `juto_agent::VIEW_TAGS`; this file only
//! decides how they look.

use gpui::{
    AnyElement, Div, FontWeight, InteractiveElement, IntoElement, ParentElement, PathBuilder,
    Pixels, Point, SharedString, Styled, canvas, div, point, px, relative, rgb,
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

/// Renders one element and its subtree.
pub fn render_view(node: &ViewNode) -> AnyElement {
    match node.tag.as_str() {
        "Column" => div()
            .flex()
            .flex_col()
            .gap_3()
            .children(column_children(node))
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
            .text_color(rgb(theme::TEXT))
            .children(render_children(node))
            .into_any_element(),
        "Badge" => badge(prop(node, "label"), prop(node, "tone")),
        "Stats" => stats(node),
        "Stat" => stat(prop(node, "label"), prop(node, "value")),
        "Timeline" => timeline(node),
        "Chart" => chart(node),
        "Subagent" => subagent_list(vec![node]),
        unknown => fallback(format!("Unknown element <{unknown}>")),
    }
}

fn render_children(node: &ViewNode) -> Vec<AnyElement> {
    node.children.iter().map(render_child).collect()
}

fn render_child(child: &ViewChild) -> AnyElement {
    match child {
        ViewChild::Text(text) => SharedString::from(text.clone()).into_any_element(),
        ViewChild::Node(node) => render_view(node),
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

/// Column children. Each run of adjacent `Subagent` nodes becomes one list, so
/// the rows sit flush with hairline dividers instead of gapped cards.
fn column_children(node: &ViewNode) -> Vec<AnyElement> {
    let mut out = Vec::new();
    let mut run: Vec<&ViewNode> = Vec::new();
    for child in &node.children {
        match node_child(child) {
            Some(subagent) if subagent.tag == "Subagent" => run.push(subagent),
            _ => {
                if !run.is_empty() {
                    out.push(subagent_list(std::mem::take(&mut run)));
                }
                out.push(render_child(child));
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
fn stats(node: &ViewNode) -> AnyElement {
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
        strip = strip.child(div().flex_1().min_w_0().child(render_view(cell)));
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

    let mut body = div().flex().flex_col();
    for lane in &lanes {
        body = body.child(lane_row(lane, &series, max, &ticks, lane_height));
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
/// stack by series, one row per series.
fn lane_row(
    lane: &ViewNode,
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
    for bar in lane.children.iter().filter_map(node_child) {
        let start = number(bar, "start");
        let end = number(bar, "end");
        let index = series_index(series, prop(bar, "series"));
        let left = (start / max).clamp(0.0, 1.0);
        let width = ((end - start) / max).clamp(0.0, 1.0 - left);
        track = track.child(
            div()
                .absolute()
                .top(px(LANE_PAD + index as f32 * BAR_PITCH))
                .h(px(BAR_HEIGHT))
                .left(relative(left as f32))
                .w(relative(width as f32))
                .rounded_sm()
                .bg(rgb(series_color(index))),
        );
    }
    div()
        .flex()
        .flex_row()
        .items_center()
        .border_t_1()
        .border_color(rgb(theme::BORDER))
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

/// A `Chart` the host can draw: bar or line, optional x labels, and its series.
struct ChartData {
    line: bool,
    labels: Vec<String>,
    series: Vec<(String, Vec<f64>)>,
}

/// Parses a `Chart` and its `Series` children. The validator rejects these
/// trees before they reach the host; this guards persisted sessions that skip it.
fn chart_data(node: &ViewNode) -> Result<ChartData, String> {
    let line = match prop(node, "kind") {
        "bar" => false,
        "line" => true,
        other => return Err(format!("<Chart> kind `{other}` is not bar or line")),
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
        line,
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
    card = card.child(if data.line {
        line_plot(&data.series, lo, hi)
    } else {
        bar_plot(&data.series, lo, hi)
    });
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
fn bar_plot(series: &[(String, Vec<f64>)], lo: f64, hi: f64) -> AnyElement {
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
    plot.into_any_element()
}

/// One polyline per series, stroked on a canvas laid over the plot. Points sit at
/// the centre of their slot, matching the bar and label positions.
fn line_plot(series: &[(String, Vec<f64>)], lo: f64, hi: f64) -> AnyElement {
    let points = series[0].1.len();
    let lines: Vec<(u32, Vec<f32>)> = series
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
            for (color, fractions) in &lines {
                let mut dots: Vec<Point<Pixels>> = fractions
                    .iter()
                    .enumerate()
                    .map(|(i, fraction)| {
                        point(
                            bounds.origin.x
                                + bounds.size.width * ((i as f32 + 0.5) / points as f32),
                            bounds.origin.y + bounds.size.height * *fraction,
                        )
                    })
                    .collect();
                // A single point has no segment; a short dash marks it.
                if dots.len() == 1 {
                    dots.push(dots[0] + point(px(LINE_WIDTH * 2.0), px(0.0)));
                }
                let Some((first, rest)) = dots.split_first() else {
                    continue;
                };
                let mut builder = PathBuilder::stroke(px(LINE_WIDTH));
                builder.move_to(*first);
                for dot in rest {
                    builder.line_to(*dot);
                }
                if let Ok(path) = builder.build() {
                    window.paint_path(path, rgb(*color));
                }
            }
        },
    );
    plot_box(value_fraction(0.0, lo, hi))
        .child(
            div()
                .absolute()
                .top_0()
                .left_0()
                .w_full()
                .h(px(CHART_PLOT_HEIGHT))
                .child(overlay.size_full()),
        )
        .into_any_element()
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
