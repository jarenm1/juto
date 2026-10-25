//! Declarative view trees that tools hand to the host. A tool attaches one under
//! `ToolOutput::details` as `{"type": "view", "view": <node>}`; the host maps each
//! `tag` to a component. The model never sees the tree; it reads `content`.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};

/// `details["type"]` value that marks a view tree.
pub const VIEW_DETAILS_TYPE: &str = "view";

/// One element: a component `tag`, its `props`, and its ordered `children`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ViewNode {
    pub tag: String,
    #[serde(default, skip_serializing_if = "Map::is_empty")]
    pub props: Map<String, Value>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub children: Vec<ViewChild>,
}

/// A child is either inline text or a nested element.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ViewChild {
    Text(String),
    Node(ViewNode),
}

/// Builds the `details` value that carries `view` to the host.
pub fn view_details(view: ViewNode) -> Value {
    json!({"type": VIEW_DETAILS_TYPE, "view": view})
}

/// Reads a view tree from `details`. Returns `None` for other detail kinds and
/// for trees that do not decode.
pub fn view_from_details(details: &Value) -> Option<ViewNode> {
    if details.get("type").and_then(Value::as_str) != Some(VIEW_DETAILS_TYPE) {
        return None;
    }
    serde_json::from_value(details.get("view")?.clone()).ok()
}

/// Component tags the host renders, each with the only props it accepts. Container
/// tags accept only the child tag listed in `NESTING`. The host renderer
/// (`apps/juto/src/view_element.rs`) must handle every tag listed here.
pub const VIEW_TAGS: &[(&str, &[&str])] = &[
    ("Column", &[]),
    ("Row", &[]),
    ("Text", &[]),
    ("Badge", &["label", "tone"]),
    ("Stats", &[]),
    ("Stat", &["label", "value"]),
    ("Timeline", &["title", "max", "tick"]),
    ("Lane", &["label"]),
    ("Bar", &["start", "end", "series"]),
    ("Chart", &["kind", "title", "labels"]),
    ("Series", &["name", "values"]),
    ("Button", &["label", "prompt", "tone"]),
    ("Quiz", &["question", "prompt"]),
    ("Choice", &["correct"]),
];

/// Tones a `Badge` accepts. The renderer shows any other tone as neutral, so the
/// validator rejects it instead.
pub const BADGE_TONES: &[&str] = &["success", "warning", "error", "info"];

/// Tones a `Button` accepts. The renderer shows the tone as its style; `primary`
/// is the accent, `danger` the destructive style, and `neutral` the default.
pub const BUTTON_TONES: &[&str] = &["primary", "neutral", "danger"];

/// Deepest allowed nesting, counting the root as depth 1.
pub const MAX_VIEW_DEPTH: usize = 8;
/// Most elements allowed in one tree.
pub const MAX_VIEW_NODES: usize = 200;
/// Longest allowed text child or prop string, in bytes.
pub const MAX_VIEW_TEXT_BYTES: usize = 4096;
/// Most tick marks a `Timeline` may draw.
pub const MAX_TIMELINE_TICKS: usize = 48;

/// Most series one `Chart` may draw.
pub const MAX_CHART_SERIES: usize = 4;
/// Most values one `Series` may hold, and so most x positions one `Chart` may draw.
pub const MAX_CHART_POINTS: usize = 48;
/// Kinds a `Chart` accepts.
pub const CHART_KINDS: &[&str] = &["bar", "line", "scatter", "area"];

/// Fewest and most `Choice` options a `Quiz` may offer.
const MIN_QUIZ_CHOICES: usize = 2;
const MAX_QUIZ_CHOICES: usize = 5;
/// Option texts that make a quiz answer obvious, compared case-insensitively.
const BANNED_QUIZ_OPTIONS: &[&str] = &["all of the above", "none of the above"];

/// Tags that exist only inside another tag, as `(parent, child)` pairs. A parent
/// accepts only that child, and the child must sit directly under the parent.
const NESTING: &[(&str, &str)] = &[
    ("Stats", "Stat"),
    ("Timeline", "Lane"),
    ("Lane", "Bar"),
    ("Chart", "Series"),
    ("Quiz", "Choice"),
];

/// Checks a tree before the host renders it. Errors name the offending tag or prop
/// so the model can correct the tree.
pub fn validate_view(root: &ViewNode) -> Result<(), String> {
    let mut count = 0;
    validate_node(root, None, 1, &mut count)
}

fn validate_node(
    node: &ViewNode,
    parent: Option<&str>,
    depth: usize,
    count: &mut usize,
) -> Result<(), String> {
    if depth > MAX_VIEW_DEPTH {
        return Err(format!(
            "view is nested deeper than {MAX_VIEW_DEPTH} levels"
        ));
    }
    *count += 1;
    if *count > MAX_VIEW_NODES {
        return Err(format!("view has more than {MAX_VIEW_NODES} elements"));
    }
    let Some((_, allowed)) = VIEW_TAGS.iter().find(|(tag, _)| *tag == node.tag) else {
        let known: Vec<&str> = VIEW_TAGS.iter().map(|(tag, _)| *tag).collect();
        return Err(format!(
            "unknown element <{}>; allowed tags: {}",
            node.tag,
            known.join(", ")
        ));
    };
    for (key, value) in &node.props {
        if !allowed.contains(&key.as_str()) {
            return Err(format!("<{}> does not accept prop `{key}`", node.tag));
        }
        let Some(text) = value.as_str() else {
            return Err(format!("prop `{key}` on <{}> must be a string", node.tag));
        };
        check_text(text)?;
    }
    if let Some((required_parent, _)) = NESTING.iter().find(|(_, child)| *child == node.tag)
        && parent != Some(required_parent)
    {
        return Err(format!("<{}> must be inside <{required_parent}>", node.tag));
    }
    match node.tag.as_str() {
        "Badge" => check_badge(node)?,
        "Stat" => {
            required_text(node, "label")?;
            required_text(node, "value")?;
        }
        "Lane" => {
            required_text(node, "label")?;
        }
        "Timeline" => check_timeline(node)?,
        "Bar" => check_bar(node)?,
        "Chart" => check_chart(node)?,
        "Series" => check_series(node)?,
        "Button" => check_button(node)?,
        "Quiz" => check_quiz(node)?,
        "Choice" => check_choice(node)?,
        _ => {}
    }
    if matches!(node.tag.as_str(), "Stat" | "Bar" | "Series" | "Button")
        && !node.children.is_empty()
    {
        return Err(format!("<{}> takes no children", node.tag));
    }
    let required_child = NESTING
        .iter()
        .find(|(parent, _)| *parent == node.tag)
        .map(|(_, child)| *child);
    for child in &node.children {
        match child {
            ViewChild::Text(text) => {
                check_text(text)?;
                if let Some(required) = required_child {
                    return Err(format!("<{}> accepts only <{required}> children", node.tag));
                }
            }
            ViewChild::Node(child) => {
                if let Some(required) = required_child
                    && child.tag != required
                {
                    return Err(format!("<{}> accepts only <{required}> children", node.tag));
                }
                validate_node(child, Some(&node.tag), depth + 1, count)?;
            }
        }
    }
    Ok(())
}

fn check_badge(node: &ViewNode) -> Result<(), String> {
    required_text(node, "label")?;
    if let Some(tone) = node.props.get("tone").and_then(Value::as_str)
        && !BADGE_TONES.contains(&tone)
    {
        return Err(format!(
            "<Badge> tone must be one of: {}",
            BADGE_TONES.join(", ")
        ));
    }
    Ok(())
}

fn check_timeline(node: &ViewNode) -> Result<(), String> {
    let max = positive_number(node, "max")?;
    let tick = positive_number(node, "tick")?;
    if max / tick > MAX_TIMELINE_TICKS as f64 {
        return Err(format!(
            "<Timeline> tick is too small: at most {MAX_TIMELINE_TICKS} ticks"
        ));
    }
    for lane in node.children.iter().filter_map(child_node) {
        for bar in lane.children.iter().filter_map(child_node) {
            let end = bar
                .props
                .get("end")
                .and_then(Value::as_str)
                .and_then(|text| text.parse::<f64>().ok());
            if end.is_some_and(|end| end > max) {
                return Err(format!("<Bar> end is past <Timeline> max {max}"));
            }
        }
    }
    Ok(())
}

fn check_bar(node: &ViewNode) -> Result<(), String> {
    let start = number_prop(node, "start")?;
    let end = number_prop(node, "end")?;
    if start < 0.0 || end <= start {
        return Err("<Bar> needs 0 <= start < end".into());
    }
    Ok(())
}

fn check_button(node: &ViewNode) -> Result<(), String> {
    for key in ["label", "prompt"] {
        if required_text(node, key)?.trim().is_empty() {
            return Err(format!("<Button> prop `{key}` must not be empty"));
        }
    }
    if let Some(tone) = node.props.get("tone").and_then(Value::as_str)
        && !BUTTON_TONES.contains(&tone)
    {
        return Err(format!(
            "<Button> tone must be one of: {}",
            BUTTON_TONES.join(", ")
        ));
    }
    Ok(())
}

fn check_quiz(node: &ViewNode) -> Result<(), String> {
    for key in ["question", "prompt"] {
        if required_text(node, key)?.trim().is_empty() {
            return Err(format!("<Quiz> prop `{key}` must not be empty"));
        }
    }
    let choices: Vec<&ViewNode> = node.children.iter().filter_map(child_node).collect();
    if !(MIN_QUIZ_CHOICES..=MAX_QUIZ_CHOICES).contains(&choices.len()) {
        return Err(format!(
            "<Quiz> needs {MIN_QUIZ_CHOICES} to {MAX_QUIZ_CHOICES} <Choice> children"
        ));
    }
    let correct = choices
        .iter()
        .filter(|choice| choice.props.get("correct").and_then(Value::as_str) == Some("true"))
        .count();
    if correct != 1 {
        return Err("<Quiz> needs exactly one <Choice> with correct=\"true\"".into());
    }
    let mut seen: Vec<String> = Vec::new();
    for text in choices.iter().filter_map(|choice| choice_text(choice)) {
        let normalized = text.to_lowercase();
        if BANNED_QUIZ_OPTIONS.contains(&normalized.as_str()) {
            return Err(format!("<Quiz> option \"{text}\" makes the answer obvious"));
        }
        if seen.contains(&normalized) {
            return Err(format!("<Quiz> options must differ; \"{text}\" repeats"));
        }
        seen.push(normalized);
    }
    Ok(())
}

fn check_choice(node: &ViewNode) -> Result<(), String> {
    match required_text(node, "correct")? {
        "true" | "false" => {}
        _ => return Err("<Choice> prop `correct` must be \"true\" or \"false\"".into()),
    }
    if choice_text(node).is_none() {
        return Err("<Choice> needs exactly one non-empty text child".into());
    }
    Ok(())
}

/// A `Choice`'s text, when it holds exactly one non-blank text child.
fn choice_text(node: &ViewNode) -> Option<&str> {
    match node.children.as_slice() {
        [ViewChild::Text(text)] if !text.trim().is_empty() => Some(text.trim()),
        _ => None,
    }
}

/// Parses a `Series` `values` prop: comma-separated finite numbers, or `None`.
pub fn parse_chart_values(text: &str) -> Option<Vec<f64>> {
    text.split(',')
        .map(|part| part.trim().parse::<f64>().ok().filter(|n| n.is_finite()))
        .collect()
}

/// Parses a `Chart` `labels` prop: comma-separated x-axis labels.
pub fn parse_chart_labels(text: &str) -> Vec<String> {
    text.split(',').map(|part| part.trim().to_owned()).collect()
}

/// A `Series` node's parsed values.
fn series_values(node: &ViewNode) -> Result<Vec<f64>, String> {
    parse_chart_values(required_text(node, "values")?)
        .ok_or_else(|| "<Series> values must be comma-separated numbers".to_owned())
}

fn check_series(node: &ViewNode) -> Result<(), String> {
    required_text(node, "name")?;
    if series_values(node)?.len() > MAX_CHART_POINTS {
        return Err(format!("<Series> needs 1 to {MAX_CHART_POINTS} values"));
    }
    Ok(())
}

fn check_chart(node: &ViewNode) -> Result<(), String> {
    let kind = required_text(node, "kind")?;
    if !CHART_KINDS.contains(&kind) {
        return Err(format!(
            "<Chart> kind must be one of: {}",
            CHART_KINDS.join(", ")
        ));
    }
    let series: Vec<&ViewNode> = node.children.iter().filter_map(child_node).collect();
    if series.is_empty() || series.len() > MAX_CHART_SERIES {
        return Err(format!("<Chart> needs 1 to {MAX_CHART_SERIES} <Series>"));
    }
    let mut points = None;
    for item in &series {
        let count = series_values(item)?.len();
        if points.is_some_and(|expected| expected != count) {
            return Err("every <Series> in a <Chart> needs the same number of values".into());
        }
        points = Some(count);
    }
    if let Some(labels) = node.props.get("labels").and_then(Value::as_str)
        && Some(parse_chart_labels(labels).len()) != points
    {
        return Err("<Chart> labels must match the number of values".into());
    }
    Ok(())
}

fn child_node(child: &ViewChild) -> Option<&ViewNode> {
    match child {
        ViewChild::Node(node) => Some(node),
        ViewChild::Text(_) => None,
    }
}

/// A required string prop.
fn required_text<'a>(node: &'a ViewNode, key: &str) -> Result<&'a str, String> {
    node.props
        .get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("<{}> requires a string prop `{key}`", node.tag))
}

/// A required prop holding a finite decimal number, sent as a string.
fn number_prop(node: &ViewNode, key: &str) -> Result<f64, String> {
    required_text(node, key)?
        .parse::<f64>()
        .ok()
        .filter(|number| number.is_finite())
        .ok_or_else(|| format!("<{}> prop `{key}` must be a number", node.tag))
}

/// A required prop holding a finite number greater than zero.
fn positive_number(node: &ViewNode, key: &str) -> Result<f64, String> {
    let number = number_prop(node, key)?;
    if number <= 0.0 {
        return Err(format!(
            "<{}> prop `{key}` must be greater than zero",
            node.tag
        ));
    }
    Ok(number)
}

fn check_text(text: &str) -> Result<(), String> {
    if text.len() > MAX_VIEW_TEXT_BYTES {
        return Err(format!("text exceeds {MAX_VIEW_TEXT_BYTES} bytes"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn card() -> ViewNode {
        ViewNode {
            tag: "Column".into(),
            props: Map::new(),
            children: vec![
                ViewChild::Text("hello".into()),
                ViewChild::Node(ViewNode {
                    tag: "Badge".into(),
                    props: json!({"label": "done"}).as_object().unwrap().clone(),
                    children: Vec::new(),
                }),
            ],
        }
    }

    #[test]
    fn view_survives_details_round_trip() {
        let details = view_details(card());
        assert_eq!(view_from_details(&details), Some(card()));
    }

    #[test]
    fn other_detail_kinds_carry_no_view() {
        let artifact = json!({"type": "desktop_session_artifact", "artifact": {}});
        assert_eq!(view_from_details(&artifact), None);
        assert_eq!(view_from_details(&Value::Null), None);
    }

    #[test]
    fn malformed_view_is_not_decoded() {
        let details = json!({"type": "view", "view": {"tag": "Text", "children": [42]}});
        assert_eq!(view_from_details(&details), None);
    }

    fn leaf(tag: &str) -> ViewNode {
        ViewNode {
            tag: tag.into(),
            props: Map::new(),
            children: Vec::new(),
        }
    }

    fn with_props(tag: &str, props: Value) -> ViewNode {
        ViewNode {
            tag: tag.into(),
            props: props.as_object().unwrap().clone(),
            children: Vec::new(),
        }
    }

    fn nested(levels: usize) -> ViewNode {
        let mut node = leaf("Text");
        for _ in 1..levels {
            node = ViewNode {
                tag: "Column".into(),
                props: Map::new(),
                children: vec![ViewChild::Node(node)],
            };
        }
        node
    }

    fn flat(children: usize) -> ViewNode {
        ViewNode {
            tag: "Column".into(),
            props: Map::new(),
            children: (0..children)
                .map(|_| ViewChild::Node(leaf("Text")))
                .collect(),
        }
    }

    #[test]
    fn validator_accepts_renderable_tree() {
        assert_eq!(validate_view(&card()), Ok(()));
    }

    #[test]
    fn validator_rejects_unknown_tag_and_lists_allowed_tags() {
        let err = validate_view(&leaf("Gizmo")).unwrap_err();
        assert!(err.contains("<Gizmo>"), "{err}");
        assert!(err.contains("Badge"), "{err}");
    }

    #[test]
    fn validator_rejects_props_the_tag_does_not_accept() {
        let err = validate_view(&with_props("Column", json!({"label": "x"}))).unwrap_err();
        assert!(err.contains("does not accept prop `label`"), "{err}");
    }

    #[test]
    fn validator_requires_badge_label_and_known_tone() {
        assert!(validate_view(&leaf("Badge")).is_err());
        let bad_tone = with_props("Badge", json!({"label": "x", "tone": "purple"}));
        let err = validate_view(&bad_tone).unwrap_err();
        assert!(err.contains("tone must be one of"), "{err}");
        let good = with_props("Badge", json!({"label": "x", "tone": "info"}));
        assert_eq!(validate_view(&good), Ok(()));
    }

    #[test]
    fn validator_bounds_depth_and_element_count() {
        assert_eq!(validate_view(&nested(MAX_VIEW_DEPTH)), Ok(()));
        assert!(validate_view(&nested(MAX_VIEW_DEPTH + 1)).is_err());
        assert_eq!(validate_view(&flat(MAX_VIEW_NODES - 1)), Ok(()));
        assert!(validate_view(&flat(MAX_VIEW_NODES)).is_err());
    }

    #[test]
    fn validator_bounds_text_length() {
        let node = ViewNode {
            tag: "Text".into(),
            props: Map::new(),
            children: vec![ViewChild::Text("x".repeat(MAX_VIEW_TEXT_BYTES + 1))],
        };
        assert!(validate_view(&node).is_err());
    }

    fn parse(value: Value) -> ViewNode {
        serde_json::from_value(value).expect("test tree decodes")
    }

    #[test]
    fn validator_accepts_stat_strip_and_timeline() {
        let tree = parse(json!({"tag": "Column", "children": [
            {"tag": "Stats", "children": [
                {"tag": "Stat", "props": {"label": "Steps", "value": "2,400"}},
                {"tag": "Stat", "props": {"label": "Errors", "value": "0"}}
            ]},
            {"tag": "Timeline", "props": {"title": "Build", "max": "180", "tick": "30"},
                "children": [
                {"tag": "Lane", "props": {"label": "Setup"}, "children": [
                    {"tag": "Bar", "props": {"start": "0", "end": "12", "series": "Latest"}},
                    {"tag": "Bar", "props": {"start": "2", "end": "20", "series": "Previous"}}
                ]}
            ]}
        ]}));
        assert_eq!(validate_view(&tree), Ok(()));
    }

    #[test]
    fn validator_places_stat_lane_and_bar_inside_their_container() {
        let stat = parse(json!({"tag": "Stat", "props": {"label": "a", "value": "1"}}));
        assert!(
            validate_view(&stat)
                .unwrap_err()
                .contains("must be inside <Stats>")
        );
        let bar = parse(json!({"tag": "Column", "children": [
            {"tag": "Bar", "props": {"start": "0", "end": "1"}}
        ]}));
        assert!(
            validate_view(&bar)
                .unwrap_err()
                .contains("must be inside <Lane>")
        );
    }

    #[test]
    fn validator_limits_containers_to_their_child_tag() {
        let stats = parse(json!({"tag": "Stats", "children": ["loose text"]}));
        assert!(
            validate_view(&stats)
                .unwrap_err()
                .contains("accepts only <Stat>")
        );
        let timeline = parse(
            json!({"tag": "Timeline", "props": {"max": "10", "tick": "5"},
            "children": [{"tag": "Badge", "props": {"label": "x"}}]}),
        );
        assert!(
            validate_view(&timeline)
                .unwrap_err()
                .contains("accepts only <Lane>")
        );
    }

    #[test]
    fn validator_rejects_bars_past_timeline_max_or_backwards() {
        let past_end = parse(
            json!({"tag": "Timeline", "props": {"max": "60", "tick": "10"},
            "children": [{"tag": "Lane", "props": {"label": "Build"},
                "children": [{"tag": "Bar", "props": {"start": "50", "end": "90"}}]}]}),
        );
        assert!(
            validate_view(&past_end)
                .unwrap_err()
                .contains("past <Timeline> max")
        );
        let backwards = parse(
            json!({"tag": "Timeline", "props": {"max": "60", "tick": "10"},
            "children": [{"tag": "Lane", "props": {"label": "Build"},
                "children": [{"tag": "Bar", "props": {"start": "9", "end": "3"}}]}]}),
        );
        assert!(
            validate_view(&backwards)
                .unwrap_err()
                .contains("0 <= start < end")
        );
    }

    #[test]
    fn validator_requires_numeric_timeline_props_and_bounded_ticks() {
        let no_tick = parse(json!({"tag": "Timeline", "props": {"max": "60"}}));
        assert!(validate_view(&no_tick).is_err());
        let text_max = parse(json!({"tag": "Timeline", "props": {"max": "soon", "tick": "5"}}));
        assert!(
            validate_view(&text_max)
                .unwrap_err()
                .contains("must be a number")
        );
        let too_many = parse(json!({"tag": "Timeline", "props": {"max": "1000", "tick": "1"}}));
        assert!(validate_view(&too_many).unwrap_err().contains("ticks"));
    }

    #[test]
    fn validator_accepts_aligned_bar_and_line_charts() {
        let bar = parse(
            json!({"tag": "Chart", "props": {"kind": "bar", "labels": "Mon, Tue,Wed"},
            "children": [
                {"tag": "Series", "props": {"name": "Steps", "values": "1, 2.5,3"}},
                {"tag": "Series", "props": {"name": "Errors", "values": "0,-1,4"}}
            ]}),
        );
        assert_eq!(validate_view(&bar), Ok(()));
        let line = parse(json!({"tag": "Chart", "props": {"kind": "line"},
            "children": [{"tag": "Series", "props": {"name": "Reward", "values": "0.1,0.4"}}]}));
        assert_eq!(validate_view(&line), Ok(()));
    }

    #[test]
    fn validator_accepts_scatter_and_area_charts() {
        for kind in ["scatter", "area"] {
            let node = parse(json!({"tag": "Chart", "props": {"kind": kind},
                "children": [{"tag": "Series", "props": {"name": "Latency", "values": "3,1,2.5"}}]}));
            assert_eq!(validate_view(&node), Ok(()), "{kind}");
        }
    }

    #[test]
    fn validator_rejects_unknown_chart_kind_and_missing_series() {
        let pie = parse(json!({"tag": "Chart", "props": {"kind": "pie"},
            "children": [{"tag": "Series", "props": {"name": "x", "values": "1"}}]}));
        assert!(
            validate_view(&pie)
                .unwrap_err()
                .contains("kind must be one of: bar, line, scatter, area")
        );
        let empty = parse(json!({"tag": "Chart", "props": {"kind": "bar"}}));
        assert!(
            validate_view(&empty)
                .unwrap_err()
                .contains("<Chart> needs 1 to")
        );
    }

    #[test]
    fn validator_rejects_ragged_or_mislabelled_charts() {
        let ragged = parse(json!({"tag": "Chart", "props": {"kind": "line"},
        "children": [
            {"tag": "Series", "props": {"name": "a", "values": "1,2"}},
            {"tag": "Series", "props": {"name": "b", "values": "1,2,3"}}
        ]}));
        assert!(
            validate_view(&ragged)
                .unwrap_err()
                .contains("same number of values")
        );
        let labels = parse(
            json!({"tag": "Chart", "props": {"kind": "bar", "labels": "a,b,c"},
            "children": [{"tag": "Series", "props": {"name": "a", "values": "1,2"}}]}),
        );
        assert!(
            validate_view(&labels)
                .unwrap_err()
                .contains("labels must match")
        );
    }

    #[test]
    fn validator_rejects_non_finite_or_oversized_series_values() {
        for values in ["", "1,abc", "1,NaN", "1,inf"] {
            let node = parse(json!({"tag": "Chart", "props": {"kind": "bar"},
                "children": [{"tag": "Series", "props": {"name": "a", "values": values}}]}));
            let err = validate_view(&node).unwrap_err();
            assert!(err.contains("comma-separated numbers"), "{values:?}: {err}");
        }
        let many = (0..=MAX_CHART_POINTS)
            .map(|n| n.to_string())
            .collect::<Vec<_>>()
            .join(",");
        let too_many = parse(json!({"tag": "Chart", "props": {"kind": "bar"},
            "children": [{"tag": "Series", "props": {"name": "a", "values": many}}]}));
        assert!(
            validate_view(&too_many)
                .unwrap_err()
                .contains("<Series> needs 1 to")
        );
    }

    #[test]
    fn validator_places_series_inside_chart() {
        let loose = parse(json!({"tag": "Series", "props": {"name": "a", "values": "1"}}));
        assert!(
            validate_view(&loose)
                .unwrap_err()
                .contains("must be inside <Chart>")
        );
    }

    #[test]
    fn validator_accepts_buttons_in_any_position() {
        let tree = parse(json!({"tag": "Row", "children": [
            {"tag": "Button", "props": {"label": "Ship", "prompt": "Ship the build", "tone": "primary"}},
            {"tag": "Button", "props": {"label": "Explain", "prompt": "Explain the diff"}},
            {"tag": "Button", "props": {"label": "Delete", "prompt": "Delete it", "tone": "danger"}}
        ]}));
        assert_eq!(validate_view(&tree), Ok(()));
    }

    #[test]
    fn validator_requires_nonblank_button_label_and_prompt() {
        for props in [
            json!({"prompt": "Run"}),
            json!({"label": "Run"}),
            json!({"label": "", "prompt": "Run"}),
            json!({"label": "Run", "prompt": "  "}),
        ] {
            let err = validate_view(&with_props("Button", props.clone())).unwrap_err();
            assert!(err.contains("<Button>"), "{props}: {err}");
        }
    }

    #[test]
    fn validator_rejects_unknown_button_tone() {
        let node = with_props(
            "Button",
            json!({"label": "Run", "prompt": "Run", "tone": "success"}),
        );
        let err = validate_view(&node).unwrap_err();
        assert!(
            err.contains("tone must be one of: primary, neutral, danger"),
            "{err}"
        );
    }

    #[test]
    fn validator_rejects_button_children() {
        let mut node = with_props("Button", json!({"label": "Run", "prompt": "Run"}));
        node.children.push(ViewChild::Text("extra".into()));
        assert!(
            validate_view(&node)
                .unwrap_err()
                .contains("takes no children")
        );
    }

    #[test]
    fn validator_bounds_button_prompt_length() {
        let button =
            |len: usize| with_props("Button", json!({"label": "Run", "prompt": "x".repeat(len)}));
        assert_eq!(validate_view(&button(MAX_VIEW_TEXT_BYTES)), Ok(()));
        assert!(validate_view(&button(MAX_VIEW_TEXT_BYTES + 1)).is_err());
    }

    fn choice(text: &str, correct: &str) -> ViewNode {
        ViewNode {
            tag: "Choice".into(),
            props: json!({"correct": correct}).as_object().unwrap().clone(),
            children: vec![ViewChild::Text(text.into())],
        }
    }

    fn quiz(choices: Vec<ViewNode>) -> ViewNode {
        ViewNode {
            tag: "Quiz".into(),
            props: json!({"question": "Which step?", "prompt": "Quiz: step"})
                .as_object()
                .unwrap()
                .clone(),
            children: choices.into_iter().map(ViewChild::Node).collect(),
        }
    }

    fn three_way() -> Vec<ViewNode> {
        vec![
            choice("alpha", "false"),
            choice("beta", "true"),
            choice("gamma", "false"),
        ]
    }

    fn options(count: usize) -> Vec<ViewNode> {
        (0..count)
            .map(|index| {
                let correct = if index == 0 { "true" } else { "false" };
                choice(&format!("option {index}"), correct)
            })
            .collect()
    }

    #[test]
    fn quiz_with_one_correct_choice_is_valid() {
        assert_eq!(validate_view(&quiz(three_way())), Ok(()));
    }

    #[test]
    fn quiz_option_count_is_bounded() {
        assert_eq!(validate_view(&quiz(options(2))), Ok(()));
        assert_eq!(validate_view(&quiz(options(5))), Ok(()));
        assert!(
            validate_view(&quiz(options(1)))
                .unwrap_err()
                .contains("2 to 5")
        );
        assert!(
            validate_view(&quiz(options(6)))
                .unwrap_err()
                .contains("2 to 5")
        );
    }

    #[test]
    fn quiz_needs_exactly_one_correct_choice() {
        let none = vec![choice("alpha", "false"), choice("beta", "false")];
        let two = vec![choice("alpha", "true"), choice("beta", "true")];
        assert!(
            validate_view(&quiz(none))
                .unwrap_err()
                .contains("exactly one")
        );
        assert!(
            validate_view(&quiz(two))
                .unwrap_err()
                .contains("exactly one")
        );
    }

    #[test]
    fn quiz_rejects_repeated_option_text_ignoring_case_and_space() {
        let repeated = vec![choice("Yes", "true"), choice(" yes ", "false")];
        assert!(
            validate_view(&quiz(repeated))
                .unwrap_err()
                .contains("repeats")
        );
    }

    #[test]
    fn quiz_rejects_answers_that_are_obvious() {
        let all = vec![choice("alpha", "false"), choice("All of the above", "true")];
        assert!(validate_view(&quiz(all)).unwrap_err().contains("obvious"));
    }

    #[test]
    fn quiz_needs_a_question_and_a_prompt() {
        let mut blank_question = quiz(three_way());
        blank_question.props.insert("question".into(), json!("   "));
        assert!(validate_view(&blank_question).is_err());

        let mut missing_prompt = quiz(three_way());
        missing_prompt.props.remove("prompt");
        assert!(validate_view(&missing_prompt).is_err());
    }

    #[test]
    fn quiz_accepts_only_choice_children() {
        let mut with_text = quiz(three_way());
        with_text.children.push(ViewChild::Text("hint".into()));
        assert!(
            validate_view(&with_text)
                .unwrap_err()
                .contains("accepts only <Choice>")
        );
    }

    #[test]
    fn choice_must_sit_directly_inside_quiz() {
        assert!(
            validate_view(&choice("alpha", "true"))
                .unwrap_err()
                .contains("inside <Quiz>")
        );
        let column = ViewNode {
            tag: "Column".into(),
            props: Map::new(),
            children: vec![ViewChild::Node(choice("alpha", "true"))],
        };
        assert!(validate_view(&column).is_err());
    }

    #[test]
    fn choice_flag_must_be_true_or_false() {
        let mut options = three_way();
        options[0].props.insert("correct".into(), json!("yes"));
        assert!(
            validate_view(&quiz(options))
                .unwrap_err()
                .contains("\"true\" or \"false\"")
        );
    }

    #[test]
    fn choice_needs_exactly_one_text_child() {
        let mut empty = three_way();
        empty[0].children.clear();
        assert!(
            validate_view(&quiz(empty))
                .unwrap_err()
                .contains("one non-empty text child")
        );

        let mut nested_child = three_way();
        nested_child[0].children = vec![ViewChild::Node(leaf("Text"))];
        assert!(
            validate_view(&quiz(nested_child))
                .unwrap_err()
                .contains("one non-empty text child")
        );
    }
}
