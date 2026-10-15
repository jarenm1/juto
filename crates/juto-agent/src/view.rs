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
];

/// Tones a `Badge` accepts. The renderer shows any other tone as neutral, so the
/// validator rejects it instead.
pub const BADGE_TONES: &[&str] = &["success", "warning", "error", "info"];

/// Deepest allowed nesting, counting the root as depth 1.
pub const MAX_VIEW_DEPTH: usize = 8;
/// Most elements allowed in one tree.
pub const MAX_VIEW_NODES: usize = 200;
/// Longest allowed text child or prop string, in bytes.
pub const MAX_VIEW_TEXT_BYTES: usize = 4096;
/// Most tick marks a `Timeline` may draw.
pub const MAX_TIMELINE_TICKS: usize = 48;

/// Tags that exist only inside another tag, as `(parent, child)` pairs. A parent
/// accepts only that child, and the child must sit directly under the parent.
const NESTING: &[(&str, &str)] = &[("Stats", "Stat"), ("Timeline", "Lane"), ("Lane", "Bar")];

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
        _ => {}
    }
    if matches!(node.tag.as_str(), "Stat" | "Bar") && !node.children.is_empty() {
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
}
