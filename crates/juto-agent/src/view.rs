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

/// Component tags the host renders, each with the only props it accepts. The host
/// renderer (`apps/juto/src/view_element.rs`) must handle every tag listed here.
pub const VIEW_TAGS: &[(&str, &[&str])] = &[
    ("Column", &[]),
    ("Row", &[]),
    ("Text", &[]),
    ("Badge", &["label", "tone"]),
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

/// Checks a tree before the host renders it. Errors name the offending tag or prop
/// so the model can correct the tree.
pub fn validate_view(root: &ViewNode) -> Result<(), String> {
    let mut count = 0;
    validate_node(root, 1, &mut count)
}

fn validate_node(node: &ViewNode, depth: usize, count: &mut usize) -> Result<(), String> {
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
    if node.tag == "Badge" {
        if node.props.get("label").and_then(Value::as_str).is_none() {
            return Err("<Badge> requires a string prop `label`".into());
        }
        if let Some(tone) = node.props.get("tone").and_then(Value::as_str)
            && !BADGE_TONES.contains(&tone)
        {
            return Err(format!(
                "<Badge> tone must be one of: {}",
                BADGE_TONES.join(", ")
            ));
        }
    }
    for child in &node.children {
        match child {
            ViewChild::Text(text) => check_text(text)?,
            ViewChild::Node(child) => validate_node(child, depth + 1, count)?,
        }
    }
    Ok(())
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
}
