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
}
