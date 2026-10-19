//! Transcript cards for subagents. A card is a host-built `Subagent` element:
//! created when a child is first seen, then updated in place as its status changes.

use juto_agent::ViewNode;
use juto_runtime::subagents::SubagentStatus;
use serde_json::{Map, json};

use crate::chat::{Message, MessageRole, view_row};
use crate::task_card::status_label;

/// Element tag that `view_element` renders as a subagent card. Host-built, so it
/// is not in the agent-facing `view` tag set.
const SUBAGENT_TAG: &str = "Subagent";

/// Id of a `Subagent` card, or `None` for any other view.
pub fn card_id(view: &ViewNode) -> Option<&str> {
    if view.tag != SUBAGENT_TAG {
        return None;
    }
    view.props.get("id")?.as_str()
}

/// Badge tone for a status. Groups match the sidebar's status colors.
fn status_tone(status: &SubagentStatus) -> &'static str {
    match status {
        SubagentStatus::Queued
        | SubagentStatus::Running
        | SubagentStatus::Verifying
        | SubagentStatus::Reviewing => "info",
        SubagentStatus::Blocked { .. } | SubagentStatus::Cancelled => "warning",
        SubagentStatus::Completed => "success",
        SubagentStatus::Failed => "error",
    }
}

fn set_card_status(card: &mut ViewNode, name: Option<&str>, status: &SubagentStatus) {
    if let Some(name) = name {
        card.props.insert("name".to_owned(), json!(name));
    }
    card.props
        .insert("status".to_owned(), json!(status_label(status)));
    card.props
        .insert("tone".to_owned(), json!(status_tone(status)));
}

/// Creates the card for child `id` on first sight, or updates its name and status
/// in place. A name of `None` keeps the existing name, and a new card uses the id.
///
/// While a run streams, a new card goes before the streaming assistant row, so the
/// next `TextDelta` still lands on that row. Otherwise the card is appended.
pub fn upsert_subagent_card(
    messages: &mut Vec<Message>,
    id: &str,
    name: Option<&str>,
    status: &SubagentStatus,
    streaming: bool,
) {
    let existing = messages
        .iter_mut()
        .flat_map(|message| message.views.iter_mut())
        .find(|view| card_id(view) == Some(id));
    if let Some(card) = existing {
        set_card_status(card, name, status);
        return;
    }

    let mut card = ViewNode {
        tag: SUBAGENT_TAG.to_owned(),
        props: Map::new(),
        children: Vec::new(),
    };
    card.props.insert("id".to_owned(), json!(id));
    set_card_status(&mut card, Some(name.unwrap_or(id)), status);

    let streaming_assistant = streaming
        && messages
            .last()
            .is_some_and(|last| last.role == MessageRole::Assistant);
    if streaming_assistant {
        let at = messages.len() - 1;
        messages.insert(at, view_row(card));
    } else {
        messages.push(view_row(card));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assistant(text: &str) -> Message {
        Message {
            role: MessageRole::Assistant,
            text: text.to_owned().into(),
            attachments: Vec::new(),
            desktop_artifacts: Vec::new(),
            approval: None,
            views: Vec::new(),
        }
    }

    fn card_count(messages: &[Message]) -> usize {
        messages
            .iter()
            .flat_map(|message| message.views.iter())
            .filter(|view| card_id(view).is_some())
            .count()
    }

    #[test]
    fn finishing_a_spawned_child_updates_its_one_card() {
        let mut messages = Vec::new();
        upsert_subagent_card(
            &mut messages,
            "w1",
            Some("builder"),
            &SubagentStatus::Queued,
            false,
        );
        upsert_subagent_card(&mut messages, "w1", None, &SubagentStatus::Completed, false);

        assert_eq!(card_count(&messages), 1);
        let card = &messages[0].views[0];
        assert_eq!(card.props.get("name"), Some(&json!("builder")));
        assert_eq!(
            card.props.get("status"),
            Some(&json!(status_label(&SubagentStatus::Completed)))
        );
        assert_eq!(card.props.get("tone"), Some(&json!("success")));
    }

    #[test]
    fn card_arriving_mid_stream_keeps_text_deltas_on_the_assistant_row() {
        let mut messages = vec![assistant("")];
        upsert_subagent_card(&mut messages, "w1", None, &SubagentStatus::Running, true);

        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].views.len(), 1);
        assert_eq!(messages[1].role, MessageRole::Assistant);
        assert!(messages[1].views.is_empty());
    }

    #[test]
    fn card_between_runs_is_appended_after_the_transcript() {
        let mut messages = vec![assistant("done")];
        upsert_subagent_card(&mut messages, "w1", None, &SubagentStatus::Running, false);

        assert_eq!(messages.len(), 2);
        assert!(messages[0].views.is_empty());
        assert_eq!(messages[1].views.len(), 1);
    }

    #[test]
    fn status_change_without_spawn_names_the_card_by_id() {
        let mut messages = Vec::new();
        upsert_subagent_card(&mut messages, "w2", None, &SubagentStatus::Running, false);

        assert_eq!(card_count(&messages), 1);
        assert_eq!(messages[0].views[0].props.get("name"), Some(&json!("w2")));
    }
}
