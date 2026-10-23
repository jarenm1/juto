//! Transcript cards for subagents. A card is a host-built `Subagent` element:
//! created when a child is first seen, then updated in place as its status changes.

use juto_agent::ViewNode;
use juto_runtime::DesktopSessionArtifact;
use juto_runtime::subagents::SubagentStatus;
use serde_json::{Map, json};

use crate::chat::{Message, MessageRole, desktop_row, view_row};
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

/// Name for a card whose child has no configured name. Ids are UUIDv7 strings:
/// their head is a timestamp, identical for siblings spawned together, so the
/// tail is what tells them apart.
pub(crate) fn fallback_name(id: &str) -> String {
    let tail: Vec<char> = id.chars().rev().take(6).collect();
    let tail: String = tail.into_iter().rev().collect();
    format!("Subagent {tail}")
}

/// Creates the card for child `id` on first sight, or updates its name and status
/// in place. A name of `None` keeps the existing name, and a new card without one
/// is labeled from the tail of its id (see `fallback_name`).
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
    let fallback = fallback_name(id);
    set_card_status(&mut card, Some(name.unwrap_or(&fallback)), status);

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

/// Places a desktop that child `id` reported in its own row directly below the
/// child's card, so the canvas stays with the child it came from. An artifact
/// already in the transcript is not added again. With no card yet, it goes last.
pub fn attach_desktop_artifact(
    messages: &mut Vec<Message>,
    id: &str,
    artifact: DesktopSessionArtifact,
) {
    let already_shown = messages.iter().any(|message| {
        message
            .desktop_artifacts
            .iter()
            .any(|shown| shown.artifact_id == artifact.artifact_id)
    });
    if already_shown {
        return;
    }
    let card_row = messages
        .iter()
        .position(|message| message.views.iter().any(|view| card_id(view) == Some(id)));
    match card_row {
        Some(index) => messages.insert(index + 1, desktop_row(artifact)),
        None => messages.push(desktop_row(artifact)),
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
    fn unnamed_card_is_labeled_by_id_tail_not_the_full_id() {
        let id = "0195c2a4-7f3e-7d10-9b8a-3c1e5f2a9d47";
        let mut messages = Vec::new();
        upsert_subagent_card(&mut messages, id, None, &SubagentStatus::Running, false);

        assert_eq!(card_count(&messages), 1);
        assert_eq!(
            messages[0].views[0].props.get("name"),
            Some(&json!("Subagent 2a9d47"))
        );
    }

    fn desktop(artifact_id: &str) -> DesktopSessionArtifact {
        DesktopSessionArtifact {
            version: 1,
            artifact_id: artifact_id.to_owned(),
            title: "Build desktop".to_owned(),
            host_id: "host-local".to_owned(),
            environment_id: "desk-1".to_owned(),
            grant_id: "grant-1".to_owned(),
        }
    }

    #[test]
    fn reported_desktop_sits_below_its_card_and_shows_once() {
        let mut messages = Vec::new();
        upsert_subagent_card(&mut messages, "w1", None, &SubagentStatus::Running, false);
        upsert_subagent_card(&mut messages, "w2", None, &SubagentStatus::Running, false);
        attach_desktop_artifact(&mut messages, "w1", desktop("desk-1-artifact"));
        attach_desktop_artifact(&mut messages, "w1", desktop("desk-1-artifact"));

        assert_eq!(messages.len(), 3);
        assert_eq!(card_count(&messages), 2);
        assert!(messages[0].desktop_artifacts.is_empty());
        assert!(messages[1].views.is_empty());
        assert_eq!(
            messages[1].desktop_artifacts,
            vec![desktop("desk-1-artifact")]
        );
        assert!(messages[2].desktop_artifacts.is_empty());
    }
}
