//! Transcript cards for subagents. A card is a host-built `Subagent` element:
//! created when a child is first seen, then updated in place as its status changes.

use juto_agent::ViewNode;
use juto_runtime::DesktopSessionArtifact;
use juto_runtime::subagents::{
    MainMessageHeader, MainMessageKind, SubagentStatus, parse_main_message_header,
};
use serde_json::{Map, json};

use crate::chat::{Message, MessageRole, desktop_row, user_row, view_row};
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

/// A `Subagent` card for child `id`, labeled `name` and showing `status`.
fn new_card(id: &str, name: &str, status: &SubagentStatus) -> ViewNode {
    let mut card = ViewNode {
        tag: SUBAGENT_TAG.to_owned(),
        props: Map::new(),
        children: Vec::new(),
    };
    card.props.insert("id".to_owned(), json!(id));
    set_card_status(&mut card, Some(name), status);
    card
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

    let fallback = fallback_name(id);
    let card = new_card(id, name.unwrap_or(&fallback), status);

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

/// Transcript rows for a user message the runtime framed for main. Each framed
/// report opens with the header `main_message_text` writes. A terminal report
/// becomes a card without its body: the card carries the status, and the child's
/// focus thread keeps the report. Any other report keeps its text as a row
/// attributed to the child, except a user turn, which stays "You".
/// `None` unless the text opens with a framed header, so typed text is unchanged.
pub fn report_rows(text: &str) -> Option<Vec<Message>> {
    let mut segments: Vec<(MainMessageHeader, String)> = Vec::new();
    for line in text.lines() {
        match parse_main_message_header(line) {
            Some(header) => segments.push((header, line.to_owned())),
            None => {
                let (_, raw) = segments.last_mut()?;
                raw.push('\n');
                raw.push_str(line);
            }
        }
    }
    if segments.is_empty() {
        return None;
    }
    let rows = segments
        .into_iter()
        .map(|(header, raw)| {
            let name = header
                .name
                .clone()
                .unwrap_or_else(|| fallback_name(&header.from));
            match terminal_status(header.kind) {
                Some(status) => view_row(new_card(&header.from, &name, &status)),
                None => {
                    let mut row = user_row(raw.trim_end().to_owned());
                    if header.kind != MainMessageKind::UserTurn {
                        row.author = Some(name.into());
                    }
                    row
                }
            }
        })
        .collect();
    Some(rows)
}

/// Drops the card rows in `rows` whose child already has a card in `shown`, or an
/// earlier row in `rows`. A child has one card, updated in place as its status
/// changes, so a report for a child that already has one adds no second card.
pub fn without_repeated_cards(shown: &[Message], rows: Vec<Message>) -> Vec<Message> {
    let mut known: Vec<String> = shown
        .iter()
        .flat_map(|message| message.views.iter())
        .filter_map(card_id)
        .map(str::to_owned)
        .collect();
    rows.into_iter()
        .filter(|row| match row.views.first().and_then(card_id) {
            Some(id) if known.iter().any(|seen| seen == id) => false,
            Some(id) => {
                known.push(id.to_owned());
                true
            }
            None => true,
        })
        .collect()
}

/// The status a terminal report leaves its child in. Other kinds say nothing
/// final about the child, so they have none.
fn terminal_status(kind: MainMessageKind) -> Option<SubagentStatus> {
    match kind {
        MainMessageKind::Completed => Some(SubagentStatus::Completed),
        MainMessageKind::Failed => Some(SubagentStatus::Failed),
        MainMessageKind::Cancelled => Some(SubagentStatus::Cancelled),
        MainMessageKind::Message | MainMessageKind::Desktop | MainMessageKind::UserTurn => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assistant(text: &str) -> Message {
        Message {
            role: MessageRole::Assistant,
            author: None,
            text: text.to_owned().into(),
            thinking: Default::default(),
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

    #[test]
    fn terminal_reports_render_as_cards_without_their_bodies() {
        let text =
            "[subagent w1 Completed]\nsecret full report\n\n[subagent w2 Failed]\nbuild error";
        let rows = report_rows(text).unwrap();

        assert_eq!(rows.len(), 2);
        assert_eq!(card_count(&rows), 2);
        assert_eq!(rows[0].role, MessageRole::Assistant);
        assert!(rows[0].text.as_ref().is_empty());
        let card = &rows[0].views[0];
        assert_eq!(card_id(card), Some("w1"));
        assert_eq!(card.props.get("name"), Some(&json!(fallback_name("w1"))));
        assert_eq!(
            card.props.get("status"),
            Some(&json!(status_label(&SubagentStatus::Completed)))
        );
        assert!(
            !serde_json::to_string(&card.props)
                .unwrap()
                .contains("secret")
        );
    }

    #[test]
    fn framed_header_with_name_and_digest_still_makes_a_card() {
        let text = r#"[subagent w3 name="api \"v2\"" Cancelled (digest of 9 bytes)]
clipped"#;
        let rows = report_rows(text).unwrap();

        assert_eq!(card_count(&rows), 1);
        let card = &rows[0].views[0];
        assert_eq!(card_id(card), Some("w3"));
        assert_eq!(
            card.props.get("status"),
            Some(&json!(status_label(&SubagentStatus::Cancelled)))
        );
    }

    #[test]
    fn non_terminal_reports_keep_their_text_as_a_user_row() {
        let rows =
            report_rows("[subagent w1 Completed]\ndone\n\n[subagent w2 Message]\nneed input")
                .unwrap();

        assert_eq!(card_count(&rows), 1);
        assert_eq!(rows[1].role, MessageRole::User);
        assert_eq!(rows[1].text.as_ref(), "[subagent w2 Message]\nneed input");
    }

    #[test]
    fn text_not_opening_with_a_framed_header_is_not_a_report() {
        assert!(report_rows("hello").is_none());
        assert!(report_rows("").is_none());
        assert!(report_rows("note\n[subagent w1 Completed]\ndone").is_none());
    }

    #[test]
    fn a_report_adds_no_card_for_a_child_already_shown() {
        let mut shown = Vec::new();
        upsert_subagent_card(
            &mut shown,
            "w1",
            Some("builder"),
            &SubagentStatus::Running,
            false,
        );
        let rows =
            report_rows("[subagent w1 Completed]\ndone\n\n[subagent w2 Failed]\nbad").unwrap();

        let kept = without_repeated_cards(&shown, rows);
        assert_eq!(kept.len(), 1);
        assert_eq!(card_id(&kept[0].views[0]), Some("w2"));
    }

    #[test]
    fn one_card_per_child_within_a_batch() {
        let rows = report_rows("[subagent w1 Completed]\na\n\n[subagent w1 Failed]\nb").unwrap();

        let kept = without_repeated_cards(&[], rows);
        assert_eq!(kept.len(), 1);
        assert_eq!(
            kept[0].views[0].props.get("status"),
            Some(&json!(status_label(&SubagentStatus::Completed)))
        );
    }

    #[test]
    fn a_child_report_is_attributed_to_the_child_and_a_user_turn_is_not() {
        let rows = report_rows(
            "[subagent w2 name=\"builder\" Message]\nneed input\n[subagent w3 Message]\nhi\n[subagent w4 UserTurn]\nthanks",
        )
        .unwrap();

        assert_eq!(rows[0].author, Some("builder".into()));
        assert_eq!(rows[1].author, Some("Subagent w3".into()));
        assert_eq!(rows[2].author, None);
    }
}
