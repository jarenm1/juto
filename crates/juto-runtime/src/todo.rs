//! `todo`: the checklist a subagent keeps for its own run.
//!
//! Behavioral reference: oh-my-pi 579da1d6 `packages/coding-agent/src/tools/todo.ts`
//! (ops, batch rollback, single `in_progress` normalization). Divergence: the
//! list belongs to the run that holds the tool, and only roles whose tool scope
//! grants `todo` register it. The main thread's scope does not, so main never
//! owns a list.
//!
//! Durability: a successful op returns the full phases in `ToolOutput::details`.
//! The agent journals that output with the `ToolResult` message, so
//! [`phases_from_entries`] rebuilds the list on resume, fork, or branch switch.
//! There is no second store.

use async_trait::async_trait;
use juto_agent::{Tool, ToolConcurrency, ToolContext, ToolOutput};
use juto_ai::{Message, ToolDefinition};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::session::{EntryKind, SessionEntry};

/// Name the model calls the tool by.
pub(crate) const TOOL_NAME: &str = "todo";

/// Phase a flat `items` init lands in when no `phase` is named.
const DEFAULT_INIT_PHASE: &str = "Tasks";

const DESCRIPTION: &str = "Per-run checklist: one task in_progress at a time. Ops: init, start, \
    done, drop, block, unblock, rm, append, view. Tasks match by exact text; a phase targets \
    all its tasks. A call with any error changes nothing.";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum TodoStatus {
    Pending,
    InProgress,
    Completed,
    Abandoned,
    Blocked,
}

impl TodoStatus {
    fn label(self) -> &'static str {
        match self {
            Self::Pending => "pending",
            Self::InProgress => "in_progress",
            Self::Completed => "completed",
            Self::Abandoned => "abandoned",
            Self::Blocked => "blocked",
        }
    }

    /// Work still to do: counted as remaining, and the only states `block` accepts.
    fn is_open(self) -> bool {
        matches!(self, Self::Pending | Self::InProgress)
    }

    fn is_closed(self) -> bool {
        matches!(self, Self::Completed | Self::Abandoned)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct TodoItem {
    pub content: String,
    pub status: TodoStatus,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub blocker: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct TodoPhase {
    pub name: String,
    pub tasks: Vec<TodoItem>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
enum Op {
    Init,
    Start,
    Done,
    Rm,
    Drop,
    Block,
    Unblock,
    Append,
    View,
}

impl Op {
    fn name(self) -> &'static str {
        match self {
            Self::Init => "init",
            Self::Start => "start",
            Self::Done => "done",
            Self::Rm => "rm",
            Self::Drop => "drop",
            Self::Block => "block",
            Self::Unblock => "unblock",
            Self::Append => "append",
            Self::View => "view",
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct InitEntry {
    phase: String,
    items: Vec<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Args {
    op: Op,
    #[serde(default)]
    list: Option<Vec<InitEntry>>,
    #[serde(default)]
    task: Option<String>,
    #[serde(default)]
    phase: Option<String>,
    #[serde(default)]
    items: Option<Vec<String>>,
    #[serde(default)]
    reason: Option<String>,
}

/// Find the first task with `content`, as (phase index, task index).
fn find_task(
    phases: &[TodoPhase],
    content: Option<&str>,
    errors: &mut Vec<String>,
) -> Option<(usize, usize)> {
    let Some(content) = content.filter(|c| !c.is_empty()) else {
        errors.push("Missing task content".to_string());
        return None;
    };
    let hit = phases.iter().enumerate().find_map(|(pi, phase)| {
        phase
            .tasks
            .iter()
            .position(|task| task.content == content)
            .map(|ti| (pi, ti))
    });
    if hit.is_none() {
        errors.push(format!("Task \"{content}\" not found"));
    }
    hit
}

fn find_phase(phases: &[TodoPhase], name: Option<&str>, errors: &mut Vec<String>) -> Option<usize> {
    let Some(name) = name.filter(|n| !n.is_empty()) else {
        errors.push("Missing phase name".to_string());
        return None;
    };
    let hit = phases.iter().position(|phase| phase.name == name);
    if hit.is_none() {
        errors.push(format!("Phase \"{name}\" not found"));
    }
    hit
}

/// Tasks an op targets: one task, every task in one phase, or every task.
fn targets(phases: &[TodoPhase], args: &Args, errors: &mut Vec<String>) -> Vec<(usize, usize)> {
    if args.task.is_some() {
        return find_task(phases, args.task.as_deref(), errors)
            .into_iter()
            .collect();
    }
    if args.phase.is_some() {
        return match find_phase(phases, args.phase.as_deref(), errors) {
            Some(pi) => (0..phases[pi].tasks.len()).map(|ti| (pi, ti)).collect(),
            None => Vec::new(),
        };
    }
    phases
        .iter()
        .enumerate()
        .flat_map(|(pi, phase)| (0..phase.tasks.len()).map(move |ti| (pi, ti)))
        .collect()
}

fn init_phases(args: &Args, errors: &mut Vec<String>) -> Vec<TodoPhase> {
    let list: Vec<(String, Vec<String>)> = match (&args.list, &args.items) {
        (Some(list), _) => list
            .iter()
            .map(|entry| (entry.phase.clone(), entry.items.clone()))
            .collect(),
        (None, Some(items)) if !items.is_empty() => {
            let phase = args
                .phase
                .clone()
                .filter(|name| !name.is_empty())
                .unwrap_or_else(|| DEFAULT_INIT_PHASE.to_string());
            vec![(phase, items.clone())]
        }
        _ => {
            errors.push("Missing list for init operation".to_string());
            return Vec::new();
        }
    };
    // A duplicate name would be unaddressable: every targeting op resolves the first match.
    let mut seen_phases = std::collections::BTreeSet::new();
    let mut seen_tasks = std::collections::BTreeSet::new();
    for (phase, items) in &list {
        if !seen_phases.insert(phase.as_str()) {
            errors.push(format!("Duplicate phase \"{phase}\" in init list"));
        }
        if items.is_empty() {
            errors.push(format!("Phase \"{phase}\" has no items"));
        }
        for content in items {
            if !seen_tasks.insert(content.as_str()) {
                errors.push(format!("Duplicate task \"{content}\" in init list"));
            }
        }
    }
    list.into_iter()
        .map(|(name, items)| TodoPhase {
            name,
            tasks: items.into_iter().map(pending).collect(),
        })
        .collect()
}

fn pending(content: String) -> TodoItem {
    TodoItem {
        content,
        status: TodoStatus::Pending,
        blocker: None,
    }
}

/// Keep at most one `in_progress` task. If none exists, promote the first pending one.
fn normalize_in_progress(phases: &mut [TodoPhase]) {
    let mut kept_active = false;
    for task in phases.iter_mut().flat_map(|phase| phase.tasks.iter_mut()) {
        if task.status == TodoStatus::InProgress {
            if kept_active {
                task.status = TodoStatus::Pending;
            } else {
                kept_active = true;
            }
        }
    }
    if kept_active {
        return;
    }
    if let Some(task) = phases
        .iter_mut()
        .flat_map(|phase| phase.tasks.iter_mut())
        .find(|task| task.status == TodoStatus::Pending)
    {
        task.status = TodoStatus::InProgress;
    }
}

/// Apply one op to a copy of `current`. Any error rejects the whole call.
fn apply_op(current: &[TodoPhase], args: &Args) -> Result<Vec<TodoPhase>, Vec<String>> {
    let mut next = current.to_vec();
    if args.op == Op::View {
        return Ok(next);
    }
    let mut errors = Vec::new();
    match args.op {
        Op::View => unreachable!("view returns before the batch"),
        Op::Init => next = init_phases(args, &mut errors),
        Op::Start => {
            if let Some((pi, ti)) = find_task(&next, args.task.as_deref(), &mut errors) {
                for task in next.iter_mut().flat_map(|phase| phase.tasks.iter_mut()) {
                    if task.status == TodoStatus::InProgress {
                        task.status = TodoStatus::Pending;
                    }
                }
                next[pi].tasks[ti].status = TodoStatus::InProgress;
            }
        }
        Op::Done | Op::Drop => {
            let status = if args.op == Op::Done {
                TodoStatus::Completed
            } else {
                TodoStatus::Abandoned
            };
            for (pi, ti) in targets(&next, args, &mut errors) {
                next[pi].tasks[ti].status = status;
            }
        }
        Op::Block => {
            if args.task.is_none() && args.phase.is_none() {
                errors.push("block requires a task or phase target".to_string());
            } else {
                // Collapse whitespace so a multi-line reason fits one checklist line.
                let reason = args
                    .reason
                    .as_deref()
                    .map(|reason| reason.split_whitespace().collect::<Vec<_>>().join(" "))
                    .filter(|reason| !reason.is_empty());
                for (pi, ti) in targets(&next, args, &mut errors) {
                    let task = &mut next[pi].tasks[ti];
                    // Blocking never reopens finished work; a blocked task may refine its note.
                    if task.status.is_open() || task.status == TodoStatus::Blocked {
                        task.status = TodoStatus::Blocked;
                        task.blocker = reason.clone();
                    }
                }
            }
        }
        Op::Unblock => {
            if args.task.is_none() && args.phase.is_none() {
                errors.push("unblock requires a task or phase target".to_string());
            } else {
                for (pi, ti) in targets(&next, args, &mut errors) {
                    let task = &mut next[pi].tasks[ti];
                    if task.status == TodoStatus::Blocked {
                        task.status = TodoStatus::Pending;
                        task.blocker = None;
                    }
                }
            }
        }
        Op::Rm => {
            if args.task.is_some() {
                if let Some((pi, ti)) = find_task(&next, args.task.as_deref(), &mut errors) {
                    next[pi].tasks.remove(ti);
                }
            } else if args.phase.is_some() {
                if let Some(pi) = find_phase(&next, args.phase.as_deref(), &mut errors) {
                    next[pi].tasks.clear();
                }
            } else {
                for phase in &mut next {
                    phase.tasks.clear();
                }
            }
        }
        Op::Append => match (args.phase.as_deref(), args.items.as_deref()) {
            (None | Some(""), _) => {
                errors.push("Missing phase name for append operation".to_string())
            }
            (_, None | Some([])) => errors.push("Missing items for append operation".to_string()),
            (Some(name), Some(items)) => {
                let mut seen = std::collections::BTreeSet::new();
                for content in items {
                    let duplicate_in_call = !seen.insert(content.as_str());
                    let exists = next
                        .iter()
                        .any(|phase| phase.tasks.iter().any(|task| task.content == *content));
                    if duplicate_in_call || exists {
                        errors.push(format!("Task \"{content}\" already exists"));
                    }
                }
                if errors.is_empty() {
                    let pi = match next.iter().position(|phase| phase.name == name) {
                        Some(pi) => pi,
                        None => {
                            next.push(TodoPhase {
                                name: name.to_string(),
                                tasks: Vec::new(),
                            });
                            next.len() - 1
                        }
                    };
                    next[pi].tasks.extend(items.iter().cloned().map(pending));
                }
            }
        },
    }
    if !errors.is_empty() {
        return Err(errors);
    }
    normalize_in_progress(&mut next);
    Ok(next)
}

/// Human-readable state after a call, with any errors first.
fn format_summary(phases: &[TodoPhase], errors: &[String]) -> String {
    let mut lines = Vec::new();
    if !errors.is_empty() {
        lines.push(format!("Errors: {}", errors.join("; ")));
    }
    let remaining: Vec<(&str, &TodoItem)> = phases
        .iter()
        .flat_map(|phase| {
            phase
                .tasks
                .iter()
                .filter(|task| task.status.is_open())
                .map(move |task| (phase.name.as_str(), task))
        })
        .collect();
    if remaining.is_empty() {
        lines.push("Remaining items: none.".to_string());
    } else {
        lines.push(format!("Remaining items ({}):", remaining.len()));
        for (phase, task) in &remaining {
            lines.push(format!(
                "  - {} [{}] ({phase})",
                task.content,
                task.status.label()
            ));
        }
    }
    let all = phases.iter().flat_map(|phase| phase.tasks.iter());
    let total = all.clone().count();
    let closed = all.clone().filter(|task| task.status.is_closed()).count();
    let blocked = all
        .filter(|task| task.status == TodoStatus::Blocked)
        .count();
    let blocked_note = if blocked > 0 {
        format!(", {blocked} blocked")
    } else {
        String::new()
    };
    lines.push(format!(
        "Overall: {closed}/{total} done, {} open{blocked_note}.",
        remaining.len()
    ));
    // The active phase is the earliest one with open work, else the last phase.
    let current = phases
        .iter()
        .position(|phase| phase.tasks.iter().any(|task| task.status.is_open()))
        .or(phases.len().checked_sub(1));
    if let Some(index) = current {
        let phase = &phases[index];
        let done = phase
            .tasks
            .iter()
            .filter(|task| task.status.is_closed())
            .count();
        lines.push(format!(
            "Active phase {}/{} \"{}\" ({done}/{}).",
            index + 1,
            phases.len(),
            phase.name,
            phase.tasks.len()
        ));
    }
    for phase in phases {
        lines.push(format!("  {}:", phase.name));
        for task in &phase.tasks {
            let checkbox = if task.status == TodoStatus::Completed {
                "[X]"
            } else {
                "[ ]"
            };
            let tag = match (task.status, &task.blocker) {
                (TodoStatus::InProgress, _) => " (in progress)".to_string(),
                (TodoStatus::Abandoned, _) => " (dropped)".to_string(),
                (TodoStatus::Blocked, Some(reason)) => format!(" (blocked: {reason})"),
                (TodoStatus::Blocked, None) => " (blocked)".to_string(),
                _ => String::new(),
            };
            lines.push(format!("    - {checkbox} {}{tag}", task.content));
        }
    }
    lines.join("\n")
}

/// Rebuild the list from the newest successful state-changing `todo` result.
/// `entries` is the active branch, oldest first. Errors and `view` reads carry
/// no new state, so they are skipped.
pub(crate) fn phases_from_entries(entries: &[&SessionEntry]) -> Vec<TodoPhase> {
    entries
        .iter()
        .rev()
        .find_map(|entry| {
            let EntryKind::Message {
                message:
                    Message::ToolResult {
                        tool_name,
                        is_error,
                        details,
                        ..
                    },
            } = &entry.kind
            else {
                return None;
            };
            if tool_name != TOOL_NAME
                || *is_error
                || details.get("op").and_then(Value::as_str) == Some(Op::View.name())
            {
                return None;
            }
            serde_json::from_value(details.get("phases")?.clone()).ok()
        })
        .unwrap_or_default()
}

fn definition() -> ToolDefinition {
    ToolDefinition {
        name: TOOL_NAME.to_string(),
        description: DESCRIPTION.to_string(),
        parameters: json!({
            "type": "object",
            "properties": {
                "op": {"type": "string", "enum": ["init", "start", "done", "rm", "drop", "block", "unblock", "append", "view"]},
                "list": {"type": "array", "items": {"type": "object", "properties": {"phase": {"type": "string"}, "items": {"type": "array", "items": {"type": "string"}, "minItems": 1}}, "required": ["phase", "items"], "additionalProperties": false}},
                "task": {"type": "string"},
                "phase": {"type": "string"},
                "items": {"type": "array", "items": {"type": "string"}},
                "reason": {"type": "string"}
            },
            "required": ["op"],
            "additionalProperties": false
        }),
    }
}

/// One run's checklist. Exists only in runs whose role scope grants `todo`.
pub(crate) struct TodoTool {
    phases: Mutex<Vec<TodoPhase>>,
}

impl TodoTool {
    pub(crate) fn new(phases: Vec<TodoPhase>) -> Self {
        Self {
            phases: Mutex::new(phases),
        }
    }

    /// Apply one call. A rejected call leaves the list unchanged.
    fn run_op(&self, arguments: Value) -> ToolOutput {
        let args: Args = match serde_json::from_value(arguments) {
            Ok(args) => args,
            Err(error) => return ToolOutput::error(format!("Invalid todo arguments: {error}")),
        };
        let mut phases = self.phases.lock();
        match apply_op(&phases, &args) {
            Ok(next) => {
                let mut output = ToolOutput::text(format_summary(&next, &[]));
                output.details = json!({ "op": args.op.name(), "phases": &next });
                *phases = next;
                output
            }
            Err(errors) => ToolOutput::error(format_summary(&phases, &errors)),
        }
    }
}

#[async_trait]
impl Tool for TodoTool {
    fn definition(&self) -> ToolDefinition {
        definition()
    }

    /// Every op rewrites the whole list, so calls in one batch must not overlap.
    fn concurrency(&self) -> ToolConcurrency {
        ToolConcurrency::Exclusive
    }

    async fn execute(&self, arguments: Value, _ctx: ToolContext) -> Result<ToolOutput, String> {
        Ok(self.run_op(arguments))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn call(tool: &TodoTool, arguments: Value) -> ToolOutput {
        tool.run_op(arguments)
    }

    fn snapshot(tool: &TodoTool) -> Vec<TodoPhase> {
        tool.phases.lock().clone()
    }

    fn statuses(tool: &TodoTool) -> Vec<(String, TodoStatus)> {
        snapshot(tool)
            .iter()
            .flat_map(|phase| {
                phase
                    .tasks
                    .iter()
                    .map(|task| (task.content.clone(), task.status))
            })
            .collect()
    }

    fn init_build(tool: &TodoTool, items: &[&str]) {
        let output = call(
            tool,
            json!({"op": "init", "list": [{"phase": "Build", "items": items}]}),
        );
        assert!(!output.is_error, "init failed: {:?}", output.content);
    }

    #[test]
    fn one_task_in_progress_and_done_advances_to_next_pending() {
        let tool = TodoTool::new(Vec::new());
        init_build(&tool, &["a", "b", "c"]);
        assert_eq!(statuses(&tool)[0].1, TodoStatus::InProgress);

        call(&tool, json!({"op": "start", "task": "c"}));
        assert_eq!(
            statuses(&tool),
            vec![
                ("a".into(), TodoStatus::Pending),
                ("b".into(), TodoStatus::Pending),
                ("c".into(), TodoStatus::InProgress),
            ]
        );

        call(&tool, json!({"op": "done", "task": "c"}));
        assert_eq!(
            statuses(&tool),
            vec![
                ("a".into(), TodoStatus::InProgress),
                ("b".into(), TodoStatus::Pending),
                ("c".into(), TodoStatus::Completed),
            ]
        );
    }

    #[test]
    fn rejected_call_leaves_the_list_unchanged() {
        let tool = TodoTool::new(Vec::new());
        init_build(&tool, &["a", "b"]);
        let before = snapshot(&tool);

        // "a" already exists, so the whole append is rejected, including "d".
        let output = call(
            &tool,
            json!({"op": "append", "phase": "Build", "items": ["d", "a"]}),
        );
        assert!(output.is_error);
        assert_eq!(snapshot(&tool), before);
    }

    #[test]
    fn block_collapses_reason_and_unblock_clears_it() {
        let tool = TodoTool::new(Vec::new());
        init_build(&tool, &["a", "b"]);

        call(
            &tool,
            json!({"op": "block", "task": "a", "reason": "waiting\n  on   api"}),
        );
        let blocked = &snapshot(&tool)[0].tasks[0];
        assert_eq!(blocked.status, TodoStatus::Blocked);
        assert_eq!(blocked.blocker.as_deref(), Some("waiting on api"));

        call(&tool, json!({"op": "unblock", "task": "a"}));
        let unblocked = &snapshot(&tool)[0].tasks[0];
        assert_eq!(unblocked.status, TodoStatus::Pending);
        assert_eq!(unblocked.blocker, None);
    }

    #[test]
    fn rm_without_target_clears_tasks_and_keeps_phases() {
        let tool = TodoTool::new(Vec::new());
        call(
            &tool,
            json!({"op": "init", "list": [
                {"phase": "Build", "items": ["a"]},
                {"phase": "Verify", "items": ["b"]}
            ]}),
        );

        call(&tool, json!({"op": "rm"}));
        let phases = snapshot(&tool);
        assert_eq!(phases.len(), 2);
        assert!(phases.iter().all(|phase| phase.tasks.is_empty()));
    }

    #[test]
    fn journal_rehydrates_latest_successful_list() {
        fn result(id: &str, is_error: bool, details: Value) -> SessionEntry {
            SessionEntry {
                id: id.to_string(),
                parent_id: None,
                timestamp: 0,
                kind: EntryKind::Message {
                    message: Message::ToolResult {
                        tool_call_id: id.to_string(),
                        tool_name: TOOL_NAME.to_string(),
                        content: Vec::new(),
                        is_error,
                        details,
                        timestamp: 0,
                    },
                },
            }
        }

        let tool = TodoTool::new(Vec::new());
        init_build(&tool, &["a", "b"]);
        let init = call(
            &tool,
            json!({"op": "init", "list": [{"phase": "Build", "items": ["a", "b"]}]}),
        );
        let start = call(&tool, json!({"op": "start", "task": "b"}));
        let rejected = call(&tool, json!({"op": "start", "task": "missing"}));
        let view = call(&tool, json!({"op": "view"}));
        let entries = [
            result("init", false, init.details),
            result("start", false, start.details),
            result("rejected", true, rejected.details),
            result("view", false, view.details),
        ];
        let refs: Vec<&SessionEntry> = entries.iter().collect();

        let restored = phases_from_entries(&refs);
        assert_eq!(restored, snapshot(&tool));
        assert_eq!(restored[0].tasks[1].status, TodoStatus::InProgress);
    }
}
