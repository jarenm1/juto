//! Read-only native change graph inspection. All repository I/O runs on Tokio workers.
use std::{
    collections::HashMap,
    path::PathBuf,
    sync::Arc,
    time::{Duration, SystemTime},
};

use gpui::{
    ClickEvent, Context, IntoElement, PathBuilder, Render, ScrollHandle, Task, Window, canvas, div,
    point, prelude::*, px, rgb,
};
use juto_runtime::{
    change_graph::{ChangeGraph, GraphState},
    git_transfer::GitRepository,
};

const ROW_HEIGHT: f32 = 64.;
const LANE_WIDTH: f32 = 18.;
const COLORS: [u32; 6] = [0x93b4d5, 0xb6a1cb, 0x8db6a0, 0xc9ad83, 0xc38f9e, 0x90b8be];

#[derive(Clone)]
struct GraphLayout {
    lanes: Vec<usize>,
    edges: Vec<(usize, usize, usize, usize)>,
    width: f32,
}

impl GraphLayout {
    fn new(state: &GraphState) -> Self {
        let mut threads = HashMap::new();
        let mut lanes = Vec::with_capacity(state.nodes.len());
        let positions: HashMap<_, _> = state
            .nodes
            .iter()
            .rev()
            .enumerate()
            .map(|(row, node)| (node.id.as_str(), row))
            .collect();
        for node in state.nodes.iter().rev() {
            let next = threads.len();
            lanes.push(*threads.entry(node.thread.as_str()).or_insert(next));
        }
        let mut edges = Vec::new();
        for (row, node) in state.nodes.iter().rev().enumerate() {
            if node.parents.is_empty() {
                edges.push((row, lanes[row], state.nodes.len(), 0));
            }
            for parent in &node.parents {
                if let Some(&parent_row) = positions.get(parent.as_str()) {
                    edges.push((row, lanes[row], parent_row, lanes[parent_row]));
                }
            }
        }
        Self {
            width: (threads.len().max(1) as f32 + 1.) * LANE_WIDTH,
            lanes,
            edges,
        }
    }
}

struct Snapshot {
    state: Option<GraphState>,
    workspace: PathBuf,
}

struct Reader {
    repository: GitRepository,
    fingerprint: Option<Option<(SystemTime, u64)>>,
}

impl Reader {
    fn read(&mut self) -> anyhow::Result<Option<Snapshot>> {
        let path = self.repository.git_dir().join("juto/change-graph.json");
        let fingerprint = match std::fs::metadata(path) {
            Ok(metadata) => Some((metadata.modified()?, metadata.len())),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
            Err(error) => return Err(error.into()),
        };
        if self.fingerprint == Some(fingerprint) {
            return Ok(None);
        }
        // Never initialize history merely because somebody opened the UI.
        let state = if fingerprint.is_some() {
            Some(ChangeGraph::open(self.repository.workspace())?.state()?)
        } else {
            None
        };
        self.fingerprint = Some(fingerprint);
        Ok(Some(Snapshot {
            state,
            workspace: self.repository.workspace().to_path_buf(),
        }))
    }
}

pub struct VcsSidebar {
    cwd: PathBuf,
    handle: tokio::runtime::Handle,
    state: Option<GraphState>,
    layout: Arc<GraphLayout>,
    selected: Option<String>,
    parent_index: usize,
    diff: String,
    diff_error: Option<String>,
    error: Option<String>,
    loading: bool,
    diff_loading: bool,
    scroll: ScrollHandle,
    diff_scroll: ScrollHandle,
    poll_task: Option<Task<()>>,
    diff_task: Option<Task<()>>,
    diff_request: u64,
}

impl VcsSidebar {
    pub fn new(cwd: PathBuf, handle: tokio::runtime::Handle, cx: &mut Context<Self>) -> Self {
        let mut sidebar = Self {
            cwd: cwd.clone(),
            handle: handle.clone(),
            state: None,
            layout: Arc::new(GraphLayout {
                lanes: Vec::new(),
                edges: Vec::new(),
                width: 36.,
            }),
            selected: None,
            parent_index: 0,
            diff: String::new(),
            diff_error: None,
            error: None,
            loading: true,
            diff_loading: false,
            scroll: ScrollHandle::new(),
            diff_scroll: ScrollHandle::new(),
            poll_task: None,
            diff_task: None,
            diff_request: 0,
        };
        sidebar.poll_task = Some(cx.spawn(async move |this, cx| {
            let mut reader: Option<Reader> = None;
            loop {
                let cwd = cwd.clone();
                let previous_reader = reader.take();
                let result = handle
                    .spawn_blocking(move || {
                        let mut current_reader = previous_reader;
                        let result = (|| {
                            if current_reader.is_none() {
                                current_reader = Some(Reader {
                                    repository: GitRepository::open(&cwd)?,
                                    fingerprint: None,
                                });
                            }
                            current_reader.as_mut().unwrap().read()
                        })();
                        (current_reader, result)
                    })
                    .await;
                let update = match result {
                    Ok((next_reader, result)) => {
                        reader = next_reader;
                        result.map_err(|error| format!("{error:#}"))
                    }
                    Err(error) => Err(format!("Graph worker failed: {error}")),
                };
                if this
                    .update(cx, |this, cx| {
                        this.loading = false;
                        match update {
                            Ok(Some(snapshot)) => {
                                this.cwd = snapshot.workspace;
                                this.layout = Arc::new(
                                    snapshot.state.as_ref().map(GraphLayout::new).unwrap_or(
                                        GraphLayout {
                                            lanes: Vec::new(),
                                            edges: Vec::new(),
                                            width: 36.,
                                        },
                                    ),
                                );
                                this.state = snapshot.state;
                                if this.selected.as_ref().is_some_and(|id| {
                                    !this.state.as_ref().is_some_and(|state| {
                                        state.nodes.iter().any(|node| &node.id == id)
                                    })
                                }) {
                                    this.selected = None;
                                    this.diff.clear();
                                    this.diff_request += 1;
                                }
                                this.error = None;
                                cx.notify();
                            }
                            Ok(None) => {
                                if this.error.take().is_some() {
                                    cx.notify();
                                }
                            }
                            Err(error) => {
                                this.error = Some(error);
                                cx.notify();
                            }
                        }
                    })
                    .is_err()
                {
                    break;
                }
                cx.background_executor().timer(Duration::from_secs(2)).await;
            }
        }));
        sidebar
    }

    fn select(&mut self, id: String, parent_index: usize, cx: &mut Context<Self>) {
        let Some(state) = &self.state else {
            return;
        };
        let Some(node) = state.nodes.iter().find(|node| node.id == id) else {
            return;
        };
        let before = node
            .parents
            .get(parent_index)
            .and_then(|id| state.nodes.iter().find(|node| &node.id == id))
            .map(|parent| parent.tree.clone())
            .unwrap_or_else(|| state.base_tree.clone());
        let after = node.tree.clone();
        let cwd = self.cwd.clone();
        self.selected = Some(id);
        self.parent_index = parent_index;
        self.diff.clear();
        self.diff_error = None;
        self.diff_loading = true;
        self.diff_scroll = ScrollHandle::new();
        self.diff_request += 1;
        let request = self.diff_request;
        let worker = self
            .handle
            .spawn_blocking(move || ChangeGraph::open(&cwd)?.diff_trees(&before, &after, b""));
        self.diff_task = Some(cx.spawn(async move |this, cx| {
            let result = worker.await;
            let _ = this.update(cx, |this, cx| {
                if this.diff_request != request {
                    return;
                }
                this.diff_loading = false;
                match result {
                    Ok(Ok(diff)) => this.diff = diff,
                    Ok(Err(error)) => this.diff_error = Some(format!("{error:#}")),
                    Err(error) => this.diff_error = Some(format!("Diff worker failed: {error}")),
                }
                cx.notify();
            });
        }));
        cx.notify();
    }

    fn graph(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let Some(state) = &self.state else {
            return div().p_4().text_sm().text_color(rgb(0xa0a0a0)).child(if self.loading { "Loading native changes…" } else { "No native changes yet. Tool edits create live points; shell/manual edits require capture." }).into_any_element();
        };
        let layout = self.layout.clone();
        let graph_width = layout.width;
        let graph_height = (state.nodes.len() + 1) as f32 * ROW_HEIGHT;
        let current = state
            .workspace_heads
            .get(&self.cwd.to_string_lossy().into_owned());
        let rows = state
            .nodes
            .iter()
            .rev()
            .enumerate()
            .map(|(row, node)| {
                let id = node.id.clone();
                let lane = layout.lanes[row];
                let is_selected = self.selected.as_ref() == Some(&node.id);
                let heads: Vec<_> = state
                    .heads
                    .iter()
                    .filter(|(_, head)| *head == &node.id)
                    .map(|(thread, _)| thread.as_str())
                    .collect();
                let title = node
                    .mark
                    .as_ref()
                    .map(|mark| mark.name.clone())
                    .unwrap_or_else(|| {
                        node.provenance
                            .path
                            .clone()
                            .unwrap_or_else(|| format!("{} snapshot", node.provenance.tool))
                    });
                let detail = format!(
                    "{} · {} · {}{}{}",
                    short(&node.id),
                    node.thread,
                    node.provenance.tool,
                    if node.mark.is_some() {
                        " · checkpoint"
                    } else {
                        ""
                    },
                    if current == Some(&node.id) {
                        " · HERE"
                    } else {
                        ""
                    }
                );
                div()
                    .id(("change", row))
                    .relative()
                    .flex()
                    .items_center()
                    .h(px(ROW_HEIGHT))
                    .w_full()
                    .cursor_pointer()
                    .when(is_selected, |row| row.bg(rgb(0x30343b)))
                    .hover(|style| style.bg(rgb(0x292b30)))
                    .on_click(cx.listener(move |this, _: &ClickEvent, _, cx| {
                        this.select(id.clone(), 0, cx)
                    }))
                    .child(
                        div()
                            .relative()
                            .flex_none()
                            .w(px(graph_width))
                            .h_full()
                            .child(
                                div()
                                    .absolute()
                                    .left(px(LANE_WIDTH * (lane as f32 + 1.) - 4.))
                                    .top(px(ROW_HEIGHT / 2. - 4.))
                                    .size(px(8.))
                                    .rounded_full()
                                    .bg(rgb(COLORS[lane % COLORS.len()])),
                            ),
                    )
                    .child(
                        div()
                            .flex_1()
                            .min_w_0()
                            .pr_3()
                            .text_xs()
                            .child(div().truncate().text_color(rgb(0xe4e4e4)).child(title))
                            .child(div().truncate().text_color(rgb(0xaaaaaa)).child(detail))
                            .when(!heads.is_empty(), |column| {
                                column.child(
                                    div()
                                        .truncate()
                                        .text_color(rgb(COLORS[lane % COLORS.len()]))
                                        .child(format!("head: {}", heads.join(", "))),
                                )
                            }),
                    )
            })
            .collect::<Vec<_>>();
        div()
            .relative()
            .w(px((graph_width + 290.).max(360.)))
            .h(px(graph_height))
            .child(
                canvas(
                    move |_, _, _| (),
                    move |bounds, _, window, _| {
                        for &(child_row, child_lane, parent_row, parent_lane) in &layout.edges {
                            let start = bounds.origin
                                + point(
                                    px(LANE_WIDTH * (child_lane as f32 + 1.)),
                                    px(child_row as f32 * ROW_HEIGHT + ROW_HEIGHT / 2.),
                                );
                            let end = bounds.origin
                                + point(
                                    px(LANE_WIDTH * (parent_lane as f32 + 1.)),
                                    px(parent_row as f32 * ROW_HEIGHT + ROW_HEIGHT / 2.),
                                );
                            let mut path = PathBuilder::stroke(px(1.5));
                            path.move_to(start);
                            path.line_to(point(start.x, end.y - px(ROW_HEIGHT / 2.)));
                            path.line_to(end);
                            if let Ok(path) = path.build() {
                                window.paint_path(path, rgb(COLORS[child_lane % COLORS.len()]));
                            }
                        }
                    },
                )
                .absolute()
                .size_full(),
            )
            .children(rows)
            .child(
                div()
                    .flex()
                    .items_center()
                    .h(px(ROW_HEIGHT))
                    .pl(px(graph_width))
                    .text_xs()
                    .text_color(rgb(0xaaaaaa))
                    .child(format!(
                        "○ Git base {}",
                        if state.base_commit.bytes().all(|byte| byte == b'0') {
                            "(new repository)"
                        } else {
                            short(&state.base_commit)
                        }
                    )),
            )
            .into_any_element()
    }

    fn details(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let selected = self.selected.as_ref().and_then(|id| {
            self.state
                .as_ref()?
                .nodes
                .iter()
                .find(|node| &node.id == id)
        });
        let Some(node) = selected else {
            return div()
                .p_3()
                .text_xs()
                .text_color(rgb(0xaaaaaa))
                .child("Select a point to inspect its provenance and diff.")
                .into_any_element();
        };
        let id = node.id.clone();
        let parent_buttons = node
            .parents
            .iter()
            .enumerate()
            .map(|(index, parent)| {
                let id = id.clone();
                div()
                    .id(("diff-parent", index))
                    .px_2()
                    .py_1()
                    .rounded_sm()
                    .cursor_pointer()
                    .bg(rgb(if index == self.parent_index {
                        0x404650
                    } else {
                        0x292b30
                    }))
                    .on_click(cx.listener(move |this, _: &ClickEvent, _, cx| {
                        this.select(id.clone(), index, cx)
                    }))
                    .child(format!("Parent {} {}", index + 1, short(parent)))
            })
            .collect::<Vec<_>>();
        let lines = self
            .diff
            .lines()
            .enumerate()
            .map(|(index, line)| {
                let color = if line.starts_with('+') && !line.starts_with("+++") {
                    0xa5c5ac
                } else if line.starts_with('-') && !line.starts_with("---") {
                    0xd2a0a0
                } else if line.starts_with("@@") {
                    0x93b4d5
                } else {
                    0xbcbcbc
                };
                div()
                    .id(("diff-line", index))
                    .text_color(rgb(color))
                    .child(line.to_owned())
            })
            .collect::<Vec<_>>();
        div()
            .flex()
            .flex_col()
            .min_h_0()
            .h_full()
            .border_t_1()
            .border_color(rgb(0x373737))
            .child(
                div()
                    .flex_none()
                    .p_3()
                    .text_xs()
                    .child(div().text_color(rgb(0xe4e4e4)).child(format!(
                            "{} · {}",
                            node.provenance.tool,
                            node.provenance
                                .path
                                .as_deref()
                                .unwrap_or("workspace snapshot")
                        )))
                    .child(
                        div()
                            .text_color(rgb(0xaaaaaa))
                            .child(format!("Thread: {}", node.thread)),
                    )
                    .child(div().text_color(rgb(0xaaaaaa)).child(format!(
                        "Session: {}",
                        if node.provenance.session_id.is_empty() {
                            "external / unattributed"
                        } else {
                            &node.provenance.session_id
                        }
                    )))
                    .when_some(node.provenance.selection_mode.clone(), |column, mode| {
                        column.child(format!("Selection: {mode}"))
                    })
                    .when_some(node.mark.clone(), |column, mark| {
                        column.child(format!(
                            "Checkpoint: {} · {}\n{}",
                            mark.name,
                            if mark.verified {
                                "verification attested"
                            } else {
                                "not verified"
                            },
                            mark.message
                        ))
                    })
                    .child(
                        div()
                            .mt_2()
                            .flex()
                            .flex_wrap()
                            .gap_1()
                            .children(parent_buttons)
                            .when(node.parents.is_empty(), |row| {
                                row.child("Compared with Git base")
                            }),
                    ),
            )
            .child(
                div()
                    .id("change-diff")
                    .flex_1()
                    .min_h_0()
                    .overflow_scroll()
                    .track_scroll(&self.diff_scroll)
                    .p_3()
                    .text_xs()
                    .font_family("DejaVu Sans Mono")
                    .when(self.diff_loading, |view| view.child("Loading diff…"))
                    .when_some(self.diff_error.clone(), |view, error| {
                        view.child(div().text_color(rgb(0xd2a0a0)).child(error))
                    })
                    .when(
                        !self.diff_loading && self.diff_error.is_none() && self.diff.is_empty(),
                        |view| view.child("No tree differences against this parent."),
                    )
                    .children(lines),
            )
            .into_any_element()
    }
}

fn short(id: &str) -> &str {
    id.get(..8).unwrap_or(id)
}

impl Render for VcsSidebar {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let count = self.state.as_ref().map_or(0, |state| state.nodes.len());
        let checkpoints = self
            .state
            .as_ref()
            .map_or(0, |state| state.checkpoints.len());
        div().flex().flex_col().size_full().min_h_0().bg(rgb(0x202124))
            .child(div().flex_none().p_3().text_xs().child(div().text_color(rgb(0xe4e4e4)).child(format!("LIVE CHANGES · {count} points · {checkpoints} checkpoints"))).child(div().truncate().text_color(rgb(0x999999)).child(self.cwd.display().to_string())).child(div().mt_1().text_color(rgb(0x999999)).child("Newest first · live trees, not Git commits")))
            .when_some(self.error.clone(), |view, error| view.child(div().flex_none().p_3().text_xs().text_color(rgb(0xd2a0a0)).child(format!("Graph unavailable: {error}\nRetrying automatically; any visible graph is the last successful read."))))
            .child(div().id("change-graph").flex_1().min_h_0().overflow_scroll().track_scroll(&self.scroll).child(self.graph(cx)))
            .child(div().flex_none().h(px(if self.selected.is_some() { 270. } else { 48. })).child(self.details(cx)))
            .child(div().flex_none().px_3().py_2().text_xs().text_color(rgb(0x999999)).child("Checkpoint ≠ exported commit. Git projection happens on export."))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use juto_runtime::change_graph::{ChangeNode, EditProvenance};
    use std::collections::BTreeMap;

    #[test]
    fn graph_edges_keep_fork_and_join_parents_and_shared_base() {
        let node = |id: &str, thread: &str, parents: &[&str]| ChangeNode {
            id: id.into(),
            thread: thread.into(),
            parents: parents.iter().map(|parent| (*parent).into()).collect(),
            tree: String::new(),
            provenance: EditProvenance {
                session_id: thread.into(),
                tool: "edit".into(),
                selection_mode: None,
                path: None,
            },
            mark: None,
        };
        let state = GraphState {
            version: 2,
            base_commit: String::new(),
            base_tree: String::new(),
            nodes: vec![
                node("root", "main", &[]),
                node("left", "main", &["root"]),
                node("right", "worker", &["root"]),
                node("join", "main", &["left", "right"]),
            ],
            heads: BTreeMap::new(),
            workspace_heads: BTreeMap::new(),
            checkpoints: BTreeMap::new(),
        };
        let layout = GraphLayout::new(&state);
        assert_eq!(layout.lanes, [0, 1, 0, 0]);
        assert_eq!(
            layout.edges,
            [
                (0, 0, 2, 0),
                (0, 0, 1, 1),
                (1, 1, 3, 0),
                (2, 0, 3, 0),
                (3, 0, 4, 0)
            ]
        );
    }
}
