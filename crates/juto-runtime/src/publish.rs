//! `publish`: the privileged tool that pushes a jj bookmark and opens a GitHub
//! pull request for it.
//!
//! It runs on the host, outside any run sandbox, using the operator's own `jj`
//! and `gh` credentials. Credentials never enter a sandbox. The tool is never
//! registered for `untrusted` runs, and an unattended run needs an operator
//! decision for each bookmark before it may publish, and then pushes only the commit that decision pinned.

use std::path::{Path, PathBuf};
use std::process::Stdio;

use async_trait::async_trait;
use juto_agent::{Tool, ToolConcurrency, ToolContext, ToolOutput, ToolTier};
use juto_ai::ContentBlock;
use serde::Deserialize;
use serde_json::{Value, json};
use tokio_util::sync::CancellationToken;

use crate::decisions::{self, SharedScope};

/// Name the model calls the tool by.
pub const TOOL_NAME: &str = "publish";

/// Base branch used when the call does not name one.
const DEFAULT_BASE: &str = "main";

pub struct PublishTool {
    pub cwd: PathBuf,
    /// Set for unattended runs: the push runs only if the bookmark still points at
    /// the commit its approved decision pinned.
    pub unattended: Option<SharedScope>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PublishParams {
    bookmark: String,
    title: String,
    #[serde(default)]
    body: String,
    base: Option<String>,
}

/// Branch names: letters, digits, and `._/-`, never starting with `-` and never
/// containing `..`. The value is passed as an argument, never through a shell.
fn valid_branch(name: &str) -> bool {
    !name.is_empty()
        && !name.starts_with('-')
        && !name.contains("..")
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '/' | '-'))
}

#[async_trait]
impl Tool for PublishTool {
    fn definition(&self) -> juto_ai::ToolDefinition {
        juto_ai::ToolDefinition {
            name: TOOL_NAME.to_string(),
            description: "Push a jj bookmark to the git remote and open a GitHub pull request \
				for it. Runs on the host outside the sandbox with the operator's credentials. \
				Unattended runs need an operator decision for each bookmark."
                .to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "bookmark": {"type": "string", "description": "jj bookmark (branch) to push."},
                    "title": {"type": "string", "description": "Pull request title."},
                    "body": {"type": "string", "description": "Pull request body."},
                    "base": {"type": "string", "description": "Base branch (default main)."},
                },
                "required": ["bookmark", "title"],
            }),
        }
    }

    fn tier(&self) -> ToolTier {
        ToolTier::Exec
    }

    fn concurrency(&self) -> ToolConcurrency {
        ToolConcurrency::Exclusive
    }

    async fn execute(&self, arguments: Value, ctx: ToolContext) -> Result<ToolOutput, String> {
        let digest = decisions::args_digest(TOOL_NAME, &arguments);
        let params: PublishParams = serde_json::from_value(arguments)
            .map_err(|error| format!("invalid publish arguments: {error}"))?;
        if !valid_branch(&params.bookmark) {
            return Err(format!("invalid bookmark `{}`", params.bookmark));
        }
        let base = params.base.unwrap_or_else(|| DEFAULT_BASE.to_string());
        if !valid_branch(&base) {
            return Err(format!("invalid base branch `{base}`"));
        }
        if params.title.trim().is_empty() {
            return Err("publish needs a non-empty title".to_string());
        }
        if let Some(scope) = &self.unattended {
            let grant = scope.take_admitted(&digest).ok_or_else(|| {
                format!(
                    "publish of `{}` has no approved decision in this run",
                    params.bookmark
                )
            })?;
            let pinned = grant
                .pinned_commit
                .as_deref()
                .ok_or_else(|| format!("decision {} pins no commit", grant.id))?;
            let current = resolve_bookmark(&self.cwd, &params.bookmark, &ctx.cancel).await?;
            if current != pinned {
                return Err(format!(
                    "bookmark `{}` moved from {pinned} to {current} after decision {} was recorded; nothing was pushed",
                    params.bookmark, grant.id
                ));
            }
        }
        run(
            "jj",
            &["git", "push", "--bookmark", &params.bookmark],
            &self.cwd,
            &ctx.cancel,
        )
        .await?;
        let url = run(
            "gh",
            &[
                "pr",
                "create",
                "--head",
                &params.bookmark,
                "--base",
                &base,
                "--title",
                &params.title,
                "--body",
                &params.body,
            ],
            &self.cwd,
            &ctx.cancel,
        )
        .await?;
        Ok(ToolOutput {
            content: vec![ContentBlock::Text {
                text: format!("published {} to {base}: {url}", params.bookmark),
            }],
            details: json!({"bookmark": params.bookmark, "base": base, "url": url}),
            is_error: false,
        })
    }
}

/// Runs one host command with arguments passed directly, returning trimmed stdout.
async fn run(
    program: &str,
    args: &[&str],
    cwd: &std::path::Path,
    cancel: &CancellationToken,
) -> Result<String, String> {
    let child = tokio::process::Command::new(program)
        .args(args)
        .current_dir(cwd)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true)
        .spawn()
        .map_err(|error| format!("could not run `{program}`: {error}"))?;
    let output = tokio::select! {
        output = child.wait_with_output() => output.map_err(|error| format!("`{program}` failed: {error}"))?,
        _ = cancel.cancelled() => return Err(format!("`{program}` cancelled")),
    };
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!(
            "`{program}` exited with {}: {}",
            output.status,
            stderr.trim()
        ));
    }
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

/// The commit `bookmark` points at right now, read without snapshotting the
/// working copy. Fails unless the bookmark names exactly one commit.
pub async fn resolve_bookmark(
    cwd: &Path,
    bookmark: &str,
    cancel: &CancellationToken,
) -> Result<String, String> {
    if !valid_branch(bookmark) {
        return Err(format!("invalid bookmark `{bookmark}`"));
    }
    let revset = format!("bookmarks(exact:\"{bookmark}\")");
    let stdout = run(
        "jj",
        &[
            "--ignore-working-copy",
            "log",
            "--no-graph",
            "-r",
            &revset,
            "-T",
            "commit_id",
        ],
        cwd,
        cancel,
    )
    .await?;
    let commits: Vec<&str> = stdout.lines().collect();
    match commits.as_slice() {
        [commit] => Ok((*commit).to_owned()),
        [] => Err(format!("bookmark `{bookmark}` does not exist")),
        _ => Err(format!("bookmark `{bookmark}` is conflicted")),
    }
}

#[cfg(test)]
mod tests {
    use super::valid_branch;

    #[test]
    fn branch_names_are_argument_safe() {
        assert!(valid_branch("feat/x-1.2_ok"));
        assert!(!valid_branch("-force"));
        assert!(!valid_branch("a..b"));
        assert!(!valid_branch("a;rm -rf"));
        assert!(!valid_branch(""));
    }
}
