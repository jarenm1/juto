//! Agent roles: named profiles that set which tools a run may use, which
//! model a child run defaults to, and what instructions it starts with.
//!
//! A role is a markdown file. Its frontmatter names the tools it may use and,
//! optionally, a model selector (`provider/model` or an `@role` alias such as
//! `@smol`), and its body is appended to the run's system prompt. Built-in
//! roles ship with the crate. A `<role>.md` file under `.juto/agents/` in the
//! working directory or one of its ancestors, or under the runtime data
//! directory's `agents/`, replaces the built-in role of the same name. Nearer
//! files win.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use anyhow::{Context as _, Result, bail};
use juto_agent::ToolRegistry;
use serde::Deserialize;

use crate::context::{ContextError, split_frontmatter};
use crate::todo;

/// Role used by a child run whose `task` entry names none.
pub(crate) const DEFAULT_ROLE: &str = "task";

/// Role of the top-level thread.
pub(crate) const MAIN_ROLE: &str = "main";

/// Read-only builtin tools, by registry name. `agent_read` reads other agents'
/// status and transcripts, so a reviewer can see how a builder got its result;
/// `yield` only hands a child's result back.
const READ_ONLY_TOOLS: &[&str] = &[
    "read",
    "glob",
    "grep",
    "recall",
    "zoom",
    "agent_read",
    "yield",
];

const BUILTIN_ROLES: &[(&str, &str)] = &[
    ("main", include_str!("roles/main.md")),
    ("builder", include_str!("roles/builder.md")),
    ("scout", include_str!("roles/scout.md")),
    ("reviewer", include_str!("roles/reviewer.md")),
    ("task", include_str!("roles/task.md")),
    ("browser", include_str!("roles/browser.md")),
];

/// Registry tool names a run may advertise. `None` allows every tool the run
/// registers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ToolScope {
    allowed: Option<BTreeSet<String>>,
}

impl ToolScope {
    pub(crate) fn all() -> Self {
        Self { allowed: None }
    }

    pub(crate) fn read_only() -> Self {
        Self::named(READ_ONLY_TOOLS.iter().copied())
    }

    /// `read_only()` plus `todo`: a read-only run still keeps its own checklist.
    pub(crate) fn read_only_run() -> Self {
        Self::named(READ_ONLY_TOOLS.iter().copied().chain([todo::TOOL_NAME]))
    }

    pub(crate) fn named<S: Into<String>>(names: impl IntoIterator<Item = S>) -> Self {
        Self {
            allowed: Some(names.into_iter().map(Into::into).collect()),
        }
    }

    pub(crate) fn allows(&self, name: &str) -> bool {
        self.allowed
            .as_ref()
            .is_none_or(|names| names.contains(name))
    }

    /// The tools allowed by both scopes.
    pub(crate) fn intersect(&self, other: &Self) -> Self {
        let allowed = match (&self.allowed, &other.allowed) {
            (None, None) => None,
            (Some(names), None) | (None, Some(names)) => Some(names.clone()),
            (Some(a), Some(b)) => Some(a.intersection(b).cloned().collect()),
        };
        Self { allowed }
    }

    /// Unregister every tool outside this scope.
    pub(crate) fn apply(&self, registry: &mut ToolRegistry) {
        if let Some(names) = &self.allowed {
            let names: Vec<String> = names.iter().cloned().collect();
            registry.retain_names(&names);
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct AgentRole {
    pub(crate) name: String,
    pub(crate) description: String,
    pub(crate) tools: ToolScope,
    /// Model selectors a child run of this role tries in order when its
    /// `task` entry names no model (oh-my-pi agent `model:` frontmatter).
    pub(crate) models: Vec<String>,
    /// Optional execution-tier ceiling; a role can tighten, never loosen a run.
    pub(crate) tier: Option<crate::tier::ExecutionTier>,
    /// Markdown body, appended to the run's system prompt.
    pub(crate) instructions: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RoleFrontmatter {
    description: String,
    /// Omitted means read-only, so a role must grant write access explicitly.
    tools: Option<ToolsField>,
    /// A selector or an ordered fallback list.
    model: Option<ModelField>,
    #[serde(default)]
    tier: Option<crate::tier::ExecutionTier>,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum ToolsField {
    /// `all` or `read-only`.
    Keyword(String),
    /// Explicit registry tool names.
    Named(Vec<String>),
}

#[derive(Deserialize)]
#[serde(untagged)]
enum ModelField {
    One(String),
    Fallbacks(Vec<String>),
}

fn parse_role(path: &Path, name: &str, text: &str) -> Result<AgentRole> {
    let Some((yaml, body)) = split_frontmatter(path, text)? else {
        return Err(ContextError::Frontmatter {
            path: path.to_path_buf(),
            reason: "an agent role needs a frontmatter block with a description".into(),
        }
        .into());
    };
    let fm: RoleFrontmatter =
        serde_yaml::from_str(yaml).map_err(|e| ContextError::Frontmatter {
            path: path.to_path_buf(),
            reason: e.to_string(),
        })?;
    let tools = match fm.tools {
        None => ToolScope::read_only(),
        Some(ToolsField::Keyword(keyword)) => match keyword.as_str() {
            "all" => ToolScope::all(),
            "read-only" => ToolScope::read_only(),
            other => bail!(
                "{}: unknown tools keyword `{other}`; use all, read-only, or a list of tool names",
                path.display()
            ),
        },
        Some(ToolsField::Named(names)) => ToolScope::named(names),
    };
    let models = match fm.model {
        None => Vec::new(),
        Some(ModelField::One(model)) => vec![model],
        Some(ModelField::Fallbacks(models)) => models,
    };
    if models.iter().any(|m| m.trim().is_empty()) {
        bail!("{}: model selectors must not be empty", path.display());
    }
    Ok(AgentRole {
        name: name.to_owned(),
        description: fm.description,
        tools,
        tier: fm.tier,
        models,
        instructions: body.trim().to_owned(),
    })
}

fn scan_roles_dir(dir: &Path, roles: &mut BTreeMap<String, AgentRole>) -> Result<()> {
    let entries = match std::fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(e) => {
            return Err(e).with_context(|| format!("failed to read {}", dir.display()));
        }
    };
    for entry in entries {
        let path = entry
            .with_context(|| format!("failed to read {}", dir.display()))?
            .path();
        if path.extension().and_then(|ext| ext.to_str()) != Some("md") || !path.is_file() {
            continue;
        }
        let Some(name) = path.file_stem().and_then(|stem| stem.to_str()) else {
            continue;
        };
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        let role = parse_role(&path, name, &text)?;
        roles.entry(role.name.clone()).or_insert(role);
    }
    Ok(())
}

/// Load every role visible from `cwd`: project and global role files, nearest
/// first, then the built-ins for names no file claimed.
pub(crate) fn load(cwd: &Path, global_dir: &Path) -> Result<BTreeMap<String, AgentRole>> {
    let mut roles = BTreeMap::new();
    for dir in cwd.ancestors() {
        scan_roles_dir(&dir.join(".juto/agents"), &mut roles)?;
    }
    scan_roles_dir(&global_dir.join("agents"), &mut roles)?;
    for (name, text) in BUILTIN_ROLES {
        if !roles.contains_key(*name) {
            let role = parse_role(Path::new(name), name, text)?;
            roles.insert((*name).to_owned(), role);
        }
    }
    Ok(roles)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn write_role(dir: &Path, name: &str, text: &str) {
        std::fs::create_dir_all(dir).expect("create agents dir");
        std::fs::write(dir.join(format!("{name}.md")), text).expect("write role");
    }

    #[test]
    fn builtin_roles_grant_only_their_declared_tools() {
        let root = tempdir().expect("tempdir");
        let roles = load(root.path(), &root.path().join("data")).expect("load roles");

        let main = &roles["main"];
        assert!(main.tools.allows("task") && main.tools.allows("read"));
        assert!(!main.tools.allows("write"));
        assert!(!main.tools.allows("bash"));
        assert!(!main.tools.allows("vcs"));
        assert!(roles["builder"].tools.allows("write"));
        assert!(!roles["scout"].tools.allows("write"));
        assert!(!roles["scout"].tools.allows("task"));
        assert!(!roles["reviewer"].tools.allows("bash"));
        let browser = &roles["browser"];
        assert!(browser.tools.allows("browser"));
        assert!(!browser.tools.allows("bash") && !browser.tools.allows("write"));
        assert!(!browser.tools.allows("task"));
        assert_eq!(browser.models, ["@smol"]);
        assert!(roles["task"].models.is_empty());
    }

    #[test]
    fn nearest_role_file_replaces_the_builtin() {
        let root = tempdir().expect("tempdir");
        let project = root.path().join("project");
        let nested = project.join("nested");
        write_role(
            &project.join(".juto/agents"),
            "scout",
            "---\ndescription: wide\ntools: all\n---\n",
        );
        write_role(
            &nested.join(".juto/agents"),
            "scout",
            "---\ndescription: narrow\ntools: [read]\n---\n",
        );

        let roles = load(&nested, &root.path().join("data")).expect("load roles");
        let scout = &roles["scout"];
        assert_eq!(scout.description, "narrow");
        assert!(scout.tools.allows("read"));
        assert!(!scout.tools.allows("grep"));
    }

    #[test]
    fn role_without_tools_key_is_read_only() {
        let root = tempdir().expect("tempdir");
        write_role(
            &root.path().join(".juto/agents"),
            "auditor",
            "---\ndescription: audits\n---\nLook closely.\n",
        );

        let roles = load(root.path(), &root.path().join("data")).expect("load roles");
        let auditor = &roles["auditor"];
        assert_eq!(auditor.instructions, "Look closely.");
        assert!(auditor.tools.allows("read"));
        assert!(!auditor.tools.allows("write"));
    }

    #[test]
    fn role_parses_execution_tier_ceiling() {
        let role = parse_role(
            Path::new("auditor.md"),
            "auditor",
            "---\ndescription: audits\ntier: untrusted\n---\n",
        )
        .unwrap();
        assert_eq!(role.tier, Some(crate::tier::ExecutionTier::Untrusted));
    }

    #[test]
    fn unknown_tools_keyword_fails_to_load() {
        let root = tempdir().expect("tempdir");
        write_role(
            &root.path().join(".juto/agents"),
            "bad",
            "---\ndescription: broken\ntools: everything\n---\n",
        );

        let err = load(root.path(), &root.path().join("data")).expect_err("unknown keyword");
        assert!(format!("{err:#}").contains("everything"), "{err:#}");
    }

    #[test]
    fn role_model_accepts_a_selector_or_fallback_list() {
        let root = tempdir().expect("tempdir");
        let agents = root.path().join(".juto/agents");
        write_role(
            &agents,
            "one",
            "---\ndescription: d\nmodel: openai/gpt-5\n---\n",
        );
        write_role(
            &agents,
            "many",
            "---\ndescription: d\nmodel: [\"@smol\", openai/gpt-5-mini]\n---\n",
        );
        let roles = load(root.path(), &root.path().join("data")).expect("load roles");
        assert_eq!(roles["one"].models, ["openai/gpt-5"]);
        assert_eq!(roles["many"].models, ["@smol", "openai/gpt-5-mini"]);

        write_role(&agents, "blank", "---\ndescription: d\nmodel: \"\"\n---\n");
        let err = load(root.path(), &root.path().join("data")).expect_err("empty model");
        assert!(format!("{err:#}").contains("must not be empty"), "{err:#}");
    }

    #[test]
    fn todo_is_granted_to_subagents_that_build_or_research_not_main_or_reviewer() {
        let root = tempdir().expect("tempdir");
        let roles = load(root.path(), &root.path().join("data")).expect("load roles");

        assert!(!roles["main"].tools.allows(todo::TOOL_NAME));
        assert!(roles["builder"].tools.allows(todo::TOOL_NAME));
        assert!(roles["task"].tools.allows(todo::TOOL_NAME));
        assert!(roles["scout"].tools.allows(todo::TOOL_NAME));
        assert!(!roles["reviewer"].tools.allows(todo::TOOL_NAME));
    }

    #[test]
    fn read_only_run_keeps_todo_only_where_the_role_grants_it() {
        let root = tempdir().expect("tempdir");
        let roles = load(root.path(), &root.path().join("data")).expect("load roles");
        let run = |role: &str| roles[role].tools.intersect(&ToolScope::read_only_run());

        assert!(run("scout").allows(todo::TOOL_NAME));
        assert!(run("scout").allows("read"));
        assert!(!run("scout").allows("write"));
        assert!(!run("reviewer").allows(todo::TOOL_NAME));
        assert!(!run("builder").allows("write"));
    }
}
