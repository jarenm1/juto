//! Layered runtime configuration.
//!
//! Behavioral reference: oh-my-pi 579da1d6 `config.yml` layering (global
//! `<agentDir>` + project `.omp/config.yml`, deep map merge, arrays replace)
//! and the selector grammar (`provider/model`, `@role`, `:effort` suffix).
//!
//! Juto reads `.juto/config.yml` layers; `.omp` files are never required but
//! callers may pass them as additional layering paths through `load`/`merge`.

use std::{
    collections::BTreeMap,
    fmt,
    path::{Path, PathBuf},
};

use crate::main_report::{DEFAULT_MAIN_REPORT_BYTES, MIN_MAIN_REPORT_BYTES};
use juto_catalog::Model;
use serde::{Deserialize, Serialize};

use crate::tier::ExecutionTier;

const DEFAULT_MODEL: &str = "";
const VALID_EFFORTS: &[&str] = &[
    "off", "minimal", "low", "medium", "high", "xhigh", "max", "auto", "inherit",
];

#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("config io error on {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("malformed config file {path}: {source}")]
    Parse {
        path: PathBuf,
        #[source]
        source: serde_yaml::Error,
    },
    #[error("config file {path} must contain a YAML mapping at the top level")]
    NotAMap { path: PathBuf },
    #[error("invalid config: {0}")]
    Validation(String),
    #[error("model role cycle involving @{0}")]
    RoleCycle(String),
    #[error("unknown effort level :{0} (supported: {supported})", supported = VALID_EFFORTS.join(", "))]
    UnknownEffort(String),
}

pub type Result<T, E = ConfigError> = std::result::Result<T, E>;

/// Tool approval policy. Mirrors the source `tools.approvalMode`.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub enum ApprovalMode {
    /// Write and Exec tier tools are denied outright.
    ReadOnly,
    /// Ask the host before Write/Exec.
    Ask,
    /// Run everything without prompting. Default: full permission.
    #[default]
    Allow,
}

impl ApprovalMode {
    /// Whether a tool tier is permitted without an interactive approval.
    pub fn permits(&self, tier: juto_agent::ToolTier) -> bool {
        use juto_agent::ToolTier;
        match (self, tier) {
            (Self::ReadOnly, ToolTier::Read) => true,
            (Self::ReadOnly, _) => false,
            (Self::Ask, _) => true, // allowed to proceed to the approval hook
            (Self::Allow, _) => true,
        }
    }
}

/// How compaction archives the history it discards. Set by `compactionMethod`.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum CompactionMethod {
    /// Page images when the model accepts image input, otherwise an LLM summary.
    Auto,
    /// Always archive as page images. Fails on models without image input.
    Snapcompact,
    /// Always write an LLM summary.
    #[default]
    Summary,
}

impl CompactionMethod {
    /// Whether discarded history is archived as page images for a model with this input capability.
    pub fn archives_frames(self, accepts_images: bool) -> bool {
        match self {
            Self::Auto => accepts_images,
            Self::Snapcompact => true,
            Self::Summary => false,
        }
    }
}

fn default_system_prompt() -> String {
    String::new()
}
fn default_max_concurrency() -> usize {
    32
}
fn default_recursion_depth() -> usize {
    2
}
fn default_auto_compact() -> bool {
    true
}
fn default_compaction_threshold() -> f64 {
    0.9
}
fn default_keep_recent() -> usize {
    8
}
fn default_max_active_threads() -> usize {
    6
}

/// Runtime settings. All fields carry serde defaults so partial files layer
/// cleanly; `load` merges raw YAML values before deserialization so deep maps
/// merge and arrays replace like the source.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct RuntimeConfig {
    pub model: String,
    pub model_roles: BTreeMap<String, String>,
    pub system_prompt: String,
    /// Default thinking/effort level, e.g. `"high"`.
    pub thinking: Option<String>,
    pub temperature: Option<f32>,
    pub max_tokens: Option<u64>,
    pub approval_mode: ApprovalMode,
    /// Ceiling for concurrent model requests across the whole run tree, and for
    /// concurrent isolated children (the review limit). The provider pool starts at
    /// this ceiling, halves on rate limits, and grows back on accepted streams.
    /// Ordinary children are not capped by it; each turn waits for a memory desk.
    pub max_concurrency: usize,
    /// Subagent nesting depth. Root agents run at depth 0. A run at depth >= this
    /// value gets no `task` tool. Every depth shares one admission (provider pool,
    /// review limit, desks), so nested children never multiply those limits.
    pub max_recursion_depth: usize,
    pub max_turns: Option<usize>,
    pub auto_compact: bool,
    /// Fraction of the model context window that triggers auto-compaction.
    pub compaction_threshold: f64,
    /// Messages preserved verbatim across a compaction boundary.
    pub keep_recent_messages: usize,
    /// How discarded history is archived. Defaults to `summary`; page images
    /// require explicit `auto` or `snapcompact` opt-in.
    pub compaction_method: CompactionMethod,
    /// Model selector (`provider/model[:effort]` or `@role`) that writes
    /// compaction summaries. Unset: the session's current model compacts.
    pub compactor: Option<String>,
    /// Tool allowlist; empty means all builtin tools.
    pub tools: Vec<String>,
    /// User-defined models merged over the bundled registry.
    pub custom_models: Vec<Model>,
    /// Tier of the top-level run: how confined its shell commands are. Children
    /// get at most this tier. See `ExecutionTier`.
    pub execution_tier: ExecutionTier,
    /// HTTP/HTTPS destinations for workspace-tier shells. Untrusted shells use
    /// only the built-in package mirrors, regardless of project configuration.
    pub network: crate::network::NetworkConfig,
    /// Run attended `bash` commands under resource governance (a systemd scope
    /// with CPU and memory caps). Sandboxed tiers are always governed. Off by
    /// default because it needs `systemd-run` on the host.
    pub bash_scoped: bool,
    /// Internal threads of a main session the router may choose among. Past
    /// this many, the least recently routed thread goes dormant; dormant
    /// threads are reached by history lookup.
    pub max_active_threads: usize,
    /// Byte budget for one child report entering the main thread. Larger
    /// reports are digested, and their full text stays one `wait` away.
    pub main_report_bytes: usize,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            model: DEFAULT_MODEL.into(),
            model_roles: BTreeMap::new(),
            system_prompt: default_system_prompt(),
            thinking: None,
            temperature: None,
            max_tokens: None,
            approval_mode: ApprovalMode::Allow,
            max_concurrency: default_max_concurrency(),
            max_recursion_depth: default_recursion_depth(),
            max_turns: None,
            auto_compact: default_auto_compact(),
            compaction_threshold: default_compaction_threshold(),
            keep_recent_messages: default_keep_recent(),
            compaction_method: CompactionMethod::Summary,
            compactor: None,
            tools: Vec::new(),
            custom_models: Vec::new(),
            execution_tier: ExecutionTier::Attended,
            network: crate::network::NetworkConfig::default(),
            bash_scoped: false,
            max_active_threads: default_max_active_threads(),
            main_report_bytes: DEFAULT_MAIN_REPORT_BYTES,
        }
    }
}

impl RuntimeConfig {
    /// Layer `project_file` over `global_file` and validate. Missing files
    /// are tolerated; malformed files error; the top level of each present
    /// file must be a mapping.
    pub fn load(global_file: &Path, project_file: &Path) -> Result<Self> {
        let mut layers = Vec::new();
        for path in [global_file, project_file] {
            match std::fs::read_to_string(path) {
                Ok(text) => {
                    let value: serde_yaml::Value =
                        serde_yaml::from_str(&text).map_err(|source| ConfigError::Parse {
                            path: path.to_path_buf(),
                            source,
                        })?;
                    match value {
                        serde_yaml::Value::Null => {}
                        serde_yaml::Value::Mapping(_) => layers.push((path.to_path_buf(), value)),
                        _ => {
                            return Err(ConfigError::NotAMap {
                                path: path.to_path_buf(),
                            });
                        }
                    }
                }
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                Err(source) => {
                    return Err(ConfigError::Io {
                        path: path.to_path_buf(),
                        source,
                    });
                }
            }
        }
        let mut merged = serde_yaml::Value::Mapping(Default::default());
        for (_, layer) in layers {
            merge_value(&mut merged, layer);
        }
        let config: Self = serde_yaml::from_value(merged).map_err(|source| ConfigError::Parse {
            path: project_file.to_path_buf(),
            source,
        })?;
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<()> {
        self.network.validate().map_err(ConfigError::Validation)?;
        if self.max_concurrency == 0 {
            return Err(ConfigError::Validation(
                "maxConcurrency must be positive".into(),
            ));
        }
        if self.max_active_threads == 0 {
            return Err(ConfigError::Validation(
                "maxActiveThreads must be positive".into(),
            ));
        }
        if !(self.compaction_threshold > 0.0 && self.compaction_threshold <= 1.0) {
            return Err(ConfigError::Validation(format!(
                "compactionThreshold must be in (0, 1], got {}",
                self.compaction_threshold
            )));
        }
        if let Some(t) = self.temperature
            && (!t.is_finite() || t < 0.0)
        {
            return Err(ConfigError::Validation(format!(
                "temperature must be a non-negative finite number, got {t}"
            )));
        }
        if self.main_report_bytes < MIN_MAIN_REPORT_BYTES {
            return Err(ConfigError::Validation(format!(
                "mainReportBytes must be at least {MIN_MAIN_REPORT_BYTES}, got {}",
                self.main_report_bytes
            )));
        }
        if let Some(level) = &self.thinking
            && !VALID_EFFORTS.contains(&level.as_str())
        {
            return Err(ConfigError::UnknownEffort(level.clone()));
        }
        if let Some(selector) = &self.compactor {
            let (base, _) = split_effort(selector);
            if base.trim().is_empty() {
                return Err(ConfigError::Validation(format!(
                    "compactor must name a model or @role, got {selector:?}"
                )));
            }
            // Same role resolution as modelRoles: cycles and unknown effort fail
            // here. Unknown provider/model names are a registry lookup and fail at
            // compaction (`Runtime::resolve_model`), not at load.
            let _ = self.resolve_selector(selector)?;
        }
        // Role selectors must resolve (no cycles) — validate eagerly so a
        // broken config fails at load, not mid-run.
        for role in self.model_roles.keys() {
            let _ = self.resolve_selector(&format!("@{role}"))?;
        }
        // Custom models must carry the minimum metadata the registry needs.
        for model in &self.custom_models {
            if model.id.is_empty() || model.provider.is_empty() || model.api.is_empty() {
                return Err(ConfigError::Validation(format!(
                    "custom model {:?} needs non-empty id/provider/api",
                    model.id
                )));
            }
            if model.compat.get("apiKey").is_some_and(|v| v.is_string()) {
                // Secrets belong in the credential store, not in a config file.
                return Err(ConfigError::Validation(format!(
                    "custom model {:?} embeds a credential field in compat.apiKey",
                    model.id
                )));
            }
        }
        Ok(())
    }

    /// Resolve a model selector of the form `[provider/]model[:effort]` or
    /// `@role[:effort]` into `(model_selector, thinking_level)`.
    ///
    /// `@role` follows `model_roles` transitively with cycle detection and a
    /// source-compatible inheritance chain (`tiny→smol→default`,
    /// `memory→tiny`). The `:effort` suffix is only split on the last colon
    /// when it names a supported effort, so Windows drives and `host:port`
    /// base URLs inside selectors are not misparsed.
    pub fn resolve_selector(&self, selector: &str) -> Result<(String, Option<String>)> {
        let (base, inline_effort) = split_effort(selector);
        let mut effort = inline_effort.map(str::to_string);
        let mut current = base.to_string();
        let mut seen = std::collections::HashSet::new();
        while let Some(role) = current.strip_prefix('@') {
            // A role may itself carry an effort suffix; later suffixes win
            // only when the user did not already specify one.
            let (role_name, role_effort) = split_effort(role);
            if effort.is_none() {
                effort = role_effort.map(str::to_string);
            }
            if !seen.insert(role_name.to_string()) {
                return Err(ConfigError::RoleCycle(role_name.to_string()));
            }
            match self.model_roles.get(role_name) {
                Some(target) => {
                    let (target_base, target_effort) = split_effort(target);
                    if effort.is_none() {
                        effort = target_effort.map(str::to_string);
                    }
                    current = target_base.to_string();
                }
                None => {
                    // Source role inheritance: tiny→smol→default,
                    // memory→tiny. The `default` role without an explicit
                    // entry is the configured `model` field itself.
                    let inherited = match role_name {
                        "tiny" => Some("smol"),
                        "smol" => Some("default"),
                        "memory" => Some("tiny"),
                        _ => None,
                    };
                    match inherited {
                        Some(next) => {
                            current = format!("@{next}");
                        }
                        None if role_name == "default" && !self.model.is_empty() => {
                            current = self.model.clone();
                        }
                        _ => {
                            // Unknown role or exhausted chain: pass the
                            // literal selector through so the registry
                            // lookup reports the real failure.
                            return Ok((format!("@{role_name}"), effort));
                        }
                    }
                }
            }
        }
        if let Some(level) = &effort
            && !VALID_EFFORTS.contains(&level.as_str())
        {
            return Err(ConfigError::UnknownEffort(level.clone()));
        }
        Ok((current, effort))
    }

    /// Role → raw selector for hosts that need the map entry unchanged.
    pub fn role_selector(&self, role: &str) -> Option<&str> {
        self.model_roles.get(role).map(String::as_str)
    }
}

/// Split a `:effort` suffix only when it names a supported level, so
/// `provider/host:8080/model` and `c:\path` selectors keep their colons.
fn split_effort(selector: &str) -> (&str, Option<&str>) {
    match selector.rsplit_once(':') {
        Some((base, suffix)) if VALID_EFFORTS.contains(&suffix) => (base, Some(suffix)),
        _ => (selector, None),
    }
}

/// Deep merge `overlay` into `base`: mappings merge recursively, every other
/// value (including arrays and `null` tombstones) replaces.
pub(crate) fn merge_value(base: &mut serde_yaml::Value, overlay: serde_yaml::Value) {
    use serde_yaml::Value;
    match (base, overlay) {
        (Value::Mapping(base_map), Value::Mapping(overlay_map)) => {
            for (key, value) in overlay_map {
                match base_map.get_mut(&key) {
                    Some(slot @ Value::Mapping(_)) if matches!(value, Value::Mapping(_)) => {
                        merge_value(slot, value);
                    }
                    slot => {
                        if matches!(value, Value::Null) {
                            base_map.remove(&key);
                        } else if let Some(slot) = slot {
                            *slot = value;
                        } else {
                            base_map.insert(key, value);
                        }
                    }
                }
            }
        }
        (slot, value) => *slot = value,
    }
}

/// Redacted view of settings for logs: same shape, secrets never present.
/// `RuntimeConfig` itself stores no credentials; this guard exists so future
/// string fields can't accidentally surface keys through `Debug`.
impl fmt::Display for RuntimeConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "RuntimeConfig(model={:?}, roles={})",
            self.model,
            self.model_roles.len()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write(dir: &Path, name: &str, body: &str) -> PathBuf {
        let path = dir.join(name);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, body).unwrap();
        path
    }

    #[test]
    fn defaults_are_source_shaped() {
        let cfg = RuntimeConfig::default();
        assert_eq!(cfg.approval_mode, ApprovalMode::Allow);
        assert!(cfg.max_concurrency > 0);
        assert!((0.0..=1.0).contains(&cfg.compaction_threshold));
        assert!(cfg.approval_mode.permits(juto_agent::ToolTier::Read));
        assert!(!ApprovalMode::ReadOnly.permits(juto_agent::ToolTier::Write));
        assert!(!ApprovalMode::ReadOnly.permits(juto_agent::ToolTier::Exec));
        assert!(ApprovalMode::Allow.permits(juto_agent::ToolTier::Exec));
    }

    #[test]
    fn compaction_method_selects_archive_by_model_input() {
        assert!(CompactionMethod::Auto.archives_frames(true));
        assert!(!CompactionMethod::Auto.archives_frames(false));
        assert!(CompactionMethod::Snapcompact.archives_frames(false));
        assert!(!CompactionMethod::Summary.archives_frames(true));
        let parsed: CompactionMethod = serde_yaml::from_str("snapcompact").unwrap();
        assert_eq!(parsed, CompactionMethod::Snapcompact);
    }

    #[test]
    fn layered_load_merges_maps_and_replaces_arrays() {
        let dir = tempfile::tempdir().unwrap();
        let global = write(
            dir.path(),
            "global.yml",
            "model: openai/gpt-5\nmodelRoles:\n  smol: openai/gpt-5-mini\n  slow: anthropic/claude\ntools: [read, grep]\nmaxConcurrency: 4\n",
        );
        let project = write(
            dir.path(),
            ".juto/config.yml",
            "modelRoles:\n  smol: google/gemini-flash\ntools: [bash]\n",
        );
        let cfg = RuntimeConfig::load(&global, &project).unwrap();
        assert_eq!(cfg.model, "openai/gpt-5");
        assert_eq!(cfg.model_roles["smol"], "google/gemini-flash");
        assert_eq!(cfg.model_roles["slow"], "anthropic/claude");
        assert_eq!(cfg.tools, vec!["bash".to_string()]);
        assert_eq!(cfg.max_concurrency, 4);
    }

    #[test]
    fn missing_files_tolerated_malformed_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let missing = dir.path().join("none.yml");
        let cfg = RuntimeConfig::load(&missing, &missing).unwrap();
        assert_eq!(cfg.model, RuntimeConfig::default().model);
        assert_eq!(
            cfg.max_concurrency,
            RuntimeConfig::default().max_concurrency
        );
        let bad = write(dir.path(), "bad.yml", "model: [unclosed");
        assert!(matches!(
            RuntimeConfig::load(&bad, &missing),
            Err(ConfigError::Parse { .. })
        ));
        let nonmap = write(dir.path(), "list.yml", "- a\n- b\n");
        assert!(matches!(
            RuntimeConfig::load(&nonmap, &missing),
            Err(ConfigError::NotAMap { .. })
        ));
    }

    #[test]
    fn validation_bounds() {
        let dir = tempfile::tempdir().unwrap();
        let missing = dir.path().join("none.yml");
        let bad = write(dir.path(), "bad.yml", "maxConcurrency: 0\n");
        assert!(RuntimeConfig::load(&bad, &missing).is_err());
        let bad = write(dir.path(), "bad2.yml", "compactionThreshold: 1.5\n");
        assert!(RuntimeConfig::load(&bad, &missing).is_err());
        let bad = write(dir.path(), "bad3.yml", "compactionThreshold: 0\n");
        assert!(RuntimeConfig::load(&bad, &missing).is_err());
    }

    #[test]
    fn selector_roles_and_cycles() {
        let dir = tempfile::tempdir().unwrap();
        let missing = dir.path().join("none.yml");
        let file = write(
            dir.path(),
            "cfg.yml",
            "modelRoles:\n  smol: openai/mini\n  task: \"@smol:low\"\n  a: \"@b\"\n  b: \"@a\"\n",
        );
        // Cyclic roles fail at load.
        assert!(matches!(
            RuntimeConfig::load(&file, &missing),
            Err(ConfigError::RoleCycle(_))
        ));
        let file = write(
            dir.path(),
            "ok.yml",
            "modelRoles:\n  smol: openai/mini\n  task: \"@smol\"\n  default: anthropic/claude\n",
        );
        let cfg = RuntimeConfig::load(&file, &missing).unwrap();
        assert_eq!(cfg.resolve_selector("@task").unwrap().0, "openai/mini");
        // tiny inherits smol→default per source chain.
        assert_eq!(cfg.resolve_selector("@tiny").unwrap().0, "openai/mini");
        let (model, effort) = cfg.resolve_selector("@task:high").unwrap();
        assert_eq!(model, "openai/mini");
        assert_eq!(effort.as_deref(), Some("high"));
        let (model, effort) = cfg.resolve_selector("openai/gpt-5:low").unwrap();
        assert_eq!(model, "openai/gpt-5");
        assert_eq!(effort.as_deref(), Some("low"));
        // Colons that are not efforts are preserved (host:port ids).
        let (model, effort) = cfg.resolve_selector("local/llama:8080").unwrap();
        assert_eq!(model, "local/llama:8080");
        assert_eq!(effort, None);
        assert!(
            cfg.resolve_selector("m:bogus-effort-that-is-not-real")
                .is_ok()
        );
        let (model, _) = cfg.resolve_selector("plain-model").unwrap();
        assert_eq!(model, "plain-model");
        // Unknown roles pass through for the registry to reject, not silently
        // rebound to the default.
        assert_eq!(
            cfg.resolve_selector("@nosuchrole").unwrap().0,
            "@nosuchrole"
        );
    }

    #[test]
    fn default_role_falls_back_to_model_field() {
        let dir = tempfile::tempdir().unwrap();
        let missing = dir.path().join("none.yml");
        let file = write(dir.path(), "cfg.yml", "model: acme/main\n");
        let cfg = RuntimeConfig::load(&file, &missing).unwrap();
        assert_eq!(cfg.resolve_selector("@default").unwrap().0, "acme/main");
        // Chain end: memory→tiny→smol→default→model.
        assert_eq!(cfg.resolve_selector("@memory").unwrap().0, "acme/main");
    }

    #[test]
    fn effort_on_plain_selector() {
        let cfg = RuntimeConfig::default();
        let (model, effort) = cfg.resolve_selector("plain-model:medium").unwrap();
        assert_eq!(model, "plain-model");
        assert_eq!(effort.as_deref(), Some("medium"));
    }

    #[test]
    fn explicit_effort_suffixes() {
        let cfg = RuntimeConfig::default();
        let (m, e) = cfg.resolve_selector("p/m:max").unwrap();
        assert_eq!((m.as_str(), e.as_deref()), ("p/m", Some("max")));
        let (m, e) = cfg.resolve_selector("p/m:off").unwrap();
        assert_eq!((m.as_str(), e.as_deref()), ("p/m", Some("off")));
    }

    #[test]
    fn custom_models_validated() {
        let dir = tempfile::tempdir().unwrap();
        let missing = dir.path().join("none.yml");
        let file = write(
            dir.path(),
            "m.yml",
            "customModels:\n  - id: \"\"\n    provider: x\n    api: openai\n",
        );
        assert!(RuntimeConfig::load(&file, &missing).is_err());
    }

    #[test]
    fn compactor_layers_and_stays_unset_by_default() {
        let dir = tempfile::tempdir().unwrap();
        let missing = dir.path().join("none.yml");
        assert_eq!(
            RuntimeConfig::load(&missing, &missing).unwrap().compactor,
            None
        );
        let global = write(
            dir.path(),
            "global.yml",
            "compactor: openai/gpt-5-mini:low\n",
        );
        let cfg = RuntimeConfig::load(&global, &missing).unwrap();
        assert_eq!(cfg.compactor.as_deref(), Some("openai/gpt-5-mini:low"));
        let project = write(dir.path(), ".juto/config.yml", "compactor: \"@smol\"\n");
        let cfg = RuntimeConfig::load(&global, &project).unwrap();
        assert_eq!(cfg.compactor.as_deref(), Some("@smol"));
        // A null tombstone in the project layer unsets the global value.
        let unset = write(dir.path(), "unset.yml", "compactor: null\n");
        assert_eq!(
            RuntimeConfig::load(&global, &unset).unwrap().compactor,
            None
        );
    }

    #[test]
    fn invalid_compactor_selector_fails_load() {
        let dir = tempfile::tempdir().unwrap();
        let missing = dir.path().join("none.yml");
        for body in [
            "compactor: \"\"\n",
            "compactor: \"  \"\n",
            "compactor: \":high\"\n",
        ] {
            let file = write(dir.path(), "bad.yml", body);
            assert!(
                matches!(
                    RuntimeConfig::load(&file, &missing),
                    Err(ConfigError::Validation(_))
                ),
                "selector {body:?} must be rejected"
            );
        }
    }
}
