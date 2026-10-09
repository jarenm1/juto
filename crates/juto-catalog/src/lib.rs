//! Model catalog: bundled OMP model metadata, custom overrides, and exact
//! selector resolution.
//!
//! The bundled catalog is a lossless gzip snapshot of upstream
//! `packages/catalog/src/models.json` (see `data/source.json` for provenance)
//! decoded once via [`Registry::bundled`]. Entries are metadata only: presence
//! in the catalog does not imply the runtime has a transport for the model's
//! `api` family.

use std::collections::BTreeMap;

use flate2::read::GzDecoder;
use serde::{Deserialize, Deserializer, Serialize};

/// Gzip-compressed upstream catalog, stored at `data/models.json.gz`.
const BUNDLE: &[u8] = include_bytes!("../data/models.json.gz");

/// Model kind used when the source row leaves `kind` absent or null.
const DEFAULT_KIND: &str = "chat";

/// Per-million-token rates for one model.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct ModelCost {
    #[serde(default)]
    pub input: f64,
    #[serde(default)]
    pub output: f64,
    #[serde(default)]
    pub cache_read: f64,
    #[serde(default)]
    pub cache_write: f64,
}

/// Catalog metadata for a single provider model.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct Model {
    pub id: String,
    pub name: String,
    /// Wire transport family (e.g. `openai-responses`, `anthropic-messages`).
    pub api: String,
    pub provider: String,
    pub base_url: String,
    #[serde(default)]
    pub reasoning: bool,
    /// Accepted input modalities (`text`, `image`).
    #[serde(default)]
    pub input: Vec<String>,
    #[serde(default)]
    pub cost: ModelCost,
    #[serde(default)]
    pub context_window: Option<u64>,
    #[serde(default)]
    pub max_tokens: Option<u64>,
    /// Catalog kind; absent or null means `chat`.
    #[serde(default = "default_kind", deserialize_with = "deserialize_kind")]
    pub kind: String,
    /// `false` is the only "unsupported" signal: absent means tools are usable.
    #[serde(
        default = "default_supports_tools",
        deserialize_with = "deserialize_supports_tools"
    )]
    pub supports_tools: bool,
    /// Provider compatibility flags; shape is upstream-defined and forwarded
    /// verbatim to transports. Absent rows deserialize as null.
    #[serde(default)]
    pub compat: serde_json::Value,
}

fn default_kind() -> String {
    DEFAULT_KIND.to_string()
}

fn deserialize_kind<'de, D>(deserializer: D) -> Result<String, D::Error>
where
    D: Deserializer<'de>,
{
    Ok(Option::<String>::deserialize(deserializer)?.unwrap_or_else(default_kind))
}

fn default_supports_tools() -> bool {
    true
}

fn deserialize_supports_tools<'de, D>(deserializer: D) -> Result<bool, D::Error>
where
    D: Deserializer<'de>,
{
    Ok(Option::<bool>::deserialize(deserializer)?.unwrap_or(true))
}

/// Selector resolution and bundle decode failures.
#[derive(Debug, thiserror::Error)]
pub enum CatalogError {
    /// The bundled snapshot could not be decompressed or parsed.
    #[error("bundled model catalog is corrupt: {0}")]
    Decode(String),
    /// No model matched the selector.
    #[error("model not found: {0}")]
    NotFound(String),
    /// A bare model id exists under more than one provider and requires a
    /// `provider/id` selector.
    #[error("model id {selector:?} is ambiguous across providers: {}", providers.join(", "))]
    Ambiguous {
        selector: String,
        providers: Vec<String>,
    },
}

/// Model collection grouped by lowercased provider. Rows keep their source
/// order and casing (the bundle contains same-provider ids that differ only by
/// case, e.g. nanogpt `gemma-4-31B-Fabled`/`Gemma-4-31B-Fabled`, so ids are not
/// a unique key); lookups compare case-insensitively like upstream selectors.
#[derive(Debug, Clone, Default)]
pub struct Registry {
    models: BTreeMap<String, Vec<Model>>,
}

impl Registry {
    /// Decode the bundled upstream snapshot (`data/models.json.gz`). Rows are
    /// appended verbatim so same-provider case-variant ids survive losslessly.
    pub fn bundled() -> Result<Self, CatalogError> {
        let decoder = GzDecoder::new(BUNDLE);
        let raw: BTreeMap<String, BTreeMap<String, Model>> = serde_json::from_reader(decoder)
            .map_err(|error| CatalogError::Decode(error.to_string()))?;
        let mut registry = Self::empty();
        for models in raw.into_values() {
            for model in models.into_values() {
                registry
                    .models
                    .entry(model.provider.to_lowercase())
                    .or_default()
                    .push(model);
            }
        }
        Ok(registry)
    }

    /// Registry with no models; fills come from [`Registry::upsert`].
    pub fn empty() -> Self {
        Self::default()
    }

    /// Insert a model or replace the first row with the same provider and id
    /// (compared case-insensitively). Custom model overrides and bundled rows
    /// share the same table.
    pub fn upsert(&mut self, model: Model) {
        let models = self
            .models
            .entry(model.provider.to_lowercase())
            .or_default();
        match models
            .iter()
            .position(|existing| existing.id.eq_ignore_ascii_case(&model.id))
        {
            Some(index) => models[index] = model,
            None => models.push(model),
        }
    }

    /// All models grouped by provider in deterministic provider order.
    pub fn iter(&self) -> impl Iterator<Item = &Model> {
        self.models.values().flat_map(|models| models.iter())
    }

    /// Resolve a selector. `provider/id` splits on the first slash so ids may
    /// contain slashes; when the prefix is not a known provider the whole
    /// selector is matched as a bare id. Bare ids resolve only when a single
    /// row matches; a selector naming a known provider is locked to that
    /// provider and never falls back to another provider's same-id model.
    pub fn resolve(&self, selector: &str) -> Result<&Model, CatalogError> {
        if let Some((provider, id)) = selector.split_once('/')
            && let Some(models) = self.models.get(&provider.to_lowercase())
        {
            return models
                .iter()
                .find(|model| model.id.eq_ignore_ascii_case(id))
                .ok_or_else(|| CatalogError::NotFound(selector.to_string()));
        }
        let found: Vec<&Model> = self
            .iter()
            .filter(|model| model.id.eq_ignore_ascii_case(selector))
            .collect();
        match found.as_slice() {
            [model] => Ok(model),
            [] => Err(CatalogError::NotFound(selector.to_string())),
            _ => {
                let mut providers: Vec<String> = Vec::new();
                for model in &found {
                    if !providers.contains(&model.provider) {
                        providers.push(model.provider.clone());
                    }
                }
                Err(CatalogError::Ambiguous {
                    selector: selector.to_string(),
                    providers,
                })
            }
        }
    }

    /// Exact `(provider, id)` lookup; `None` instead of an error on a miss.
    pub fn get(&self, provider: &str, id: &str) -> Option<&Model> {
        self.models
            .get(&provider.to_lowercase())
            .and_then(|models| {
                models
                    .iter()
                    .find(|model| model.id.eq_ignore_ascii_case(id))
            })
    }

    /// Number of models in the registry.
    pub fn len(&self) -> usize {
        self.models.values().map(Vec::len).sum()
    }

    /// Whether the registry has no models.
    pub fn is_empty(&self) -> bool {
        self.models.values().all(Vec::is_empty)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model(provider: &str, id: &str) -> Model {
        Model {
            id: id.to_string(),
            name: id.to_string(),
            api: "openai-completions".to_string(),
            provider: provider.to_string(),
            base_url: "https://example.test/v1".to_string(),
            reasoning: false,
            input: vec!["text".to_string()],
            cost: ModelCost::default(),
            context_window: Some(128_000),
            max_tokens: Some(8_192),
            kind: DEFAULT_KIND.to_string(),
            supports_tools: true,
            compat: serde_json::Value::Null,
        }
    }

    fn bundled() -> Registry {
        Registry::bundled().expect("bundled catalog must decode")
    }

    #[test]
    fn bundled_catalog_decodes_fully() {
        let registry = bundled();
        // The pinned snapshot (data/source.json) carries 5671 rows.
        assert_eq!(registry.len(), 5671);
        let sonnet = registry
            .get("anthropic", "claude-sonnet-4-5")
            .expect("anthropic claude-sonnet-4-5 must exist");
        assert_eq!(sonnet.provider, "anthropic");
        assert_eq!(sonnet.kind, "chat");
        assert!(sonnet.context_window.is_some());
    }

    #[test]
    fn bundled_catalog_preserves_case_variant_rows() {
        let registry = bundled();
        // nanogpt ships `gemma-4-31B-Fabled` and `Gemma-4-31B-Fabled` as
        // distinct rows; lossless decoding keeps both.
        let variants: Vec<&str> = registry
            .iter()
            .filter(|model| {
                model.provider == "nanogpt" && model.id.eq_ignore_ascii_case("gemma-4-31b-fabled")
            })
            .map(|model| model.id.as_str())
            .collect();
        assert_eq!(variants.len(), 2);
        // Case-insensitive provider-scoped lookup still resolves deterministically.
        assert!(registry.get("nanogpt", "GEMMA-4-31B-FABLED").is_some());
        // A bare selector that only hits same-provider case variants is ambiguous.
        assert!(matches!(
            registry.resolve("gemma-4-31b-fabled"),
            Err(CatalogError::Ambiguous { .. })
        ));
    }

    #[test]
    fn resolve_exact_provider_id() {
        let registry = bundled();
        let model = registry.resolve("anthropic/claude-sonnet-4-5").unwrap();
        assert_eq!(model.provider, "anthropic");
        assert_eq!(model.id, "claude-sonnet-4-5");
    }

    #[test]
    fn resolve_is_case_insensitive() {
        let registry = bundled();
        let model = registry.resolve("ANTHROPIC/Claude-Sonnet-4-5").unwrap();
        assert_eq!(model.id, "claude-sonnet-4-5");
        assert!(registry.get("OpenAI", "GPT-4O").is_some());
    }

    #[test]
    fn resolve_split_keeps_slashes_in_model_id() {
        let registry = bundled();
        // openrouter carries `google/gemma-4-31b-it`: first slash separates
        // the provider, the rest is the model id verbatim.
        let model = registry
            .resolve("openrouter/google/gemma-4-31b-it")
            .unwrap();
        assert_eq!(model.provider, "openrouter");
        assert_eq!(model.id, "google/gemma-4-31b-it");
    }

    #[test]
    fn resolve_unique_bare_id() {
        let registry = bundled();
        // Only aimlapi carries this id, including the slash.
        let model = registry.resolve("alibaba/qwen-max").unwrap();
        assert_eq!(model.provider, "aimlapi");
        assert_eq!(model.id, "alibaba/qwen-max");
    }

    #[test]
    fn resolve_ambiguous_bare_id_fails() {
        let registry = bundled();
        let err = registry.resolve("claude-sonnet-4-5").unwrap_err();
        match err {
            CatalogError::Ambiguous { providers, .. } => {
                assert!(providers.contains(&"anthropic".to_string()));
                assert!(providers.len() > 1);
            }
            other => panic!("expected Ambiguous, got {other:?}"),
        }
    }

    #[test]
    fn resolve_known_provider_locks_selector() {
        let registry = bundled();
        // `anthropic` is a real provider but does not carry `gpt-4o`; the
        // selector must not fall back to another provider's same-id model.
        let err = registry.resolve("anthropic/gpt-4o").unwrap_err();
        assert!(matches!(err, CatalogError::NotFound(_)));
    }

    #[test]
    fn resolve_unknown_selector_fails() {
        let registry = bundled();
        assert!(matches!(
            registry.resolve("nonexistent-provider/whatever"),
            Err(CatalogError::NotFound(_))
        ));
        assert!(matches!(
            registry.resolve("definitely-not-a-model"),
            Err(CatalogError::NotFound(_))
        ));
        assert!(bundled().resolve("").is_err());
    }

    #[test]
    fn upsert_adds_and_overrides_models() {
        let mut registry = Registry::empty();
        assert!(registry.is_empty());
        assert!(matches!(
            registry.resolve("custom/thing"),
            Err(CatalogError::NotFound(_))
        ));

        registry.upsert(model("custom", "thing"));
        assert_eq!(registry.resolve("custom/thing").unwrap().provider, "custom");
        assert_eq!(registry.resolve("thing").unwrap().provider, "custom");

        // Replacing the same (provider, id) key keeps one entry.
        let mut updated = model("custom", "thing");
        updated.base_url = "https://override.test/v2".to_string();
        updated.reasoning = true;
        registry.upsert(updated);
        assert_eq!(registry.len(), 1);
        let resolved = registry.get("custom", "thing").unwrap();
        assert_eq!(resolved.base_url, "https://override.test/v2");
        assert!(resolved.reasoning);
    }

    #[test]
    fn upsert_override_changes_ambiguity() {
        let mut registry = Registry::empty();
        registry.upsert(model("a", "shared"));
        registry.upsert(model("b", "shared"));
        assert!(matches!(
            registry.resolve("shared"),
            Err(CatalogError::Ambiguous { .. })
        ));
        assert_eq!(registry.resolve("b/shared").unwrap().provider, "b");
    }

    #[test]
    fn iter_covers_every_model_once() {
        let mut registry = Registry::empty();
        registry.upsert(model("b", "two"));
        registry.upsert(model("a", "one"));
        let ids: Vec<&str> = registry.iter().map(|m| m.id.as_str()).collect();
        assert_eq!(ids, ["one", "two"]);
    }

    #[test]
    fn model_deserialization_defaults() {
        let json = r#"{
            "id": "m",
            "name": "M",
            "api": "openai-completions",
            "provider": "p",
            "baseUrl": "http://localhost:1",
            "cost": {"input": 1.5, "output": 2.0},
            "contextWindow": null,
            "extraUpstreamField": {"anything": true}
        }"#;
        let model: Model = serde_json::from_str(json).unwrap();
        assert_eq!(model.kind, "chat");
        assert!(model.supports_tools);
        assert_eq!(model.context_window, None);
        assert_eq!(model.max_tokens, None);
        assert_eq!(model.cost.cache_read, 0.0);
        assert_eq!(model.cost.cache_write, 0.0);
        assert_eq!(model.cost.input, 1.5);
        assert!(model.input.is_empty());
        assert!(!model.reasoning);
        assert!(model.compat.is_null());
    }

    #[test]
    fn model_deserialization_explicit_values() {
        let json = r#"{
            "id": "m",
            "name": "M",
            "api": "anthropic-messages",
            "provider": "p",
            "baseUrl": "http://localhost:1",
            "reasoning": true,
            "input": ["text", "image"],
            "cost": {"input": 3, "output": 15, "cacheRead": 0.3, "cacheWrite": 3.75},
            "contextWindow": 200000,
            "maxTokens": null,
            "kind": null,
            "supportsTools": false,
            "compat": {"supportsDeveloperRole": false}
        }"#;
        let model: Model = serde_json::from_str(json).unwrap();
        // Explicit null kind still means chat.
        assert_eq!(model.kind, "chat");
        assert!(!model.supports_tools);
        assert!(model.reasoning);
        assert_eq!(model.max_tokens, None);
        assert_eq!(model.cost.cache_read, 0.3);
        assert_eq!(
            model.compat["supportsDeveloperRole"],
            serde_json::Value::Bool(false)
        );

        let kind_json = r#"{
            "id": "e",
            "name": "E",
            "api": "voyage",
            "provider": "p",
            "baseUrl": "http://localhost:1",
            "cost": {},
            "kind": "embedding"
        }"#;
        assert_eq!(
            serde_json::from_str::<Model>(kind_json).unwrap().kind,
            "embedding"
        );
    }

    #[test]
    fn model_serializes_camel_case() {
        let value = serde_json::to_value(model("p", "m")).unwrap();
        let object = value.as_object().unwrap();
        for key in ["baseUrl", "contextWindow", "maxTokens", "supportsTools"] {
            assert!(object.contains_key(key), "missing camelCase key {key}");
        }
        assert_eq!(value["kind"], serde_json::Value::String("chat".into()));
        // Round-trip preserves resolved defaults.
        let reparsed: Model = serde_json::from_value(value).unwrap();
        assert_eq!(reparsed, model("p", "m"));
    }
}
