//! Discrete decision and routing layer.
//!
//! Provides ultra-fast calibrated selection over finite choices ([`TypeSafeJudge`])
//! targeting TypeSafe's System One endpoint (`/v1/systemone`) and OpenRouter
//! (`/decisions`), as well as completion-based fallback ([`TextJudge`]) for
//! standard generative models.

use std::collections::HashSet;
use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use futures::StreamExt;
use juto_catalog::Model;
use serde_json::{Value, json};
use tokio_util::sync::CancellationToken;

use crate::{
    Choice, Context, CredentialStore, Decision, Judge, JudgeError, Message, Provider,
    ProviderError, ProviderEvent, StreamOptions, Usage,
};

pub const DEFAULT_TYPESAFE_ENDPOINT: &str = "https://api.typesafe.ai/v1/systemone";
pub const DEFAULT_OPENROUTER_ENDPOINT: &str = "https://openrouter.ai/api/v1/decisions";
pub const DEFAULT_TYPESAFE_MODEL: &str = "jev-latest";
pub const DEFAULT_OPENROUTER_MODEL: &str = "openrouter/~typesafe/jev-latest";
pub const DEFAULT_TIMEOUT: Duration = Duration::from_secs(15);

/// Discrete choice classifier targeting TypeSafe's System One API or OpenRouter decisions.
#[derive(Clone)]
pub struct TypeSafeJudge {
    client: reqwest::Client,
    endpoint: String,
    api_key: String,
    model: Option<String>,
    timeout: Duration,
}

impl TypeSafeJudge {
    /// Construct a new [`TypeSafeJudge`] targeting TypeSafe's System One endpoint.
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            client: reqwest::Client::builder()
                .redirect(reqwest::redirect::Policy::none())
                .build()
                .unwrap_or_default(),
            endpoint: DEFAULT_TYPESAFE_ENDPOINT.to_string(),
            api_key: api_key.into(),
            model: None,
            timeout: DEFAULT_TIMEOUT,
        }
    }

    /// Construct a [`TypeSafeJudge`] targeting OpenRouter's `/decisions` endpoint.
    pub fn for_openrouter(api_key: impl Into<String>) -> Self {
        Self {
            client: reqwest::Client::builder()
                .redirect(reqwest::redirect::Policy::none())
                .build()
                .unwrap_or_default(),
            endpoint: DEFAULT_OPENROUTER_ENDPOINT.to_string(),
            api_key: api_key.into(),
            model: Some(DEFAULT_OPENROUTER_MODEL.to_string()),
            timeout: DEFAULT_TIMEOUT,
        }
    }

    /// Override the destination endpoint URL.
    pub fn with_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.endpoint = endpoint.into();
        self
    }

    /// Override the model selector.
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Provide a custom HTTP client.
    pub fn with_client(mut self, client: reqwest::Client) -> Self {
        self.client = client;
        self
    }

    /// Configure the request timeout.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    pub fn endpoint(&self) -> &str {
        &self.endpoint
    }

    pub fn model(&self) -> Option<&str> {
        self.model.as_deref()
    }
}
impl std::fmt::Debug for TypeSafeJudge {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TypeSafeJudge")
            .field("endpoint", &self.endpoint)
            .field("model", &self.model)
            .field("timeout", &self.timeout)
            .finish_non_exhaustive()
    }
}

#[async_trait]
impl Judge for TypeSafeJudge {
    async fn choose(&self, context: &str, options: &[Choice]) -> Result<Decision, JudgeError> {
        validate_options(options)?;

        let start = Instant::now();
        let mut criteria = serde_json::Map::new();
        for opt in options {
            criteria.insert(opt.key.clone(), Value::String(opt.description.clone()));
        }

        let payload = json!({
            "model": self.model.as_deref().unwrap_or(DEFAULT_TYPESAFE_MODEL),
            "state": context,
            "questions": {
                "decision": {
                    "type": "choice",
                    "instructions": "Select the single best choice from criteria.",
                    "criteria": criteria,
                }
            },
            // Legacy / flat compatibility fields for simple mock servers
            "context": context,
            "choices": options,
        });

        let mut request = self
            .client
            .post(&self.endpoint)
            .timeout(self.timeout)
            .header("Content-Type", "application/json")
            .header("Authorization", format!("Bearer {}", self.api_key));

        if self.endpoint.contains("openrouter.ai") {
            request = request
                .header("HTTP-Referer", "https://juto.dev")
                .header("X-Title", "Juto");
        }

        let response = request
            .json(&payload)
            .send()
            .await
            .map_err(|e| JudgeError::Transport(e.to_string()))?;

        let status = response.status();
        let status_code = status.as_u16();

        if status_code == 401 || status_code == 403 {
            let msg = response
                .text()
                .await
                .unwrap_or_else(|_| "unauthorized".to_string());
            return Err(JudgeError::Authentication(msg));
        }

        if !status.is_success() {
            let msg = response
                .text()
                .await
                .unwrap_or_else(|_| format!("HTTP {status_code}"));
            return Err(JudgeError::Http {
                status: status_code,
                message: msg,
            });
        }

        let body: Value = response
            .json()
            .await
            .map_err(|e| JudgeError::Protocol(format!("failed to parse JSON response: {e}")))?;

        parse_decision_response(&body, options, start.elapsed())
    }
}

/// Discrete choice classifier that uses a standard completion or chat model.
pub struct TextJudge {
    provider: Arc<dyn Provider>,
    model: Model,
    timeout: Duration,
}

impl TextJudge {
    pub fn new(provider: Arc<dyn Provider>, model: Model) -> Self {
        Self {
            provider,
            model,
            timeout: Duration::from_secs(30),
        }
    }

    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    pub fn model(&self) -> &Model {
        &self.model
    }
}
impl std::fmt::Debug for TextJudge {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TextJudge")
            .field("model", &self.model.id)
            .field("timeout", &self.timeout)
            .finish()
    }
}

#[async_trait]
impl Judge for TextJudge {
    async fn choose(&self, context: &str, options: &[Choice]) -> Result<Decision, JudgeError> {
        validate_options(options)?;

        let start = Instant::now();

        // Build prompt with candidates
        let mut prompt = String::new();
        prompt.push_str("Context:\n");
        prompt.push_str(context);
        prompt.push_str("\n\nCandidate Choices:\n");
        for opt in options {
            prompt.push_str(&format!("- {}: {}\n", opt.key, opt.description));
        }
        prompt.push_str(
            "\nSelect the SINGLE best choice from the candidates above.\n\
            Respond ONLY with a JSON object in this exact schema:\n\
            {\n  \"key\": \"<EXACT_CHOICE_KEY>\",\n  \"confidence\": <number between 0.0 and 1.0>,\n  \"rationale\": \"<brief reason>\"\n}\n",
        );

        let stream_context = Context {
            system_prompt: vec![
                "You are an expert discrete decision classifier. Always answer with valid JSON only."
                    .to_string(),
            ],
            messages: vec![Message::user(prompt)],
            tools: vec![],
        };

        let stream_options = StreamOptions {
            temperature: Some(0.0),
            max_tokens: Some(256),
            timeout: self.timeout,
            ..Default::default()
        };

        let cancel = CancellationToken::new();
        let stream_res = tokio::time::timeout(
            self.timeout,
            self.provider.stream(
                &self.model,
                &stream_context,
                &stream_options,
                cancel.clone(),
            ),
        )
        .await
        .map_err(|_| JudgeError::Http {
            status: 408,
            message: "decision request timed out".to_string(),
        })?;

        let mut stream = stream_res.map_err(map_provider_error)?;

        let mut accumulated_text = String::new();
        let mut final_usage = Usage::default();

        while let Some(event_res) = stream.next().await {
            let event = event_res.map_err(map_provider_error)?;
            match event {
                ProviderEvent::TextDelta { delta, .. } => {
                    accumulated_text.push_str(&delta);
                }
                ProviderEvent::Done { message } => {
                    final_usage = message.usage.clone();
                    if accumulated_text.is_empty() {
                        accumulated_text = message.text();
                    }
                }
                _ => {}
            }
        }

        let parsed_val = parse_json_from_text(&accumulated_text, options)?;
        let mut decision = parse_decision_response(&parsed_val, options, start.elapsed())?;
        if decision.usage.total_tokens == 0 && final_usage.total_tokens > 0 {
            decision.usage = final_usage;
        }
        Ok(decision)
    }
}

/// Automatically constructs the best available [`Judge`] implementation based on
/// credentials and fallback configuration.
///
/// Order of precedence:
/// 1. TypeSafe System One (`TYPESAFE_API_KEY` in store or environment)
/// 2. OpenRouter Decisions (`OPENROUTER_API_KEY` in store or environment)
/// 3. Fallback [`TextJudge`] using the provided provider and model
pub async fn create_judge(
    credentials: &CredentialStore,
    fallback_provider: Option<(Arc<dyn Provider>, Model)>,
) -> Result<Arc<dyn Judge>, JudgeError> {
    // 1. Check for TypeSafe API key
    match credentials.resolve_provider("typesafe").await {
        Ok(Some(cred)) => {
            let secret = cred.expose_secret();
            if !secret.trim().is_empty() {
                return Ok(Arc::new(TypeSafeJudge::new(secret)));
            }
        }
        Ok(None) => {}
        Err(err) => {
            return Err(JudgeError::Authentication(format!(
                "failed to resolve TypeSafe credentials: {err}"
            )));
        }
    }

    // 2. Check for OpenRouter API key
    match credentials.resolve_provider("openrouter").await {
        Ok(Some(cred)) => {
            let secret = cred.expose_secret();
            if !secret.trim().is_empty() {
                return Ok(Arc::new(TypeSafeJudge::for_openrouter(secret)));
            }
        }
        Ok(None) => {}
        Err(err) => {
            return Err(JudgeError::Authentication(format!(
                "failed to resolve OpenRouter credentials: {err}"
            )));
        }
    }

    // 3. Fallback to TextJudge
    if let Some((provider, model)) = fallback_provider {
        return Ok(Arc::new(TextJudge::new(provider, model)));
    }

    Err(JudgeError::Authentication(
        "no TypeSafe or OpenRouter credentials found and no fallback completion model provided"
            .to_string(),
    ))
}

/// Catalog provider whose judge-kind rows [`judge_for_model`] serves.
const TYPESAFE_PROVIDER: &str = "typesafe";

/// Whether `model` is a judge-kind row served by the TypeSafe System One transport.
/// Other judge-kind rows (e.g. `opencode-zen` `jev-1.13`) are catalog metadata only.
pub fn supports_judge_model(model: &Model) -> bool {
    model.kind == "judge" && model.api == "typesafe" && model.provider == TYPESAFE_PROVIDER
}

/// Build the [`Judge`] for a model accepted by [`supports_judge_model`]. The catalog
/// model id is sent as the wire `model` selector.
pub async fn judge_for_model(
    credentials: &CredentialStore,
    model: &Model,
) -> Result<Arc<dyn Judge>, JudgeError> {
    if !supports_judge_model(model) {
        return Err(JudgeError::Protocol(format!(
            "{}/{} is not a supported judge model",
            model.provider, model.id
        )));
    }
    match credentials.resolve_provider(TYPESAFE_PROVIDER).await {
        Ok(Some(cred)) => {
            let secret = cred.expose_secret();
            if !secret.trim().is_empty() {
                return Ok(Arc::new(
                    TypeSafeJudge::new(secret).with_model(model.id.clone()),
                ));
            }
        }
        Ok(None) => {}
        Err(err) => {
            return Err(JudgeError::Authentication(format!(
                "failed to resolve TypeSafe credentials: {err}"
            )));
        }
    }
    Err(JudgeError::Authentication(format!(
        "no TypeSafe credentials for judge model {}",
        model.id
    )))
}

fn validate_options(options: &[Choice]) -> Result<(), JudgeError> {
    if options.is_empty() {
        return Err(JudgeError::InvalidOptions(
            "at least one choice must be provided".to_string(),
        ));
    }

    let mut seen = HashSet::with_capacity(options.len());
    for opt in options {
        if opt.key.trim().is_empty() {
            return Err(JudgeError::InvalidOptions(
                "choice key cannot be empty".to_string(),
            ));
        }
        if !seen.insert(opt.key.as_str()) {
            return Err(JudgeError::InvalidOptions(format!(
                "duplicate choice key: '{}'",
                opt.key
            )));
        }
    }

    Ok(())
}

fn parse_decision_response(
    body: &Value,
    options: &[Choice],
    elapsed: Duration,
) -> Result<Decision, JudgeError> {
    // 1. Extract choice key
    let raw_key = extract_decision_key(body)
        .ok_or_else(|| JudgeError::Protocol(format!("missing choice key in response: {body}")))?;

    // 2. Match against options (exact match first, then case-insensitive)
    let matched_option = options
        .iter()
        .find(|opt| opt.key == raw_key)
        .or_else(|| {
            options
                .iter()
                .find(|opt| opt.key.eq_ignore_ascii_case(&raw_key))
        })
        .ok_or_else(|| {
            JudgeError::NoMatch(format!(
                "model returned key '{raw_key}' which does not match any candidate choice"
            ))
        })?;

    // 3. Extract confidence
    let confidence = extract_confidence(body);

    // 4. Extract rationale
    let rationale = extract_rationale(body);

    // 5. Extract usage
    let usage = extract_usage(body);

    Ok(Decision {
        key: matched_option.key.clone(),
        confidence,
        rationale,
        usage,
        latency_ms: Some(elapsed.as_millis() as u64),
    })
}

fn extract_decision_key(body: &Value) -> Option<String> {
    // 1. Official TypeSafe / OpenRouter System One format: answers.<name>.choice
    if let Some(answers) = body.get("answers").and_then(Value::as_object) {
        for (_q_name, answer) in answers {
            if let Some(s) = answer.get("choice").and_then(Value::as_str) {
                return Some(s.to_string());
            }
            if let Some(s) = answer.get("key").and_then(Value::as_str) {
                return Some(s.to_string());
            }
        }
    }
    if let Some(s) = body.get("key").and_then(Value::as_str) {
        return Some(s.to_string());
    }
    if let Some(s) = body.get("choice").and_then(Value::as_str) {
        return Some(s.to_string());
    }
    if let Some(s) = body.get("result").and_then(Value::as_str) {
        return Some(s.to_string());
    }
    if let Some(s) = body.get("selected").and_then(Value::as_str) {
        return Some(s.to_string());
    }
    if let Some(decision) = body.get("decision") {
        if let Some(s) = decision.as_str() {
            return Some(s.to_string());
        }
        if let Some(s) = decision.get("key").and_then(Value::as_str) {
            return Some(s.to_string());
        }
        if let Some(s) = decision.get("choice").and_then(Value::as_str) {
            return Some(s.to_string());
        }
    }
    None
}

fn extract_confidence(body: &Value) -> Option<f64> {
    if let Some(answers) = body.get("answers").and_then(Value::as_object) {
        for (_q_name, answer) in answers {
            if let Some(c) = answer.get("confidence").and_then(Value::as_f64) {
                return Some(c);
            }
        }
    }
    body.get("confidence")
        .or_else(|| body.get("score"))
        .or_else(|| body.get("probability"))
        .or_else(|| {
            body.get("decision").and_then(|d| {
                d.get("confidence")
                    .or_else(|| d.get("score"))
                    .or_else(|| d.get("probability"))
            })
        })
        .and_then(Value::as_f64)
}

fn extract_rationale(body: &Value) -> Option<String> {
    if let Some(answers) = body.get("answers").and_then(Value::as_object) {
        for (_q_name, answer) in answers {
            if let Some(s) = answer
                .get("rationale")
                .or_else(|| answer.get("explanation"))
                .or_else(|| answer.get("reason"))
                .and_then(Value::as_str)
            {
                return Some(s.to_string());
            }
        }
    }
    body.get("rationale")
        .or_else(|| body.get("explanation"))
        .or_else(|| body.get("reason"))
        .or_else(|| {
            body.get("decision").and_then(|d| {
                d.get("rationale")
                    .or_else(|| d.get("explanation"))
                    .or_else(|| d.get("reason"))
            })
        })
        .and_then(Value::as_str)
        .map(str::to_string)
}

fn extract_usage(body: &Value) -> Usage {
    let usage_val = body.get("usage");
    let input = usage_val
        .and_then(|u| {
            u.get("input_tokens")
                .or_else(|| u.get("input"))
                .or_else(|| u.get("prompt_tokens"))
        })
        .and_then(Value::as_u64)
        .unwrap_or(0);

    let output = usage_val
        .and_then(|u| {
            u.get("output_tokens")
                .or_else(|| u.get("output"))
                .or_else(|| u.get("completion_tokens"))
        })
        .and_then(Value::as_u64)
        .unwrap_or(0);

    let total = usage_val
        .and_then(|u| u.get("total_tokens").or_else(|| u.get("totalTokens")))
        .and_then(Value::as_u64)
        .unwrap_or(input + output);

    let cost = usage_val
        .and_then(|u| u.get("cost"))
        .and_then(Value::as_f64)
        .unwrap_or(0.0);

    Usage {
        input,
        output,
        cache_read: 0,
        cache_write: 0,
        total_tokens: total,
        cost,
    }
}

fn parse_json_from_text(text: &str, options: &[Choice]) -> Result<Value, JudgeError> {
    let trimmed = text.trim();

    // 1. Try direct serde_json parse
    if let Some(val) = try_parse_object(trimmed) {
        return Ok(val);
    }

    // 2. Strip markdown code fences e.g. ```json ... ``` or ``` ... ```
    if let Some(val) = extract_markdown_code_block(trimmed).and_then(|s| try_parse_object(&s)) {
        return Ok(val);
    }

    // 3. Find outermost curly braces { ... }
    if let (Some(start), Some(end)) = (trimmed.find('{'), trimmed.rfind('}')) {
        let slice = if start < end {
            &trimmed[start..=end]
        } else {
            ""
        };
        if let Some(val) = try_parse_object(slice) {
            return Ok(val);
        }
    }
    // 4. Fallback: search for candidate keys in the raw text
    let mut matched_keys = Vec::new();
    for opt in options {
        if trimmed.contains(&opt.key) {
            matched_keys.push(opt.key.clone());
        }
    }

    if matched_keys.len() == 1 {
        return Ok(json!({
            "key": matched_keys[0],
            "confidence": 0.75,
            "rationale": trimmed,
        }));
    }

    Err(JudgeError::Protocol(format!(
        "could not parse decision JSON from model output: '{text}'"
    )))
}

fn extract_markdown_code_block(text: &str) -> Option<String> {
    let start_idx = text.find("```")?;
    let after_start = &text[start_idx + 3..];
    let content_start = if let Some(newline) = after_start.find('\n') {
        newline + 1
    } else {
        0
    };
    let rest = &after_start[content_start..];
    let end_idx = rest.find("```")?;
    Some(rest[..end_idx].trim().to_string())
}
fn try_parse_object(s: &str) -> Option<Value> {
    match serde_json::from_str::<Value>(s) {
        Ok(val) if val.is_object() => Some(val),
        _ => None,
    }
}

fn map_provider_error(err: ProviderError) -> JudgeError {
    match err {
        ProviderError::Authentication(msg) => JudgeError::Authentication(msg),
        ProviderError::Http {
            status, message, ..
        } => JudgeError::Http { status, message },
        ProviderError::Transport(msg) => JudgeError::Transport(msg),
        ProviderError::Aborted => JudgeError::Aborted,
        ProviderError::Protocol(msg) | ProviderError::Unsupported(msg) => JudgeError::Protocol(msg),
    }
}

#[cfg(test)]
mod tests {
    use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
    use tokio::sync::Mutex;

    use super::*;

    /// Spawns a mock HTTP server returning `status_code` and `body`.
    /// Returns `(base_url, recorded_requests)`.
    async fn spawn_mock_server(
        status_code: u16,
        body: String,
    ) -> (String, Arc<Mutex<Vec<String>>>) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let requests: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
        let seen = requests.clone();
        tokio::spawn(async move {
            while let Ok((stream, _)) = listener.accept().await {
                let mut reader = BufReader::new(stream);
                let mut request = String::new();
                let mut line = String::new();
                let mut content_length = 0usize;
                loop {
                    line.clear();
                    if reader.read_line(&mut line).await.unwrap_or(0) == 0 {
                        break;
                    }
                    request.push_str(&line);
                    if let Some(v) = line.to_lowercase().strip_prefix("content-length:") {
                        content_length = v.trim().parse().unwrap_or(0);
                    }
                    if line.trim().is_empty() {
                        break;
                    }
                }
                let mut bodybuf = vec![0u8; content_length];
                let _ = reader.read_exact(&mut bodybuf).await;
                request.push_str(&String::from_utf8_lossy(&bodybuf));
                seen.lock().await.push(request);
                let status_line = match status_code {
                    200 => "200 OK",
                    401 => "401 Unauthorized",
                    500 => "500 Internal Server Error",
                    _ => "400 Bad Request",
                };
                let response = format!(
                    "HTTP/1.1 {status_line}\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
                    body.len()
                );
                let _ = reader.get_mut().write_all(response.as_bytes()).await;
            }
        });
        (format!("http://127.0.0.1:{port}"), requests)
    }

    #[test]
    fn validate_options_rejects_empty_or_duplicates() {
        assert!(matches!(
            validate_options(&[]),
            Err(JudgeError::InvalidOptions(_))
        ));

        let empty_key = vec![Choice::new("", "desc")];
        assert!(matches!(
            validate_options(&empty_key),
            Err(JudgeError::InvalidOptions(_))
        ));

        let dups = vec![
            Choice::new("OPTION_A", "First"),
            Choice::new("OPTION_A", "Duplicate"),
        ];
        assert!(matches!(
            validate_options(&dups),
            Err(JudgeError::InvalidOptions(_))
        ));

        let valid = vec![
            Choice::new("OPTION_A", "First"),
            Choice::new("OPTION_B", "Second"),
        ];
        assert!(validate_options(&valid).is_ok());
    }

    #[tokio::test]
    async fn typesafe_judge_systemone_success() {
        let (url, requests) = spawn_mock_server(
            200,
            json!({
                "choice": "MECHANICAL_EDIT",
                "confidence": 0.96,
                "rationale": "Single file localized change.",
                "usage": {
                    "input_tokens": 42,
                    "output_tokens": 4,
                    "total_tokens": 46
                }
            })
            .to_string(),
        )
        .await;

        let judge =
            TypeSafeJudge::new("sk-typesafe-test-123").with_endpoint(format!("{url}/v1/systemone"));

        let options = vec![
            Choice::new("DIRECT_REPLY", "Conversational response"),
            Choice::new("MECHANICAL_EDIT", "Deterministic code edit"),
            Choice::new("END_TO_END_TASK", "Multi-file implementation"),
        ];

        let decision = judge
            .choose("Please rename `foo` to `bar` in auth.rs", &options)
            .await
            .expect("choose succeeds");

        assert_eq!(decision.key, "MECHANICAL_EDIT");
        assert_eq!(decision.confidence, Some(0.96));
        assert_eq!(
            decision.rationale.as_deref(),
            Some("Single file localized change.")
        );
        assert_eq!(decision.usage.total_tokens, 46);
        assert!(decision.latency_ms.is_some());

        // Verify request was sent with proper headers
        let recorded = requests.lock().await;
        assert_eq!(recorded.len(), 1);
        let req_text = &recorded[0];
        assert!(req_text.contains("authorization: Bearer sk-typesafe-test-123"));
        assert!(req_text.contains("content-type: application/json"));
        assert!(req_text.contains("\"model\":\"jev-latest\""));
    }

    #[tokio::test]
    async fn typesafe_judge_official_systemone_schema_success() {
        let (url, requests) = spawn_mock_server(
            200,
            json!({
                "model": "jev-latest",
                "answers": {
                    "decision": {
                        "type": "choice",
                        "choice": "MECHANICAL_EDIT",
                        "confidence": 0.98,
                        "probabilities": {
                            "DIRECT_REPLY": 0.01,
                            "MECHANICAL_EDIT": 0.98,
                            "END_TO_END_TASK": 0.01
                        }
                    }
                },
                "usage": {
                    "input_tokens": 52,
                    "output_tokens": 5,
                    "total_tokens": 57
                }
            })
            .to_string(),
        )
        .await;

        let judge = TypeSafeJudge::new("sk-typesafe-official-test")
            .with_endpoint(format!("{url}/v1/systemone"));

        let options = vec![
            Choice::new("DIRECT_REPLY", "Conversational response"),
            Choice::new("MECHANICAL_EDIT", "Deterministic code edit"),
            Choice::new("END_TO_END_TASK", "Multi-file implementation"),
        ];

        let decision = judge
            .choose("Please rename `foo` to `bar` in auth.rs", &options)
            .await
            .expect("choose succeeds");

        assert_eq!(decision.key, "MECHANICAL_EDIT");
        assert_eq!(decision.confidence, Some(0.98));
        assert_eq!(decision.usage.total_tokens, 57);

        // Verify official request payload fields were sent
        let recorded = requests.lock().await;
        assert_eq!(recorded.len(), 1);
        let req_text = &recorded[0];
        assert!(req_text.contains("\"state\":\"Please rename `foo` to `bar` in auth.rs\""));
        assert!(req_text.contains("\"questions\":{"));
        assert!(req_text.contains("\"criteria\":{"));
        assert!(req_text.contains("\"MECHANICAL_EDIT\":\"Deterministic code edit\""));
    }
    #[tokio::test]
    async fn typesafe_judge_openrouter_decisions_success() {
        let (url, requests) = spawn_mock_server(
            200,
            json!({
                "key": "EXPLORE_SCOUT",
                "confidence": 0.91,
                "usage": {
                    "prompt_tokens": 20,
                    "completion_tokens": 2,
                    "total_tokens": 22
                }
            })
            .to_string(),
        )
        .await;

        let judge = TypeSafeJudge::for_openrouter("sk-or-test-456")
            .with_endpoint(format!("{url}/api/v1/decisions"));

        let options = vec![
            Choice::new("EXPLORE_SCOUT", "Read-only exploration"),
            Choice::new("DIRECT_REPLY", "Direct chat reply"),
        ];

        let decision = judge
            .choose("Where is the user authentication flow defined?", &options)
            .await
            .expect("choose succeeds");

        assert_eq!(decision.key, "EXPLORE_SCOUT");
        assert_eq!(decision.confidence, Some(0.91));
        assert_eq!(decision.usage.total_tokens, 22);

        let recorded = requests.lock().await;
        assert_eq!(recorded.len(), 1);
        let req_text = &recorded[0];
        assert!(req_text.contains("authorization: Bearer sk-or-test-456"));
        assert!(req_text.contains("openrouter/~typesafe/jev-latest"));
    }

    #[tokio::test]
    async fn typesafe_judge_authentication_failure() {
        let (url, _) = spawn_mock_server(401, "Invalid API token".to_string()).await;

        let judge = TypeSafeJudge::new("bad-key").with_endpoint(url);
        let options = vec![Choice::new("A", "Option A"), Choice::new("B", "Option B")];

        let err = judge.choose("test context", &options).await.unwrap_err();
        assert!(matches!(err, JudgeError::Authentication(_)));
    }

    #[tokio::test]
    async fn typesafe_judge_server_error() {
        let (url, _) = spawn_mock_server(500, "Internal Server Error".to_string()).await;

        let judge = TypeSafeJudge::new("key").with_endpoint(url);
        let options = vec![Choice::new("A", "Option A"), Choice::new("B", "Option B")];

        let err = judge.choose("test context", &options).await.unwrap_err();
        match err {
            JudgeError::Http { status, .. } => assert_eq!(status, 500),
            other => panic!("expected JudgeError::Http, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn typesafe_judge_unmatched_choice_returns_nomatch() {
        let (url, _) = spawn_mock_server(
            200,
            json!({
                "choice": "NON_EXISTENT_OPTION"
            })
            .to_string(),
        )
        .await;

        let judge = TypeSafeJudge::new("key").with_endpoint(url);
        let options = vec![Choice::new("OPTION_A", "A"), Choice::new("OPTION_B", "B")];

        let err = judge.choose("test context", &options).await.unwrap_err();
        assert!(matches!(err, JudgeError::NoMatch(_)));
    }

    struct MockProvider {
        output: String,
    }

    #[async_trait]
    impl Provider for MockProvider {
        async fn stream(
            &self,
            _model: &Model,
            _context: &Context,
            _options: &StreamOptions,
            _cancel: CancellationToken,
        ) -> Result<crate::ProviderStream, ProviderError> {
            let output = self.output.clone();
            let stream = async_stream::stream! {
                yield Ok(ProviderEvent::TextDelta { index: 0, delta: output });
                yield Ok(ProviderEvent::Done {
                    message: crate::AssistantMessage {
                        content: vec![],
                        api: "mock".to_string(),
                        provider: "mock".to_string(),
                        model: "mock".to_string(),
                        usage: Usage {
                            input: 10,
                            output: 5,
                            cache_read: 0,
                            cache_write: 0,
                            total_tokens: 15,
                            cost: 0.0001,
                        },
                        stop_reason: crate::StopReason::Stop,
                        timestamp: crate::timestamp_ms(),
                        response_id: None,
                        error: None,
                    },
                });
            };
            Ok(Box::pin(stream))
        }
    }

    fn test_model() -> Model {
        Model {
            id: "mock/test".to_string(),
            name: "Mock Model".to_string(),
            api: "openai-completions".to_string(),
            provider: "mock".to_string(),
            base_url: "http://localhost".to_string(),
            reasoning: false,
            input: vec!["text".to_string()],
            cost: juto_catalog::ModelCost::default(),
            context_window: Some(8192),
            max_tokens: Some(1024),
            kind: "chat".to_string(),
            supports_tools: true,
            compat: serde_json::Value::Null,
        }
    }

    #[tokio::test]
    async fn text_judge_parses_markdown_json() {
        let mock_provider = Arc::new(MockProvider {
            output: r#"
I have analyzed the prompt.
```json
{
  "key": "BLOCKED_USER_INPUT",
  "confidence": 0.89,
  "rationale": "Missing database password in parameters"
}
```
"#
            .to_string(),
        });

        let judge = TextJudge::new(mock_provider, test_model());
        let options = vec![
            Choice::new("CONTINUE", "Proceed with execution"),
            Choice::new("BLOCKED_USER_INPUT", "Escalate to human"),
        ];

        let decision = judge
            .choose("Database connection requires password", &options)
            .await
            .expect("choose succeeds");

        assert_eq!(decision.key, "BLOCKED_USER_INPUT");
        assert_eq!(decision.confidence, Some(0.89));
        assert_eq!(
            decision.rationale.as_deref(),
            Some("Missing database password in parameters")
        );
        assert_eq!(decision.usage.total_tokens, 15);
    }

    #[tokio::test]
    async fn text_judge_parses_fallback_key() {
        let mock_provider = Arc::new(MockProvider {
            output: "Based on the input, the correct route is CONTINUE.".to_string(),
        });

        let judge = TextJudge::new(mock_provider, test_model());
        let options = vec![
            Choice::new("CONTINUE", "Proceed"),
            Choice::new("STOP", "Halt"),
        ];

        let decision = judge
            .choose("Everything looks good", &options)
            .await
            .expect("choose succeeds");

        assert_eq!(decision.key, "CONTINUE");
    }

    #[tokio::test]
    async fn create_judge_precedence() {
        let temp_dir = tempfile::tempdir().unwrap();
        let store = CredentialStore::open(temp_dir.path().join("creds.json")).unwrap();

        // 1. Neither key present, no fallback -> Error
        let err = create_judge(&store, None).await.unwrap_err();
        assert!(matches!(err, JudgeError::Authentication(_)));

        // 2. Fallback provider provided -> TextJudge
        let mock_provider = Arc::new(MockProvider {
            output: json!({"key": "A"}).to_string(),
        });
        let judge = create_judge(&store, Some((mock_provider, test_model())))
            .await
            .expect("creates fallback judge");
        let decision = judge
            .choose("context", &[Choice::new("A", "desc")])
            .await
            .unwrap();
        assert_eq!(decision.key, "A");

        // 3. Typesafe key in store -> TypeSafeJudge
        store.set_api_key("typesafe", "sk-ts-live-123").unwrap();
        let judge = create_judge(&store, None)
            .await
            .expect("creates TypeSafeJudge");
        // Verify it was created
        assert!(judge.choose("", &[Choice::new("A", "desc")]).await.is_err()); // Points to real endpoint with fake key
    }
}
