//! One managed browser tab driven over CDP.
//!
//! Behavioral reference: oh-my-pi `tools/browser/tern/tern-tab.ts` and
//! `tab-worker.ts` at 579da1d6 (MIT, see licenses/OMP-MIT.txt). Element work
//! runs through the vendored page kit in an isolated world; pointer and
//! keyboard input are trusted CDP `Input.*` events at kit-computed points,
//! like the Tern backend's native input.

use std::{
    collections::{HashMap, HashSet, VecDeque},
    path::{Path, PathBuf},
    sync::{Arc, LazyLock},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use base64::Engine as _;
use juto_ai::ContentBlock;
use parking_lot::Mutex;
use serde_json::{Map, Value, json};
use tokio::sync::broadcast::error::RecvError;
use tokio_util::sync::CancellationToken;

use super::cdp::{Cdp, CdpEvent};
use super::launch::Spawned;
use super::support::{self, Key, SnapshotBaseline, SnapshotOptions};

// ---------------------------------------------------------------------------
// Page kit
// ---------------------------------------------------------------------------

const ARIA_BUNDLE: &str = include_str!("assets/aria-snapshot.bundle.js");
const KIT_BODY: &str = include_str!("assets/page_kit.js");
const SEMANTIC_HANDLERS: &str = include_str!("assets/semantic_handlers.js");
/// Isolated world hosting the kit; page scripts cannot see or alter it.
const KIT_WORLD: &str = "__juto_browser_kit";
/// Attribute the kit's `mark()` writes; shared DOM, so the main world can find it.
const HANDLE_ATTR: &str = "data-juto-handle";

const INTERACTIVE_ROLES: &str = r#"["button","link","textbox","combobox","listbox","option","checkbox","radio","switch","tab","menuitem","menuitemcheckbox","menuitemradio","slider","spinbutton","searchbox","treeitem"]"#;
const DEFAULT_STYLES: &str = r#"["font-family","font-size","font-weight","color","background-color","display","visibility","opacity","position","z-index","width","height","margin","padding"]"#;

/// Installs `globalThis.__jutoBrowserKit` in the current realm if absent
/// (`TERN_KIT_SOURCE`).
static KIT_SOURCE: LazyLock<String> = LazyLock::new(|| {
    [
        "(function () {",
        "\"use strict\";",
        "if (globalThis.__jutoBrowserKit) return;",
        "const loadAria = function () {",
        "var module = { exports: {} };",
        ARIA_BUNDLE,
        ";",
        "return module.exports;",
        "};",
        "const semantic = {",
        SEMANTIC_HANDLERS,
        "};",
        &format!("const INTERACTIVE_ROLES = new Set({INTERACTIVE_ROLES});"),
        &format!("const DEFAULT_STYLES = {DEFAULT_STYLES};"),
        KIT_BODY,
        "})();",
    ]
    .join("\n")
});

/// Page-world helper for `extract`: candidate roots with `blockText`
/// (`readable.ts`) and their inner HTML.
const EXTRACT_SCRIPT: &str = r#"(function (selector, articleHtml) {
const BLOCK = new Set(["ADDRESS","ARTICLE","ASIDE","BLOCKQUOTE","CAPTION","DD","DETAILS","DIALOG","DIV","DL","DT","FIELDSET","FIGCAPTION","FIGURE","FOOTER","FORM","H1","H2","H3","H4","H5","H6","HEADER","HR","LI","MAIN","NAV","OL","P","PRE","SECTION","SUMMARY","TABLE","TR","UL"]);
const SKIP = new Set(["SCRIPT","STYLE","NOSCRIPT","TEMPLATE"]);
const blockText = root => {
	if (SKIP.has(root.tagName.toUpperCase())) return (root.textContent ?? "").trimEnd();
	const parts = [];
	let pendingBreaks = 0;
	let cellsInRow = 0;
	const append = text => {
		if (pendingBreaks > 0) {
			while (parts.length > 0) {
				const last = parts[parts.length - 1].replace(/[ \t]+$/, "");
				if (last) { parts[parts.length - 1] = last; parts.push("\n".repeat(pendingBreaks)); break; }
				parts.pop();
			}
			pendingBreaks = 0;
		}
		parts.push(text);
	};
	const walk = (node, pre) => {
		if (node.nodeType === 3) {
			let text = node.textContent ?? "";
			if (!pre) {
				text = text.replace(/[ \t\n\r\f]+/g, " ");
				const last = parts[parts.length - 1];
				if (pendingBreaks > 0 || !last || " \n\t".includes(last.at(-1))) text = text.replace(/^ /, "");
			}
			if (text) append(text);
			return;
		}
		if (node.nodeType !== 1) return;
		const tag = node.tagName.toUpperCase();
		if (node !== root && SKIP.has(tag)) return;
		if (tag === "BR") { if (parts.length > 0) pendingBreaks++; return; }
		if (tag === "TR") cellsInRow = 0;
		if (tag === "TD" || tag === "TH") { if (cellsInRow > 0) append("\t"); cellsInRow++; }
		const breaks = tag === "P" ? 2 : BLOCK.has(tag) ? 1 : 0;
		pendingBreaks = Math.max(pendingBreaks, breaks);
		for (const child of node.childNodes) walk(child, pre || tag === "PRE");
		pendingBreaks = Math.max(pendingBreaks, breaks);
	};
	walk(root, root.parentElement?.closest("pre") != null);
	return parts.join("").trimEnd();
};
if (articleHtml !== null) {
	const doc = new DOMParser().parseFromString("<body>" + articleHtml, "text/html");
	return blockText(doc.body);
}
const selected = selector ? document.querySelector(selector) : null;
if (selector && !selected) return null;
const roots = selected ? [selected] : [
	document.querySelector("[data-pagefind-body]"),
	document.querySelector("main article"),
	document.querySelector("article"),
	document.querySelector("main"),
	document.querySelector("[role='main']"),
	document.body,
];
return {
	url: location.href,
	title: document.title,
	html: document.documentElement.outerHTML,
	selected: !!selected,
	candidates: roots.filter(Boolean).map(el => ({ html: (el.innerHTML ?? "").trim(), text: blockText(el), raw: (el.textContent ?? "").trim() })),
};
})"#;

/// `pushStateInPage` (navigation.ts).
const PUSH_STATE_SCRIPT: &str = r#"(async function (destination) {
const routerPush = globalThis.next?.router?.push;
if (typeof routerPush === "function") {
	await routerPush.call(globalThis.next.router, destination);
} else {
	history.pushState({}, "", destination);
	dispatchEvent(new PopStateEvent("popstate", { state: history.state }));
	dispatchEvent(new Event("navigate"));
}
return location.href;
})"#;

// ---------------------------------------------------------------------------
// Tab state
// ---------------------------------------------------------------------------

/// Retained console/error entries and network records per tab.
const CAPTURE_LIMIT: usize = 500;

/// Browser a tab belongs to.
pub(crate) struct Browser {
    pub key: String,
    /// `headless` (managed Chromium), `spawned` (`app.path`), or `connected` (`app.cdp_url`).
    pub kind: &'static str,
    pub headed: bool,
    pub cdp: Arc<Cdp>,
    pub spawned: tokio::sync::Mutex<Option<Spawned>>,
    downloads_dir: Mutex<Option<PathBuf>>,
}

impl Browser {
    pub(crate) fn new(
        key: String,
        kind: &'static str,
        headed: bool,
        cdp: Arc<Cdp>,
        spawned: Option<Spawned>,
    ) -> Self {
        Self {
            key,
            kind,
            headed,
            cdp,
            spawned: tokio::sync::Mutex::new(spawned),
            downloads_dir: Mutex::new(None),
        }
    }

    /// Route downloads of this browser to `dir` and emit download events.
    async fn enable_downloads(&self, dir: &Path) -> Result<(), String> {
        if self.downloads_dir.lock().as_deref() == Some(dir) {
            return Ok(());
        }
        std::fs::create_dir_all(dir)
            .map_err(|e| format!("cannot create download directory {}: {e}", dir.display()))?;
        self.cdp
            .send(
                None,
                "Browser.setDownloadBehavior",
                json!({ "behavior": "allow", "downloadPath": dir, "eventsEnabled": true }),
            )
            .await?;
        *self.downloads_dir.lock() = Some(dir.to_path_buf());
        Ok(())
    }

    fn downloads_dir(&self) -> Option<PathBuf> {
        self.downloads_dir.lock().clone()
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum DialogPolicy {
    Accept,
    Dismiss,
}

struct PendingDialog {
    kind: String,
    message: String,
    default_value: Option<String>,
}

struct RequestRecord {
    id: String,
    seq: u64,
    ts: f64,
    method: String,
    url: String,
    resource_type: String,
    status: Option<i64>,
    failure: Option<String>,
    started: f64,
    duration_ms: Option<f64>,
    request_headers: Value,
    response_headers: Option<Value>,
    encoded_bytes: Option<f64>,
}

impl RequestRecord {
    fn to_json(&self) -> Value {
        let mut out = json!({
            "id": self.id,
            "seq": self.seq,
            "ts": self.ts,
            "method": self.method,
            "url": self.url,
            "resourceType": self.resource_type,
            "requestHeaders": self.request_headers,
            "sizes": { "responseBody": self.encoded_bytes },
        });
        if let Some(status) = self.status {
            out["status"] = json!(status);
            out["ok"] = json!((200..300).contains(&status));
        }
        if let Some(failure) = &self.failure {
            out["failureText"] = json!(failure);
        }
        if let Some(duration) = self.duration_ms {
            out["durationMs"] = json!(duration);
        }
        if let Some(headers) = &self.response_headers {
            out["responseHeaders"] = headers.clone();
        }
        out
    }
}

struct Download {
    guid: String,
    url: String,
    suggested: String,
    bytes: f64,
    state: String,
    claimed: bool,
}

#[derive(Default)]
struct TabState {
    url: String,
    title: String,
    main_loader: String,
    lifecycle: HashSet<String>,
    /// Bumped on every main-frame navigation (cross- or same-document).
    nav_seq: u64,
    same_document_seq: u64,
    kit_context: Option<i64>,
    console: VecDeque<Value>,
    console_seq: u64,
    console_dropped: u64,
    requests: VecDeque<RequestRecord>,
    request_seq: u64,
    dialog: Option<PendingDialog>,
    dialog_policy: Option<DialogPolicy>,
    allowed_domains: Vec<String>,
    downloads: Vec<Download>,
    detached: bool,
}

impl TabState {
    fn push_console(&mut self, mut entry: Value) {
        self.console_seq += 1;
        entry["seq"] = json!(self.console_seq);
        entry["ts"] = json!(now_ms());
        if self.console.len() == CAPTURE_LIMIT {
            self.console.pop_front();
            self.console_dropped += 1;
        }
        self.console.push_back(entry);
    }

    fn request_mut(&mut self, id: &str) -> Option<&mut RequestRecord> {
        self.requests.iter_mut().rev().find(|r| r.id == id)
    }
}

#[derive(Default)]
struct InputState {
    pointer: Option<(f64, f64)>,
    buttons: Vec<&'static str>,
    mods: u8,
}

struct InitScript {
    id: String,
    identifier: String,
    source: String,
}

/// Per-call execution context: deadline, cancellation, cwd, and display output.
pub(crate) struct OpCtx {
    pub cwd: PathBuf,
    pub cell_ms: u64,
    pub cancel: CancellationToken,
    pub output: Vec<ContentBlock>,
    pub screenshots: Vec<Value>,
}

/// Slack reserved inside the call budget so named op timeouts fire first.
const BUDGET_SLACK_MS: u64 = 1_000;
const ACTION_OP_MS: u64 = 8_000;
const ZERO_MATCH_FAIL_FAST_MS: u64 = 2_000;

impl OpCtx {
    fn budget_ms(&self) -> u64 {
        self.cell_ms.saturating_sub(BUDGET_SLACK_MS).max(1)
    }
    fn action_ms(&self) -> u64 {
        self.budget_ms().min(ACTION_OP_MS)
    }
    /// `resolveWaitTimeout`: explicit ms, 0 = whole budget, default = action timeout.
    fn wait_ms(&self, explicit: Option<f64>) -> u64 {
        match explicit {
            None => self.action_ms(),
            Some(v) if v == 0.0 || v.is_infinite() => self.budget_ms(),
            Some(v) if v.is_finite() && v > 0.0 => (v as u64).min(self.budget_ms()),
            Some(_) => self.action_ms(),
        }
    }
    async fn sleep(&self, ms: u64) -> Result<(), String> {
        tokio::select! {
            _ = tokio::time::sleep(Duration::from_millis(ms)) => Ok(()),
            _ = self.cancel.cancelled() => Err("cancelled".to_string()),
        }
    }
}

fn now_ms() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64() * 1000.0)
        .unwrap_or_default()
}

// ---------------------------------------------------------------------------
// Argument helpers
// ---------------------------------------------------------------------------

fn arg(args: &[Value], index: usize) -> Option<&Value> {
    args.get(index).filter(|v| !v.is_null())
}

fn arg_str<'a>(args: &'a [Value], index: usize, label: &str) -> Result<&'a str, String> {
    arg(args, index).and_then(Value::as_str).ok_or_else(|| {
        format!(
            "{label} requires a string argument at position {}",
            index + 1
        )
    })
}

fn arg_f64(args: &[Value], index: usize, label: &str) -> Result<f64, String> {
    arg(args, index)
        .and_then(Value::as_f64)
        .filter(|v| v.is_finite())
        .ok_or_else(|| format!("{label} requires a finite number at position {}", index + 1))
}

fn arg_opts(args: &[Value], index: usize) -> Map<String, Value> {
    arg(args, index)
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default()
}

fn opt_f64(opts: &Map<String, Value>, key: &str) -> Option<f64> {
    opts.get(key).and_then(Value::as_f64)
}

fn opt_str<'a>(opts: &'a Map<String, Value>, key: &str) -> Option<&'a str> {
    opts.get(key).and_then(Value::as_str)
}

fn opt_bool(opts: &Map<String, Value>, key: &str) -> bool {
    opts.get(key).and_then(Value::as_bool).unwrap_or(false)
}

/// Selector argument: a string parsed like upstream, or an already-parsed spec.
fn selector_arg(args: &[Value], index: usize, label: &str) -> Result<Value, String> {
    match arg(args, index) {
        Some(Value::String(s)) => support::parse_selector(s),
        Some(other) => Err(format!(
            "Browser selector must be a string; got {}. {label} takes string selectors only — call the handle method directly (e.g. tab.id(n).click()) or pass a string like \"aria-ref=eN\".",
            match other {
                Value::Object(_) => "an object",
                Value::Array(_) => "an array",
                Value::Number(_) => "a number",
                Value::Bool(_) => "a boolean",
                _ => "a value",
            }
        )),
        None => Err(format!("{label} requires a selector")),
    }
}

fn describe(spec: &Value) -> String {
    match spec.get("engine").and_then(Value::as_str) {
        Some("css") => json!(spec["query"]).to_string(),
        Some("id") => format!("id {}", spec["id"]),
        Some("ariaRef") => format!("ref {}", spec["ref"]),
        Some("handle") => "element".to_string(),
        Some(engine) => format!(
            "{engine}/{}",
            spec.get("query")
                .or_else(|| spec.get("name"))
                .and_then(Value::as_str)
                .unwrap_or("")
        ),
        None => spec.to_string(),
    }
}

/// `waitUntil` → CDP lifecycle event name.
fn lifecycle_event(wait_until: Option<&str>) -> Result<&'static str, String> {
    Ok(match wait_until.unwrap_or("load") {
        "load" => "load",
        "domcontentloaded" => "DOMContentLoaded",
        "networkidle0" => "networkIdle",
        "networkidle2" => "networkAlmostIdle",
        other => {
            return Err(format!(
                "Unknown waitUntil {}; use load, domcontentloaded, networkidle0, or networkidle2",
                json!(other)
            ));
        }
    })
}

fn mouse_button(name: Option<&str>) -> Result<&'static str, String> {
    match name.unwrap_or("left") {
        "left" => Ok("left"),
        "right" => Ok("right"),
        "middle" => Ok("middle"),
        "back" => Ok("back"),
        "forward" => Ok("forward"),
        other => Err(format!("Unknown mouse button {}", json!(other))),
    }
}

fn buttons_mask(buttons: &[&str]) -> u32 {
    buttons.iter().fold(0, |mask, b| {
        mask | match *b {
            "left" => 1,
            "right" => 2,
            "middle" => 4,
            "back" => 8,
            "forward" => 16,
            _ => 0,
        }
    })
}

fn exception_text(details: &Value) -> String {
    details
        .pointer("/exception/description")
        .and_then(Value::as_str)
        .or_else(|| details.pointer("/exception/value").and_then(Value::as_str))
        .or_else(|| details.get("text").and_then(Value::as_str))
        .unwrap_or("JavaScript exception")
        .to_string()
}

/// First line of a page exception without the `Error: ` prefix.
fn kit_error_text(text: &str) -> String {
    let first = text.lines().next().unwrap_or(text);
    first.strip_prefix("Error: ").unwrap_or(first).to_string()
}

fn is_context_error(error: &str) -> bool {
    error.contains("Cannot find context with specified id")
        || error.contains("Execution context was destroyed")
        || error.contains("Cannot find default execution context")
        || error.contains("Inspected target navigated or closed")
}

// ---------------------------------------------------------------------------
// Tab
// ---------------------------------------------------------------------------

pub(crate) struct Tab {
    pub name: String,
    pub browser: Arc<Browser>,
    pub target_id: String,
    pub session: String,
    /// Created by this runtime: release closes the page target.
    pub owned: bool,
    state: Arc<Mutex<TabState>>,
    input: tokio::sync::Mutex<InputState>,
    init_scripts: Mutex<Vec<InitScript>>,
    next_init_script: Mutex<u64>,
    aria_baselines: Mutex<HashMap<String, SnapshotBaseline>>,
    emulation: Mutex<Map<String, Value>>,
    /// Serializes calls on one tab; input state is not reentrant.
    pub busy: tokio::sync::Mutex<()>,
    pump: tokio::task::JoinHandle<()>,
}

impl Drop for Tab {
    fn drop(&mut self) {
        self.pump.abort();
    }
}

impl Tab {
    /// Attach to `target_id` with a flattened session and start capturing.
    pub(crate) async fn attach(
        browser: Arc<Browser>,
        name: &str,
        target_id: &str,
        owned: bool,
    ) -> Result<Arc<Self>, String> {
        let cdp = browser.cdp.clone();
        let attached = cdp
            .send(
                None,
                "Target.attachToTarget",
                json!({ "targetId": target_id, "flatten": true }),
            )
            .await?;
        let session = attached["sessionId"]
            .as_str()
            .ok_or("Target.attachToTarget returned no sessionId")?
            .to_string();
        let state = Arc::new(Mutex::new(TabState::default()));
        let events = cdp.subscribe();
        let pump = tokio::spawn(pump_events(
            cdp.clone(),
            session.clone(),
            target_id.to_string(),
            state.clone(),
            events,
        ));
        let tab = Arc::new(Self {
            name: name.to_string(),
            browser,
            target_id: target_id.to_string(),
            session,
            owned,
            state,
            input: tokio::sync::Mutex::new(InputState::default()),
            init_scripts: Mutex::new(Vec::new()),
            next_init_script: Mutex::new(1),
            aria_baselines: Mutex::new(HashMap::new()),
            emulation: Mutex::new(Map::new()),
            busy: tokio::sync::Mutex::new(()),
            pump,
        });
        for (method, params) in [
            ("Page.enable", json!({})),
            ("Runtime.enable", json!({})),
            ("Network.enable", json!({})),
            ("Page.setLifecycleEventsEnabled", json!({ "enabled": true })),
        ] {
            tab.cdp(method, params).await?;
        }
        let tree = tab.cdp("Page.getFrameTree", json!({})).await?;
        {
            let mut state = tab.state.lock();
            state.url = tree
                .pointer("/frameTree/frame/url")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string();
            state.main_loader = tree
                .pointer("/frameTree/frame/loaderId")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string();
        }
        let _ = tab.cdp("Runtime.runIfWaitingForDebugger", json!({})).await;
        Ok(tab)
    }

    async fn cdp(&self, method: &str, params: Value) -> Result<Value, String> {
        self.browser
            .cdp
            .send(Some(&self.session), method, params)
            .await
    }

    pub(crate) fn url(&self) -> String {
        self.state.lock().url.clone()
    }

    pub(crate) fn is_detached(&self) -> bool {
        self.state.lock().detached || self.browser.cdp.closed().is_cancelled()
    }

    pub(crate) fn info(&self) -> Value {
        let state = self.state.lock();
        json!({
            "name": self.name,
            "url": state.url,
            "title": state.title,
            "targetId": self.target_id,
            "kind": self.browser.kind,
            "persist": false,
        })
    }

    fn dialog_hint(&self) -> &'static str {
        if self.state.lock().dialog.is_some() {
            " (a confirm/prompt dialog is pending; call tab.handleDialog({accept}) first)"
        } else {
            ""
        }
    }

    // ----- configuration applied at open ---------------------------------

    pub(crate) async fn set_viewport(
        &self,
        width: f64,
        height: f64,
        scale: Option<f64>,
    ) -> Result<(), String> {
        self.cdp(
            "Emulation.setDeviceMetricsOverride",
            json!({
                "width": width as i64,
                "height": height as i64,
                "deviceScaleFactor": scale.unwrap_or(1.0),
                "mobile": false,
            }),
        )
        .await?;
        Ok(())
    }

    pub(crate) async fn set_user_agent(&self, user_agent: &str) -> Result<(), String> {
        self.cdp(
            "Emulation.setUserAgentOverride",
            json!({ "userAgent": user_agent }),
        )
        .await?;
        Ok(())
    }

    pub(crate) async fn set_ignore_https_errors(&self, ignore: bool) -> Result<(), String> {
        self.cdp(
            "Security.setIgnoreCertificateErrors",
            json!({ "ignore": ignore }),
        )
        .await?;
        Ok(())
    }

    pub(crate) fn set_dialog_policy(&self, policy: Option<DialogPolicy>) {
        self.state.lock().dialog_policy = policy;
    }

    pub(crate) async fn set_allowed_domains(&self, domains: Vec<String>) -> Result<(), String> {
        let enable = !domains.is_empty();
        self.state.lock().allowed_domains = domains;
        if enable {
            self.cdp(
                "Fetch.enable",
                json!({ "patterns": [{ "urlPattern": "*", "requestStage": "Request" }] }),
            )
            .await?;
        } else {
            self.cdp("Fetch.disable", json!({})).await?;
        }
        Ok(())
    }

    pub(crate) async fn set_downloads(&self, dir: &Path) -> Result<(), String> {
        self.browser.enable_downloads(dir).await
    }

    pub(crate) async fn add_init_script(&self, source: &str) -> Result<String, String> {
        let added = self
            .cdp(
                "Page.addScriptToEvaluateOnNewDocument",
                json!({ "source": source }),
            )
            .await?;
        let identifier = added["identifier"].as_str().unwrap_or_default().to_string();
        let id = {
            let mut next = self.next_init_script.lock();
            let id = format!("init-{}", *next);
            *next += 1;
            id
        };
        self.init_scripts.lock().push(InitScript {
            id: id.clone(),
            identifier,
            source: source.to_string(),
        });
        Ok(id)
    }

    // ----- evaluation ----------------------------------------------------

    /// `run` action: evaluate agent JavaScript in the page's main world.
    pub(crate) async fn run_expression(&self, expression: &str) -> Result<Value, String> {
        self.eval_main("browser run", expression).await
    }

    /// Evaluate `expression` in the page's main world; returns the JSON value
    /// (`Null` for `undefined`).
    async fn eval_main(&self, label: &str, expression: &str) -> Result<Value, String> {
        let result = self
            .cdp(
                "Runtime.evaluate",
                json!({
                    "expression": expression,
                    "awaitPromise": true,
                    "returnByValue": true,
                    "userGesture": true,
                }),
            )
            .await
            .map_err(|e| format!("{label}: {e}"))?;
        if let Some(details) = result.get("exceptionDetails") {
            return Err(format!(
                "{label} threw a JavaScript exception:\n{}",
                exception_text(details)
            ));
        }
        Ok(result
            .pointer("/result/value")
            .cloned()
            .unwrap_or(Value::Null))
    }

    /// Call `fn_source(...args)` in the main world.
    async fn call_main(
        &self,
        label: &str,
        fn_source: &str,
        args: &[Value],
    ) -> Result<Value, String> {
        let args = serde_json::to_string(args).map_err(|e| e.to_string())?;
        self.eval_main(label, &format!("({fn_source})(...{args})"))
            .await
    }

    async fn kit_context(&self) -> Result<i64, String> {
        if let Some(id) = self.state.lock().kit_context {
            return Ok(id);
        }
        let world = self
            .cdp(
                "Page.createIsolatedWorld",
                json!({ "frameId": self.target_id, "worldName": KIT_WORLD, "grantUniveralAccess": true }),
            )
            .await?;
        let id = world["executionContextId"]
            .as_i64()
            .ok_or("Page.createIsolatedWorld returned no context")?;
        self.state.lock().kit_context = Some(id);
        Ok(id)
    }

    async fn eval_in_kit_world(&self, expression: &str) -> Result<Value, String> {
        let context = self.kit_context().await?;
        let result = self
            .cdp(
                "Runtime.evaluate",
                json!({
                    "expression": expression,
                    "contextId": context,
                    "awaitPromise": true,
                    "returnByValue": true,
                    "userGesture": true,
                }),
            )
            .await?;
        if let Some(details) = result.get("exceptionDetails") {
            return Err(format!("\u{0}js:{}", exception_text(details)));
        }
        Ok(result
            .pointer("/result/value")
            .cloned()
            .unwrap_or(Value::Null))
    }

    /// Call `kit[method](...args)` in the isolated world, installing the kit
    /// into fresh documents.
    async fn kit(&self, method: &str, args: Value) -> Result<Value, String> {
        let call = format!(
            "(async () => {{ const k = globalThis.__jutoBrowserKit; if (!k) return {{ missing: true }}; return {{ value: await k[{}](...{}) }}; }})()",
            json!(method),
            args
        );
        for attempt in 0..3 {
            match self.eval_in_kit_world(&call).await {
                Ok(result) if result.get("missing").is_some() => {
                    self.eval_in_kit_world(&KIT_SOURCE)
                        .await
                        .map_err(|e| e.trim_start_matches("\u{0}js:").to_string())?;
                }
                Ok(result) => return Ok(result.get("value").cloned().unwrap_or(Value::Null)),
                Err(error) if is_context_error(&error) && attempt < 2 => {
                    self.state.lock().kit_context = None;
                }
                Err(error) => {
                    return Err(match error.strip_prefix("\u{0}js:") {
                        Some(js) => kit_error_text(js),
                        None => error,
                    });
                }
            }
        }
        Err(format!("page kit {method} did not become available"))
    }

    // ----- input ---------------------------------------------------------

    /// Dispatch one `Input.*` command. A confirm/prompt opened by the event
    /// blocks the renderer until it is handled, so a pending dialog ends the
    /// wait; the command completes in the background after `handleDialog`.
    async fn input_event(&self, method: &'static str, params: Value) -> Result<(), String> {
        let cdp = self.browser.cdp.clone();
        let session = self.session.clone();
        let mut dispatch =
            tokio::spawn(async move { cdp.send(Some(&session), method, params).await });
        loop {
            tokio::select! {
                result = &mut dispatch => {
                    return result.map_err(|e| e.to_string())?.map(|_| ());
                }
                _ = tokio::time::sleep(Duration::from_millis(25)) => {
                    if self.state.lock().dialog.is_some() {
                        return Ok(());
                    }
                }
            }
        }
    }

    async fn mouse(
        &self,
        kind: &str,
        x: f64,
        y: f64,
        button: &str,
        click_count: u32,
        input: &InputState,
    ) -> Result<(), String> {
        self.input_event(
            "Input.dispatchMouseEvent",
            json!({
                "type": kind,
                "x": x,
                "y": y,
                "button": if kind == "mouseMoved" && input.buttons.is_empty() { "none" } else { button },
                "buttons": buttons_mask(&input.buttons),
                "clickCount": click_count,
                "modifiers": input.mods,
            }),
        )
        .await
    }

    async fn click_point(&self, x: f64, y: f64, button: &str, count: u32) -> Result<(), String> {
        let mut input = self.input.lock().await;
        self.mouse("mouseMoved", x, y, button, 0, &input).await?;
        for click in 1..=count {
            input.buttons.push(leak_button(button));
            self.mouse("mousePressed", x, y, button, click, &input)
                .await?;
            input.buttons.retain(|b| *b != button);
            self.mouse("mouseReleased", x, y, button, click, &input)
                .await?;
        }
        input.pointer = Some((x, y));
        Ok(())
    }

    async fn key(&self, key: &Key, down: bool, mods: u8) -> Result<(), String> {
        self.input_event(
            "Input.dispatchKeyEvent",
            support::key_event(key, down, mods),
        )
        .await
    }

    /// Press `combo` (`Enter`, `Control+a`) with held modifiers (`pressSteps`).
    async fn press_combo(&self, combo: &str) -> Result<(), String> {
        let keys = support::split_key_combo(combo)
            .into_iter()
            .map(support::resolve_key)
            .collect::<Result<Vec<_>, _>>()?;
        let held = self.input.lock().await.mods;
        let mut mods = held;
        for key in &keys {
            self.key(key, true, mods).await?;
            mods |= key.modifier;
        }
        for key in keys.iter().rev() {
            if key.modifier != 0 && held & key.modifier == 0 {
                mods &= !key.modifier;
            }
            self.key(key, false, mods).await?;
        }
        Ok(())
    }

    /// Type like a keyboard: printable ASCII, Enter and Tab as key presses,
    /// anything else as inserted text (`typeSteps`).
    async fn type_text(&self, text: &str) -> Result<(), String> {
        let mut pending = String::new();
        for c in text.chars() {
            if c == '\n' || c == '\r' || c == '\t' || (' '..='~').contains(&c) {
                if !pending.is_empty() {
                    self.cdp("Input.insertText", json!({ "text": pending }))
                        .await?;
                    pending.clear();
                }
                let mut buf = [0u8; 4];
                self.press_combo(c.encode_utf8(&mut buf)).await?;
            } else {
                pending.push(c);
            }
        }
        if !pending.is_empty() {
            self.cdp("Input.insertText", json!({ "text": pending }))
                .await?;
        }
        Ok(())
    }

    // ----- targeting -----------------------------------------------------

    /// Wait for the element to be actionable for `action` and return its
    /// point (`#target`): fail fast after 2 s of zero matches.
    async fn target(
        &self,
        ctx: &OpCtx,
        label: &str,
        spec: &Value,
        action: &str,
        explicit_timeout: Option<f64>,
    ) -> Result<(f64, f64, f64, f64), String> {
        let timeout = match explicit_timeout {
            None => ctx.action_ms(),
            Some(_) => ctx.wait_ms(explicit_timeout),
        };
        let started = Instant::now();
        let mut delay = 30u64;
        let mut last;
        loop {
            last = self.kit("target", json!([spec, action])).await?;
            if last["ok"] == true {
                return Ok((
                    last["x"].as_f64().unwrap_or(0.0),
                    last["y"].as_f64().unwrap_or(0.0),
                    last["width"].as_f64().unwrap_or(0.0),
                    last["height"].as_f64().unwrap_or(0.0),
                ));
            }
            let elapsed = started.elapsed().as_millis() as u64;
            if elapsed >= timeout {
                break;
            }
            if last["reason"] == "missing"
                && explicit_timeout.is_none()
                && elapsed >= ZERO_MATCH_FAIL_FAST_MS
            {
                break;
            }
            ctx.sleep(delay).await?;
            delay = (delay * 3 / 2).min(250);
        }
        let detail = last["detail"].as_str();
        let why = match last["reason"].as_str().unwrap_or("") {
            "missing" => "no element matches (wrong page or selector?)".to_string(),
            "covered" => format!(
                "the element is covered by {}",
                detail.unwrap_or("another element")
            ),
            "hidden" => "the element is not visible".to_string(),
            "disabled" => "the element is disabled".to_string(),
            "notEditable" => "the element is not editable".to_string(),
            "offViewport" => "the element is outside the viewport".to_string(),
            _ => "the element keeps moving".to_string(),
        };
        Err(format!(
            "{label} timed out after {}ms: {why} ({} match(es)){}",
            started.elapsed().as_millis(),
            last["count"].as_u64().unwrap_or(0),
            self.dialog_hint()
        ))
    }

    async fn read(&self, spec: &Value, prop: &str, arg: Value) -> Result<Value, String> {
        self.kit("read", json!([spec, prop, arg])).await
    }

    /// Poll `check` until it yields a value or `timeout_ms` passes.
    async fn poll<F, Fut>(
        &self,
        ctx: &OpCtx,
        label: &str,
        timeout_ms: u64,
        mut check: F,
    ) -> Result<Value, String>
    where
        F: FnMut() -> Fut,
        Fut: Future<Output = Result<Option<Value>, String>>,
    {
        let started = Instant::now();
        loop {
            if let Some(value) = check().await? {
                return Ok(value);
            }
            if started.elapsed().as_millis() as u64 >= timeout_ms {
                return Err(format!(
                    "{label} timed out after {timeout_ms}ms{}",
                    self.dialog_hint()
                ));
            }
            ctx.sleep(100).await?;
        }
    }

    // ----- navigation ----------------------------------------------------

    /// Wait until the main frame's document `loader` reached `event`.
    async fn wait_lifecycle(
        &self,
        ctx: &OpCtx,
        label: &str,
        loader: Option<&str>,
        nav_before: u64,
        event: &str,
    ) -> Result<(), String> {
        let timeout = ctx.budget_ms();
        let started = Instant::now();
        loop {
            {
                let state = self.state.lock();
                if state.detached {
                    return Err(format!("{label}: the tab was closed"));
                }
                let reached = match loader {
                    Some(loader) => state.main_loader == loader && state.lifecycle.contains(event),
                    None => {
                        state.nav_seq > nav_before
                            && (state.same_document_seq == state.nav_seq
                                || state.lifecycle.contains(event))
                    }
                };
                if reached {
                    return Ok(());
                }
            }
            if started.elapsed().as_millis() as u64 >= timeout {
                let _ = self.cdp("Page.stopLoading", json!({})).await;
                return Err(format!(
                    "{label} timed out after {timeout}ms; pending navigation stopped — retry with a longer tool timeout or waitUntil:\"domcontentloaded\""
                ));
            }
            ctx.sleep(50).await?;
        }
    }

    pub(crate) async fn goto(
        &self,
        ctx: &OpCtx,
        url: &str,
        wait_until: Option<&str>,
    ) -> Result<(), String> {
        let label = format!("tab.goto({})", json!(url));
        let event = lifecycle_event(wait_until)?;
        let navigated = self
            .cdp("Page.navigate", json!({ "url": url }))
            .await
            .map_err(|e| format!("{label}: {e}"))?;
        if let Some(error) = navigated.get("errorText").and_then(Value::as_str)
            && !error.is_empty()
        {
            return Err(format!("{label} failed: {error}"));
        }
        match navigated.get("loaderId").and_then(Value::as_str) {
            Some(loader) => {
                self.wait_lifecycle(ctx, &label, Some(loader), 0, event)
                    .await
            }
            // Same-document navigation commits immediately.
            None => Ok(()),
        }
    }

    async fn traverse(
        &self,
        ctx: &OpCtx,
        delta: i64,
        wait_until: Option<&str>,
    ) -> Result<Value, String> {
        let label = if delta < 0 {
            "tab.back()"
        } else {
            "tab.forward()"
        };
        let event = lifecycle_event(wait_until)?;
        let history = self.cdp("Page.getNavigationHistory", json!({})).await?;
        let index = history["currentIndex"].as_i64().unwrap_or(0) + delta;
        let entries = history["entries"].as_array().cloned().unwrap_or_default();
        let Some(entry) = usize::try_from(index).ok().and_then(|i| entries.get(i)) else {
            return Ok(json!(self.url()));
        };
        let before = self.state.lock().nav_seq;
        self.cdp(
            "Page.navigateToHistoryEntry",
            json!({ "entryId": entry["id"] }),
        )
        .await?;
        self.wait_lifecycle(ctx, label, None, before, event).await?;
        Ok(json!(self.url()))
    }

    async fn reload(&self, ctx: &OpCtx, wait_until: Option<&str>) -> Result<Value, String> {
        let event = lifecycle_event(wait_until)?;
        let before = self.state.lock().nav_seq;
        self.cdp("Page.reload", json!({})).await?;
        self.wait_lifecycle(ctx, "tab.reload()", None, before, event)
            .await?;
        Ok(json!(self.url()))
    }

    async fn title(&self) -> Result<String, String> {
        let title = self.eval_main("tab.title()", "document.title").await?;
        let title = title.as_str().unwrap_or_default().to_string();
        self.state.lock().title = title.clone();
        Ok(title)
    }

    // ----- dispatch ------------------------------------------------------

    /// Run one allowlisted tab helper.
    pub(crate) async fn call(
        &self,
        ctx: &mut OpCtx,
        method: &str,
        args: &[Value],
    ) -> Result<Value, String> {
        match method {
            "url" => Ok(json!(self.url())),
            "title" => Ok(json!(self.title().await?)),
            "goto" => {
                let url = arg_str(args, 0, "tab.goto")?;
                let opts = arg_opts(args, 1);
                self.goto(ctx, url, opt_str(&opts, "waitUntil")).await?;
                Ok(Value::Null)
            }
            "back" | "forward" => {
                let opts = arg_opts(args, 0);
                self.traverse(
                    ctx,
                    if method == "back" { -1 } else { 1 },
                    opt_str(&opts, "waitUntil"),
                )
                .await
            }
            "reload" => {
                let opts = arg_opts(args, 0);
                self.reload(ctx, opt_str(&opts, "waitUntil")).await
            }
            "pushState" => {
                let url = arg_str(args, 0, "tab.pushState")?;
                self.call_main("tab.pushState()", PUSH_STATE_SCRIPT, &[json!(url)])
                    .await
            }
            "waitForUrl" => {
                let pattern = arg_str(args, 0, "tab.waitForUrl")?.to_string();
                let opts = arg_opts(args, 1);
                let timeout = ctx.wait_ms(opt_f64(&opts, "timeout"));
                let label = format!("tab.waitForUrl({pattern})");
                self.poll(ctx, &label, timeout, || async {
                    let url = self.eval_main("tab.waitForUrl()", "location.href").await?;
                    let url = url.as_str().unwrap_or_default().to_string();
                    Ok(url.contains(&pattern).then(|| json!(url)))
                })
                .await
            }
            "frames" => self.frames().await,
            "dialog" => Ok(self.dialog_state()),
            "handleDialog" => {
                let opts = arg_opts(args, 0);
                self.handle_dialog(opt_bool(&opts, "accept"), opt_str(&opts, "text"))
                    .await?;
                Ok(Value::Null)
            }
            "setDialogs" => {
                let policy = match arg(args, 0).and_then(Value::as_str) {
                    None => None,
                    Some("accept") => Some(DialogPolicy::Accept),
                    Some("dismiss") => Some(DialogPolicy::Dismiss),
                    Some(other) => {
                        return Err(format!(
                            "tab.setDialogs expects \"accept\", \"dismiss\", or null; got {}",
                            json!(other)
                        ));
                    }
                };
                self.set_dialog_policy(policy);
                let pending = self.state.lock().dialog.is_some();
                if let (Some(policy), true) = (policy, pending) {
                    self.handle_dialog(policy == DialogPolicy::Accept, None)
                        .await?;
                }
                Ok(Value::Null)
            }
            "observe" => {
                let opts = arg_opts(args, 0);
                let mut request = json!({
                    "includeAll": opt_bool(&opts, "includeAll"),
                    "viewportOnly": opt_bool(&opts, "viewportOnly"),
                    "compact": opt_bool(&opts, "compact"),
                });
                if let Some(selector) = opt_str(&opts, "selector") {
                    request["root"] = support::parse_selector(selector)?;
                }
                let observation = self.kit("observe", json!([request])).await?;
                let mut state = self.state.lock();
                if let Some(url) = observation["url"].as_str() {
                    state.url = url.to_string();
                }
                if let Some(title) = observation["title"].as_str() {
                    state.title = title.to_string();
                }
                Ok(observation)
            }
            "ariaSnapshot" => {
                let selector = arg(args, 0).and_then(Value::as_str).map(str::to_string);
                // `ariaSnapshot({…})` with options only is accepted too.
                let opts = match arg(args, 0) {
                    Some(Value::Object(map)) => map.clone(),
                    _ => arg_opts(args, 1),
                };
                self.aria_snapshot(selector.as_deref(), &opts).await
            }
            "screenshot" => {
                let opts = arg_opts(args, 0);
                self.screenshot(ctx, &opts).await
            }
            "pdf" => {
                let opts = arg_opts(args, 0);
                self.pdf(ctx, &opts).await
            }
            "extract" => {
                let format = arg(args, 0)
                    .and_then(Value::as_str)
                    .unwrap_or("markdown")
                    .to_string();
                let opts = arg_opts(args, 1);
                self.extract(&format, &opts).await
            }
            "click" | "dblclick" => {
                let spec = selector_arg(args, 0, &format!("tab.{method}"))?;
                let opts = arg_opts(args, 1);
                let count = if method == "dblclick" {
                    2
                } else {
                    opt_f64(&opts, "count").map_or(1, |c| c.max(1.0) as u32)
                };
                self.click_spec(ctx, &spec, count, opt_str(&opts, "button"))
                    .await?;
                Ok(Value::Null)
            }
            "hover" => {
                let spec = selector_arg(args, 0, "tab.hover")?;
                self.hover_spec(ctx, &spec).await?;
                Ok(Value::Null)
            }
            "focus" => {
                let spec = selector_arg(args, 0, "tab.focus")?;
                self.focus_spec(ctx, &spec).await?;
                Ok(Value::Null)
            }
            "check" | "uncheck" => {
                let spec = selector_arg(args, 0, &format!("tab.{method}"))?;
                self.set_checked(ctx, &spec, method == "check").await?;
                Ok(Value::Null)
            }
            "type" => {
                let spec = selector_arg(args, 0, "tab.type")?;
                let text = value_text(arg(args, 1));
                self.type_spec(ctx, &spec, &text).await?;
                Ok(Value::Null)
            }
            "fill" => {
                let spec = selector_arg(args, 0, "tab.fill")?;
                let text = value_text(arg(args, 1));
                self.fill_spec(ctx, &spec, &text).await?;
                Ok(Value::Null)
            }
            "press" => {
                let key = arg_str(args, 0, "tab.press")?;
                let opts = arg_opts(args, 1);
                if let Some(selector) = opt_str(&opts, "selector") {
                    let spec = support::parse_selector(selector)?;
                    self.focus_spec(ctx, &spec).await?;
                }
                self.press_combo(key).await?;
                Ok(Value::Null)
            }
            "keyDown" | "keyUp" => {
                let key = support::resolve_key(arg_str(args, 0, &format!("tab.{method}"))?)?;
                let mut input = self.input.lock().await;
                if method == "keyDown" {
                    self.key(&key, true, input.mods).await?;
                    input.mods |= key.modifier;
                } else {
                    input.mods &= !key.modifier;
                    self.key(&key, false, input.mods).await?;
                }
                Ok(Value::Null)
            }
            "mouseMove" => {
                let x = arg_f64(args, 0, "mouseMove")?;
                let y = arg_f64(args, 1, "mouseMove")?;
                let steps = opt_f64(&arg_opts(args, 2), "steps").map_or(1, |s| s.max(1.0) as u32);
                let mut input = self.input.lock().await;
                let (fx, fy) = input.pointer.unwrap_or((x, y));
                let held = input.buttons.first().copied().unwrap_or("left");
                for step in 1..=steps {
                    let t = step as f64 / steps as f64;
                    self.mouse(
                        "mouseMoved",
                        fx + (x - fx) * t,
                        fy + (y - fy) * t,
                        held,
                        0,
                        &input,
                    )
                    .await?;
                }
                input.pointer = Some((x, y));
                Ok(Value::Null)
            }
            "mouseDown" | "mouseUp" => {
                let opts = arg_opts(args, 0);
                let button = mouse_button(opt_str(&opts, "button"))?;
                let count = opt_f64(&opts, "clickCount").map_or(1, |c| c.max(1.0) as u32);
                let (x, y) = match self.input.lock().await.pointer {
                    Some(point) => point,
                    None => self.viewport_centre().await?,
                };
                let mut input = self.input.lock().await;
                if method == "mouseDown" {
                    input.buttons.push(button);
                    self.mouse("mousePressed", x, y, button, count, &input)
                        .await?;
                } else {
                    input.buttons.retain(|b| *b != button);
                    self.mouse("mouseReleased", x, y, button, count, &input)
                        .await?;
                }
                input.pointer = Some((x, y));
                Ok(Value::Null)
            }
            "clickAt" => {
                let x = arg_f64(args, 0, "clickAt")?;
                let y = arg_f64(args, 1, "clickAt")?;
                let opts = arg_opts(args, 2);
                let button = mouse_button(opt_str(&opts, "button"))?;
                let count = opt_f64(&opts, "clickCount").map_or(1, |c| c.max(1.0) as u32);
                self.click_point(x, y, button, count).await?;
                Ok(Value::Null)
            }
            "wheel" => {
                let dx = arg_f64(args, 0, "wheel")?;
                let dy = arg_f64(args, 1, "wheel")?;
                let (x, y) = match self.input.lock().await.pointer {
                    Some(point) => point,
                    None => self.viewport_centre().await?,
                };
                self.wheel_at(x, y, dx, dy).await?;
                Ok(Value::Null)
            }
            "scroll" => {
                let dx = arg_f64(args, 0, "scroll")?;
                let dy = arg_f64(args, 1, "scroll")?;
                let opts = arg_opts(args, 2);
                if let Some(selector) = opt_str(&opts, "selector") {
                    let spec = support::parse_selector(selector)?;
                    self.target(
                        ctx,
                        &format!("tab.scroll({})", describe(&spec)),
                        &spec,
                        "point",
                        None,
                    )
                    .await?;
                    self.kit("scrollBy", json!([spec, dx, dy])).await?;
                } else {
                    let (x, y) = self.viewport_centre().await?;
                    self.wheel_at(x, y, dx, dy).await?;
                }
                Ok(Value::Null)
            }
            "drag" => {
                let from = self.drag_point(ctx, arg(args, 0), "from").await?;
                let to = self.drag_point(ctx, arg(args, 1), "to").await?;
                let mut input = self.input.lock().await;
                self.mouse("mouseMoved", from.0, from.1, "left", 0, &input)
                    .await?;
                input.buttons.push("left");
                self.mouse("mousePressed", from.0, from.1, "left", 1, &input)
                    .await?;
                for step in 1..=10 {
                    let t = step as f64 / 10.0;
                    self.mouse(
                        "mouseMoved",
                        from.0 + (to.0 - from.0) * t,
                        from.1 + (to.1 - from.1) * t,
                        "left",
                        0,
                        &input,
                    )
                    .await?;
                }
                input.buttons.retain(|b| *b != "left");
                self.mouse("mouseReleased", to.0, to.1, "left", 1, &input)
                    .await?;
                input.pointer = Some(to);
                Ok(Value::Null)
            }
            "highlight" => {
                let spec = selector_arg(args, 0, "tab.highlight")?;
                let duration = opt_f64(&arg_opts(args, 1), "duration").unwrap_or(2000.0);
                self.highlight_spec(ctx, &spec, duration).await?;
                Ok(Value::Null)
            }
            "scrollIntoView" => {
                let spec = selector_arg(args, 0, "tab.scrollIntoView")?;
                self.target(
                    ctx,
                    &format!("tab.scrollIntoView({})", describe(&spec)),
                    &spec,
                    "point",
                    None,
                )
                .await?;
                self.kit("scrollIntoView", json!([spec])).await?;
                Ok(Value::Null)
            }
            "select" => {
                let spec = selector_arg(args, 0, "tab.select")?;
                self.select_spec(ctx, &spec, &args[1.min(args.len())..])
                    .await
            }
            "uploadFile" => {
                let spec = selector_arg(args, 0, "tab.uploadFile")?;
                self.upload_spec(ctx, &spec, &args[1.min(args.len())..])
                    .await?;
                Ok(Value::Null)
            }
            "text" | "html" | "value" => {
                let spec = selector_arg(args, 0, &format!("tab.{method}"))?;
                self.read(&spec, method, Value::Null).await
            }
            "attr" => {
                let spec = selector_arg(args, 0, "tab.attr")?;
                let name = arg_str(args, 1, "tab.attr")?;
                self.read(&spec, "attr", json!(name)).await
            }
            "count" => {
                let spec = selector_arg(args, 0, "tab.count")?;
                self.kit("count", json!([spec])).await
            }
            "box" => {
                let spec = selector_arg(args, 0, "tab.box")?;
                self.read(&spec, "box", Value::Null).await
            }
            "styles" => {
                let spec = selector_arg(args, 0, "tab.styles")?;
                self.read(
                    &spec,
                    "styles",
                    arg(args, 1).cloned().unwrap_or(Value::Null),
                )
                .await
            }
            "isVisible" | "isEnabled" | "isChecked" => {
                let spec = selector_arg(args, 0, &format!("tab.{method}"))?;
                let prop = match method {
                    "isVisible" => "visible",
                    "isEnabled" => "enabled",
                    _ => "checked",
                };
                Ok(json!(self.read(&spec, prop, Value::Null).await? == true))
            }
            "waitFor" => {
                let spec = selector_arg(args, 0, "tab.waitFor")?;
                let timeout = ctx.wait_ms(opt_f64(&arg_opts(args, 1), "timeout"));
                let label = format!("tab.waitFor({})", describe(&spec));
                self.poll(ctx, &label, timeout, || async {
                    Ok((self.read(&spec, "visible", Value::Null).await? == true)
                        .then_some(json!(true)))
                })
                .await
            }
            "waitForSelector" => {
                let spec = selector_arg(args, 0, "tab.waitForSelector")?;
                let opts = arg_opts(args, 1);
                let timeout = ctx.wait_ms(opt_f64(&opts, "timeout"));
                let label = format!("tab.waitForSelector({})", describe(&spec));
                let hidden = opt_bool(&opts, "hidden");
                let visible = opt_bool(&opts, "visible");
                self.poll(ctx, &label, timeout, || async {
                    if hidden {
                        let count = self.kit("count", json!([spec])).await?;
                        if count.as_u64() == Some(0) {
                            return Ok(Some(json!(false)));
                        }
                        return Ok((self.read(&spec, "visible", Value::Null).await? != true)
                            .then_some(json!(false)));
                    }
                    if visible {
                        return Ok((self.read(&spec, "visible", Value::Null).await? == true)
                            .then_some(json!(true)));
                    }
                    let count = self.kit("count", json!([spec])).await?;
                    Ok((count.as_u64().unwrap_or(0) > 0).then_some(json!(true)))
                })
                .await
            }
            "waitForText" => {
                let text = arg_str(args, 0, "tab.waitForText")?.to_string();
                let opts = arg_opts(args, 1);
                let root = match opt_str(&opts, "selector") {
                    Some(selector) => support::parse_selector(selector)?,
                    None => Value::Null,
                };
                let exact = opt_bool(&opts, "exact");
                let timeout = ctx.wait_ms(opt_f64(&opts, "timeout"));
                let label = format!("tab.waitForText({})", json!(text));
                self.poll(ctx, &label, timeout, || async {
                    Ok(
                        (self.kit("hasText", json!([text, root, exact])).await? == true)
                            .then_some(Value::Null),
                    )
                })
                .await
                .map(|_| Value::Null)
            }
            "evaluate" => {
                let source = arg_str(args, 0, "tab.evaluate")?;
                // Puppeteer evaluates strings as page-global expressions.
                self.eval_main("tab.evaluate()", source).await
            }
            "emulate" => {
                let opts = arg_opts(args, 0);
                self.emulate(&opts).await
            }
            "cookies" => {
                let opts = arg_opts(args, 0);
                let urls = match opts.get("urls") {
                    Some(Value::Array(urls)) => Value::Array(urls.clone()),
                    Some(_) => return Err("tab.cookies() expects urls to be an array".into()),
                    None => json!([self.url()]),
                };
                let cookies = self
                    .cdp("Network.getCookies", json!({ "urls": urls }))
                    .await?;
                Ok(cookies["cookies"].clone())
            }
            "setCookies" => {
                self.set_cookies(args).await?;
                Ok(Value::Null)
            }
            "clearCookies" => {
                let opts = arg_opts(args, 0);
                let names: Option<Vec<String>> = match opts.get("names") {
                    Some(Value::Array(names)) => Some(
                        names
                            .iter()
                            .filter_map(Value::as_str)
                            .map(str::to_string)
                            .collect(),
                    ),
                    Some(_) => return Err("tab.clearCookies() expects names to be an array".into()),
                    None => None,
                };
                let cookies = self
                    .cdp("Network.getCookies", json!({ "urls": [self.url()] }))
                    .await?;
                for cookie in cookies["cookies"].as_array().into_iter().flatten() {
                    let name = cookie["name"].as_str().unwrap_or_default();
                    if names.as_ref().is_some_and(|n| !n.iter().any(|w| w == name)) {
                        continue;
                    }
                    self.cdp(
                        "Network.deleteCookies",
                        json!({ "name": name, "domain": cookie["domain"], "path": cookie["path"] }),
                    )
                    .await?;
                }
                Ok(Value::Null)
            }
            "storage" | "setStorage" | "clearStorage" => self.storage(method, args).await,
            "addInitScript" => {
                let source = arg_str(args, 0, "tab.addInitScript(source)")?;
                Ok(json!({ "id": self.add_init_script(source).await? }))
            }
            "removeInitScript" => {
                let id = arg_str(args, 0, "tab.removeInitScript(id)")?;
                let identifier = {
                    let mut scripts = self.init_scripts.lock();
                    let index = scripts
                        .iter()
                        .position(|s| s.id == id)
                        .ok_or_else(|| format!("Unknown init script {}", json!(id)))?;
                    scripts.remove(index).identifier
                };
                self.cdp(
                    "Page.removeScriptToEvaluateOnNewDocument",
                    json!({ "identifier": identifier }),
                )
                .await?;
                Ok(Value::Null)
            }
            "initScripts" => Ok(Value::Array(
                self.init_scripts
                    .lock()
                    .iter()
                    .map(|s| json!({ "id": s.id, "source": s.source }))
                    .collect(),
            )),
            "waitForDownload" => {
                self.ensure_downloads().await?;
                let opts = arg_opts(args, 0);
                let timeout = ctx.wait_ms(Some(
                    opt_f64(&opts, "timeout").unwrap_or(ctx.budget_ms() as f64),
                ));
                let failed_before: HashSet<String> = self
                    .state
                    .lock()
                    .downloads
                    .iter()
                    .filter(|d| d.state == "canceled")
                    .map(|d| d.guid.clone())
                    .collect();
                let dir = self.browser.downloads_dir();
                self.poll(ctx, "tab.waitForDownload()", timeout, || {
                    let ready = {
                        let mut state = self.state.lock();
                        let failed = state
                            .downloads
                            .iter()
                            .find(|d| d.state == "canceled" && !failed_before.contains(&d.guid))
                            .map(|d| d.url.clone());
                        match failed {
                            Some(url) => Err(format!("Download of {url} failed: canceled")),
                            None => Ok(state
                                .downloads
                                .iter_mut()
                                .find(|d| d.state == "completed" && !d.claimed)
                                .map(|d| {
                                    d.claimed = true;
                                    download_json(d, dir.as_deref())
                                })),
                        }
                    };
                    async move { ready }
                })
                .await
            }
            "downloads" => {
                self.ensure_downloads().await?;
                let dir = self.browser.downloads_dir();
                Ok(Value::Array(
                    self.state
                        .lock()
                        .downloads
                        .iter()
                        .filter(|d| d.state == "completed")
                        .map(|d| download_json(d, dir.as_deref()))
                        .collect(),
                ))
            }
            "console" | "errors" => {
                let opts = arg_opts(args, 0);
                Ok(self.capture_log(method == "console", &opts))
            }
            "clearConsole" => {
                let mut state = self.state.lock();
                state.console.clear();
                state.console_dropped = 0;
                Ok(Value::Null)
            }
            "metrics" => {
                let _ = self.cdp("Performance.enable", json!({})).await;
                let metrics = self.cdp("Performance.getMetrics", json!({})).await?;
                let mut out = Map::new();
                for metric in metrics["metrics"].as_array().into_iter().flatten() {
                    if let Some(name) = metric["name"].as_str() {
                        out.insert(name.to_string(), metric["value"].clone());
                    }
                }
                Ok(Value::Object(out))
            }
            "requests" => {
                let opts = arg_opts(args, 0);
                Ok(self.requests(&opts))
            }
            "clearRequests" => {
                self.state.lock().requests.clear();
                Ok(Value::Null)
            }
            "allowedDomains" => Ok(json!(self.state.lock().allowed_domains)),
            other => Err(not_ported(other)),
        }
    }

    /// Run one method on an element handle (`tab.id(n)` / `tab.ref(id)`).
    pub(crate) async fn element_call(
        &self,
        ctx: &mut OpCtx,
        spec: &Value,
        method: &str,
        args: &[Value],
    ) -> Result<Value, String> {
        match method {
            "click" => {
                let opts = arg_opts(args, 0);
                let count = opt_f64(&opts, "count").map_or(1, |c| c.max(1.0) as u32);
                self.click_spec(ctx, spec, count, opt_str(&opts, "button"))
                    .await?;
                Ok(Value::Null)
            }
            "dblclick" => {
                self.click_spec(ctx, spec, 2, None).await?;
                Ok(Value::Null)
            }
            "check" | "uncheck" => {
                self.set_checked(ctx, spec, method == "check").await?;
                Ok(Value::Null)
            }
            "highlight" => {
                let duration = opt_f64(&arg_opts(args, 0), "duration").unwrap_or(2000.0);
                self.highlight_spec(ctx, spec, duration).await?;
                Ok(Value::Null)
            }
            "type" => {
                self.type_spec(ctx, spec, &value_text(arg(args, 0))).await?;
                Ok(Value::Null)
            }
            "fill" => {
                self.fill_spec(ctx, spec, &value_text(arg(args, 0))).await?;
                Ok(Value::Null)
            }
            "press" => {
                let key = arg_str(args, 0, "element.press")?;
                self.focus_spec(ctx, spec).await?;
                self.press_combo(key).await?;
                Ok(Value::Null)
            }
            "hover" => {
                self.hover_spec(ctx, spec).await?;
                Ok(Value::Null)
            }
            "focus" => {
                self.focus_spec(ctx, spec).await?;
                Ok(Value::Null)
            }
            "select" => self.select_spec(ctx, spec, args).await,
            "uploadFile" => {
                self.upload_spec(ctx, spec, args).await?;
                Ok(Value::Null)
            }
            "scrollIntoView" => {
                self.target(ctx, "element.scrollIntoView()", spec, "point", None)
                    .await?;
                self.kit("scrollIntoView", json!([spec])).await?;
                Ok(Value::Null)
            }
            "boundingBox" => self.read(spec, "box", Value::Null).await,
            "isVisible" => Ok(json!(
                self.read(spec, "visible", Value::Null).await? == true
            )),
            "isHidden" => Ok(json!(
                self.read(spec, "visible", Value::Null).await? != true
            )),
            "text" | "html" | "value" => self.read(spec, method, Value::Null).await,
            "attr" => {
                let name = arg_str(args, 0, "element.attr")?;
                self.read(spec, "attr", json!(name)).await
            }
            "styles" => {
                self.read(spec, "styles", arg(args, 0).cloned().unwrap_or(Value::Null))
                    .await
            }
            "isEnabled" => Ok(json!(
                self.read(spec, "enabled", Value::Null).await? == true
            )),
            "isChecked" => Ok(json!(
                self.read(spec, "checked", Value::Null).await? == true
            )),
            "evaluate" => {
                let source = arg_str(args, 0, "element.evaluate")?;
                self.evaluate_on_element(spec, source, &args[1..]).await
            }
            other => Err(format!(
                "Unknown element method \"{other}\". Element handles support: click, dblclick, check, uncheck, highlight, type, fill, press, hover, focus, select, uploadFile, scrollIntoView, boundingBox, isVisible, isHidden, text, html, value, attr, styles, isEnabled, isChecked, evaluate."
            )),
        }
    }

    /// Validate an observation id or ARIA ref and return its selector spec.
    pub(crate) async fn handle(&self, method: &str, args: &[Value]) -> Result<Value, String> {
        match method {
            "id" => {
                let id = arg(args, 0)
                    .and_then(Value::as_u64)
                    .ok_or("tab.id(n) requires a numeric observation id")?;
                let spec = json!({ "engine": "id", "id": id });
                self.kit("count", json!([spec])).await?;
                Ok(spec)
            }
            "ref" => {
                let raw = arg_str(args, 0, "tab.ref")?;
                let reference =
                    support::parse_aria_ref(raw).unwrap_or_else(|| raw.trim().to_string());
                let spec = json!({ "engine": "ariaRef", "ref": reference });
                if self.kit("count", json!([spec])).await? == json!(0) {
                    return Err(format!(
                        "Unknown ARIA ref {}. Run tab.ariaSnapshot() to refresh refs (they renumber each snapshot).",
                        json!(reference)
                    ));
                }
                Ok(spec)
            }
            other => Err(not_ported(other)),
        }
    }

    // ----- interactions --------------------------------------------------

    async fn click_spec(
        &self,
        ctx: &OpCtx,
        spec: &Value,
        count: u32,
        button: Option<&str>,
    ) -> Result<(), String> {
        let button = mouse_button(button)?;
        let label = format!(
            "tab.{}({})",
            if count == 2 { "dblclick" } else { "click" },
            describe(spec)
        );
        let (x, y, ..) = self
            .target(
                ctx,
                &label,
                spec,
                if count == 2 { "dblclick" } else { "click" },
                None,
            )
            .await?;
        self.click_point(x, y, button, count).await
    }

    async fn hover_spec(&self, ctx: &OpCtx, spec: &Value) -> Result<(), String> {
        let (x, y, ..) = self
            .target(
                ctx,
                &format!("tab.hover({})", describe(spec)),
                spec,
                "hover",
                None,
            )
            .await?;
        let mut input = self.input.lock().await;
        self.mouse("mouseMoved", x, y, "left", 0, &input).await?;
        input.pointer = Some((x, y));
        Ok(())
    }

    async fn focus_spec(&self, ctx: &OpCtx, spec: &Value) -> Result<(), String> {
        self.target(
            ctx,
            &format!("tab.focus({})", describe(spec)),
            spec,
            "focus",
            None,
        )
        .await?;
        self.kit("focus", json!([spec])).await?;
        Ok(())
    }

    async fn set_checked(&self, ctx: &OpCtx, spec: &Value, desired: bool) -> Result<(), String> {
        let label = format!(
            "tab.{}({})",
            if desired { "check" } else { "uncheck" },
            describe(spec)
        );
        let (x, y, ..) = self.target(ctx, &label, spec, "check", None).await?;
        let before = self.kit("checkState", json!([spec])).await?;
        if before["checked"] == desired {
            return Ok(());
        }
        self.click_point(x, y, "left", 1).await?;
        let after = self.kit("checkState", json!([spec])).await?;
        if after["checked"] != desired {
            return Err(format!(
                "{label} clicked the element but it is still {}",
                if desired { "unchecked" } else { "checked" }
            ));
        }
        Ok(())
    }

    async fn type_spec(&self, ctx: &OpCtx, spec: &Value, text: &str) -> Result<(), String> {
        self.target(
            ctx,
            &format!("tab.type({})", describe(spec)),
            spec,
            "type",
            None,
        )
        .await?;
        self.kit("caretToEnd", json!([spec])).await?;
        self.type_text(text).await
    }

    async fn fill_spec(&self, ctx: &OpCtx, spec: &Value, text: &str) -> Result<(), String> {
        self.target(
            ctx,
            &format!("tab.fill({})", describe(spec)),
            spec,
            "fill",
            None,
        )
        .await?;
        let prepared = self.kit("prepareFill", json!([spec, text])).await?;
        if prepared["mode"] == "done" {
            return Ok(());
        }
        if text.is_empty() {
            self.press_combo("Backspace").await
        } else {
            self.cdp("Input.insertText", json!({ "text": text }))
                .await?;
            Ok(())
        }
    }

    async fn highlight_spec(&self, ctx: &OpCtx, spec: &Value, duration: f64) -> Result<(), String> {
        if !duration.is_finite() || duration < 0.0 {
            return Err("highlight duration must be a non-negative number".into());
        }
        self.target(
            ctx,
            &format!("tab.highlight({})", describe(spec)),
            spec,
            "point",
            None,
        )
        .await?;
        let id = format!("juto-highlight-{}", uuid::Uuid::now_v7());
        self.kit("highlight", json!([spec, id])).await?;
        let waited = ctx.sleep(duration as u64).await;
        let _ = self.kit("removeOverlay", json!([id])).await;
        waited
    }

    async fn select_spec(
        &self,
        ctx: &OpCtx,
        spec: &Value,
        values: &[Value],
    ) -> Result<Value, String> {
        let values: Vec<String> = values
            .iter()
            .flat_map(|v| match v {
                Value::Array(items) => items.iter().map(|i| value_text(Some(i))).collect(),
                other => vec![value_text(Some(other))],
            })
            .collect();
        self.target(
            ctx,
            &format!("tab.select({})", describe(spec)),
            spec,
            "point",
            None,
        )
        .await?;
        self.kit("select", json!([spec, values])).await
    }

    async fn upload_spec(&self, ctx: &OpCtx, spec: &Value, paths: &[Value]) -> Result<(), String> {
        let files: Vec<PathBuf> = paths
            .iter()
            .flat_map(|v| match v {
                Value::Array(items) => items.clone(),
                other => vec![other.clone()],
            })
            .filter_map(|v| v.as_str().map(|p| super::resolve_path(&ctx.cwd, p)))
            .collect();
        if files.is_empty() {
            return Err("tab.uploadFile() requires at least one file path".into());
        }
        for file in &files {
            if !file.is_file() {
                return Err(format!("tab.uploadFile() cannot read {}", file.display()));
            }
        }
        self.target(
            ctx,
            &format!("tab.uploadFile({})", describe(spec)),
            spec,
            "upload",
            None,
        )
        .await?;
        if self.kit("isFileInput", json!([spec])).await? == true {
            let token = self
                .kit("mark", json!([spec]))
                .await?
                .as_str()
                .map(str::to_string)
                .ok_or("Element handle selector no longer resolves")?;
            let found = self
                .cdp(
                    "Runtime.evaluate",
                    json!({ "expression": find_marked_expression(&token, true) }),
                )
                .await;
            let _ = self.kit("unmark", json!([token])).await;
            let object_id = found?
                .pointer("/result/objectId")
                .and_then(Value::as_str)
                .map(str::to_string)
                .ok_or("Element handle selector no longer resolves")?;
            self.cdp(
                "DOM.setFileInputFiles",
                json!({ "files": files, "objectId": object_id }),
            )
            .await?;
            let _ = self
                .cdp("Runtime.releaseObject", json!({ "objectId": object_id }))
                .await;
            return Ok(());
        }
        // Not a file input: drop the files on the element.
        let mut payload = Vec::with_capacity(files.len());
        for file in &files {
            let data = std::fs::read(file)
                .map_err(|e| format!("tab.uploadFile() cannot read {}: {e}", file.display()))?;
            payload.push(json!({
                "name": file.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default(),
                "type": mime_for(file),
                "data": base64::engine::general_purpose::STANDARD.encode(data),
            }));
        }
        self.kit("setFiles", json!([spec, payload])).await?;
        Ok(())
    }

    async fn evaluate_on_element(
        &self,
        spec: &Value,
        source: &str,
        args: &[Value],
    ) -> Result<Value, String> {
        let token = self
            .kit("mark", json!([spec]))
            .await?
            .as_str()
            .map(str::to_string)
            .ok_or("Element handle selector no longer resolves")?;
        let wrapper = format!(
            "(async (args) => {{ const element = {}; if (!element) throw new Error(\"Element handle selector no longer resolves\"); return await ({source})(element, ...args); }})({})",
            find_marked_expression(&token, false),
            serde_json::to_string(args).map_err(|e| e.to_string())?
        );
        let result = self.eval_main("elementHandle.evaluate()", &wrapper).await;
        let _ = self.kit("unmark", json!([token])).await;
        result
    }

    async fn viewport_centre(&self) -> Result<(f64, f64), String> {
        let geometry = self.kit("geometry", json!([])).await?;
        Ok((
            geometry["innerWidth"].as_f64().unwrap_or(0.0) / 2.0,
            geometry["innerHeight"].as_f64().unwrap_or(0.0) / 2.0,
        ))
    }

    async fn wheel_at(&self, x: f64, y: f64, dx: f64, dy: f64) -> Result<(), String> {
        let mods = self.input.lock().await.mods;
        self.cdp(
            "Input.dispatchMouseEvent",
            json!({ "type": "mouseWheel", "x": x, "y": y, "deltaX": dx, "deltaY": dy, "modifiers": mods }),
        )
        .await?;
        Ok(())
    }

    async fn drag_point(
        &self,
        ctx: &OpCtx,
        target: Option<&Value>,
        role: &str,
    ) -> Result<(f64, f64), String> {
        match target {
            Some(Value::String(selector)) => {
                let spec = support::parse_selector(selector)?;
                let (x, y, ..) = self
                    .target(
                        ctx,
                        &format!("tab.drag() {role} {}", json!(selector)),
                        &spec,
                        "drag",
                        None,
                    )
                    .await?;
                Ok((x, y))
            }
            Some(Value::Object(point)) => match (opt_f64(point, "x"), opt_f64(point, "y")) {
                (Some(x), Some(y)) if x.is_finite() && y.is_finite() => Ok((x, y)),
                _ => Err(
                    "Drag target must be a selector string or { x: number, y: number } point"
                        .into(),
                ),
            },
            _ => Err(
                "Drag target must be a selector string or { x: number, y: number } point".into(),
            ),
        }
    }

    // ----- inspection ----------------------------------------------------

    async fn frames(&self) -> Result<Value, String> {
        let children = self.kit("frames", json!([])).await?;
        let mut out =
            vec![json!({ "id": "main", "name": "", "url": self.url(), "parentId": null })];
        for child in children.as_array().into_iter().flatten() {
            let path = child["path"].as_str().unwrap_or_default();
            let parent = path.rsplit_once('.').map_or("main", |(p, _)| p);
            let mut info = json!({
                "id": path,
                "name": child["name"].as_str().or(child["id"].as_str()).unwrap_or(""),
                "url": child["url"].as_str().or(child["src"].as_str()).unwrap_or(""),
                "parentId": parent,
            });
            if let Some(id) = child["id"].as_str() {
                info["selector"] = json!(format!("iframe[id={}]", json!(id)));
            } else if let Some(name) = child["name"].as_str() {
                info["selector"] = json!(format!("iframe[name={}]", json!(name)));
            }
            out.push(info);
        }
        Ok(Value::Array(out))
    }

    async fn aria_snapshot(
        &self,
        selector: Option<&str>,
        opts: &Map<String, Value>,
    ) -> Result<Value, String> {
        let root = match selector {
            Some(selector) => support::parse_selector(selector)?,
            None => Value::Null,
        };
        let request = json!({ "depth": opts.get("depth"), "boxes": opts.get("boxes") });
        let urls = opt_bool(opts, "urls");
        let payload = self
            .kit("ariaSnapshot", json!([root, request, urls]))
            .await?;
        let hrefs: HashMap<String, String> = payload["hrefs"]
            .as_object()
            .map(|m| {
                m.iter()
                    .filter_map(|(k, v)| v.as_str().map(|v| (k.clone(), v.to_string())))
                    .collect()
            })
            .unwrap_or_default();
        let snapshot = support::post_process_snapshot(
            payload["snapshot"].as_str().unwrap_or_default(),
            SnapshotOptions {
                interactive: opt_bool(opts, "interactive"),
                compact: opt_bool(opts, "compact"),
                urls,
            },
            &hrefs,
        );
        if !opt_bool(opts, "diff") {
            return Ok(json!(snapshot));
        }
        let url = self
            .eval_main("tab.ariaSnapshot()", "location.href")
            .await?
            .as_str()
            .unwrap_or_default()
            .to_string();
        let key = json!({
            "selector": selector,
            "depth": opts.get("depth"),
            "boxes": opt_bool(opts, "boxes"),
            "interactive": opt_bool(opts, "interactive"),
            "compact": opt_bool(opts, "compact"),
            "urls": urls,
        })
        .to_string();
        Ok(support::diff_snapshot(
            &mut self.aria_baselines.lock(),
            &key,
            &url,
            &snapshot,
        ))
    }

    /// Capture one image clip as raw bytes.
    async fn capture(
        &self,
        format: &str,
        quality: Option<i64>,
        clip: Value,
        beyond_viewport: bool,
    ) -> Result<Vec<u8>, String> {
        let mut params = json!({ "format": format, "clip": clip, "captureBeyondViewport": beyond_viewport, "fromSurface": true });
        if format != "png" {
            params["quality"] = json!(quality.unwrap_or(80));
        }
        let shot = self.cdp("Page.captureScreenshot", params).await?;
        base64::engine::general_purpose::STANDARD
            .decode(shot["data"].as_str().unwrap_or_default())
            .map_err(|e| format!("invalid screenshot data: {e}"))
    }

    async fn screenshot(
        &self,
        ctx: &mut OpCtx,
        opts: &Map<String, Value>,
    ) -> Result<Value, String> {
        if opts.contains_key("ifChanged") || opts.contains_key("threshold") {
            return Err(not_ported("screenshot ifChanged/threshold"));
        }
        let requested_format = opt_str(opts, "format");
        let format = match requested_format.unwrap_or("png") {
            "png" => "png",
            "jpeg" | "jpg" => "jpeg",
            "webp" => "webp",
            other => return Err(format!("Unknown screenshot format {}", json!(other))),
        };
        let quality = opt_f64(opts, "quality").map(|q| q.clamp(0.0, 100.0) as i64);
        let mut legend = Vec::new();
        let mut annotation: Option<String> = None;
        if opt_bool(opts, "annotate") {
            let observation = self.kit("observe", json!([{}])).await?;
            let elements = observation["elements"]
                .as_array()
                .cloned()
                .unwrap_or_default();
            let ids: Vec<Value> = elements.iter().map(|e| e["id"].clone()).collect();
            let installed = self.kit("annotate", json!([ids])).await?;
            annotation = installed["token"].as_str().map(str::to_string);
            for b in installed["boxes"].as_array().into_iter().flatten() {
                let element = elements.iter().find(|e| e["id"] == b["id"]);
                let role = element
                    .and_then(|e| e["role"].as_str())
                    .unwrap_or("generic");
                let name = element.and_then(|e| e["name"].as_str());
                legend.push(match name {
                    Some(name) => format!("[{}] {role} {}", b["id"], json!(name)),
                    None => format!("[{}] {role}", b["id"]),
                });
            }
        }
        let result = self.screenshot_bytes(ctx, opts, format, quality).await;
        if let Some(token) = annotation {
            let _ = self.kit("removeOverlay", json!([token])).await;
        }
        let (full, model, model_mime, (width, height)) = result?;
        let save_full = requested_format.is_some();
        let (saved, saved_mime) = if save_full {
            (&full, format_mime(format))
        } else {
            (&model, model_mime)
        };
        let ext = match saved_mime {
            "image/jpeg" => "jpg",
            "image/webp" => "webp",
            _ => "png",
        };
        let dest = std::env::temp_dir().join(format!("juto-sshots-{}.{ext}", uuid::Uuid::now_v7()));
        std::fs::write(&dest, saved)
            .map_err(|e| format!("cannot save screenshot {}: {e}", dest.display()))?;
        ctx.screenshots.push(json!({
            "dest": dest,
            "mimeType": saved_mime,
            "bytes": saved.len(),
            "width": width,
            "height": height,
        }));
        if !opt_bool(opts, "silent") {
            let mut text = format!(
                "Screenshot: {width}x{height} {model_mime} sent to the model; {} saved to {} ({} bytes)",
                if save_full {
                    "full resolution"
                } else {
                    "same image"
                },
                dest.display(),
                saved.len()
            );
            if !legend.is_empty() {
                text.push_str("\nAnnotations:\n");
                text.push_str(&legend.join("\n"));
            }
            ctx.output.push(ContentBlock::Text { text });
            ctx.output.push(ContentBlock::Image {
                data: base64::engine::general_purpose::STANDARD.encode(&model),
                mime_type: model_mime.to_string(),
            });
        }
        Ok(json!(dest))
    }

    /// Full-resolution capture plus a model-sized image (≤1024px, ≤150 KB;
    /// `resizeImage` budget), resized by Chromium's clip scale.
    async fn screenshot_bytes(
        &self,
        ctx: &OpCtx,
        opts: &Map<String, Value>,
        format: &str,
        quality: Option<i64>,
    ) -> Result<(Vec<u8>, Vec<u8>, &'static str, (u64, u64)), String> {
        let geometry = self.kit("geometry", json!([])).await?;
        let dpr = geometry["dpr"].as_f64().unwrap_or(1.0).max(0.1);
        let (scroll_x, scroll_y) = (
            geometry["scrollX"].as_f64().unwrap_or(0.0),
            geometry["scrollY"].as_f64().unwrap_or(0.0),
        );
        let (clip_x, clip_y, clip_w, clip_h, beyond) =
            if let Some(selector) = opt_str(opts, "selector") {
                let spec = support::parse_selector(selector)?;
                self.target(
                    ctx,
                    &format!("tab.screenshot({})", describe(&spec)),
                    &spec,
                    "point",
                    None,
                )
                .await?;
                let rect = self.read(&spec, "box", Value::Null).await?;
                if rect.is_null() {
                    return Err("Screenshot selector did not resolve to a visible element".into());
                }
                (
                    rect["x"].as_f64().unwrap_or(0.0) + scroll_x,
                    rect["y"].as_f64().unwrap_or(0.0) + scroll_y,
                    rect["width"].as_f64().unwrap_or(1.0),
                    rect["height"].as_f64().unwrap_or(1.0),
                    true,
                )
            } else if opt_bool(opts, "fullPage") {
                (
                    0.0,
                    0.0,
                    geometry["scrollWidth"].as_f64().unwrap_or(1.0),
                    geometry["scrollHeight"].as_f64().unwrap_or(1.0),
                    true,
                )
            } else {
                (
                    scroll_x,
                    scroll_y,
                    geometry["innerWidth"].as_f64().unwrap_or(1.0),
                    geometry["innerHeight"].as_f64().unwrap_or(1.0),
                    false,
                )
            };
        // Let the page paint the state the caller just produced.
        let _ = self
            .eval_main(
                "tab.screenshot()",
                "new Promise(r => requestAnimationFrame(() => r(true)))",
            )
            .await;
        let clip = |scale: f64| json!({ "x": clip_x, "y": clip_y, "width": clip_w.max(1.0), "height": clip_h.max(1.0), "scale": scale });
        let full = self.capture(format, quality, clip(1.0), beyond).await?;
        let pixels_w = clip_w * dpr;
        let pixels_h = clip_h * dpr;
        let scale = (1024.0 / pixels_w).min(1024.0 / pixels_h).min(1.0);
        let size = (
            (pixels_w * scale).round().max(1.0) as u64,
            (pixels_h * scale).round().max(1.0) as u64,
        );
        let mut model = if scale >= 1.0 && format == "png" {
            full.clone()
        } else {
            self.capture("png", None, clip(scale), beyond).await?
        };
        let mut mime = "image/png";
        if model.len() > 150 * 1024 {
            model = self.capture("jpeg", Some(70), clip(scale), beyond).await?;
            mime = "image/jpeg";
        }
        Ok((full, model, mime, size))
    }

    async fn pdf(&self, ctx: &OpCtx, opts: &Map<String, Value>) -> Result<Value, String> {
        let (mut width, mut height) = match opt_str(opts, "format")
            .unwrap_or("letter")
            .to_ascii_lowercase()
            .as_str()
        {
            "letter" => (8.5, 11.0),
            "legal" => (8.5, 14.0),
            "tabloid" => (11.0, 17.0),
            "ledger" => (17.0, 11.0),
            "a0" => (33.1102, 46.811),
            "a1" => (23.3858, 33.1102),
            "a2" => (16.5354, 23.3858),
            "a3" => (11.6929, 16.5354),
            "a4" => (8.2677, 11.6929),
            "a5" => (5.8268, 8.2677),
            "a6" => (4.1339, 5.8268),
            other => return Err(format!("Unknown PDF format {}", json!(other))),
        };
        let landscape = opt_bool(opts, "landscape");
        if landscape {
            std::mem::swap(&mut width, &mut height);
        }
        let margin = arg_opts(&[opts.get("margin").cloned().unwrap_or(Value::Null)], 0);
        let mut params = json!({
            "landscape": landscape,
            "printBackground": opt_bool(opts, "printBackground"),
            "scale": opt_f64(opts, "scale").unwrap_or(1.0),
            "paperWidth": width,
            "paperHeight": height,
            "pageRanges": opt_str(opts, "pageRanges").unwrap_or(""),
        });
        for side in ["top", "bottom", "left", "right"] {
            if let Some(value) = margin.get(side) {
                params[format!("margin{}{}", side[..1].to_ascii_uppercase(), &side[1..])] =
                    json!(css_inches(value)?);
            }
        }
        let printed = self.cdp("Page.printToPDF", params).await.map_err(|e| {
            format!(
                "tab.pdf(): {e}{}",
                if self.browser.headed {
                    " (PDF printing requires a headless browser)"
                } else {
                    ""
                }
            )
        })?;
        let data = base64::engine::general_purpose::STANDARD
            .decode(printed["data"].as_str().unwrap_or_default())
            .map_err(|e| format!("invalid PDF data: {e}"))?;
        let dest = match opt_str(opts, "path") {
            Some(path) => super::resolve_path(&ctx.cwd, path),
            None => std::env::temp_dir().join(format!("juto-browser-{}.pdf", uuid::Uuid::now_v7())),
        };
        if let Some(parent) = dest.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        std::fs::write(&dest, data).map_err(|e| format!("cannot write {}: {e}", dest.display()))?;
        Ok(json!(dest))
    }

    /// Readable text/Markdown (`extractReadableFromHtml`): Readability first,
    /// then the selector/landmark fallback chain.
    async fn extract(&self, format: &str, opts: &Map<String, Value>) -> Result<Value, String> {
        if format != "markdown" && format != "text" {
            return Err(format!(
                "tab.extract format must be \"markdown\" or \"text\"; got {}",
                json!(format)
            ));
        }
        let selector = opt_str(opts, "selector");
        let payload = self
            .call_main(
                "tab.extract()",
                EXTRACT_SCRIPT,
                &[json!(selector), Value::Null],
            )
            .await?;
        let url = payload["url"].as_str().unwrap_or_default().to_string();
        if payload.is_null() {
            return Err(format!(
                "tab.extract({}) found no readable content on {}",
                json!(format),
                self.url()
            ));
        }
        let filter = opt_str(opts, "filter").map(str::to_string);
        let outline = opt_bool(opts, "outline");
        if payload["selected"] != true {
            let html = payload["html"].as_str().unwrap_or_default().to_string();
            let doc_url =
                (url.starts_with("http://") || url.starts_with("https://")).then(|| url.clone());
            let article = tokio::task::spawn_blocking(move || {
                dom_smoothie::Readability::new(html, doc_url.as_deref(), None)
                    .ok()?
                    .parse()
                    .ok()
                    .map(|a| (a.content.to_string(), a.text_content.to_string()))
            })
            .await
            .map_err(|e| e.to_string())?;
            if let Some((content, text_content)) = article {
                let text = if format == "text" && !content.is_empty() {
                    self.call_main(
                        "tab.extract()",
                        EXTRACT_SCRIPT,
                        &[Value::Null, json!(content)],
                    )
                    .await?
                    .as_str()
                    .unwrap_or_default()
                    .to_string()
                } else {
                    text_content
                };
                if let Some(result) =
                    readable_content(format, &text, &content, filter.as_deref(), outline)
                {
                    return Ok(json!(result));
                }
            }
        }
        for candidate in payload["candidates"].as_array().into_iter().flatten() {
            let html = candidate["html"].as_str().unwrap_or_default();
            let text = if format == "text" {
                candidate["text"].as_str()
            } else {
                candidate["raw"].as_str()
            }
            .unwrap_or_default();
            if html.is_empty() || text.trim().is_empty() {
                continue;
            }
            if let Some(result) = readable_content(format, text, html, filter.as_deref(), outline) {
                return Ok(json!(result));
            }
        }
        Err(format!(
            "tab.extract({}) found no readable content on {url}",
            json!(format)
        ))
    }

    async fn emulate(&self, opts: &Map<String, Value>) -> Result<Value, String> {
        if opts.contains_key("device") {
            return Err(not_ported("emulate device profiles"));
        }
        if opts.get("credentials").is_some_and(|v| !v.is_null()) {
            return Err(not_ported("emulate credentials"));
        }
        if let Some(viewport) = opts.get("viewport").and_then(Value::as_object) {
            let width = opt_f64(viewport, "width").ok_or("emulate viewport needs width")?;
            let height = opt_f64(viewport, "height").ok_or("emulate viewport needs height")?;
            self.set_viewport(width, height, opt_f64(viewport, "scale"))
                .await?;
        }
        if let Some(user_agent) = opts.get("userAgent") {
            self.set_user_agent(user_agent.as_str().unwrap_or(""))
                .await?;
        }
        let mut features = Vec::new();
        if let Some(scheme) = opt_str(opts, "colorScheme") {
            features.push(json!({ "name": "prefers-color-scheme", "value": if scheme == "no-preference" { "" } else { scheme } }));
        }
        if let Some(reduced) = opts.get("reducedMotion").and_then(Value::as_bool) {
            features.push(json!({ "name": "prefers-reduced-motion", "value": if reduced { "reduce" } else { "" } }));
        }
        if !features.is_empty() {
            self.cdp(
                "Emulation.setEmulatedMedia",
                json!({ "features": features }),
            )
            .await?;
        }
        if let Some(geo) = opts.get("geolocation") {
            match geo.as_object() {
                Some(geo) => {
                    let origin = self.url();
                    let _ = self
                        .browser
                        .cdp
                        .send(
                            None,
                            "Browser.grantPermissions",
                            json!({ "permissions": ["geolocation"], "origin": origin }),
                        )
                        .await;
                    self.cdp(
                        "Emulation.setGeolocationOverride",
                        json!({ "latitude": geo.get("latitude"), "longitude": geo.get("longitude"), "accuracy": geo.get("accuracy").cloned().unwrap_or(json!(1)) }),
                    )
                    .await?;
                }
                None => {
                    self.cdp("Emulation.clearGeolocationOverride", json!({}))
                        .await?;
                }
            }
        }
        if let Some(locale) = opts.get("locale") {
            self.cdp(
                "Emulation.setLocaleOverride",
                json!({ "locale": locale.as_str().unwrap_or("") }),
            )
            .await?;
        }
        if let Some(timezone) = opts.get("timezone") {
            self.cdp(
                "Emulation.setTimezoneOverride",
                json!({ "timezoneId": timezone.as_str().unwrap_or("") }),
            )
            .await?;
        }
        if let Some(headers) = opts.get("headers") {
            self.cdp(
                "Network.setExtraHTTPHeaders",
                json!({ "headers": if headers.is_null() { json!({}) } else { headers.clone() } }),
            )
            .await?;
        }
        if let Some(rate) = opts.get("cpuThrottling") {
            self.cdp(
                "Emulation.setCPUThrottlingRate",
                json!({ "rate": rate.as_f64().unwrap_or(1.0) }),
            )
            .await?;
        }
        if opts.contains_key("offline") || opts.contains_key("network") {
            let offline = opt_bool(opts, "offline");
            let (latency, download, upload) = match opts.get("network") {
                Some(Value::String(preset)) if preset == "slow3g" => (2000.0, 50_000.0, 50_000.0),
                Some(Value::String(preset)) if preset == "fast3g" => (562.5, 180_000.0, 84_375.0),
                Some(Value::Object(custom)) => (
                    opt_f64(custom, "latency").unwrap_or(0.0),
                    opt_f64(custom, "download").unwrap_or(-1.0),
                    opt_f64(custom, "upload").unwrap_or(-1.0),
                ),
                Some(Value::String(other)) => {
                    return Err(format!("Unknown network preset {}", json!(other)));
                }
                _ => (0.0, -1.0, -1.0),
            };
            self.cdp(
                "Network.emulateNetworkConditions",
                json!({ "offline": offline, "latency": latency, "downloadThroughput": download, "uploadThroughput": upload }),
            )
            .await?;
        }
        let mut emulation = self.emulation.lock();
        for (key, value) in opts {
            emulation.insert(key.clone(), value.clone());
        }
        Ok(Value::Object(emulation.clone()))
    }

    async fn set_cookies(&self, args: &[Value]) -> Result<(), String> {
        let current = self.url();
        let mut cookies = Vec::new();
        for value in args {
            collect_cookies(value, &mut cookies)?;
        }
        for mut cookie in cookies {
            let name = cookie
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string();
            if !cookie.contains_key("url") && !cookie.contains_key("domain") {
                if current.starts_with("http") {
                    cookie.insert("url".into(), json!(current));
                } else {
                    return Err(format!(
                        "tab.setCookies() needs a domain or url for cookie {}",
                        json!(name)
                    ));
                }
            }
            if cookie
                .get("expires")
                .and_then(Value::as_f64)
                .is_some_and(|e| e < 0.0)
            {
                cookie.remove("expires");
            }
            let set = self.cdp("Network.setCookie", Value::Object(cookie)).await?;
            if set["success"] == false {
                return Err(format!("tab.setCookies() rejected cookie {}", json!(name)));
            }
        }
        Ok(())
    }

    async fn storage(&self, method: &str, args: &[Value]) -> Result<Value, String> {
        let kind = arg_str(args, 0, &format!("tab.{method}"))?;
        let area = match kind {
            "local" | "localStorage" => "localStorage",
            "session" | "sessionStorage" => "sessionStorage",
            other => {
                return Err(format!(
                    "Storage kind must be \"local\" or \"session\"; got {}",
                    json!(other)
                ));
            }
        };
        match method {
            "storage" => {
                let key = opt_str(&arg_opts(args, 1), "key").map(str::to_string);
                self.call_main(
                    "tab.storage()",
                    &format!("(key) => {{ const s = {area}; if (key != null) return s.getItem(key); const out = {{}}; for (let i = 0; i < s.length; i++) {{ const k = s.key(i); out[k] = s.getItem(k); }} return out; }}"),
                    &[json!(key)],
                )
                .await
            }
            "setStorage" => {
                let entries: Vec<(String, String)> = match arg(args, 1) {
                    Some(Value::String(key)) => vec![(key.clone(), value_text(arg(args, 2)))],
                    Some(Value::Object(map)) => map
                        .iter()
                        .map(|(k, v)| (k.clone(), value_text(Some(v))))
                        .collect(),
                    _ => return Err(
                        "tab.setStorage(kind, key, value) or tab.setStorage(kind, {key: value})"
                            .into(),
                    ),
                };
                self.call_main(
                    "tab.setStorage()",
                    &format!(
                        "(entries) => {{ for (const [k, v] of entries) {area}.setItem(k, v); }}"
                    ),
                    &[json!(entries)],
                )
                .await?;
                Ok(Value::Null)
            }
            _ => {
                self.eval_main("tab.clearStorage()", &format!("{area}.clear()"))
                    .await?;
                Ok(Value::Null)
            }
        }
    }

    async fn ensure_downloads(&self) -> Result<(), String> {
        if self.browser.downloads_dir().is_none() {
            let dir = std::env::temp_dir().join(format!("juto-downloads-{}", uuid::Uuid::now_v7()));
            self.browser.enable_downloads(&dir).await?;
        }
        Ok(())
    }

    fn capture_log(&self, console: bool, opts: &Map<String, Value>) -> Value {
        let since = opt_f64(opts, "since").unwrap_or(0.0).max(0.0) as u64;
        let limit = opt_f64(opts, "limit").map_or(500, |l| l.max(0.0) as usize);
        let level = opt_str(opts, "level");
        let mut state = self.state.lock();
        let entries: Vec<Value> = state
            .console
            .iter()
            .filter(|e| e["seq"].as_u64().unwrap_or(0) > since)
            .filter(|e| {
                if console {
                    e["type"] == "console" && level.is_none_or(|l| e["level"] == l)
                } else {
                    e["type"] != "console"
                }
            })
            .take(limit)
            .cloned()
            .collect();
        let next_seq = entries
            .last()
            .and_then(|e| e["seq"].as_u64())
            .unwrap_or(state.console_seq);
        let result =
            json!({ "entries": entries, "nextSeq": next_seq, "dropped": state.console_dropped });
        if opt_bool(opts, "clear") {
            state.console.clear();
            state.console_dropped = 0;
        }
        result
    }

    fn requests(&self, opts: &Map<String, Value>) -> Value {
        let filter = opt_str(opts, "filter");
        let since = opt_f64(opts, "since").unwrap_or(0.0);
        let limit = opt_f64(opts, "limit").map_or(usize::MAX, |l| l.max(0.0) as usize);
        let list = |key: &str| -> Option<Vec<String>> {
            match opts.get(key) {
                Some(Value::String(s)) => Some(vec![s.to_ascii_lowercase()]),
                Some(Value::Array(items)) => Some(
                    items
                        .iter()
                        .filter_map(Value::as_str)
                        .map(str::to_ascii_lowercase)
                        .collect(),
                ),
                _ => None,
            }
        };
        let types = list("type");
        let methods = list("method");
        let status = opts.get("status").cloned();
        let mut state = self.state.lock();
        let matched: Vec<Value> = state
            .requests
            .iter()
            .filter(|r| filter.is_none_or(|f| r.url.contains(f)))
            .filter(|r| r.ts >= since)
            .filter(|r| {
                types
                    .as_ref()
                    .is_none_or(|t| t.contains(&r.resource_type.to_ascii_lowercase()))
            })
            .filter(|r| {
                methods
                    .as_ref()
                    .is_none_or(|m| m.contains(&r.method.to_ascii_lowercase()))
            })
            .filter(|r| status_matches(status.as_ref(), r.status))
            .map(RequestRecord::to_json)
            .collect();
        let skip = matched.len().saturating_sub(limit);
        let out = Value::Array(matched.into_iter().skip(skip).collect());
        if opt_bool(opts, "clear") {
            state.requests.clear();
        }
        out
    }

    fn dialog_state(&self) -> Value {
        match &self.state.lock().dialog {
            None => json!({ "open": false }),
            Some(dialog) => {
                let mut out =
                    json!({ "open": true, "type": dialog.kind, "message": dialog.message });
                if let Some(default) = &dialog.default_value {
                    out["defaultValue"] = json!(default);
                }
                out
            }
        }
    }

    async fn handle_dialog(&self, accept: bool, text: Option<&str>) -> Result<(), String> {
        if self.state.lock().dialog.is_none() {
            return Err("tab.handleDialog() found no pending confirm or prompt".into());
        }
        let mut params = json!({ "accept": accept });
        if let Some(text) = text {
            params["promptText"] = json!(text);
        }
        self.cdp("Page.handleJavaScriptDialog", params).await?;
        self.state.lock().dialog = None;
        Ok(())
    }
}

/// Input state stores `&'static str` buttons; map dynamic names onto statics.
fn leak_button(button: &str) -> &'static str {
    match button {
        "right" => "right",
        "middle" => "middle",
        "back" => "back",
        "forward" => "forward",
        _ => "left",
    }
}

fn value_text(value: Option<&Value>) -> String {
    match value {
        None | Some(Value::Null) => String::new(),
        Some(Value::String(s)) => s.clone(),
        Some(other) => other.to_string(),
    }
}

/// Main-world expression locating the element marked with `token` (through
/// open shadow roots); `file_control` resolves a `<label>` to its control.
fn find_marked_expression(token: &str, file_control: bool) -> String {
    let selector = format!("[{HANDLE_ATTR}~={}]", json!(token));
    format!(
        "(() => {{ const sel = {}; const find = root => {{ const direct = root.querySelector(sel); if (direct) return direct; for (const node of root.querySelectorAll(\"*\")) {{ if (node.shadowRoot) {{ const found = find(node.shadowRoot); if (found) return found; }} }} return null; }}; const el = find(document); {} }})()",
        json!(selector),
        if file_control {
            "return el && el.localName === \"label\" && el.control ? el.control : el;"
        } else {
            "return el;"
        }
    )
}

fn mime_for(path: &Path) -> &'static str {
    match path
        .extension()
        .map(|e| e.to_string_lossy().to_ascii_lowercase())
        .as_deref()
    {
        Some("png") => "image/png",
        Some("jpg" | "jpeg") => "image/jpeg",
        Some("gif") => "image/gif",
        Some("webp") => "image/webp",
        Some("svg") => "image/svg+xml",
        Some("pdf") => "application/pdf",
        Some("txt") => "text/plain",
        Some("html" | "htm") => "text/html",
        Some("json") => "application/json",
        Some("csv") => "text/csv",
        _ => "application/octet-stream",
    }
}

fn format_mime(format: &str) -> &'static str {
    match format {
        "jpeg" => "image/jpeg",
        "webp" => "image/webp",
        _ => "image/png",
    }
}

/// CSS length (`"1cm"`, `"10px"`, number px) in inches.
fn css_inches(value: &Value) -> Result<f64, String> {
    if let Some(px) = value.as_f64() {
        return Ok(px / 96.0);
    }
    let raw = value.as_str().unwrap_or("").trim().to_ascii_lowercase();
    let split = raw
        .find(|c: char| c.is_ascii_alphabetic())
        .unwrap_or(raw.len());
    let (number, unit) = raw.split_at(split);
    let number: f64 = number
        .trim()
        .parse()
        .map_err(|_| format!("invalid PDF margin {}", json!(raw)))?;
    Ok(match unit {
        "" | "px" => number / 96.0,
        "in" => number,
        "cm" => number / 2.54,
        "mm" => number / 25.4,
        _ => return Err(format!("invalid PDF margin unit {}", json!(unit))),
    })
}

/// `markdownContent` + `toReadableResult` content selection.
fn readable_content(
    format: &str,
    text: &str,
    html: &str,
    filter: Option<&str>,
    outline: bool,
) -> Option<String> {
    let text = if format == "text" {
        text.trim_end().to_string()
    } else {
        text.trim().to_string()
    };
    if format == "text" && filter.is_none() && !outline {
        return (!text.is_empty()).then_some(text);
    }
    let cleaned = strip_tags(html, "script");
    let cleaned = strip_tags(&cleaned, "style");
    let mut markdown = htmd::convert(&cleaned)
        .ok()
        .map(|m| m.trim().to_string())
        .filter(|m| !m.is_empty())
        .unwrap_or(text);
    if markdown.is_empty() {
        return None;
    }
    if let Some(filter) = filter {
        markdown = support::filter_markdown_sections(&markdown, filter);
        if markdown.is_empty() {
            return None;
        }
    }
    if outline {
        markdown = support::markdown_outline(&markdown).trim().to_string();
        if markdown.is_empty() {
            return None;
        }
    }
    Some(if outline || format == "markdown" {
        markdown
    } else {
        support::markdown_to_plain_text(&markdown)
    })
}

fn strip_tags(html: &str, tag: &str) -> String {
    let re = regex::Regex::new(&format!(r"(?is)<{tag}[\s>].*?</{tag}>")).expect("static regex");
    re.replace_all(html, "").into_owned()
}

/// Cookie arguments: objects, arrays, JSON array strings, or raw `Cookie:` headers.
fn collect_cookies(value: &Value, out: &mut Vec<Map<String, Value>>) -> Result<(), String> {
    match value {
        Value::Array(items) => {
            for item in items {
                collect_cookies(item, out)?;
            }
        }
        Value::Object(map) => {
            if !map.contains_key("name") {
                return Err("tab.setCookies() cookie objects need name and value".into());
            }
            let mut cookie = Map::new();
            for key in [
                "name", "value", "url", "domain", "path", "secure", "httpOnly", "sameSite",
                "expires",
            ] {
                if let Some(v) = map.get(key).filter(|v| !v.is_null()) {
                    cookie.insert(key.to_string(), v.clone());
                }
            }
            cookie
                .entry("value".to_string())
                .or_insert_with(|| json!(""));
            out.push(cookie);
        }
        Value::String(text) => {
            let trimmed = text.trim();
            if trimmed.starts_with('[') || trimmed.starts_with('{') {
                let parsed: Value = serde_json::from_str(trimmed)
                    .map_err(|e| format!("tab.setCookies() invalid JSON: {e}"))?;
                return collect_cookies(&parsed, out);
            }
            let header = trimmed
                .strip_prefix("Cookie:")
                .or_else(|| trimmed.strip_prefix("cookie:"))
                .unwrap_or(trimmed);
            for pair in header.split(';') {
                if let Some((name, value)) = pair.split_once('=') {
                    let mut cookie = Map::new();
                    cookie.insert("name".into(), json!(name.trim()));
                    cookie.insert("value".into(), json!(value.trim()));
                    out.push(cookie);
                }
            }
        }
        _ => {
            return Err(
                "tab.setCookies() accepts cookie objects, arrays, or Cookie header strings".into(),
            );
        }
    }
    Ok(())
}

fn status_matches(filter: Option<&Value>, status: Option<i64>) -> bool {
    let Some(filter) = filter else {
        return true;
    };
    let Some(status) = status else {
        return false;
    };
    match filter {
        Value::Number(n) => n.as_i64() == Some(status),
        Value::String(s) => {
            let s = s.trim().to_ascii_lowercase();
            if let Some((lo, hi)) = s.split_once('-') {
                match (lo.trim().parse::<i64>(), hi.trim().parse::<i64>()) {
                    (Ok(lo), Ok(hi)) => (lo..=hi).contains(&status),
                    _ => false,
                }
            } else if s.len() == 3 && s.ends_with("xx") {
                s[..1]
                    .parse::<i64>()
                    .is_ok_and(|class| status / 100 == class)
            } else {
                s.parse::<i64>().is_ok_and(|exact| exact == status)
            }
        }
        _ => true,
    }
}

fn download_json(download: &Download, dir: Option<&Path>) -> Value {
    json!({
        "path": dir.map(|d| d.join(&download.suggested)).unwrap_or_else(|| PathBuf::from(&download.suggested)),
        "suggestedFilename": download.suggested,
        "url": download.url,
        "bytes": download.bytes,
    })
}

pub(crate) fn not_ported(method: &str) -> String {
    format!(
        "tab.{method} is not available in Juto's browser tool. Supported helpers are listed in the tool description."
    )
}

// ---------------------------------------------------------------------------
// Event pump
// ---------------------------------------------------------------------------

fn console_text(params: &Value) -> String {
    params["args"]
        .as_array()
        .into_iter()
        .flatten()
        .map(|arg| match arg.get("value") {
            Some(Value::String(s)) => s.clone(),
            Some(value) if !value.is_null() => value.to_string(),
            _ => arg
                .get("description")
                .and_then(Value::as_str)
                .or_else(|| arg.get("unserializableValue").and_then(Value::as_str))
                .or_else(|| arg.get("type").and_then(Value::as_str))
                .unwrap_or("")
                .to_string(),
        })
        .collect::<Vec<_>>()
        .join(" ")
}

async fn pump_events(
    cdp: Arc<Cdp>,
    session: String,
    target_id: String,
    state: Arc<Mutex<TabState>>,
    mut events: tokio::sync::broadcast::Receiver<CdpEvent>,
) {
    loop {
        let event = match events.recv().await {
            Ok(event) => event,
            Err(RecvError::Lagged(_)) => continue,
            Err(RecvError::Closed) => break,
        };
        let ours = event.session.as_deref() == Some(session.as_str());
        let params = &event.params;
        if event.session.is_none() {
            match event.method.as_str() {
                "Target.detachedFromTarget" if params["sessionId"] == session.as_str() => {
                    state.lock().detached = true;
                    break;
                }
                "Target.targetDestroyed" if params["targetId"] == target_id.as_str() => {
                    state.lock().detached = true;
                    break;
                }
                "Browser.downloadWillBegin" if params["frameId"] == target_id.as_str() => {
                    state.lock().downloads.push(Download {
                        guid: params["guid"].as_str().unwrap_or_default().to_string(),
                        url: params["url"].as_str().unwrap_or_default().to_string(),
                        suggested: params["suggestedFilename"]
                            .as_str()
                            .unwrap_or_default()
                            .to_string(),
                        bytes: 0.0,
                        state: "inProgress".into(),
                        claimed: false,
                    });
                }
                "Browser.downloadProgress" => {
                    let mut state = state.lock();
                    if let Some(download) = state
                        .downloads
                        .iter_mut()
                        .find(|d| params["guid"] == d.guid.as_str())
                    {
                        download.bytes = params["receivedBytes"].as_f64().unwrap_or(download.bytes);
                        if let Some(s) = params["state"].as_str() {
                            download.state = s.to_string();
                        }
                    }
                }
                _ => {}
            }
            continue;
        }
        if !ours {
            continue;
        }
        match event.method.as_str() {
            "Page.lifecycleEvent" if params["frameId"] == target_id.as_str() => {
                let mut state = state.lock();
                let loader = params["loaderId"].as_str().unwrap_or_default();
                let name = params["name"].as_str().unwrap_or_default();
                if name == "init" {
                    state.main_loader = loader.to_string();
                    state.lifecycle.clear();
                } else if state.main_loader == loader {
                    state.lifecycle.insert(name.to_string());
                }
            }
            "Page.frameNavigated" if params.pointer("/frame/parentId").is_none() => {
                let mut state = state.lock();
                if let Some(url) = params.pointer("/frame/url").and_then(Value::as_str) {
                    state.url = url.to_string();
                }
                if let Some(loader) = params.pointer("/frame/loaderId").and_then(Value::as_str)
                    && state.main_loader != loader
                {
                    state.main_loader = loader.to_string();
                    state.lifecycle.clear();
                }
                // A back/forward-cache restore revives a loaded document
                // without replaying lifecycle events.
                if params["type"] == "BackForwardCacheRestore" {
                    for name in [
                        "DOMContentLoaded",
                        "load",
                        "networkAlmostIdle",
                        "networkIdle",
                    ] {
                        state.lifecycle.insert(name.to_string());
                    }
                }
                state.nav_seq += 1;
                state.kit_context = None;
            }
            "Page.navigatedWithinDocument" if params["frameId"] == target_id.as_str() => {
                let mut state = state.lock();
                if let Some(url) = params["url"].as_str() {
                    state.url = url.to_string();
                }
                state.nav_seq += 1;
                state.same_document_seq = state.nav_seq;
            }
            "Runtime.executionContextsCleared" => {
                state.lock().kit_context = None;
            }
            "Runtime.consoleAPICalled" => {
                let level = match params["type"].as_str().unwrap_or("log") {
                    "warning" => "warn",
                    "error" | "assert" => "error",
                    "debug" => "debug",
                    "info" => "info",
                    _ => "log",
                };
                let frame = params.pointer("/stackTrace/callFrames/0");
                let location = frame.map(|f| {
                    format!(
                        "{}:{}:{}",
                        f["url"].as_str().unwrap_or(""),
                        f["lineNumber"].as_i64().unwrap_or(0) + 1,
                        f["columnNumber"].as_i64().unwrap_or(0) + 1
                    )
                });
                let args: Vec<Value> = params["args"]
                    .as_array()
                    .into_iter()
                    .flatten()
                    .map(|a| {
                        a.get("value").cloned().unwrap_or_else(|| {
                            json!(a.get("description").and_then(Value::as_str).unwrap_or(""))
                        })
                    })
                    .collect();
                let mut entry = json!({ "type": "console", "level": level, "text": console_text(params), "args": args });
                if let Some(location) = location {
                    entry["location"] = json!(location);
                }
                state.lock().push_console(entry);
            }
            "Runtime.exceptionThrown" => {
                let details = &params["exceptionDetails"];
                let text = exception_text(details);
                let mut entry = json!({ "type": "pageerror", "level": "error", "text": kit_error_text(&text), "stack": text });
                if let Some(url) = details["url"].as_str() {
                    entry["location"] = json!(format!(
                        "{url}:{}:{}",
                        details["lineNumber"].as_i64().unwrap_or(0) + 1,
                        details["columnNumber"].as_i64().unwrap_or(0) + 1
                    ));
                }
                state.lock().push_console(entry);
            }
            "Network.requestWillBeSent" => {
                let mut state = state.lock();
                let id = params["requestId"].as_str().unwrap_or_default().to_string();
                state.request_seq += 1;
                let seq = state.request_seq;
                if state.requests.len() == CAPTURE_LIMIT {
                    state.requests.pop_front();
                }
                let started = params["timestamp"].as_f64().unwrap_or(0.0);
                state.requests.push_back(RequestRecord {
                    id,
                    seq,
                    ts: params["wallTime"]
                        .as_f64()
                        .map_or_else(now_ms, |w| w * 1000.0),
                    method: params
                        .pointer("/request/method")
                        .and_then(Value::as_str)
                        .unwrap_or("GET")
                        .to_string(),
                    url: params
                        .pointer("/request/url")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_string(),
                    resource_type: params["type"]
                        .as_str()
                        .unwrap_or("Other")
                        .to_ascii_lowercase(),
                    status: None,
                    failure: None,
                    started,
                    duration_ms: None,
                    request_headers: params
                        .pointer("/request/headers")
                        .cloned()
                        .unwrap_or(json!({})),
                    response_headers: None,
                    encoded_bytes: None,
                });
            }
            "Network.responseReceived" => {
                let mut state = state.lock();
                if let Some(record) =
                    state.request_mut(params["requestId"].as_str().unwrap_or_default())
                {
                    record.status = params
                        .pointer("/response/status")
                        .and_then(Value::as_f64)
                        .map(|s| s as i64);
                    record.response_headers = params.pointer("/response/headers").cloned();
                }
            }
            "Network.loadingFinished" => {
                let mut state = state.lock();
                if let Some(record) =
                    state.request_mut(params["requestId"].as_str().unwrap_or_default())
                {
                    record.encoded_bytes = params["encodedDataLength"].as_f64();
                    record.duration_ms = params["timestamp"]
                        .as_f64()
                        .map(|t| (t - record.started) * 1000.0);
                }
            }
            "Network.loadingFailed" => {
                let mut state = state.lock();
                let failure = params["blockedReason"]
                    .as_str()
                    .map(|r| format!("blocked: {r}"))
                    .or_else(|| params["errorText"].as_str().map(str::to_string))
                    .unwrap_or_else(|| "failed".into());
                let mut url = None;
                if let Some(record) =
                    state.request_mut(params["requestId"].as_str().unwrap_or_default())
                {
                    record.failure = Some(failure.clone());
                    record.duration_ms = params["timestamp"]
                        .as_f64()
                        .map(|t| (t - record.started) * 1000.0);
                    url = Some(record.url.clone());
                }
                if params["canceled"] != true {
                    let mut entry =
                        json!({ "type": "requestfailed", "level": "error", "text": failure });
                    if let Some(url) = url {
                        entry["location"] = json!(url);
                    }
                    state.push_console(entry);
                }
            }
            "Page.javascriptDialogOpening" => {
                let kind = params["type"].as_str().unwrap_or("alert").to_string();
                let policy = state.lock().dialog_policy;
                let auto = match policy {
                    Some(policy) => Some(policy == DialogPolicy::Accept),
                    None if kind == "alert" || kind == "beforeunload" => Some(true),
                    None => None,
                };
                match auto {
                    Some(accept) => {
                        let cdp = cdp.clone();
                        let session = session.clone();
                        tokio::spawn(async move {
                            let _ = cdp
                                .send(
                                    Some(&session),
                                    "Page.handleJavaScriptDialog",
                                    json!({ "accept": accept }),
                                )
                                .await;
                        });
                    }
                    None => {
                        state.lock().dialog = Some(PendingDialog {
                            kind,
                            message: params["message"].as_str().unwrap_or_default().to_string(),
                            default_value: params["defaultPrompt"].as_str().map(str::to_string),
                        });
                    }
                }
            }
            "Page.javascriptDialogClosed" => {
                state.lock().dialog = None;
            }
            "Fetch.requestPaused" => {
                let allowed = state.lock().allowed_domains.clone();
                let url = params
                    .pointer("/request/url")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                let permit =
                    support::url_host(url).is_none_or(|host| support::host_allowed(host, &allowed));
                let request_id = params["requestId"].clone();
                let cdp = cdp.clone();
                let session = session.clone();
                tokio::spawn(async move {
                    let _ = if permit {
                        cdp.send(
                            Some(&session),
                            "Fetch.continueRequest",
                            json!({ "requestId": request_id }),
                        )
                        .await
                    } else {
                        cdp.send(
                            Some(&session),
                            "Fetch.failRequest",
                            json!({ "requestId": request_id, "errorReason": "BlockedByClient" }),
                        )
                        .await
                    };
                });
            }
            "Inspector.detached" => {
                state.lock().detached = true;
                break;
            }
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kit_source_assembles_vendored_assets() {
        assert!(KIT_SOURCE.contains("__jutoBrowserKit"));
        assert!(KIT_SOURCE.contains("resolveAriaRef"));
        assert!(!KIT_SOURCE.contains("webkit.messageHandlers"));
    }

    #[test]
    fn readable_content_filters_and_formats() {
        let html = "<h1>Guide</h1><p>intro</p><h2>Install</h2><p>run <code>x</code></p><script>bad()</script>";
        let md = readable_content("markdown", "Guide intro", html, None, false).unwrap();
        assert!(md.starts_with("# Guide"));
        assert!(!md.contains("bad()"));
        let filtered = readable_content("text", "", html, Some("install"), false).unwrap();
        assert_eq!(filtered, "Install\n\nrun x");
        let outline = readable_content("markdown", "", html, None, true).unwrap();
        assert_eq!(outline, "# Guide\n\n## Install".replace("\n\n", "\n"));
        assert_eq!(
            readable_content("text", "plain  \n", html, None, false).as_deref(),
            Some("plain")
        );
    }

    #[test]
    fn cookie_arguments_normalize() {
        let mut out = Vec::new();
        collect_cookies(&json!("Cookie: a=1; b=two"), &mut out).unwrap();
        collect_cookies(
            &json!([{ "name": "c", "value": "3", "domain": "x.test" }]),
            &mut out,
        )
        .unwrap();
        collect_cookies(&json!("[{\"name\":\"d\"}]"), &mut out).unwrap();
        let names: Vec<_> = out.iter().map(|c| c["name"].as_str().unwrap()).collect();
        assert_eq!(names, ["a", "b", "c", "d"]);
        assert_eq!(out[3]["value"], "");
        assert!(collect_cookies(&json!(5), &mut out).is_err());
    }

    #[test]
    fn status_filters_and_margins() {
        assert!(status_matches(Some(&json!("2xx")), Some(204)));
        assert!(!status_matches(Some(&json!("2xx")), Some(404)));
        assert!(status_matches(Some(&json!("400-499")), Some(404)));
        assert!(status_matches(Some(&json!(301)), Some(301)));
        assert!(!status_matches(Some(&json!(301)), None));
        assert!((css_inches(&json!("2.54cm")).unwrap() - 1.0).abs() < 1e-9);
        assert!((css_inches(&json!(96)).unwrap() - 1.0).abs() < 1e-9);
        assert!(css_inches(&json!("1em")).is_err());
    }
}
