//! `browser` tool: drive real Chromium tabs over the DevTools protocol.
//!
//! Behavioral reference: oh-my-pi `packages/coding-agent/src/tools/browser.ts`
//! and `tools/browser/` at 579da1d661c5cb8d43bc2ddd429ab72e67165ad8 (MIT, see
//! licenses/OMP-MIT.txt). The upstream tool runs agent JavaScript in a
//! Puppeteer worker; Juto has no JavaScript host, so `run` evaluates code in
//! the page's main world and tab helpers (`call`) are implemented in Rust over
//! CDP plus OMP's vendored in-page kit. Backends: managed Chromium (headless
//! or headed), `app.path` spawns, and `app.cdp_url` attach. The relay, Tern,
//! and cmux backends are not ported.

mod cdp;
mod launch;
mod support;
mod tab;

use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    sync::Arc,
    time::Duration,
};

use async_trait::async_trait;
use juto_agent::{AgentError, Tool, ToolContext, ToolOutput, ToolRegistry, ToolTier};
use juto_ai::ContentBlock;
use serde::Deserialize;
use serde_json::{Value, json};

use cdp::Cdp;
use tab::{Browser, DialogPolicy, OpCtx, Tab};

const DEFAULT_TAB_NAME: &str = "main";
/// Timeout bounds in seconds (`TOOL_TIMEOUTS.browser`).
const DEFAULT_TIMEOUT: f64 = 30.0;
const MIN_TIMEOUT: f64 = 1.0;
const MAX_TIMEOUT: f64 = 300.0;
/// Inline text cap for run/call output (DEFAULT_MAX_BYTES upstream).
const OUTPUT_MAX_BYTES: usize = 50 * 1024;
/// Managed Chromium viewport (`launch.ts` DEFAULT_VIEWPORT).
const DEFAULT_VIEWPORT: (f64, f64, f64) = (1365.0, 768.0, 1.25);

/// Tab helpers whose resolved value is returned directly (`TAB_VALUE_METHODS`).
const TAB_VALUE_METHODS: &[&str] = &[
    "url",
    "title",
    "goto",
    "back",
    "forward",
    "reload",
    "pushState",
    "frames",
    "dialog",
    "handleDialog",
    "setDialogs",
    "observe",
    "ariaSnapshot",
    "a11y",
    "webmcpList",
    "webmcpInvoke",
    "webmcpEvents",
    "screenshot",
    "diffScreenshot",
    "pdf",
    "extract",
    "click",
    "dblclick",
    "hover",
    "focus",
    "check",
    "uncheck",
    "keyDown",
    "keyUp",
    "mouseMove",
    "mouseDown",
    "mouseUp",
    "clickAt",
    "wheel",
    "highlight",
    "type",
    "fill",
    "press",
    "scroll",
    "drag",
    "scrollIntoView",
    "select",
    "uploadFile",
    "waitForUrl",
    "text",
    "html",
    "value",
    "attr",
    "count",
    "box",
    "styles",
    "isVisible",
    "isEnabled",
    "isChecked",
    "waitForText",
    "evaluate",
    "emulate",
    "devices",
    "clipboardRead",
    "clipboardWrite",
    "clipboardCopy",
    "clipboardPaste",
    "cookies",
    "setCookies",
    "clearCookies",
    "storage",
    "setStorage",
    "clearStorage",
    "saveState",
    "loadState",
    "addInitScript",
    "removeInitScript",
    "initScripts",
    "waitForDownload",
    "downloads",
    "console",
    "errors",
    "clearConsole",
    "traceStart",
    "traceStop",
    "profileStart",
    "profileStop",
    "metrics",
    "route",
    "unroute",
    "routes",
    "requests",
    "request",
    "clearRequests",
    "harStart",
    "harStop",
    "allowedDomains",
    "vitals",
    "reactEnable",
    "reactTree",
    "reactInspect",
    "reactRenders",
    "reactSuspense",
    "recordStart",
    "recordStop",
    "recordRestart",
    "recording",
];
/// Helpers whose handle-or-null result becomes a boolean (`TAB_PRESENCE_METHODS`).
const TAB_PRESENCE_METHODS: &[&str] = &["waitFor", "waitForSelector"];
/// Helpers returning a handle for one chained call (`TAB_HANDLE_METHODS`).
const TAB_HANDLE_METHODS: &[&str] = &["id", "ref", "frame"];

/// Resolve `raw` against `base`; absolute paths are used verbatim.
fn resolve_path(base: &Path, raw: &str) -> PathBuf {
    let path = Path::new(raw);
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        base.join(raw)
    }
}

// ---------------------------------------------------------------------------
// Parameters
// ---------------------------------------------------------------------------

#[derive(Debug, Default, Deserialize)]
struct AppParams {
    path: Option<String>,
    cdp_url: Option<String>,
    relay: Option<bool>,
    tern: Option<bool>,
    #[serde(default)]
    args: Vec<String>,
    target: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ViewportParams {
    width: f64,
    height: f64,
    scale: Option<f64>,
}

#[derive(Debug, Deserialize)]
struct CallStep {
    method: String,
    #[serde(default)]
    args: Vec<Value>,
}

#[derive(Debug, Deserialize)]
struct BrowserParams {
    action: String,
    name: Option<String>,
    url: Option<String>,
    app: Option<AppParams>,
    viewport: Option<ViewportParams>,
    wait_until: Option<String>,
    dialogs: Option<String>,
    allowed_domains: Option<Vec<String>>,
    init_scripts: Option<Vec<String>>,
    downloads: Option<String>,
    user_agent: Option<String>,
    ignore_https_errors: Option<bool>,
    allow_file_access: Option<bool>,
    headed: Option<bool>,
    code: Option<String>,
    #[serde(rename = "fn")]
    function: Option<String>,
    #[serde(default)]
    args: Vec<Value>,
    #[serde(default)]
    chain: Vec<CallStep>,
    timeout: Option<f64>,
    #[serde(default)]
    all: bool,
    #[serde(default)]
    kill: bool,
}

// ---------------------------------------------------------------------------
// Manager
// ---------------------------------------------------------------------------

/// Which browser an open drives.
#[derive(Debug, Clone, PartialEq, Eq)]
enum BrowserKind {
    Managed {
        headed: bool,
        allow_file_access: bool,
    },
    Spawned {
        path: String,
        args: Vec<String>,
    },
    Connected {
        cdp_url: String,
    },
}

impl BrowserKind {
    fn key(&self) -> String {
        match self {
            Self::Managed {
                headed,
                allow_file_access,
            } => format!("managed:{headed}:{allow_file_access}"),
            Self::Spawned { path, args } => format!("spawned:{path}:{}", args.join("\u{1f}")),
            Self::Connected { cdp_url } => format!("connected:{cdp_url}"),
        }
    }

    fn describe(&self) -> String {
        match self {
            Self::Managed { headed: false, .. } => "headless Chromium".into(),
            Self::Managed { headed: true, .. } => "headed Chromium".into(),
            Self::Spawned { path, .. } => format!("spawned app {path}"),
            Self::Connected { cdp_url } => format!("CDP endpoint {cdp_url}"),
        }
    }
}

/// Browsers and named tabs shared by every session of one runtime; tabs
/// outlive individual runs so multi-step flows keep their page state.
pub struct BrowserManager {
    data_dir: PathBuf,
    browsers: tokio::sync::Mutex<HashMap<String, Arc<Browser>>>,
    tabs: tokio::sync::Mutex<HashMap<String, (Arc<Tab>, BrowserKind)>>,
}

impl BrowserManager {
    pub fn new(data_dir: PathBuf) -> Self {
        Self {
            data_dir,
            browsers: tokio::sync::Mutex::new(HashMap::new()),
            tabs: tokio::sync::Mutex::new(HashMap::new()),
        }
    }

    async fn acquire_browser(
        &self,
        kind: &BrowserKind,
        cwd: &Path,
    ) -> Result<Arc<Browser>, String> {
        let key = kind.key();
        let mut browsers = self.browsers.lock().await;
        if let Some(browser) = browsers.get(&key) {
            if !browser.cdp.closed().is_cancelled() {
                return Ok(browser.clone());
            }
            browsers.remove(&key);
        }
        let browser = match kind {
            BrowserKind::Managed {
                headed,
                allow_file_access,
            } => {
                let executable = launch::chromium_executable()?;
                let (args, profile) = launch::managed_args(!headed, *allow_file_access)?;
                let (spawned, ws) = launch::spawn(&executable, &args, Some(profile)).await?;
                let cdp = Cdp::connect(&ws).await?;
                Browser::new(key.clone(), "headless", *headed, cdp, Some(spawned))
            }
            BrowserKind::Spawned { path, args } => {
                let executable = resolve_path(cwd, path);
                let args = launch::app_args(
                    &executable,
                    args,
                    cwd,
                    &self.data_dir.join("browser-profiles"),
                );
                let (spawned, ws) = launch::spawn(&executable, &args, None).await?;
                let cdp = Cdp::connect(&ws).await?;
                Browser::new(key.clone(), "spawned", true, cdp, Some(spawned))
            }
            BrowserKind::Connected { cdp_url } => {
                let ws = launch::resolve_cdp_url(cdp_url).await?;
                let cdp = Cdp::connect(&ws).await?;
                Browser::new(key.clone(), "connected", true, cdp, None)
            }
        };
        let browser = Arc::new(browser);
        browsers.insert(key, browser.clone());
        Ok(browser)
    }

    /// Close or detach `tab`; shut its browser down when nothing else uses it
    /// (managed Chromium always, spawned apps only with `kill`).
    async fn release(&self, tab: &Tab, kind: &BrowserKind, kill: bool) {
        let cdp = &tab.browser.cdp;
        if tab.owned {
            let _ = cdp
                .send(
                    None,
                    "Target.closeTarget",
                    json!({ "targetId": tab.target_id }),
                )
                .await;
        } else {
            let _ = cdp
                .send(
                    None,
                    "Target.detachFromTarget",
                    json!({ "sessionId": tab.session }),
                )
                .await;
        }
        let in_use = self
            .tabs
            .lock()
            .await
            .values()
            .any(|(other, _)| Arc::ptr_eq(&other.browser, &tab.browser));
        let shutdown = match kind {
            BrowserKind::Managed { .. } => !in_use,
            BrowserKind::Spawned { .. } => kill,
            BrowserKind::Connected { .. } => !in_use,
        };
        if !shutdown {
            return;
        }
        self.browsers.lock().await.remove(&tab.browser.key);
        if let Some(mut spawned) = tab.browser.spawned.lock().await.take() {
            let _ = tokio::time::timeout(
                Duration::from_secs(2),
                cdp.send(None, "Browser.close", json!({})),
            )
            .await;
            spawned.kill().await;
        }
        cdp.close();
    }

    async fn release_named(&self, name: &str, kill: bool) -> bool {
        let removed = self.tabs.lock().await.remove(name);
        match removed {
            Some((tab, kind)) => {
                self.release(&tab, &kind, kill).await;
                true
            }
            None => false,
        }
    }

    async fn release_all(&self, kill: bool) -> usize {
        let drained: Vec<_> = self.tabs.lock().await.drain().collect();
        let count = drained.len();
        for (_, (tab, kind)) in drained {
            self.release(&tab, &kind, kill).await;
        }
        count
    }

    async fn live_tab(&self, name: &str) -> Option<(Arc<Tab>, BrowserKind)> {
        let mut tabs = self.tabs.lock().await;
        let detached = tabs.get(name).is_some_and(|(tab, _)| tab.is_detached());
        if detached {
            tabs.remove(name);
            return None;
        }
        tabs.get(name).cloned()
    }
}

impl Drop for BrowserManager {
    fn drop(&mut self) {
        // Spawned browsers are killed by `Spawned::drop`; connected sockets
        // close with their `Cdp`.
        for browser in self.browsers.get_mut().values() {
            browser.cdp.close();
        }
    }
}

// ---------------------------------------------------------------------------
// Tool
// ---------------------------------------------------------------------------

pub struct BrowserTool {
    manager: Arc<BrowserManager>,
    cwd: PathBuf,
}

/// Register the `browser` tool against a runtime-wide manager.
pub fn register_browser_tool(
    registry: &mut ToolRegistry,
    manager: Arc<BrowserManager>,
    cwd: &Path,
) -> Result<(), AgentError> {
    registry.register(Arc::new(BrowserTool {
        manager,
        cwd: cwd.to_path_buf(),
    }))
}

const DESCRIPTION: &str = r#"Drive real Chromium tabs over the DevTools protocol. Static content? Use `read`/`bash` instead; use `browser` for JavaScript-rendered pages, authenticated sessions, and interactive actions.

Actions:
- `open`: open or reuse tab `name` (default "main"), optionally navigating to `url`. Options: `viewport`, `wait_until`, `dialogs`, `allowed_domains`, `init_scripts`, `downloads`, `user_agent`, `ignore_https_errors`, `allow_file_access`, `headed`, `app`.
- `call`: run one tab helper per call: `chain: [{method:"fill", args:["label/Query", "text"]}]`; `args` is always an array of positional arguments (options objects go inside it: `[{compact:true}]`). The only two-step chain is an element handle: `[{method:"id", args:[5]}, {method:"click", args:[]}]`.
- `run`: evaluate JavaScript in the page's main world. `code` is an async function body (use `return`); `fn` is a serialized function called with `args`. Runs in the page, not a Node/Puppeteer worker: there is no `tab`/`page` object; use `call` for helpers.
- `tabs`: list managed tabs. `close`: release `name`, or every tab with `all: true`; `kill: true` also terminates spawned `app.path` browsers.

Tab helpers (`call`):
- Navigation: url, title, goto(url, {waitUntil}), back, forward, reload, pushState(url), waitForUrl(substring, {timeout}).
- Inspection: observe({selector?, compact?, includeAll?, viewportOnly?}); ariaSnapshot(selector?, {interactive?, compact?, urls?, diff?, depth?, boxes?}); screenshot({selector?, fullPage?, silent?, annotate?, format?, quality?}); pdf({path?, format?, landscape?, scale?, printBackground?, margin?, pageRanges?}) (headless only); extract(format? "markdown"|"text", {selector?, outline?, filter?}); text, html, value, attr, count, box, styles, isVisible, isEnabled, isChecked; frames.
- Interaction: click(sel, {button?, count?}), dblclick, hover, focus, check, uncheck, type(sel, text), fill(sel, value), press(key, {selector?}), keyDown, keyUp, mouseMove(x, y, {steps?}), mouseDown, mouseUp, clickAt(x, y), wheel(dx, dy), scroll(dx, dy, {selector?}), drag(from, to), highlight, scrollIntoView, select(sel, ...values), uploadFile(sel, ...paths).
- Waiting (timeouts in ms): waitFor(sel), waitForSelector(sel, {visible?, hidden?}) → boolean, waitForText(text, {selector?, exact?}).
- Dialogs: dialog, handleDialog({accept, text?}), setDialogs("accept"|"dismiss"|null). Without a policy, alerts and beforeunload are accepted; confirms and prompts stay pending.
- Emulation: emulate({viewport, userAgent, colorScheme, reducedMotion, geolocation, locale, timezone, offline, network, headers, cpuThrottling}).
- Storage: cookies({urls?}), setCookies(...objects | "Cookie: a=b" | JSON array), clearCookies({names?}), storage(kind, {key?}), setStorage(kind, key, value | entries), clearStorage(kind).
- Init scripts: addInitScript(source), removeInitScript(id), initScripts. Downloads: waitForDownload({timeout?}), downloads.
- Diagnostics: console({level?, since?, limit?, clear?}), errors, clearConsole, metrics, requests({filter?, type?, method?, status?, since?, limit?, clear?}), clearRequests, allowedDomains.
- Page execution: evaluate(expression) — a page-global expression; top-level `return` is invalid.
- Element handles: id(n) from observe, ref("e5") from ariaSnapshot, then one of click, dblclick, check, uncheck, highlight, type, fill, press, hover, focus, select, uploadFile, scrollIntoView, boundingBox, isVisible, isHidden, text, html, value, attr, styles, isEnabled, isChecked, evaluate (a function source called with the element first).

Selectors: CSS plus `aria/Name[role="button"]`, `text/…`, `xpath/…`, `pierce/…`, `label/…`, `placeholder/…`, `testid/…`, `alt/…`, `title/…`, `role/<role>[name="…" exact]`, and snapshot refs `e5`/`aria-ref=e5`. Navigation and re-renders invalidate ids and refs: re-observe, then act.

Default to `observe`; use screenshots for visual confirmation. `allowed_domains` accepts exact hosts and `*.example.com` (including the bare domain) and blocks other HTTP(S) requests. `app.path` launches a browser or Electron executable; `app.cdp_url` attaches to an existing endpoint (`app.target` picks a page by URL/title substring) — these act on real sessions, and closing never closes attached pages. Not available in Juto: relay/Tern/cmux backends, a11y, diffScreenshot, webmcp*, devices, clipboard*, saveState/loadState, trace/profile, route/HAR, request, vitals, react*, record*, frame handles, screenshot ifChanged/threshold."#;

#[async_trait]
impl Tool for BrowserTool {
    fn definition(&self) -> juto_ai::ToolDefinition {
        juto_ai::ToolDefinition {
            name: "browser".to_string(),
            description: DESCRIPTION.to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "action": { "type": "string", "enum": ["open", "close", "run", "call", "tabs"], "description": "operation" },
                    "name": { "type": "string", "description": "tab id (default 'main')" },
                    "url": { "type": "string", "description": "url to open" },
                    "app": {
                        "type": "object",
                        "properties": {
                            "path": { "type": "string", "description": "binary path to spawn" },
                            "cdp_url": { "type": "string", "description": "existing cdp endpoint" },
                            "relay": { "type": "boolean", "description": "not supported in Juto" },
                            "tern": { "type": "boolean", "description": "not supported in Juto" },
                            "args": { "type": "array", "items": { "type": "string" }, "description": "extra cli args" },
                            "target": { "type": "string", "description": "substring to pick a window" }
                        },
                        "additionalProperties": false
                    },
                    "viewport": {
                        "type": "object",
                        "properties": {
                            "width": { "type": "number" },
                            "height": { "type": "number" },
                            "scale": { "type": "number" }
                        },
                        "required": ["width", "height"],
                        "additionalProperties": false
                    },
                    "wait_until": { "type": "string", "enum": ["load", "domcontentloaded", "networkidle0", "networkidle2"], "description": "navigation wait condition" },
                    "dialogs": { "type": "string", "enum": ["accept", "dismiss"], "description": "auto-handle dialogs" },
                    "allowed_domains": { "type": "array", "items": { "type": "string" }, "description": "allowed request hostnames" },
                    "init_scripts": { "type": "array", "items": { "type": "string" }, "description": "document-start JavaScript sources or cwd-relative file paths" },
                    "downloads": { "type": "string", "description": "cwd-relative download directory" },
                    "user_agent": { "type": "string", "description": "tab user agent override" },
                    "ignore_https_errors": { "type": "boolean", "description": "ignore invalid HTTPS certificates" },
                    "allow_file_access": { "type": "boolean", "description": "allow file URLs to read local files" },
                    "headed": { "type": "boolean", "description": "visible Chromium window instead of headless" },
                    "code": { "type": "string", "description": "async function body evaluated in the page" },
                    "fn": { "type": "string", "description": "serialized JavaScript function evaluated in the page with `args`" },
                    "args": { "type": "array", "description": "arguments passed to `fn`" },
                    "chain": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "method": { "type": "string" },
                                "args": { "type": "array" }
                            },
                            "required": ["method", "args"]
                        }
                    },
                    "timeout": { "type": "number", "description": "timeout in seconds (default 30, max 300)" },
                    "all": { "type": "boolean", "description": "release every managed tab" },
                    "kill": { "type": "boolean", "description": "also kill spawned-app browsers" }
                },
                "required": ["action"],
                "additionalProperties": false
            }),
        }
    }

    fn tier(&self) -> ToolTier {
        ToolTier::Exec
    }

    async fn execute(&self, arguments: Value, ctx: ToolContext) -> Result<ToolOutput, String> {
        let params: BrowserParams = serde_json::from_value(arguments)
            .map_err(|e| format!("browser received invalid arguments: {e}"))?;
        let timeout = params
            .timeout
            .unwrap_or(DEFAULT_TIMEOUT)
            .clamp(MIN_TIMEOUT, MAX_TIMEOUT);
        let cwd = if ctx.cwd.as_os_str().is_empty() {
            self.cwd.clone()
        } else {
            ctx.cwd.clone()
        };
        let mut op = OpCtx {
            cwd,
            cell_ms: (timeout * 1000.0) as u64,
            cancel: ctx.cancel.clone(),
            output: Vec::new(),
            screenshots: Vec::new(),
        };
        let action = params.action.clone();
        let name = params
            .name
            .clone()
            .unwrap_or_else(|| DEFAULT_TAB_NAME.to_string());
        let work = self.dispatch(&params, &name, &mut op);
        let result = tokio::select! {
            result = tokio::time::timeout(Duration::from_secs_f64(timeout), work) => match result {
                Ok(result) => result,
                Err(_) => Err(format!("browser {action} timed out after {timeout}s")),
            },
            _ = ctx.cancel.cancelled() => Err("cancelled".to_string()),
        };
        let (text, mut details) = result?;
        let mut content = std::mem::take(&mut op.output);
        if !op.screenshots.is_empty() {
            details["screenshots"] = Value::Array(op.screenshots);
        }
        if let Some(text) = text {
            content.push(ContentBlock::Text {
                text: cap_output(text),
            });
        }
        if content.is_empty() {
            content.push(ContentBlock::Text {
                text: "(no return value)".to_string(),
            });
        }
        Ok(ToolOutput {
            content,
            details,
            is_error: false,
        })
    }
}

/// Cap inline text; the full text is saved to a temp file named in the footer.
fn cap_output(text: String) -> String {
    if text.len() <= OUTPUT_MAX_BYTES {
        return text;
    }
    let mut end = OUTPUT_MAX_BYTES;
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    let path =
        std::env::temp_dir().join(format!("juto-browser-output-{}.txt", uuid::Uuid::now_v7()));
    let saved = std::fs::write(&path, &text).is_ok();
    let mut capped = text[..end].to_string();
    capped.push_str(&format!(
        "\n... (output truncated at {OUTPUT_MAX_BYTES} of {} bytes{})",
        text.len(),
        if saved {
            format!("; full output: {}", path.display())
        } else {
            String::new()
        }
    ));
    capped
}

fn render_value(value: &Value) -> Option<String> {
    match value {
        Value::Null => None,
        Value::String(s) => Some(s.clone()),
        other => Some(serde_json::to_string_pretty(other).unwrap_or_else(|_| other.to_string())),
    }
}

impl BrowserTool {
    async fn dispatch(
        &self,
        params: &BrowserParams,
        name: &str,
        op: &mut OpCtx,
    ) -> Result<(Option<String>, Value), String> {
        let mut details = json!({ "action": params.action, "name": name });
        match params.action.as_str() {
            "open" => self.open(params, name, op, details).await,
            "close" => {
                let text = if params.all {
                    let count = self.manager.release_all(params.kill).await;
                    format!(
                        "Released {count} managed tab{}",
                        if count == 1 { "" } else { "s" }
                    )
                } else if self.manager.release_named(name, params.kill).await {
                    format!("Released managed tab {}", json!(name))
                } else {
                    format!("No tab named {}", json!(name))
                };
                Ok((Some(text), details))
            }
            "tabs" => {
                let tabs: Vec<Value> = {
                    let tabs = self.manager.tabs.lock().await;
                    let mut list: Vec<_> = tabs
                        .values()
                        .filter(|(tab, _)| !tab.is_detached())
                        .map(|(tab, _)| tab.info())
                        .collect();
                    list.sort_by(|a, b| a["name"].as_str().cmp(&b["name"].as_str()));
                    list
                };
                let value = Value::Array(tabs);
                details["value"] = value.clone();
                Ok((render_value(&value), details))
            }
            "run" | "call" => {
                let Some((tab, _)) = self.manager.live_tab(name).await else {
                    return Err(format!(
                        "No tab named {}. Open it first with action \"open\".",
                        json!(name)
                    ));
                };
                details["url"] = json!(tab.url());
                details["browser"] = json!(tab.browser.kind);
                let _busy = tab.busy.lock().await;
                let value = if params.action == "run" {
                    run_code(&tab, params).await?
                } else {
                    call_chain(&tab, op, &params.chain).await?
                };
                if !value.is_null() {
                    details["value"] = value.clone();
                }
                Ok((render_value(&value), details))
            }
            other => Err(format!("Unknown browser action {}", json!(other))),
        }
    }

    async fn open(
        &self,
        params: &BrowserParams,
        name: &str,
        op: &mut OpCtx,
        mut details: Value,
    ) -> Result<(Option<String>, Value), String> {
        let app = params.app.as_ref();
        if app.is_some_and(|a| a.relay == Some(true)) {
            return Err("app.relay is not supported in Juto's browser tool; use app.cdp_url to attach to a browser started with --remote-debugging-port".into());
        }
        if app.is_some_and(|a| a.tern == Some(true)) {
            return Err("app.tern is not supported in Juto's browser tool".into());
        }
        let kind = match app {
            Some(AppParams {
                cdp_url: Some(url), ..
            }) => BrowserKind::Connected {
                cdp_url: url.clone(),
            },
            Some(AppParams {
                path: Some(path),
                args,
                ..
            }) => BrowserKind::Spawned {
                path: path.clone(),
                args: args.clone(),
            },
            _ => BrowserKind::Managed {
                headed: params.headed.unwrap_or(false),
                allow_file_access: params.allow_file_access.unwrap_or(false),
            },
        };
        details["browser"] = json!(match kind {
            BrowserKind::Managed { .. } => "headless",
            BrowserKind::Spawned { .. } => "spawned",
            BrowserKind::Connected { .. } => "connected",
        });

        let existing = self.manager.live_tab(name).await;
        if let Some((_, existing_kind)) = &existing
            && existing_kind.key() != kind.key()
        {
            return Err(format!(
                "Tab {} is bound to a different browser ({}). Close it first.",
                json!(name),
                existing_kind.describe()
            ));
        }
        let mut init_sources = Vec::new();
        for source in params.init_scripts.iter().flatten() {
            let path = resolve_path(&op.cwd, source);
            init_sources.push(if path.is_file() {
                std::fs::read_to_string(&path)
                    .map_err(|e| format!("cannot read init script {}: {e}", path.display()))?
            } else {
                source.clone()
            });
        }
        // Creation-time options cannot be applied to a live tab: recycle it.
        let recycle = !init_sources.is_empty()
            || params.downloads.is_some()
            || params.user_agent.is_some()
            || params.ignore_https_errors == Some(true);
        let mut reused = None;
        if let Some((tab, _)) = existing {
            if recycle {
                self.manager.release_named(name, false).await;
            } else {
                reused = Some(tab);
            }
        }
        let created = reused.is_none();
        let tab = match reused {
            Some(tab) => tab,
            None => {
                let browser = self.manager.acquire_browser(&kind, &op.cwd).await?;
                let (target_id, owned) = pick_target(&browser, &kind, app, &self.manager).await?;
                let tab = Tab::attach(browser, name, &target_id, owned).await?;
                if let Err(error) =
                    configure_new_tab(&tab, &kind, params, &init_sources, &op.cwd).await
                {
                    self.manager.release(&tab, &kind, false).await;
                    return Err(error);
                }
                self.manager
                    .tabs
                    .lock()
                    .await
                    .insert(name.to_string(), (tab.clone(), kind.clone()));
                tab
            }
        };
        let _busy = tab.busy.lock().await;
        if !created {
            if let Some(viewport) = &params.viewport {
                tab.set_viewport(viewport.width, viewport.height, viewport.scale)
                    .await?;
            }
            if let Some(dialogs) = &params.dialogs {
                tab.set_dialog_policy(dialog_policy(dialogs)?);
            }
            if let Some(domains) = &params.allowed_domains {
                tab.set_allowed_domains(domains.clone()).await?;
            }
        }
        if let Some(url) = &params.url {
            tab.goto(op, url, params.wait_until.as_deref()).await?;
        }
        let info = tab.call(op, "title", &[]).await.unwrap_or(Value::Null);
        let url = tab.url();
        details["url"] = json!(url);
        let mut lines = vec![
            format!(
                "{} tab {} on {}",
                if created { "Opened" } else { "Reused" },
                json!(name),
                kind.describe()
            ),
            format!("URL: {url}"),
        ];
        if let Some(title) = info.as_str().filter(|t| !t.is_empty()) {
            lines.push(format!("Title: {title}"));
        }
        Ok((Some(lines.join("\n")), details))
    }
}

fn dialog_policy(raw: &str) -> Result<Option<DialogPolicy>, String> {
    match raw {
        "accept" => Ok(Some(DialogPolicy::Accept)),
        "dismiss" => Ok(Some(DialogPolicy::Dismiss)),
        other => Err(format!(
            "dialogs must be \"accept\" or \"dismiss\"; got {}",
            json!(other)
        )),
    }
}

/// Choose the page target a new tab drives: an `app.target` match, the
/// spawned app's first unmanaged window, or a fresh page.
async fn pick_target(
    browser: &Arc<Browser>,
    kind: &BrowserKind,
    app: Option<&AppParams>,
    manager: &BrowserManager,
) -> Result<(String, bool), String> {
    let wanted = app.and_then(|a| a.target.as_deref());
    if wanted.is_some() || matches!(kind, BrowserKind::Spawned { .. }) {
        let managed: Vec<String> = manager
            .tabs
            .lock()
            .await
            .values()
            .map(|(tab, _)| tab.target_id.clone())
            .collect();
        let targets = browser
            .cdp
            .send(None, "Target.getTargets", json!({}))
            .await?;
        let pages: Vec<&Value> = targets["targetInfos"]
            .as_array()
            .into_iter()
            .flatten()
            .filter(|t| t["type"] == "page")
            .filter(|t| !managed.iter().any(|m| t["targetId"] == m.as_str()))
            .collect();
        let found = match wanted {
            Some(needle) => pages.iter().find(|t| {
                t["url"].as_str().is_some_and(|u| u.contains(needle))
                    || t["title"].as_str().is_some_and(|u| u.contains(needle))
            }),
            None => pages.first(),
        };
        match (found, wanted) {
            (Some(target), _) => {
                return Ok((
                    target["targetId"].as_str().unwrap_or_default().to_string(),
                    false,
                ));
            }
            (None, Some(needle)) => {
                return Err(format!(
                    "No page target matches app.target {}",
                    json!(needle)
                ));
            }
            (None, None) => {}
        }
    }
    let created = browser
        .cdp
        .send(None, "Target.createTarget", json!({ "url": "about:blank" }))
        .await?;
    Ok((
        created["targetId"]
            .as_str()
            .ok_or("Target.createTarget returned no targetId")?
            .to_string(),
        true,
    ))
}

async fn configure_new_tab(
    tab: &Tab,
    kind: &BrowserKind,
    params: &BrowserParams,
    init_sources: &[String],
    cwd: &Path,
) -> Result<(), String> {
    match (&params.viewport, kind) {
        (Some(viewport), _) => {
            tab.set_viewport(viewport.width, viewport.height, viewport.scale)
                .await?
        }
        (None, BrowserKind::Managed { headed: false, .. }) => {
            let (w, h, scale) = DEFAULT_VIEWPORT;
            tab.set_viewport(w, h, Some(scale)).await?
        }
        _ => {}
    }
    if let Some(user_agent) = &params.user_agent {
        tab.set_user_agent(user_agent).await?;
    }
    if params.ignore_https_errors == Some(true) {
        tab.set_ignore_https_errors(true).await?;
    }
    if let Some(dialogs) = &params.dialogs {
        tab.set_dialog_policy(dialog_policy(dialogs)?);
    }
    if let Some(domains) = &params.allowed_domains
        && !domains.is_empty()
    {
        tab.set_allowed_domains(domains.clone()).await?;
    }
    for source in init_sources {
        tab.add_init_script(source).await?;
    }
    if let Some(dir) = &params.downloads {
        tab.set_downloads(&resolve_path(cwd, dir)).await?;
    }
    Ok(())
}

/// `run`: exactly one of `code` (async function body) or `fn` (+ `args`).
async fn run_code(tab: &Tab, params: &BrowserParams) -> Result<Value, String> {
    let code = params
        .code
        .as_deref()
        .map(str::trim)
        .filter(|c| !c.is_empty());
    let function = params
        .function
        .as_deref()
        .map(str::trim)
        .filter(|f| !f.is_empty());
    let expression = match (code, function) {
        (Some(code), None) => format!("(async () => {{\n{code}\n}})()"),
        (None, Some(function)) => format!(
            "(async () => await ({function})(...{}))()",
            serde_json::to_string(&params.args).map_err(|e| e.to_string())?
        ),
        _ => return Err("Action 'run' requires exactly one of 'code' or 'fn'.".into()),
    };
    tab.run_expression(&expression).await
}

/// `call`: one allowlisted helper, or one element-handle hop (`renderTabCall`).
async fn call_chain(tab: &Tab, op: &mut OpCtx, chain: &[CallStep]) -> Result<Value, String> {
    let Some(root) = chain.first() else {
        return Err("Action 'call' requires a non-empty 'chain'.".into());
    };
    if chain.len() > 2 {
        return Err(
            "Call chains support one element-handle hop at most; use action \"run\" for longer page-side sequences."
                .into(),
        );
    }
    let method = root.method.as_str();
    if TAB_VALUE_METHODS.contains(&method) || TAB_PRESENCE_METHODS.contains(&method) {
        if chain.len() == 2 {
            return Err(format!(
                "Only tab.id(n)/tab.ref(id) results accept a chained call; got tab.{method}()."
            ));
        }
        let value = tab.call(op, method, &root.args).await?;
        return Ok(if TAB_PRESENCE_METHODS.contains(&method) {
            json!(value != Value::Bool(false))
        } else {
            value
        });
    }
    if TAB_HANDLE_METHODS.contains(&method) {
        if method == "frame" {
            return Err(tab::not_ported("frame"));
        }
        let Some(step) = chain.get(1) else {
            return Err(format!(
                "tab.{method}() returns an element handle; call a method on it (tab.id(5).click())."
            ));
        };
        let spec = tab.handle(method, &root.args).await?;
        return tab.element_call(op, &spec, &step.method, &step.args).await;
    }
    Err(format!(
        "Unknown tab helper \"{method}\". Direct helpers: {}; handles via tab.id(n)/tab.ref(id).",
        TAB_VALUE_METHODS
            .iter()
            .chain(TAB_PRESENCE_METHODS)
            .copied()
            .collect::<Vec<_>>()
            .join(", ")
    ))
}

#[cfg(test)]
mod live_tests;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn browser_kinds_key_separately() {
        let headless = BrowserKind::Managed {
            headed: false,
            allow_file_access: false,
        };
        let headed = BrowserKind::Managed {
            headed: true,
            allow_file_access: false,
        };
        assert_ne!(headless.key(), headed.key());
        assert_eq!(headless.describe(), "headless Chromium");
    }

    #[test]
    fn output_cap_truncates_on_char_boundary() {
        let text = "é".repeat(OUTPUT_MAX_BYTES);
        let capped = cap_output(text);
        assert!(capped.contains("output truncated"));
        assert!(capped.len() < OUTPUT_MAX_BYTES + 200);
    }
}
