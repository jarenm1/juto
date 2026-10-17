//! End-to-end tool behavior against a real Chromium (`JUTO_CHROMIUM` or
//! `PATH`) and a loopback HTTP fixture. Skipped when no Chromium exists.

use std::{path::Path, sync::Arc, time::Duration};

use juto_agent::{Tool, ToolContext, ToolOutput};
use juto_ai::ContentBlock;
use serde_json::{Value, json};
use tokio::io::{AsyncReadExt as _, AsyncWriteExt as _};
use tokio_util::sync::CancellationToken;

use super::{BrowserManager, BrowserTool, launch};

const PAGE: &str = r#"<!doctype html><title>Fixture</title>
<label for="q">Query</label><input id="q">
<button id="go" onclick="document.getElementById('out').textContent='got:'+document.getElementById('q').value">Search</button>
<p id="out"></p>
<input type="checkbox" id="agree" aria-label="Agree">
<button id="ask" onclick="document.getElementById('answer').textContent=String(confirm('Sure?'))">Ask</button>
<p id="answer"></p>
<a href="/two">Next page</a>
<script>console.log('hello', 42)</script>"#;

async fn serve() -> u16 {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    tokio::spawn(async move {
        loop {
            let Ok((mut socket, _)) = listener.accept().await else {
                break;
            };
            tokio::spawn(async move {
                let mut buf = vec![0u8; 4096];
                let n = socket.read(&mut buf).await.unwrap_or(0);
                let request = String::from_utf8_lossy(&buf[..n]);
                let path = request.split_whitespace().nth(1).unwrap_or("/");
                let (status, body) = match path {
                    "/" => ("200 OK", PAGE.to_string()),
                    "/two" => ("200 OK", "<title>Two</title><h1>Second</h1>".to_string()),
                    _ => ("404 Not Found", "missing".to_string()),
                };
                let response = format!(
                    "HTTP/1.1 {status}\r\nContent-Type: text/html\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                );
                let _ = socket.write_all(response.as_bytes()).await;
            });
        }
    });
    port
}

fn ctx(cwd: &Path, cancel: CancellationToken) -> ToolContext {
    ToolContext {
        cwd: cwd.to_path_buf(),
        session_id: "test".into(),
        cancel,
    }
}

fn text(output: &ToolOutput) -> String {
    output
        .content
        .iter()
        .filter_map(|block| match block {
            ContentBlock::Text { text } => Some(text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

async fn call(tool: &BrowserTool, cwd: &Path, chain: Value) -> Result<ToolOutput, String> {
    tool.execute(
        json!({ "action": "call", "chain": chain }),
        ctx(cwd, CancellationToken::new()),
    )
    .await
}

async fn value(tool: &BrowserTool, cwd: &Path, chain: Value) -> Value {
    let output = call(tool, cwd, chain.clone())
        .await
        .unwrap_or_else(|e| panic!("{chain}: {e}"));
    output.details.get("value").cloned().unwrap_or(Value::Null)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn drives_a_real_chromium_tab() {
    if launch::chromium_executable().is_err() {
        eprintln!("skipping: no Chromium executable");
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let cwd = dir.path();
    let port = serve().await;
    let base = format!("http://127.0.0.1:{port}/");
    let tool = BrowserTool {
        manager: Arc::new(BrowserManager::new(cwd.join("data"))),
        cwd: cwd.to_path_buf(),
    };

    let opened = tool
        .execute(
            json!({ "action": "open", "url": base }),
            ctx(cwd, CancellationToken::new()),
        )
        .await
        .unwrap();
    let opened_text = text(&opened);
    assert!(
        opened_text.starts_with("Opened tab \"main\" on headless Chromium"),
        "{opened_text}"
    );
    assert!(opened_text.contains("Title: Fixture"), "{opened_text}");

    // Semantic selectors, trusted input, and page reads.
    value(
        &tool,
        cwd,
        json!([{ "method": "fill", "args": ["label/Query", "rust"] }]),
    )
    .await;
    value(
        &tool,
        cwd,
        json!([{ "method": "click", "args": ["text/Search"] }]),
    )
    .await;
    assert_eq!(
        value(&tool, cwd, json!([{ "method": "text", "args": ["#out"] }])).await,
        "got:rust"
    );

    // Observation ids drive element handles.
    let observation = value(&tool, cwd, json!([{ "method": "observe", "args": [] }])).await;
    let checkbox = observation["elements"]
        .as_array()
        .unwrap()
        .iter()
        .find(|e| e["role"] == "checkbox")
        .expect("checkbox observed")["id"]
        .clone();
    value(
        &tool,
        cwd,
        json!([{ "method": "id", "args": [checkbox] }, { "method": "check", "args": [] }]),
    )
    .await;
    assert_eq!(
        value(
            &tool,
            cwd,
            json!([{ "method": "isChecked", "args": ["#agree"] }])
        )
        .await,
        true
    );

    // A confirm stays pending until handled; the click itself returns.
    value(&tool, cwd, json!([{ "method": "click", "args": ["#ask"] }])).await;
    let dialog = value(&tool, cwd, json!([{ "method": "dialog", "args": [] }])).await;
    assert_eq!(dialog["open"], true);
    assert_eq!(dialog["type"], "confirm");
    value(
        &tool,
        cwd,
        json!([{ "method": "handleDialog", "args": [{ "accept": true }] }]),
    )
    .await;
    assert_eq!(
        value(
            &tool,
            cwd,
            json!([{ "method": "text", "args": ["#answer"] }])
        )
        .await,
        "true"
    );

    // Console capture from document start.
    let console = value(&tool, cwd, json!([{ "method": "console", "args": [] }])).await;
    assert!(
        console["entries"]
            .as_array()
            .unwrap()
            .iter()
            .any(|e| e["text"] == "hello 42"),
        "{console}"
    );

    // Screenshots reach the model as images and are saved.
    let shot = call(&tool, cwd, json!([{ "method": "screenshot", "args": [] }]))
        .await
        .unwrap();
    assert!(
        shot.content.iter().any(
            |b| matches!(b, ContentBlock::Image { mime_type, .. } if mime_type == "image/png")
        )
    );
    assert!(Path::new(shot.details["value"].as_str().unwrap()).is_file());

    // ARIA refs resolve; navigation invalidates observation ids.
    let snapshot = value(
        &tool,
        cwd,
        json!([{ "method": "ariaSnapshot", "args": [{ "interactive": true }] }]),
    )
    .await;
    let snapshot = snapshot.as_str().unwrap();
    let line = snapshot
        .lines()
        .find(|l| l.contains("link \"Next page\""))
        .expect("link in snapshot");
    let reference = line
        .split("[ref=")
        .nth(1)
        .unwrap()
        .split(']')
        .next()
        .unwrap();
    value(
        &tool,
        cwd,
        json!([{ "method": "ref", "args": [reference] }, { "method": "click", "args": [] }]),
    )
    .await;
    let url = value(
        &tool,
        cwd,
        json!([{ "method": "waitForUrl", "args": ["/two"] }]),
    )
    .await;
    assert_eq!(url, format!("{base}two"));
    let stale = call(
        &tool,
        cwd,
        json!([{ "method": "id", "args": [checkbox] }, { "method": "click", "args": [] }]),
    )
    .await;
    assert!(stale.unwrap_err().contains("Unknown element id"));
    let markdown = value(
        &tool,
        cwd,
        json!([{ "method": "extract", "args": ["markdown"] }]),
    )
    .await;
    assert!(markdown.as_str().unwrap().contains("Second"), "{markdown}");
    assert_eq!(
        value(&tool, cwd, json!([{ "method": "back", "args": [] }])).await,
        base
    );

    // Page-world run: values and exceptions.
    let ran = tool
        .execute(
            json!({ "action": "run", "code": "return document.title + '!'" }),
            ctx(cwd, CancellationToken::new()),
        )
        .await
        .unwrap();
    assert_eq!(text(&ran), "Fixture!");
    let thrown = tool
        .execute(
            json!({ "action": "run", "fn": "(a) => { throw new Error('boom ' + a) }", "args": [7] }),
            ctx(cwd, CancellationToken::new()),
        )
        .await
        .unwrap_err();
    assert!(thrown.contains("boom 7"), "{thrown}");

    // allowed_domains blocks other hosts.
    tool.execute(
        json!({ "action": "open", "name": "locked", "url": base, "allowed_domains": ["127.0.0.1"] }),
        ctx(cwd, CancellationToken::new()),
    )
    .await
    .unwrap();
    let fetched = tool
        .execute(
            json!({
                "action": "run",
                "name": "locked",
                "code": format!("try {{ await fetch('http://localhost:{port}/two'); return 'fetched'; }} catch (e) {{ return 'blocked'; }}"),
            }),
            ctx(cwd, CancellationToken::new()),
        )
        .await
        .unwrap();
    assert_eq!(text(&fetched), "blocked");

    // Cancellation ends a wait promptly.
    let cancel = CancellationToken::new();
    let trigger = cancel.clone();
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(300)).await;
        trigger.cancel();
    });
    let started = std::time::Instant::now();
    let cancelled = tool
        .execute(
            json!({ "action": "call", "chain": [{ "method": "waitForText", "args": ["never", { "timeout": 0 }] }] }),
            ctx(cwd, cancel),
        )
        .await
        .unwrap_err();
    assert_eq!(cancelled, "cancelled");
    assert!(started.elapsed() < Duration::from_secs(5));

    let tabs = tool
        .execute(
            json!({ "action": "tabs" }),
            ctx(cwd, CancellationToken::new()),
        )
        .await
        .unwrap();
    assert_eq!(tabs.details["value"].as_array().unwrap().len(), 2);
    let closed = tool
        .execute(
            json!({ "action": "close", "all": true }),
            ctx(cwd, CancellationToken::new()),
        )
        .await
        .unwrap();
    assert_eq!(text(&closed), "Released 2 managed tabs");
    let missing = tool
        .execute(
            json!({ "action": "call", "chain": [{ "method": "url", "args": [] }] }),
            ctx(cwd, CancellationToken::new()),
        )
        .await
        .unwrap_err();
    assert!(missing.starts_with("No tab named \"main\""));
}
