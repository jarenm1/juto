use juto_ai::{Choice, CredentialStore, Judge, JudgeError, TypeSafeJudge, create_judge};
use serde_json::json;
use std::sync::Arc;
use std::time::Instant;
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::sync::Mutex;

async fn spawn_mock_server() -> (String, Arc<Mutex<Vec<String>>>) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let requests = Arc::new(Mutex::new(Vec::new()));
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
            seen.lock().await.push(request.clone());

            let (status, body) =
                if request.contains("unauthorized") {
                    (
                        "401 Unauthorized",
                        json!({"error": "unauthorized"}).to_string(),
                    )
                } else if request.contains("openrouter/~typesafe/jev-latest") {
                    (
                        "200 OK",
                        json!({
                            "key": "EXPLORE_SCOUT",
                            "confidence": 0.94,
                            "rationale": "Read-only question about authentication architecture",
                            "usage": { "input_tokens": 18, "output_tokens": 2, "total_tokens": 20 }
                        })
                        .to_string(),
                    )
                } else {
                    ("200 OK", json!({
                    "choice": "END_TO_END_TASK",
                    "confidence": 0.98,
                    "rationale": "User requested multi-file implementation across repository",
                    "usage": { "input_tokens": 35, "output_tokens": 4, "total_tokens": 39 }
                }).to_string())
                };

            let response = format!(
                "HTTP/1.1 {status}\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
                body.len()
            );
            let _ = reader.get_mut().write_all(response.as_bytes()).await;
        }
    });

    (format!("http://127.0.0.1:{port}"), requests)
}

#[tokio::test]
async fn smoke_decision_layer_end_to_end() {
    let (base_url, requests) = spawn_mock_server().await;

    // 1. System One Intent Triage
    let ts_judge = TypeSafeJudge::new("sk-typesafe-smoke-key")
        .with_endpoint(format!("{base_url}/v1/systemone"));

    let triage_choices = vec![
        Choice::new("DIRECT_REPLY", "Conversational answer or clarification"),
        Choice::new("EXPLORE_SCOUT", "Fast read-only codebase exploration"),
        Choice::new("MECHANICAL_EDIT", "Single localized deterministic change"),
        Choice::new(
            "END_TO_END_TASK",
            "Multi-file implementation with subagents",
        ),
    ];

    let t0 = Instant::now();
    let decision = ts_judge
        .choose(
            "Implement the discrete decision layer and wire up System One Jev",
            &triage_choices,
        )
        .await
        .expect("triage choice succeeds");
    let dur = t0.elapsed();

    assert_eq!(decision.key, "END_TO_END_TASK");
    assert!(decision.confidence.unwrap() > 0.9);
    assert!(decision.usage.total_tokens > 0);
    assert!(dur.as_millis() < 500);

    // 2. OpenRouter Decisions Persona Routing
    let or_judge = TypeSafeJudge::for_openrouter("sk-or-smoke-key")
        .with_endpoint(format!("{base_url}/decisions"));

    let persona_choices = vec![
        Choice::new("EXPLORE_SCOUT", "Read-only exploration"),
        Choice::new("BUILDER", "Code authoring in isolated workspace"),
        Choice::new("REVIEWER", "Adversarial code review and security audit"),
    ];

    let decision = or_judge
        .choose(
            "Search for all references to CredentialStore in juto-runtime",
            &persona_choices,
        )
        .await
        .expect("persona choice succeeds");

    assert_eq!(decision.key, "EXPLORE_SCOUT");
    assert_eq!(decision.confidence, Some(0.94));

    // 3. Auth failure error handling
    let bad_judge =
        TypeSafeJudge::new("unauthorized").with_endpoint(format!("{base_url}/v1/systemone"));

    let err = bad_judge
        .choose("Some task", &triage_choices)
        .await
        .unwrap_err();
    assert!(matches!(err, JudgeError::Authentication(_)));

    // 4. CredentialStore auto-resolution precedence
    let temp_dir = tempfile::tempdir().unwrap();
    let store = CredentialStore::open(temp_dir.path().join("credentials.json")).unwrap();
    store
        .set_api_key("typesafe", "sk-stored-typesafe-key")
        .unwrap();

    let resolved_judge = create_judge(&store, None).await.expect("resolved judge");
    assert!(format!("{resolved_judge:?}").contains("TypeSafeJudge"));

    let recorded = requests.lock().await;
    assert_eq!(recorded.len(), 3);
}
