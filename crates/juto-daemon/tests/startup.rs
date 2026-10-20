//! Daemon startup beside an unreadable session journal. This is its own test
//! binary because `serve()` reads its data and runtime directories from the
//! process environment.

use std::time::Duration;

use juto_daemon::protocol::{ClientFrame, PROTOCOL_VERSION, Reply, Request, ServerFrame, paths};
use juto_daemon::server::serve;
use juto_runtime::Session;
use tokio::{
    io::{AsyncBufReadExt, AsyncWriteExt, BufReader},
    net::UnixStream,
};

#[tokio::test]
async fn startup_skips_an_unreadable_journal_and_serves_the_good_one() {
    let root = tempfile::tempdir().unwrap();
    let workspace = root.path().join("workspace");
    let data_dir = root.path().join("data");
    let runtime_dir = root.path().join("run");
    let sessions = data_dir.join("sessions");
    std::fs::create_dir_all(&workspace).unwrap();
    std::fs::create_dir_all(&sessions).unwrap();
    std::fs::create_dir_all(&runtime_dir).unwrap();
    let workspace = workspace.canonicalize().unwrap();

    let good = Session::create(&sessions, &workspace).unwrap();
    let good_id = good.header().id.clone();
    drop(good);
    let corrupt = sessions.join("corrupt.jsonl");
    std::fs::write(&corrupt, "{ not a journal\n").unwrap();

    // SAFETY: this test binary runs a single test, before any other thread reads
    // these variables.
    unsafe {
        std::env::set_var("JUTO_DATA_DIR", &data_dir);
        std::env::set_var("XDG_RUNTIME_DIR", &runtime_dir);
    }

    let mut served = tokio::spawn(serve(workspace.clone()));
    let socket = paths::daemon_socket(&workspace).unwrap();
    let stream = tokio::time::timeout(Duration::from_secs(10), async {
        loop {
            if served.is_finished() {
                let exit = (&mut served).await;
                panic!("daemon exited during startup: {exit:?}");
            }
            if let Ok(stream) = UnixStream::connect(&socket).await {
                break stream;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("daemon must accept clients within 10s");

    let (read_half, mut write_half) = stream.into_split();
    let mut lines = BufReader::new(read_half).lines();
    for (id, request) in [
        (
            1,
            Request::Hello {
                client: "startup-test".into(),
                protocol_version: PROTOCOL_VERSION,
            },
        ),
        (2, Request::ListSessions),
    ] {
        let frame = ClientFrame { id, request };
        write_half
            .write_all(format!("{}\n", serde_json::to_string(&frame).unwrap()).as_bytes())
            .await
            .unwrap();
    }
    let mut listed = None;
    for _ in 0..2 {
        let line = lines.next_line().await.unwrap().unwrap();
        if let ServerFrame::Response {
            id: 2,
            result: Ok(Reply::Sessions(sessions)),
        } = serde_json::from_str(&line).unwrap()
        {
            listed = Some(sessions);
        }
    }
    let listed = listed.expect("list_sessions must be answered");
    assert_eq!(
        listed.iter().map(|s| s.id.as_str()).collect::<Vec<_>>(),
        vec![good_id.as_str()],
        "the good session must be loaded and the corrupt journal skipped"
    );
    assert!(
        !served.is_finished(),
        "startup must keep serving past the corrupt journal"
    );
    assert!(
        corrupt.exists(),
        "the corrupt journal must be left in place"
    );

    served.abort();
}
