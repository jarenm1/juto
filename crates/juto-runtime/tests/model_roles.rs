//! Role assignments write `modelRoles` into the global config layer and take
//! effect in the live runtime without a reload.

use juto_runtime::{Runtime, RuntimeConfig};

fn runtime_in(dir: &tempfile::TempDir) -> (Runtime, std::path::PathBuf) {
    let data_dir = dir.path().join("data");
    let cwd = dir.path().join("work");
    std::fs::create_dir_all(&cwd).unwrap();
    let runtime = Runtime::from_config(
        &cwd,
        &data_dir,
        RuntimeConfig {
            model: "anthropic/claude-sonnet-4-5".into(),
            ..Default::default()
        },
    )
    .unwrap();
    (runtime, data_dir)
}

#[test]
fn role_assignment_persists_and_resolves_immediately() {
    let dir = tempfile::tempdir().unwrap();
    let (runtime, data_dir) = runtime_in(&dir);

    runtime
        .set_model_role("task", "openai/gpt-4o")
        .expect("assign @task");

    let text = std::fs::read_to_string(data_dir.join("config.yml")).unwrap();
    assert!(
        text.contains("task: openai/gpt-4o"),
        "config.yml must record modelRoles.task, got:\n{text}"
    );

    let config = runtime.config();
    assert_eq!(config.model_roles["task"], "openai/gpt-4o");
    assert_eq!(config.resolve_selector("@task").unwrap().0, "openai/gpt-4o");
}

#[test]
fn default_role_assignment_updates_model_and_switches_sessions() {
    let dir = tempfile::tempdir().unwrap();
    let (runtime, data_dir) = runtime_in(&dir);

    runtime
        .set_model_role("default", "openai/gpt-4o")
        .expect("assign @default");

    let text = std::fs::read_to_string(data_dir.join("config.yml")).unwrap();
    assert!(
        text.contains("model: openai/gpt-4o"),
        "config.yml must update top-level model, got:\n{text}"
    );
    // No modelRoles.default is written when none existed.
    assert!(
        !text.contains("modelRoles"),
        "unexpected modelRoles: {text}"
    );

    assert_eq!(runtime.config().model, "openai/gpt-4o");
    assert_eq!(
        runtime.config().resolve_selector("@default").unwrap().0,
        "openai/gpt-4o"
    );
}

#[test]
fn invalid_selector_is_rejected_without_touching_config() {
    let dir = tempfile::tempdir().unwrap();
    let (runtime, data_dir) = runtime_in(&dir);

    assert!(runtime.set_model_role("smol", "@smol").is_err());
    assert!(
        runtime
            .set_model_role("smol", "ghost/not-a-real-model")
            .is_err()
    );

    let path = data_dir.join("config.yml");
    assert!(
        !path.exists() || !std::fs::read_to_string(&path).unwrap().contains("smol"),
        "rejected assignments must not be persisted"
    );
    assert!(!runtime.config().model_roles.contains_key("smol"));
}
