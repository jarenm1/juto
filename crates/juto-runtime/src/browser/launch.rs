//! Locate, launch, and attach to Chromium-family browsers.
//!
//! Behavioral reference: oh-my-pi `tools/browser/launch.ts` and `attach.ts`
//! at 579da1d6 (see licenses/OMP-MIT.txt). Juto does not download a managed
//! Chromium: the executable comes from `JUTO_CHROMIUM` (set by the Nix
//! package) or `PATH`.

use std::{
    path::{Path, PathBuf},
    process::Stdio,
    time::Duration,
};

use serde_json::Value;
use tokio::{
    io::{AsyncBufReadExt as _, BufReader},
    process::{Child, Command},
};

/// Environment variable naming the managed Chromium executable.
pub(crate) const CHROMIUM_ENV: &str = "JUTO_CHROMIUM";
/// Executable names searched on `PATH` when `JUTO_CHROMIUM` is unset.
const CHROMIUM_NAMES: &[&str] = &[
    "chromium",
    "chromium-browser",
    "google-chrome-stable",
    "google-chrome",
    "chrome",
];
/// How long a launched browser may take to print its DevTools endpoint.
const LAUNCH_TIMEOUT: Duration = Duration::from_secs(30);

/// Resolve the managed Chromium executable.
pub(crate) fn chromium_executable() -> Result<PathBuf, String> {
    if let Some(explicit) = std::env::var_os(CHROMIUM_ENV).filter(|v| !v.is_empty()) {
        let path = PathBuf::from(explicit);
        if path.is_file() {
            return Ok(path);
        }
        return Err(format!(
            "{CHROMIUM_ENV}={} does not name an executable file",
            path.display()
        ));
    }
    let path_var = std::env::var_os("PATH").unwrap_or_default();
    for dir in std::env::split_paths(&path_var) {
        for name in CHROMIUM_NAMES {
            let candidate = dir.join(name);
            if candidate.is_file() {
                return Ok(candidate);
            }
        }
    }
    Err(format!(
        "no Chromium executable found: set {CHROMIUM_ENV} or put one of {} on PATH",
        CHROMIUM_NAMES.join(", ")
    ))
}

/// A browser process this runtime spawned. Killed on drop.
pub(crate) struct Spawned {
    child: Child,
    /// Owned throwaway profile; removed after the process exits.
    _profile: Option<tempfile::TempDir>,
}

impl Spawned {
    /// Kill the process (and its process group) and reap it.
    pub(crate) async fn kill(&mut self) {
        #[cfg(unix)]
        if let Some(pid) = self.child.id() {
            // SAFETY: signalling our own child's process group.
            unsafe {
                libc::kill(-(pid as i32), libc::SIGKILL);
            }
        }
        let _ = self.child.start_kill();
        let _ = tokio::time::timeout(Duration::from_secs(5), self.child.wait()).await;
    }
}

impl Drop for Spawned {
    fn drop(&mut self) {
        #[cfg(unix)]
        if let Ok(None) = self.child.try_wait()
            && let Some(pid) = self.child.id()
        {
            // SAFETY: signalling our own child's process group.
            unsafe {
                libc::kill(-(pid as i32), libc::SIGKILL);
            }
        }
    }
}

/// Chromium-family executable basenames (`attach.ts` CHROMIUM_BROWSER_BASENAME).
fn is_chromium_family(executable: &Path) -> bool {
    let base = executable
        .file_name()
        .map(|n| n.to_string_lossy().to_ascii_lowercase())
        .unwrap_or_default();
    let base = base.trim_end_matches(".exe");
    [
        "chrome",
        "chromium",
        "chromium-browser",
        "google-chrome",
        "google-chrome-stable",
        "google-chrome-beta",
        "google-chrome-unstable",
        "brave",
        "brave-browser",
        "microsoft-edge",
        "microsoft-edge-stable",
        "msedge",
        "vivaldi",
        "opera",
    ]
    .contains(&base)
}

/// Arguments and profile for the managed Chromium (`launch.ts`).
pub(crate) fn managed_args(
    headless: bool,
    allow_file_access: bool,
) -> Result<(Vec<String>, tempfile::TempDir), String> {
    let profile = tempfile::Builder::new()
        .prefix("juto-browser-")
        .tempdir()
        .map_err(|e| format!("cannot create browser profile: {e}"))?;
    let mut args: Vec<String> = [
        "--no-first-run",
        "--no-default-browser-check",
        "--disable-background-networking",
        "--disable-sync",
        "--disable-default-apps",
        "--disable-component-update",
        "--disable-popup-blocking",
        "--disable-dev-shm-usage",
        "--password-store=basic",
        "--use-mock-keychain",
    ]
    .map(String::from)
    .to_vec();
    args.push(format!("--user-data-dir={}", profile.path().display()));
    if headless {
        args.push("--headless=new".into());
        args.push("--hide-scrollbars".into());
    }
    if allow_file_access {
        args.push("--allow-file-access-from-files".into());
    }
    args.push("about:blank".into());
    Ok((args, profile))
}

/// Arguments for an `app.path` spawn (`attach.ts` resolveSpawnArgs):
/// Chromium-family browsers get a persistent per-executable profile under
/// `profiles_dir` unless the caller passes `--user-data-dir`; other apps
/// (Electron) receive their arguments unchanged.
pub(crate) fn app_args(
    executable: &Path,
    user_args: &[String],
    cwd: &Path,
    profiles_dir: &Path,
) -> Vec<String> {
    if !is_chromium_family(executable) {
        return user_args.to_vec();
    }
    let mut args = Vec::with_capacity(user_args.len() + 5);
    let mut requested_profile: Option<String> = None;
    let mut iter = user_args.iter().peekable();
    while let Some(arg) = iter.next() {
        if arg == "--user-data-dir" {
            if let Some(next) = iter.next_if(|next| !next.starts_with("--")) {
                requested_profile = Some(next.clone());
            }
        } else if let Some(value) = arg.strip_prefix("--user-data-dir=") {
            requested_profile = Some(value.to_string());
        } else {
            args.push(arg.clone());
        }
    }
    if let Some(profile) = requested_profile {
        args.push(format!("--user-data-dir={}", cwd.join(profile).display()));
        return args;
    }
    for flag in [
        "--no-first-run",
        "--no-default-browser-check",
        "--use-mock-keychain",
    ] {
        if !user_args.iter().any(|arg| arg == flag) {
            args.push(flag.into());
        }
    }
    if !user_args
        .iter()
        .any(|arg| arg.starts_with("--password-store"))
    {
        args.push("--password-store=basic".into());
    }
    let base = executable
        .file_name()
        .map(|n| n.to_string_lossy().to_ascii_lowercase())
        .unwrap_or_default();
    let slug: String = base
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '-' })
        .collect();
    let hash = {
        use std::hash::{Hash as _, Hasher as _};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        executable.hash(&mut hasher);
        hasher.finish()
    };
    args.push(format!(
        "--user-data-dir={}",
        profiles_dir.join(format!("{slug}-{hash:016x}")).display()
    ));
    args
}

/// Spawn `executable` with remote debugging on an ephemeral port and return
/// it with its browser WebSocket endpoint.
pub(crate) async fn spawn(
    executable: &Path,
    args: &[String],
    profile: Option<tempfile::TempDir>,
) -> Result<(Spawned, String), String> {
    let mut command = Command::new(executable);
    command
        .arg("--remote-debugging-port=0")
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    #[cfg(unix)]
    command.process_group(0);
    let mut child = command
        .spawn()
        .map_err(|e| format!("cannot launch {}: {e}", executable.display()))?;
    let stderr = child.stderr.take().expect("stderr piped");
    let mut lines = BufReader::new(stderr).lines();
    let mut tail: Vec<String> = Vec::new();
    let endpoint = tokio::time::timeout(LAUNCH_TIMEOUT, async {
        while let Ok(Some(line)) = lines.next_line().await {
            if let Some(url) = line.strip_prefix("DevTools listening on ") {
                return Some(url.trim().to_string());
            }
            if tail.len() == 20 {
                tail.remove(0);
            }
            tail.push(line);
        }
        None
    })
    .await;
    let mut spawned = Spawned {
        child,
        _profile: profile,
    };
    match endpoint {
        Ok(Some(url)) => {
            // Keep draining stderr so the browser never blocks on a full pipe.
            tokio::spawn(async move { while let Ok(Some(_)) = lines.next_line().await {} });
            Ok((spawned, url))
        }
        Ok(None) => {
            spawned.kill().await;
            Err(format!(
                "{} exited before exposing DevTools:\n{}",
                executable.display(),
                tail.join("\n")
            ))
        }
        Err(_) => {
            spawned.kill().await;
            Err(format!(
                "{} did not expose DevTools within {}s:\n{}",
                executable.display(),
                LAUNCH_TIMEOUT.as_secs(),
                tail.join("\n")
            ))
        }
    }
}

/// Resolve an `app.cdp_url` (`ws://…` browser endpoint or `http://host:port`)
/// to a browser WebSocket URL.
pub(crate) async fn resolve_cdp_url(raw: &str) -> Result<String, String> {
    let trimmed = raw.trim();
    if trimmed.starts_with("ws://") {
        return Ok(trimmed.to_string());
    }
    if trimmed.starts_with("wss://") {
        return Err("wss:// DevTools endpoints are not supported; use ws:// or http://".into());
    }
    let base = if trimmed.starts_with("http://") || trimmed.starts_with("https://") {
        trimmed.trim_end_matches('/').to_string()
    } else {
        format!("http://{}", trimmed.trim_end_matches('/'))
    };
    let url = format!("{base}/json/version");
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(10))
        .build()
        .map_err(|e| e.to_string())?;
    let body: Value = client
        .get(&url)
        .send()
        .await
        .map_err(|e| format!("cannot reach {url}: {e}"))?
        .error_for_status()
        .map_err(|e| format!("{url}: {e}"))?
        .json()
        .await
        .map_err(|e| format!("{url}: invalid JSON: {e}"))?;
    body.get("webSocketDebuggerUrl")
        .and_then(Value::as_str)
        .map(str::to_string)
        .ok_or_else(|| format!("{url} has no webSocketDebuggerUrl"))
}
