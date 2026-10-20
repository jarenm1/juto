//! Standalone host for a workspace's agent runtime.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};

const USAGE: &str = "Usage: juto-daemon [--workspace <directory>]\n\nHosts Juto sessions and agents independently of the GUI.\nThe GUI starts this executable as a systemd user service when needed.\n\nOptions:\n  --workspace <directory>  Workspace to serve (default: current directory)\n  --binary-id              Print the build identity of this executable and exit\n  --help                  Print this help\n  --version               Print the version";

#[tokio::main]
async fn main() -> Result<()> {
    let mut workspace = None;
    let mut args = std::env::args_os().skip(1);
    while let Some(arg) = args.next() {
        match arg.to_str() {
            Some("--help" | "-h") => {
                println!("{USAGE}");
                return Ok(());
            }
            Some("--version" | "-V") => {
                println!("juto-daemon {}", env!("CARGO_PKG_VERSION"));
                return Ok(());
            }
            Some("--binary-id") => {
                println!(
                    "{}",
                    juto_daemon::protocol::binary_id(Path::new("/proc/self/exe"))?
                );
                return Ok(());
            }
            Some("--workspace") => {
                workspace = Some(PathBuf::from(
                    args.next().context("--workspace requires a directory")?,
                ));
            }
            _ => bail!("unrecognized argument {:?}\n\n{USAGE}", arg),
        }
    }
    let workspace = workspace.map(Ok).unwrap_or_else(std::env::current_dir)?;
    juto_daemon::server::serve(workspace).await
}
