//! Headless per-workspace host for the Juto agent runtime.
//!
//! `server` runs a [`juto_runtime::Runtime`] for one workspace and serves
//! [`protocol`] over a local Unix socket. `client` connects to that socket,
//! starting the daemon as a systemd `--user` unit when it is not running. The
//! GUI is a client only; the runtime, credentials, and desktop environments live
//! in the daemon, so runs outlive the window.

pub mod client;
pub mod protocol;
pub mod server;
