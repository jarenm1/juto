//! Process-owned HTTP/CONNECT allowlist proxy for network-isolated shell services.
//!
//! Only a filesystem Unix socket crosses the namespace boundary. The CLI helper
//! listens on loopback *inside* PrivateNetwork and forwards to that socket. This
//! is ordinary-tool confinement, not a hostile-code security boundary (notably,
//! the Nix daemon can perform work outside the namespace).
use std::{
    collections::BTreeMap,
    ffi::OsString,
    io::{self, Read, Write},
    net::{IpAddr, SocketAddr, TcpListener, TcpStream, ToSocketAddrs},
    os::unix::{
        fs::{DirBuilderExt, PermissionsExt},
        net::{UnixListener, UnixStream},
    },
    path::PathBuf,
    sync::{LazyLock, Mutex},
    thread,
    time::Duration,
};

use crate::tier::ExecutionTier;
use serde::{Deserialize, Serialize};

pub const CONFIG_KEY: &str = "network.allowlist";
pub const FORWARDER_ARGUMENT: &str = "__juto-network-forwarder";
const PACKAGE_HOSTS: &[&str] = &["crates.io", "static.crates.io", "index.crates.io"];

/// Exact DNS names only; no wildcard, URL, IP literal, or port syntax.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default, rename_all = "camelCase", deny_unknown_fields)]
pub struct NetworkConfig {
    /// Workspace-tier defaults. Untrusted commands never inherit this list.
    pub allowlist: Vec<String>,
}
impl Default for NetworkConfig {
    fn default() -> Self {
        Self {
            allowlist: PACKAGE_HOSTS
                .iter()
                .copied()
                .chain(["github.com"])
                .map(str::to_owned)
                .collect(),
        }
    }
}
impl NetworkConfig {
    pub fn validate(&self) -> Result<(), String> {
        for host in &self.allowlist {
            if !valid_host(host) {
                return Err(format!(
                    "{CONFIG_KEY} requires exact DNS hostnames, got {host:?}"
                ));
            }
        }
        Ok(())
    }
    pub fn policy(&self, tier: ExecutionTier) -> NetworkPolicy {
        let hosts = if tier == ExecutionTier::Untrusted {
            PACKAGE_HOSTS.iter().map(|s| (*s).to_owned()).collect()
        } else {
            self.allowlist.clone()
        };
        NetworkPolicy { hosts }
    }
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NetworkPolicy {
    hosts: Vec<String>,
}
impl Default for NetworkPolicy {
    fn default() -> Self {
        NetworkConfig::default().policy(ExecutionTier::Workspace)
    }
}
fn valid_host(host: &str) -> bool {
    host.len() <= 253
        && host.contains('.')
        && host.parse::<IpAddr>().is_err()
        && host.split('.').all(|label| {
            !label.is_empty()
                && label.len() <= 63
                && !label.starts_with('-')
                && !label.ends_with('-')
                && label
                    .bytes()
                    .all(|c| c.is_ascii_alphanumeric() || c == b'-')
        })
}

// Servers are owned by the runtime process, not by systemd-run's short-lived
// client or a command builder. Reuse one socket per policy for concurrent tools.
static SERVERS: LazyLock<Mutex<BTreeMap<Vec<String>, PathBuf>>> = LazyLock::new(Mutex::default);
impl NetworkPolicy {
    pub(crate) fn socket(&self) -> io::Result<PathBuf> {
        let mut hosts: Vec<_> = self.hosts.iter().map(|h| h.to_ascii_lowercase()).collect();
        hosts.sort();
        hosts.dedup();
        if hosts.iter().any(|h| !valid_host(h)) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("invalid {CONFIG_KEY}"),
            ));
        }
        let mut servers = SERVERS
            .lock()
            .map_err(|_| io::Error::other("network proxy registry poisoned"))?;
        if let Some(path) = servers.get(&hosts) {
            return Ok(path.clone());
        }
        let dir = std::env::temp_dir().join(format!("juto-proxy-{}", uuid::Uuid::now_v7()));
        std::fs::DirBuilder::new().mode(0o700).create(&dir)?;
        let path = dir.join("proxy.sock");
        let listener = UnixListener::bind(&path)?;
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600))?;
        let allowed = hosts.clone();
        thread::Builder::new()
            .name("juto-network-proxy".into())
            .spawn(move || {
                for incoming in listener.incoming() {
                    let Ok(stream) = incoming else { break };
                    let hosts = allowed.clone();
                    let _ = thread::Builder::new()
                        .name("juto-proxy-request".into())
                        .spawn(move || {
                            let _ = serve(stream, &hosts);
                        });
                }
            })?;
        servers.insert(hosts, path.clone());
        Ok(path)
    }
}

/// Resolve the shipped helper before starting a service; never fall back to an
/// unproxied command. An explicit path is useful for separately installed hosts.
pub(crate) fn helper() -> io::Result<PathBuf> {
    let explicit = std::env::var_os("JUTO_PROXY_HELPER").map(PathBuf::from);
    if let Some(path) = explicit {
        if path.is_absolute() && path.is_file() {
            return Ok(path);
        }
        return Err(io::Error::other(
            "JUTO_PROXY_HELPER must name an absolute juto-cli executable",
        ));
    }
    let exe = std::env::current_exe()?;
    if exe.file_name().is_some_and(|n| n == "juto-cli") {
        return Ok(exe);
    }
    if let Some(parent) = exe.parent() {
        let sibling = parent.join("juto-cli");
        if sibling.is_file() {
            return Ok(sibling);
        }
    }
    for dir in std::env::split_paths(&std::env::var_os("PATH").unwrap_or_default()) {
        let path = dir.join("juto-cli");
        if path.is_absolute() && path.is_file() {
            return Ok(path);
        }
    }
    Err(io::Error::other(
        "sandbox network helper unavailable; install juto-cli or set JUTO_PROXY_HELPER",
    ))
}

/// Hidden CLI entry point. Arguments are socket, executable, and its arguments.
/// The loopback listener is established before spawning the payload. The helper
/// and its children remain in the same systemd service for cancellation.
pub fn run_forwarder(mut args: impl Iterator<Item = OsString>) -> io::Result<i32> {
    let socket = PathBuf::from(
        args.next()
            .ok_or_else(|| io::Error::other("missing proxy socket"))?,
    );
    let program = args
        .next()
        .ok_or_else(|| io::Error::other("missing sandbox program"))?;
    // Validate the host end before launching a command whose downloads would hang.
    drop(UnixStream::connect(&socket)?);
    let listener = TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0))?;
    let proxy = format!("http://{}", listener.local_addr()?);
    thread::Builder::new()
        .name("juto-proxy-forwarder".into())
        .spawn(move || {
            for incoming in listener.incoming() {
                let Ok(tcp) = incoming else { break };
                let socket = socket.clone();
                let _ = thread::spawn(move || {
                    if let Ok(unix) = UnixStream::connect(socket) {
                        let _ = bridge(tcp, unix);
                    }
                });
            }
        })?;
    let mut command = std::process::Command::new(program);
    command.args(args);
    for key in [
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "http_proxy",
        "https_proxy",
        "ALL_PROXY",
        "all_proxy",
    ] {
        command.env(key, &proxy);
    }
    for key in ["NO_PROXY", "no_proxy"] {
        command.env(key, "");
    }
    let status = command.status()?;
    use std::os::unix::process::ExitStatusExt;
    Ok(status
        .code()
        .unwrap_or_else(|| 128 + status.signal().unwrap_or(1)))
}

fn bridge(mut tcp: TcpStream, mut unix: UnixStream) -> io::Result<()> {
    let mut tcp_read = tcp.try_clone()?;
    let mut unix_write = unix.try_clone()?;
    let upstream = thread::spawn(move || {
        let result = io::copy(&mut tcp_read, &mut unix_write);
        let _ = unix_write.shutdown(std::net::Shutdown::Write);
        result
    });
    let result = io::copy(&mut unix, &mut tcp);
    let _ = tcp.shutdown(std::net::Shutdown::Write);
    let _ = upstream.join();
    result.map(|_| ())
}

struct Target {
    host: String,
    port: u16,
    connect: bool,
    first_line: String,
}
fn target(line: &str) -> Result<Target, &'static str> {
    let parts: Vec<_> = line.split(' ').collect();
    if parts.len() != 3 || !matches!(parts[2], "HTTP/1.0" | "HTTP/1.1") {
        return Err("invalid proxy request");
    }
    let connect = parts[0] == "CONNECT";
    let (authority, path) = if connect {
        (parts[1], "")
    } else {
        let rest = parts[1]
            .strip_prefix("http://")
            .ok_or("HTTP URLs or HTTPS CONNECT required")?;
        rest.find(['/', '?'])
            .map_or((rest, "/"), |i| (&rest[..i], &rest[i..]))
    };
    let (host, port) = authority
        .split_once(':')
        .map_or((authority, if connect { "443" } else { "80" }), |v| v);
    let port = port.parse::<u16>().map_err(|_| "invalid proxy port")?;
    if !valid_host(host) || (connect && port != 443) || (!connect && port != 80) {
        return Err("only exact DNS hosts on HTTP port 80 or CONNECT port 443 are supported");
    }
    let path = if path.starts_with('?') {
        format!("/{path}")
    } else {
        path.to_owned()
    };
    Ok(Target {
        host: host.to_ascii_lowercase(),
        port,
        connect,
        first_line: format!("{} {} {}\r\n", parts[0], path, parts[2]),
    })
}
fn public_ip(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(ip) => {
            let [a, b, _, _] = ip.octets();
            !(ip.is_private()
                || ip.is_loopback()
                || ip.is_link_local()
                || ip.is_broadcast()
                || ip.is_documentation()
                || ip.is_unspecified()
                || ip.is_multicast()
                || a == 0
                || a >= 240
                || (a == 100 && (64..128).contains(&b))
                || (a == 198 && (b == 18 || b == 19)))
        }
        IpAddr::V6(ip) => {
            let segments = ip.segments();
            // Only global unicast; exclude mapped IPv4 and documentation ranges.
            (segments[0] & 0xe000) == 0x2000 && !(segments[0] == 0x2001 && segments[1] == 0xdb8)
        }
    }
}
fn reject(stream: &mut UnixStream, status: &str, message: &str) -> io::Result<()> {
    write!(
        stream,
        "HTTP/1.1 {status}\r\nConnection: close\r\nContent-Type: text/plain\r\nContent-Length: {}\r\n\r\n{message}",
        message.len()
    )
}
fn serve(mut client: UnixStream, hosts: &[String]) -> io::Result<()> {
    client.set_read_timeout(Some(Duration::from_secs(30)))?;
    client.set_write_timeout(Some(Duration::from_secs(30)))?;
    let mut header = Vec::new();
    while !header.ends_with(b"\r\n\r\n") {
        let mut byte = [0];
        if client.read(&mut byte)? == 0 {
            return Ok(());
        }
        header.push(byte[0]);
        if header.len() > 32768 {
            return reject(
                &mut client,
                "431 Request Header Fields Too Large",
                "proxy header too large",
            );
        }
    }
    let Ok(header) = std::str::from_utf8(&header) else {
        return reject(&mut client, "400 Bad Request", "invalid proxy header");
    };
    let mut lines = header.split("\r\n");
    let target = match target(lines.next().unwrap_or_default()) {
        Ok(target) => target,
        Err(message) => return reject(&mut client, "400 Bad Request", message),
    };
    if !hosts.contains(&target.host) {
        return reject(
            &mut client,
            "403 Forbidden",
            &format!(
                "network host '{}' denied by {CONFIG_KEY}; configure .juto/config.yml (untrusted tier is restricted to package mirrors)",
                target.host
            ),
        );
    }
    let addresses: Vec<SocketAddr> = match (target.host.as_str(), target.port).to_socket_addrs() {
        Ok(addresses) => addresses.collect(),
        Err(_) => {
            return reject(
                &mut client,
                "502 Bad Gateway",
                "allowed host DNS resolution failed",
            );
        }
    };
    if addresses.is_empty() || addresses.iter().any(|a| !public_ip(a.ip())) {
        return reject(
            &mut client,
            "403 Forbidden",
            &format!(
                "network host '{}' denied by {CONFIG_KEY}: non-public address",
                target.host
            ),
        );
    }
    let mut upstream = None;
    for addr in addresses {
        if let Ok(stream) = TcpStream::connect_timeout(&addr, Duration::from_secs(10)) {
            upstream = Some(stream);
            break;
        }
    }
    let Some(mut upstream) = upstream else {
        return reject(
            &mut client,
            "502 Bad Gateway",
            "allowed host connection failed",
        );
    };
    if target.connect {
        client.write_all(b"HTTP/1.1 200 Connection Established\r\n\r\n")?;
    } else {
        upstream.write_all(target.first_line.as_bytes())?;
        for line in lines.filter(|line| !line.is_empty()) {
            let Some((name, _)) = line.split_once(':') else {
                return reject(&mut client, "400 Bad Request", "invalid proxy header");
            };
            if [
                "host",
                "proxy-authorization",
                "proxy-connection",
                "connection",
            ]
            .iter()
            .any(|n| name.eq_ignore_ascii_case(n))
            {
                continue;
            }
            write!(upstream, "{line}\r\n")?;
        }
        write!(
            upstream,
            "Host: {}\r\nConnection: close\r\n\r\n",
            target.host
        )?;
    }
    client.set_read_timeout(None)?;
    client.set_write_timeout(None)?;
    bridge(upstream, client)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn tiers_do_not_allow_project_widening() {
        let config = NetworkConfig {
            allowlist: vec!["example.com".into()],
        };
        assert_eq!(
            config.policy(ExecutionTier::Workspace).hosts,
            ["example.com"]
        );
        assert_eq!(config.policy(ExecutionTier::Untrusted).hosts, PACKAGE_HOSTS);
        assert!(
            NetworkConfig::default()
                .allowlist
                .contains(&"github.com".into())
        );
    }
    #[test]
    fn rejects_ambiguous_authorities_and_ports() {
        for line in [
            "CONNECT user@github.com:443 HTTP/1.1",
            "CONNECT github.com:22 HTTP/1.1",
            "GET http://github.com.evil:8080/ HTTP/1.1",
            "GET http://127.0.0.1/ HTTP/1.1",
            "GET https://github.com/ HTTP/1.1",
        ] {
            assert!(target(line).is_err(), "{line}");
        }
        assert_eq!(
            target("CONNECT GitHub.com:443 HTTP/1.1").unwrap().host,
            "github.com"
        );
        assert_eq!(
            target("GET http://crates.io/api HTTP/1.1")
                .unwrap()
                .first_line,
            "GET /api HTTP/1.1\r\n"
        );
    }
    #[test]
    fn denied_response_names_host_and_setting() {
        let (mut client, server) = UnixStream::pair().unwrap();
        let worker = thread::spawn(move || serve(server, &["crates.io".into()]).unwrap());
        client
            .write_all(b"CONNECT denied.example:443 HTTP/1.1\r\nHost: denied.example\r\n\r\n")
            .unwrap();
        let mut response = String::new();
        client.read_to_string(&mut response).unwrap();
        worker.join().unwrap();
        assert!(response.contains("403 Forbidden"));
        assert!(response.contains("denied.example"));
        assert!(response.contains(CONFIG_KEY));
    }
    #[test]
    fn private_and_mapped_addresses_are_denied() {
        for ip in [
            "127.0.0.1",
            "10.0.0.1",
            "169.254.169.254",
            "100.64.0.1",
            "::1",
            "::ffff:127.0.0.1",
            "fc00::1",
            "2001:db8::1",
        ] {
            assert!(!public_ip(ip.parse().unwrap()), "{ip}");
        }
        assert!(public_ip("1.1.1.1".parse().unwrap()));
    }
}
