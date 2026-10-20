//! Cross-process admission for scoped commands.
//!
//! Memory is not compressible, so a heavy command starts only when its
//! reservation fits the measured agent budget next to the commands already
//! running. Every process that runs a scoped command (the runtime, `juto-cli`,
//! detached jobs) shares one ledger, so the budget holds across them.
//!
//! The ledger is `broker.json` under the broker directory, guarded by an
//! exclusive `flock` on `broker.lock`. Each entry records the owning process
//! (pid plus its start time, so a reused pid is not mistaken for the owner).
//! Entries whose owner has exited are pruned on every access, so a killed
//! process never holds a reservation.
//!
//! Admission rules, applied to the head of the queue only:
//!
//! - Queue order is priority (attended, verification, background, scheduled),
//!   then arrival. A higher-priority waiter is served first; the head is never
//!   skipped, so a large command is not starved by small ones.
//! - Reserved bytes of running commands plus the request must fit the budget.
//! - At most one GPU-flagged command runs at a time.
//! - No admission while the kernel reports a memory stall (PSI `full` in the
//!   last 10 s). Without PSI the ledger alone gates admission.
//!
//! A queued request is served by polling the ledger every
//! [`ADMISSION_POLL`]. Dropping the future that waits, or the [`Lease`], removes
//! its entry, so a cancelled command never blocks the queue.

use std::{
    fs::{self, File},
    io::{self, Write},
    path::{Path, PathBuf},
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use serde::{Deserialize, Serialize};
use thiserror::Error;
use uuid::Uuid;

/// Period between admission checks while a command waits in the queue.
pub const ADMISSION_POLL: Duration = Duration::from_millis(500);

/// The kernel's memory pressure file, read before each admission check.
const PSI_MEMORY: &str = "/proc/pressure/memory";

/// Time a request must wait to gain one priority level. A scheduled command that
/// has waited three steps is ordered like a fresh attended one, and an older
/// request wins that tie, so no request waits forever behind newer arrivals.
pub const AGING_STEP_MS: u128 = 60_000;

const LEDGER_FILE: &str = "broker.json";
const LOCK_FILE: &str = "broker.lock";

/// Who a command is serving; higher variants are admitted first.
#[derive(
    Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
#[serde(rename_all = "snake_case")]
pub enum Priority {
    Scheduled,
    #[default]
    Background,
    Verification,
    Attended,
}

impl std::str::FromStr for Priority {
    type Err = String;

    fn from_str(text: &str) -> Result<Self, Self::Err> {
        match text {
            "scheduled" => Ok(Self::Scheduled),
            "background" => Ok(Self::Background),
            "verification" => Ok(Self::Verification),
            "attended" => Ok(Self::Attended),
            other => Err(format!(
                "unknown priority `{other}` (expected attended, verification, background, or scheduled)"
            )),
        }
    }
}

/// What a scoped command asks the broker for.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Request {
    /// Bytes reserved while the command runs (its resolved memory ceiling).
    pub bytes: u64,
    /// Whether the command holds the exclusive GPU slot while it runs.
    pub gpu: bool,
    pub priority: Priority,
    /// Command class, for diagnostics in the ledger.
    pub label: String,
}

#[derive(Debug, Error)]
pub enum BrokerError {
    #[error("command reserves {bytes} bytes, more than the {budget}-byte agent budget")]
    TooLarge { bytes: u64, budget: u64 },
    #[error("cannot read this process's start time from /proc: {0}")]
    ProcessIdentity(String),
    #[error("admission ledger {path}: {source}")]
    Ledger {
        path: PathBuf,
        #[source]
        source: io::Error,
    },
}

/// One reservation or queued request in the ledger.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Entry {
    id: String,
    pid: u32,
    start_ticks: u64,
    bytes: u64,
    gpu: bool,
    priority: Priority,
    label: String,
    queued_ms: u128,
    running: bool,
}

/// The admission decision for one entry against the current ledger.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Decision {
    /// The entry is now running; its bytes are reserved.
    Admitted,
    /// The entry waits at this 1-based position in the queue.
    Queued { position: usize },
}

/// Decide whether `me` (already present in `entries`) may run now.
///
/// On admission the entry is marked running in place. Pure so the rules are
/// testable without a filesystem or a second process.
pub fn decide(
    entries: &mut [Entry],
    me: &str,
    budget: u64,
    pressure: bool,
    now_ms: u128,
) -> Decision {
    let mut waiting: Vec<&Entry> = entries.iter().filter(|e| !e.running).collect();
    waiting.sort_by(|a, b| {
        effective_level(b, now_ms)
            .cmp(&effective_level(a, now_ms))
            .then(a.queued_ms.cmp(&b.queued_ms))
            .then(a.id.cmp(&b.id))
    });
    let Some(index) = waiting.iter().position(|e| e.id == me) else {
        return Decision::Queued { position: 0 };
    };
    let position = index + 1;
    let head_is_me = index == 0;
    let mine = waiting[index].clone();

    let running_bytes: u64 = entries.iter().filter(|e| e.running).map(|e| e.bytes).sum();
    let gpu_busy = entries.iter().any(|e| e.running && e.gpu);
    let fits = running_bytes.saturating_add(mine.bytes) <= budget;
    let gpu_free = !(mine.gpu && gpu_busy);

    if head_is_me && fits && gpu_free && !pressure {
        if let Some(entry) = entries.iter_mut().find(|e| e.id == me) {
            entry.running = true;
        }
        Decision::Admitted
    } else {
        Decision::Queued { position }
    }
}

/// Queue level of a waiting entry at `now_ms`: its priority level plus one per
/// full [`AGING_STEP_MS`] waited, capped at the attended level.
fn effective_level(entry: &Entry, now_ms: u128) -> u128 {
    let base = match entry.priority {
        Priority::Scheduled => 0,
        Priority::Background => 1,
        Priority::Verification => 2,
        Priority::Attended => 3,
    };
    let steps = now_ms.saturating_sub(entry.queued_ms) / AGING_STEP_MS;
    (base + steps).min(3)
}

/// The process start time from `/proc/<pid>/stat`, in clock ticks since boot.
fn process_start_ticks(pid: u32) -> io::Result<u64> {
    let stat = fs::read_to_string(format!("/proc/{pid}/stat"))?;
    // The command name may contain spaces and parentheses; fields resume after
    // the last `)`. Field 3 (state) is the first token there, so starttime
    // (field 22) is the twentieth.
    let rest = stat
        .rsplit_once(')')
        .map(|(_, rest)| rest)
        .ok_or_else(|| io::Error::other("malformed /proc stat"))?;
    rest.split_whitespace()
        .nth(19)
        .and_then(|field| field.parse().ok())
        .ok_or_else(|| io::Error::other("missing starttime in /proc stat"))
}

fn is_alive(entry: &Entry) -> bool {
    process_start_ticks(entry.pid).is_ok_and(|ticks| ticks == entry.start_ticks)
}

/// Whether memory stall is rising: the kernel's `full avg10` exceeds its own
/// `full avg60`.
///
/// Reads the PSI file at `path` (normally `/proc/pressure/memory`). A stall that is
/// steady or falling does not block admission, because the ledger budget already
/// bounds what runs; a missing or unreadable file returns `false`.
fn memory_pressure(path: &Path) -> bool {
    let Ok(text) = fs::read_to_string(path) else {
        return false;
    };
    let Some(line) = text.lines().find(|line| line.starts_with("full")) else {
        return false;
    };
    let average = |key: &str| {
        line.split_whitespace()
            .find_map(|field| field.strip_prefix(key))
            .and_then(|value| value.parse::<f64>().ok())
    };
    match (average("avg10="), average("avg60=")) {
        (Some(ten), Some(sixty)) => ten > sixty,
        _ => false,
    }
}

/// The pressure file a broker reads by default. Unit tests get an empty path, so
/// `memory_pressure` reads nothing and reports no stall.
fn default_pressure_file() -> PathBuf {
    if cfg!(test) {
        PathBuf::new()
    } else {
        PathBuf::from(PSI_MEMORY)
    }
}

fn now_millis() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |elapsed| elapsed.as_millis())
}

/// The shared admission ledger in one directory.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Broker {
    dir: PathBuf,
    /// The file read before each admission check; see [`Broker::with_pressure_file`].
    pressure_file: PathBuf,
}

impl Broker {
    /// A broker whose ledger lives in `dir`. Unit tests read no kernel pressure
    /// file: a host stall from unrelated work would otherwise queue every desk.
    pub fn new(dir: impl Into<PathBuf>) -> Self {
        Self {
            dir: dir.into(),
            pressure_file: default_pressure_file(),
        }
    }

    /// Read memory pressure from `file` instead of the kernel's PSI file.
    pub fn with_pressure_file(mut self, file: impl Into<PathBuf>) -> Self {
        self.pressure_file = file.into();
        self
    }

    /// The directory holding the ledger and its lock file.
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// Join the queue for `request`. An admitted request holds its reservation
    /// at once; a queued one returns a [`Ticket`] to wait on.
    ///
    /// `budget` is the agent memory budget in bytes. Dropping the returned
    /// value before its reservation is held removes the queued entry.
    pub fn submit(&self, request: Request, budget: u64) -> Result<Submission, BrokerError> {
        if request.bytes > budget {
            return Err(BrokerError::TooLarge {
                bytes: request.bytes,
                budget,
            });
        }
        let pid = std::process::id();
        let start_ticks = process_start_ticks(pid)
            .map_err(|error| BrokerError::ProcessIdentity(error.to_string()))?;
        let id = Uuid::now_v7().to_string();
        let entry = Entry {
            id: id.clone(),
            pid,
            start_ticks,
            bytes: request.bytes,
            gpu: request.gpu,
            priority: request.priority,
            label: request.label,
            queued_ms: now_millis(),
            running: false,
        };
        // The lease exists before the first check, so dropping the ticket at any
        // point removes the queued entry.
        let lease = Lease {
            broker: self.clone(),
            id,
        };
        match self.decide_once(&entry, budget)? {
            Decision::Admitted => Ok(Submission::Admitted(lease)),
            Decision::Queued { position } => Ok(Submission::Queued(Ticket {
                lease,
                entry,
                budget,
                position,
            })),
        }
    }

    /// One admission check under the ledger lock, joining the queue if absent.
    fn decide_once(&self, entry: &Entry, budget: u64) -> Result<Decision, BrokerError> {
        let pressure = memory_pressure(&self.pressure_file);
        self.update(|entries| {
            if !entries.iter().any(|e| e.id == entry.id) {
                entries.push(entry.clone());
            }
            decide(entries, &entry.id, budget, pressure, now_millis())
        })
    }

    /// Run `change` on the ledger under the lock, pruning dead owners, then write it back.
    fn update<T>(&self, change: impl FnOnce(&mut Vec<Entry>) -> T) -> Result<T, BrokerError> {
        let ledger_error = |source: io::Error| BrokerError::Ledger {
            path: self.dir.join(LEDGER_FILE),
            source,
        };
        fs::create_dir_all(&self.dir).map_err(ledger_error)?;
        let lock = File::options()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(self.dir.join(LOCK_FILE))
            .map_err(ledger_error)?;
        lock.lock().map_err(ledger_error)?;

        let path = self.dir.join(LEDGER_FILE);
        let mut entries: Vec<Entry> = match fs::read_to_string(&path) {
            Ok(text) => serde_json::from_str(&text)
                .map_err(|error| ledger_error(io::Error::other(error)))?,
            Err(error) if error.kind() == io::ErrorKind::NotFound => Vec::new(),
            Err(error) => return Err(ledger_error(error)),
        };
        entries.retain(is_alive);
        let result = change(&mut entries);

        // Write then rename so a crash never leaves a torn ledger behind.
        let text = serde_json::to_string(&entries)
            .map_err(|error| ledger_error(io::Error::other(error)))?;
        let temp = self.dir.join(format!("{LEDGER_FILE}.tmp"));
        let mut file = File::create(&temp).map_err(ledger_error)?;
        file.write_all(text.as_bytes()).map_err(ledger_error)?;
        file.sync_all().map_err(ledger_error)?;
        fs::rename(&temp, &path).map_err(ledger_error)?;
        Ok(result)
    }
}

/// A reservation held by a running command. Dropping it releases the bytes.
#[derive(Debug)]
pub struct Lease {
    broker: Broker,
    id: String,
}

impl Drop for Lease {
    fn drop(&mut self) {
        let id = self.id.clone();
        if let Err(error) = self.broker.update(|entries| entries.retain(|e| e.id != id)) {
            // The owner has exited by the time anyone else reads the ledger, so
            // the entry is pruned on the next access; say so rather than hide it.
            eprintln!(
                "juto: could not release admission lease {}: {error}",
                self.id
            );
        }
    }
}

/// Outcome of [`Broker::submit`].
#[derive(Debug)]
pub enum Submission {
    /// The reservation is held; the command may start now.
    Admitted(Lease),
    /// The request is queued; wait on the ticket for admission.
    Queued(Ticket),
}

/// A queued request. Dropping it before admission removes its ledger entry,
/// so a cancelled caller never leaves a phantom reservation in the queue.
#[derive(Debug)]
pub struct Ticket {
    lease: Lease,
    entry: Entry,
    budget: u64,
    position: usize,
}

impl Ticket {
    /// 1-based queue position when the request joined the queue.
    pub fn position(&self) -> usize {
        self.position
    }

    /// Poll until admitted and return the lease. Cancel-safe: dropping the
    /// future removes the queued entry, so callers race it against cancellation.
    pub async fn admitted(self) -> Result<Lease, BrokerError> {
        loop {
            if matches!(
                self.lease.broker.decide_once(&self.entry, self.budget)?,
                Decision::Admitted
            ) {
                return Ok(self.lease);
            }
            tokio::time::sleep(ADMISSION_POLL).await;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(id: &str, bytes: u64, priority: Priority, queued_ms: u128) -> Entry {
        Entry {
            id: id.to_string(),
            pid: 1,
            start_ticks: 1,
            bytes,
            gpu: false,
            priority,
            label: "test".to_string(),
            queued_ms,
            running: false,
        }
    }

    #[test]
    fn admits_head_that_fits_and_queues_the_rest() {
        let mut entries = vec![entry("a", 60, Priority::Background, 1)];
        assert_eq!(decide(&mut entries, "a", 100, false, 0), Decision::Admitted);
        entries.push(entry("b", 60, Priority::Background, 2));
        // 60 is reserved, so 60 more does not fit a 100-byte budget.
        assert_eq!(
            decide(&mut entries, "b", 100, false, 0),
            Decision::Queued { position: 1 }
        );
        entries.push(entry("c", 30, Priority::Background, 3));
        // `c` fits, but `b` is ahead of it in the queue and is not skipped.
        assert_eq!(
            decide(&mut entries, "c", 100, false, 0),
            Decision::Queued { position: 2 }
        );
    }

    #[test]
    fn higher_priority_is_served_before_earlier_arrivals() {
        let mut entries = vec![
            entry("early-background", 10, Priority::Background, 1),
            entry("late-attended", 10, Priority::Attended, 9),
        ];
        assert_eq!(
            decide(&mut entries, "early-background", 100, false, 9),
            Decision::Queued { position: 2 }
        );
        assert_eq!(
            decide(&mut entries, "late-attended", 100, false, 9),
            Decision::Admitted
        );
    }

    #[test]
    fn aging_serves_a_long_waiting_command_ahead_of_fresh_higher_priority_ones() {
        let now = 3 * AGING_STEP_MS;
        let mut entries = vec![
            entry("old-scheduled", 90, Priority::Scheduled, 0),
            entry("new-attended", 10, Priority::Attended, now),
        ];
        // Three steps lift the scheduled request to the attended level; it
        // queued first, so it is the head and the attended request waits.
        assert_eq!(
            decide(&mut entries, "new-attended", 100, false, now),
            Decision::Queued { position: 2 }
        );
        assert_eq!(
            decide(&mut entries, "old-scheduled", 100, false, now),
            Decision::Admitted
        );
    }

    #[test]
    fn fresh_scheduled_command_still_waits_behind_attended() {
        let mut entries = vec![
            entry("scheduled", 10, Priority::Scheduled, 0),
            entry("attended", 10, Priority::Attended, 0),
        ];
        assert_eq!(
            decide(&mut entries, "scheduled", 100, false, 0),
            Decision::Queued { position: 2 }
        );
        assert_eq!(
            decide(&mut entries, "attended", 100, false, 0),
            Decision::Admitted
        );
    }

    #[test]
    fn same_priority_is_first_come_first_served() {
        let mut entries = vec![
            entry("second", 10, Priority::Background, 5),
            entry("first", 10, Priority::Background, 2),
        ];
        assert_eq!(
            decide(&mut entries, "second", 100, false, 5),
            Decision::Queued { position: 2 }
        );
        assert_eq!(
            decide(&mut entries, "first", 100, false, 5),
            Decision::Admitted
        );
    }

    #[test]
    fn gpu_commands_are_exclusive() {
        let mut entries = vec![entry("gpu-one", 10, Priority::Background, 1)];
        entries[0].gpu = true;
        assert_eq!(
            decide(&mut entries, "gpu-one", 100, false, 1),
            Decision::Admitted
        );
        let mut second = entry("gpu-two", 10, Priority::Background, 2);
        second.gpu = true;
        entries.push(second);
        assert_eq!(
            decide(&mut entries, "gpu-two", 100, false, 2),
            Decision::Queued { position: 1 }
        );
    }

    #[test]
    fn rising_memory_stall_blocks_admission() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("pressure");
        std::fs::write(
            &file,
            "some avg10=0.40 avg60=0.10 avg300=0.10 total=1\nfull avg10=0.40 avg60=0.10 avg300=0.10 total=1\n",
        )
        .unwrap();
        assert!(memory_pressure(&file));
        // A host's steady baseline stall (avg10 below avg60) does not block.
        std::fs::write(&file, "full avg10=0.19 avg60=0.22 avg300=0.27 total=1\n").unwrap();
        assert!(!memory_pressure(&file));
        assert!(!memory_pressure(&dir.path().join("missing")));

        let mut entries = vec![entry("a", 10, Priority::Background, 1)];
        assert_eq!(
            decide(&mut entries, "a", 100, true, 1),
            Decision::Queued { position: 1 }
        );
        assert_eq!(decide(&mut entries, "a", 100, false, 1), Decision::Admitted);
    }

    #[test]
    fn own_process_identity_is_readable_and_stable() {
        let pid = std::process::id();
        let first = process_start_ticks(pid).expect("own start time");
        assert_eq!(first, process_start_ticks(pid).expect("own start time"));
    }

    #[test]
    fn parses_priorities_and_rejects_unknown_names() {
        assert_eq!("attended".parse::<Priority>(), Ok(Priority::Attended));
        assert!("urgent".parse::<Priority>().is_err());
    }

    /// A broker over a fresh ledger that reads a quiet pressure file.
    fn quiet_broker(dir: &tempfile::TempDir) -> Broker {
        let pressure = dir.path().join("pressure");
        std::fs::write(&pressure, "some avg10=0.00\nfull avg10=0.00\n").unwrap();
        Broker::new(dir.path().join("broker")).with_pressure_file(pressure)
    }

    fn request(bytes: u64) -> Request {
        Request {
            bytes,
            gpu: false,
            priority: Priority::Background,
            label: "test".to_string(),
        }
    }

    #[tokio::test]
    async fn queued_ticket_is_admitted_once_the_reservation_is_released() {
        let dir = tempfile::tempdir().unwrap();
        let broker = quiet_broker(&dir);
        let Submission::Admitted(first) = broker.submit(request(60), 100).unwrap() else {
            panic!("an empty ledger admits the first request");
        };
        let Submission::Queued(ticket) = broker.submit(request(60), 100).unwrap() else {
            panic!("60 + 60 bytes exceed a 100-byte budget");
        };
        assert_eq!(ticket.position(), 1);
        drop(first);
        let lease = tokio::time::timeout(Duration::from_secs(5), ticket.admitted())
            .await
            .expect("admission follows the release")
            .unwrap();
        drop(lease);
    }

    #[test]
    fn dropped_ticket_leaves_no_queue_entry() {
        let dir = tempfile::tempdir().unwrap();
        let broker = quiet_broker(&dir);
        let Submission::Admitted(_held) = broker.submit(request(60), 100).unwrap() else {
            panic!("an empty ledger admits the first request");
        };
        let Submission::Queued(ticket) = broker.submit(request(60), 100).unwrap() else {
            panic!("60 + 60 bytes exceed a 100-byte budget");
        };
        drop(ticket);
        // The dropped request no longer sits ahead of a new one.
        let Submission::Queued(next) = broker.submit(request(60), 100).unwrap() else {
            panic!("the held reservation still blocks the new request");
        };
        assert_eq!(next.position(), 1);
    }
}
