//! Machine capacity measured from the host, and the agent memory budget derived
//! from it.
//!
//! The runtime's per-command grants, `juto-cli resources`, `scripts/agent-scope.sh`,
//! and the desktop compositor all derive their limits from [`MachineCapacity`],
//! so they agree on one measured number instead of carrying their own constants.

use std::fs;
use std::io;
use std::path::Path;

use serde::{Deserialize, Serialize};

/// The shared slice that agent scopes join. The runtime, the CLI and the desktop
/// all name it through this constant.
pub const AGENTS_SLICE: &str = "agents.slice";

/// Bytes in one gibibyte.
pub const GIB: u64 = 1024 * 1024 * 1024;

/// The memory a parallel `rustc` job is assumed to need. Cargo parallelism is
/// bounded by memory at one job per GiB of grant, and never beyond one per core.
pub const CARGO_JOB_MEMORY_BYTES: u64 = GIB;

/// Tasks (threads and processes) allowed per core in one scope.
const TASKS_PER_CORE: u32 = 128;

/// Measured host capacity. All byte values are exact integers derived from the
/// measurement; the derivation is documented in `docs/resources.md`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MachineCapacity {
    /// `MemTotal` from `/proc/meminfo`.
    pub mem_total_bytes: u64,
    /// Logical cores available to this process.
    pub cores: usize,
    /// Memory kept for the user's session: memory outside the agent slice,
    /// plus one tenth of total memory as host headroom.
    pub reserved_bytes: u64,
    /// Memory agents may use: total minus the reserve.
    pub budget_bytes: u64,
    /// Throttle line for the agent slice: nine tenths of the budget.
    pub slice_memory_high_bytes: u64,
    /// `TasksMax` for each scope.
    pub tasks_max: u32,
}

impl MachineCapacity {
    /// Measure this host. `agents_slice` is the slice whose memory is excluded
    /// from the session reserve, so running agents do not reduce their own budget.
    pub fn measure(agents_slice: &str) -> io::Result<Self> {
        let meminfo = fs::read_to_string("/proc/meminfo")?;
        let total = meminfo_bytes(&meminfo, "MemTotal")?;
        let session = match session_memory_outside(agents_slice) {
            Some(bytes) => bytes,
            None => total.saturating_sub(meminfo_bytes(&meminfo, "MemAvailable")?),
        };
        let cores = std::thread::available_parallelism()?.get();
        Self::from_measurements(total, cores, session)
    }

    /// Derive capacity from raw measurements. Fails when the reserve leaves
    /// nothing for agents.
    pub fn from_measurements(total: u64, cores: usize, session: u64) -> io::Result<Self> {
        let reserved = session.saturating_add(total / 10);
        let budget = total
            .checked_sub(reserved)
            .filter(|bytes| *bytes > 0)
            .ok_or_else(|| {
                io::Error::other("no memory remains for agents after the session reserve")
            })?;
        let cores = cores.max(1);
        Ok(Self {
            mem_total_bytes: total,
            cores,
            reserved_bytes: reserved,
            budget_bytes: budget,
            slice_memory_high_bytes: budget / 10 * 9,
            tasks_max: u32::try_from(cores.saturating_mul(TASKS_PER_CORE as usize))
                .unwrap_or(u32::MAX),
        })
    }

    /// Parallel jobs `cargo` may run for a command granted `memory_bytes`.
    pub fn cargo_jobs(&self, memory_bytes: u64) -> usize {
        let by_memory =
            usize::try_from(memory_bytes / CARGO_JOB_MEMORY_BYTES).unwrap_or(usize::MAX);
        self.cores.min(by_memory.max(1))
    }
}

fn meminfo_bytes(meminfo: &str, field: &str) -> io::Result<u64> {
    meminfo
        .lines()
        .find_map(|line| {
            let mut words = line.split_whitespace();
            (words.next()? == format!("{field}:")).then(|| words.next()?.parse::<u64>().ok())?
        })
        .and_then(|kib| kib.checked_mul(1024))
        .ok_or_else(|| io::Error::other(format!("missing or invalid {field} in /proc/meminfo")))
}

/// Memory of the user's cgroup subtree minus the agent slice, or `None` when
/// the process is not under a `user-<uid>.slice` (for example on a non-systemd host).
/// Both sides are measured as working set (see [`working_set`]), so page cache
/// the kernel can reclaim never reserves memory for the session.
fn session_memory_outside(agents_slice: &str) -> Option<u64> {
    let cgroup = fs::read_to_string("/proc/self/cgroup").ok()?;
    let relative = cgroup.lines().find_map(|line| line.strip_prefix("0::"))?;
    let mut dir = Path::new("/sys/fs/cgroup").join(relative.trim_start_matches('/'));
    loop {
        let name = dir.file_name()?.to_str()?;
        if let Some(uid) = name
            .strip_prefix("user-")
            .and_then(|rest| rest.strip_suffix(".slice"))
        {
            let total = working_set(&dir)?;
            let agents =
                working_set(&dir.join(format!("user@{uid}.service/{agents_slice}"))).unwrap_or(0);
            return Some(total.saturating_sub(agents));
        }
        if !dir.pop() {
            return None;
        }
    }
}

/// Working set of one cgroup: memory the kernel cannot drop without swapping or
/// killing (anonymous memory, kernel stacks, page tables). Page cache (`file`)
/// and shared memory are excluded because the kernel reclaims or re-reads them.
fn working_set(dir: &Path) -> Option<u64> {
    working_set_from_stat(&fs::read_to_string(dir.join("memory.stat")).ok()?)
}

/// Sum the working-set fields of a `memory.stat` file. `anon`, `kernel_stack`
/// and `pagetables` are required; `sec_pagetables` is absent on older kernels
/// and counts as zero.
fn working_set_from_stat(stat: &str) -> Option<u64> {
    let field = |name: &str| {
        stat.lines().find_map(|line| {
            let mut words = line.split_whitespace();
            (words.next()? == name).then(|| words.next()?.parse::<u64>().ok())?
        })
    };
    Some(
        field("anon")?
            + field("kernel_stack")?
            + field("pagetables")?
            + field("sec_pagetables").unwrap_or(0),
    )
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn budget_is_total_minus_session_and_headroom() {
        let capacity = MachineCapacity::from_measurements(30 * GIB, 24, 6 * GIB).unwrap();
        assert_eq!(capacity.reserved_bytes, 9 * GIB);
        assert_eq!(capacity.budget_bytes, 21 * GIB);
        assert_eq!(capacity.slice_memory_high_bytes, 21 * GIB / 10 * 9);
        assert_eq!(capacity.tasks_max, 24 * 128);
    }

    #[test]
    fn fails_when_no_memory_remains() {
        assert!(MachineCapacity::from_measurements(10 * GIB, 4, 10 * GIB).is_err());
    }

    #[test]
    fn cargo_jobs_are_bounded_by_cores_and_grant_memory() {
        let capacity = MachineCapacity::from_measurements(30 * GIB, 8, GIB).unwrap();
        assert_eq!(capacity.cargo_jobs(3 * GIB), 3);
        assert_eq!(capacity.cargo_jobs(20 * GIB), 8);
        assert_eq!(capacity.cargo_jobs(GIB / 2), 1);
    }

    #[test]
    fn meminfo_fields_parse_as_bytes() {
        let text = "MemTotal:       2048 kB\nMemAvailable:   1024 kB\n";
        assert_eq!(meminfo_bytes(text, "MemTotal").unwrap(), 2048 * 1024);
        assert!(meminfo_bytes(text, "SwapTotal").is_err());
    }

    #[test]
    fn session_reserve_ignores_page_cache() {
        // Shaped like the host's user slice: 6.8 GB of page cache must not count.
        let stat = "anon 20385832960\nfile 6847127552\nkernel_stack 37634048\npagetables 196128768\nshmem 515252224\n";
        assert_eq!(
            working_set_from_stat(stat),
            Some(20385832960 + 37634048 + 196128768)
        );
    }

    #[test]
    fn working_set_counts_secondary_page_tables_and_needs_anon() {
        let stat = "anon 100\nkernel_stack 10\npagetables 20\nsec_pagetables 5\n";
        assert_eq!(working_set_from_stat(stat), Some(135));
        assert_eq!(
            working_set_from_stat("kernel_stack 10\npagetables 20\n"),
            None
        );
    }

    #[test]
    fn host_measurement_is_consistent() {
        let capacity = MachineCapacity::measure("agents.slice").unwrap();
        assert!(capacity.cores >= 1);
        assert!(capacity.budget_bytes > 0);
        assert_eq!(
            capacity.budget_bytes + capacity.reserved_bytes,
            capacity.mem_total_bytes
        );
    }
}
