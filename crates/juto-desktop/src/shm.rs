use crate::error::DesktopError;
use parking_lot::Mutex;
use std::fs::File;
use std::os::fd::{FromRawFd, RawFd};
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::{Duration, Instant};

/// Number of buffers in an attachment-private triple-buffered pool.
pub const POOL_SLOT_COUNT: usize = 3;
/// Bytes at the start of every pool reserved for its [`ShmControl`] block.
const CONTROL_BYTES: usize = 4096;
/// `"JUTOSHM1"`, identifying a juto frame pool in its control block.
const POOL_MAGIC: u64 = u64::from_le_bytes(*b"JUTOSHM1");
/// Layout revision of the pool; readers refuse any other value.
const POOL_LAYOUT_VERSION: u32 = 1;

/// Control block at offset zero of every pool.
///
/// `publish_seq` is the futex word: the producer increments and wakes it once
/// per published frame, and readers sleep on it instead of polling.
#[repr(C)]
struct ShmControl {
    magic: u64,
    layout_version: u32,
    slot_count: u32,
    slot_size: u64,
    publish_seq: AtomicU32,
}

/// Outcome of [`ShmReader::wait_for_frame`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameWait {
    /// A frame was published; carries the new publish sequence.
    Published(u32),
    /// No frame was published before the timeout.
    TimedOut,
}

/// Byte offset of `slot` in a pool whose slots are `slot_size` bytes.
fn slot_offset(slot: usize, slot_size: usize) -> Option<usize> {
    slot.checked_mul(slot_size)?.checked_add(CONTROL_BYTES)
}

/// Checks the control block of a mapped pool and returns its slot size.
fn validate_layout(base: *const u8, size: usize) -> Result<usize, DesktopError> {
    if size < CONTROL_BYTES {
        return Err(DesktopError::Shm(format!(
            "pool of {size} bytes has no control block"
        )));
    }
    // SAFETY: `size` covers the control block, so `base` is readable for it.
    let control = unsafe { &*(base as *const ShmControl) };
    let slot_size = usize::try_from(control.slot_size)
        .map_err(|_| DesktopError::Shm("pool slot size out of range".to_string()))?;
    let expected = slot_size
        .checked_mul(POOL_SLOT_COUNT)
        .and_then(|slots| slots.checked_add(CONTROL_BYTES));
    let valid = control.magic == POOL_MAGIC
        && control.layout_version == POOL_LAYOUT_VERSION
        && control.slot_count as usize == POOL_SLOT_COUNT
        && slot_size >= std::mem::size_of::<ShmSlotHeader>()
        && expected == Some(size);
    if !valid {
        return Err(DesktopError::Shm(format!(
            "not a juto frame pool of {size} bytes"
        )));
    }
    Ok(slot_size)
}

/// Wakes every waiter on `word`. The futex is shared, not process-private:
/// readers wait through their own mapping of the same memfd.
fn futex_wake_all(word: &AtomicU32) {
    futex_syscall(word, libc::FUTEX_WAKE, i32::MAX as u32, std::ptr::null());
}

/// Sleeps while `word == expected`, for at most `timeout`.
fn futex_wait(word: &AtomicU32, expected: u32, timeout: Duration) -> std::io::Result<()> {
    let spec = libc::timespec {
        tv_sec: timeout.as_secs() as libc::time_t,
        tv_nsec: libc::c_long::from(timeout.subsec_nanos()),
    };
    if futex_syscall(word, libc::FUTEX_WAIT, expected, &spec) == 0 {
        Ok(())
    } else {
        Err(std::io::Error::last_os_error())
    }
}

fn futex_syscall(
    word: &AtomicU32,
    op: libc::c_int,
    val: u32,
    timeout: *const libc::timespec,
) -> libc::c_long {
    // SAFETY: `word` is a live, aligned futex word. The kernel ignores the
    // arguments an operation does not use, and they are passed as zero.
    unsafe {
        libc::syscall(
            libc::SYS_futex,
            word as *const AtomicU32 as libc::c_long,
            libc::c_long::from(op),
            libc::c_long::from(val),
            timeout as libc::c_long,
            0 as libc::c_long,
            0 as libc::c_long,
        )
    }
}

/// Header metadata prepended to each shared-memory buffer slot.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ShmSlotHeader {
    pub frame_id: u64,
    pub generation: u64,
    pub timestamp_us: u64,
    pub width: u32,
    pub height: u32,
    pub stride: u32,
    pub format: u32, // FourCC, e.g. DRM_FORMAT_ARGB8888
    pub size_bytes: u32,
}

/// A single frame buffer inside an allocated memfd pool.
#[derive(Debug)]
pub struct ShmFrame {
    pub slot: usize,
    pub header: ShmSlotHeader,
    pub offset: usize,
    pub length: usize,
}

/// An attachment-private shared-memory frame pool backed by a sealed `memfd`.
pub struct ShmFramePool {
    pool_id: String,
    fd: RawFd,
    _file: Arc<File>,
    size: usize,
    slot_size: usize,
    mmap_ptr: *mut u8,
    active_slot: Mutex<usize>,
    in_flight_readers: Mutex<[usize; POOL_SLOT_COUNT]>,
}

unsafe impl Send for ShmFramePool {}
unsafe impl Sync for ShmFramePool {}

impl ShmFramePool {
    /// Creates a new sealed memfd frame pool with 3 slots sized for the given dimensions.
    pub fn create(
        pool_id: impl Into<String>,
        width: u32,
        height: u32,
    ) -> Result<Self, DesktopError> {
        let pool = Self::create_unsealed(pool_id, width, height)?;
        pool.seal()?;
        Ok(pool)
    }

    /// Creates a pool whose producer mapping is writable but not yet sealed.
    ///
    /// A compositor worker maps the pool writable and then seals it, so the
    /// supervisor and every viewer can only map it read-only afterwards.
    pub fn create_unsealed(
        pool_id: impl Into<String>,
        width: u32,
        height: u32,
    ) -> Result<Self, DesktopError> {
        let pool_id = pool_id.into();
        let stride = width * 4;
        let pixel_bytes = (stride * height) as usize;
        let header_bytes = std::mem::size_of::<ShmSlotHeader>();
        // Align slot size to page boundary (4096 bytes)
        let slot_size = ((header_bytes + pixel_bytes) + 4095) & !4095;
        let total_size = CONTROL_BYTES + slot_size * POOL_SLOT_COUNT;

        let name = std::ffi::CString::new(format!("juto-shm-{}", pool_id))
            .map_err(|e| DesktopError::Shm(e.to_string()))?;

        // 1. Create anonymous sealed memfd
        let fd = unsafe {
            libc::memfd_create(name.as_ptr(), libc::MFD_CLOEXEC | libc::MFD_ALLOW_SEALING)
        };
        if fd < 0 {
            return Err(DesktopError::Shm(format!(
                "memfd_create failed: {}",
                std::io::Error::last_os_error()
            )));
        }

        // 2. Set file size
        let truncate_res = unsafe { libc::ftruncate(fd, total_size as libc::off_t) };
        if truncate_res != 0 {
            unsafe { libc::close(fd) };
            return Err(DesktopError::Shm(format!(
                "ftruncate failed: {}",
                std::io::Error::last_os_error()
            )));
        }

        // 3. Create writable producer mmap mapping *before* applying future write seals
        let mmap_ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                total_size,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                fd,
                0,
            )
        };
        if mmap_ptr == libc::MAP_FAILED {
            unsafe { libc::close(fd) };
            return Err(DesktopError::Shm(format!(
                "mmap failed: {}",
                std::io::Error::last_os_error()
            )));
        }
        // Lay down the control block before any reader can map the pool.
        unsafe {
            (mmap_ptr as *mut ShmControl).write(ShmControl {
                magic: POOL_MAGIC,
                layout_version: POOL_LAYOUT_VERSION,
                slot_count: POOL_SLOT_COUNT as u32,
                slot_size: slot_size as u64,
                publish_seq: AtomicU32::new(0),
            });
        }

        let file = unsafe { Arc::new(File::from_raw_fd(fd)) };

        Ok(Self {
            pool_id,
            fd,
            _file: file,
            size: total_size,
            slot_size,
            mmap_ptr: mmap_ptr as *mut u8,
            active_slot: Mutex::new(0),
            in_flight_readers: Mutex::new([0; POOL_SLOT_COUNT]),
        })
    }

    /// Applies the write and size seals.
    ///
    /// `F_SEAL_GROW | F_SEAL_SHRINK | F_SEAL_FUTURE_WRITE | F_SEAL_SEAL` denies
    /// future writable mappings to anyone receiving the descriptor, while the
    /// producer's existing mapping stays writable. `F_SEAL_WRITE` is deliberately
    /// not used: it would forbid the reusable producer mapping.
    pub fn seal(&self) -> Result<(), DesktopError> {
        const F_SEAL_SEAL: libc::c_int = 0x0001;
        const F_SEAL_SHRINK: libc::c_int = 0x0002;
        const F_SEAL_GROW: libc::c_int = 0x0004;
        const F_SEAL_FUTURE_WRITE: libc::c_int = 0x0010;

        let seals = F_SEAL_SEAL | F_SEAL_SHRINK | F_SEAL_GROW | F_SEAL_FUTURE_WRITE;
        let result = unsafe { libc::fcntl(self.fd, libc::F_ADD_SEALS, seals) };
        if result != 0 {
            return Err(DesktopError::Shm(format!(
                "F_ADD_SEALS failed: {}",
                std::io::Error::last_os_error()
            )));
        }
        Ok(())
    }

    pub fn pool_id(&self) -> &str {
        &self.pool_id
    }

    pub fn raw_fd(&self) -> RawFd {
        self.fd
    }

    pub fn size(&self) -> usize {
        self.size
    }

    /// Byte size of one slot (header plus pixel payload, page aligned).
    pub fn slot_size(&self) -> usize {
        self.slot_size
    }

    fn control(&self) -> &ShmControl {
        // SAFETY: the mapping spans the control block written by `create_unsealed`.
        unsafe { &*(self.mmap_ptr as *const ShmControl) }
    }

    /// Bumps the publish sequence and wakes readers blocked in `wait_for_frame`.
    fn publish(&self) {
        let word = &self.control().publish_seq;
        word.fetch_add(1, Ordering::Release);
        futex_wake_all(word);
    }

    /// Renders or copies new desktop content into the next available slot.
    pub fn write_frame(
        &self,
        frame_id: u64,
        generation: u64,
        timestamp_us: u64,
        width: u32,
        height: u32,
        pixels: &[u8],
    ) -> Result<ShmFrame, DesktopError> {
        let mut active = self.active_slot.lock();
        let readers = self.in_flight_readers.lock();

        // Next free slot. Never the slot holding the latest published frame:
        // a reader may still be copying it.
        let target_slot = (1..POOL_SLOT_COUNT)
            .map(|step| (*active + step) % POOL_SLOT_COUNT)
            .find(|&slot| readers[slot] == 0);
        let Some(target_slot) = target_slot else {
            // All other slots held by readers: backpressure skip (drop frame export, never block seat)
            return Err(DesktopError::LimitExceeded(
                "all SHM pool slots held by slow readers; dropped frame".to_string(),
            ));
        };

        *active = target_slot;
        let offset = CONTROL_BYTES + target_slot * self.slot_size;
        let header = ShmSlotHeader {
            frame_id,
            generation,
            timestamp_us,
            width,
            height,
            stride: width * 4,
            format: 0x34325241, // ARGB8888 FourCC
            size_bytes: pixels.len() as u32,
        };

        unsafe {
            let slot_ptr = self.mmap_ptr.add(offset);
            // Pixels first. Readers pick the slot with the highest frame_id, so the
            // header (which carries frame_id) must become visible only after its
            // pixels are complete. The release fence orders the two copies.
            let pixel_ptr = slot_ptr.add(std::mem::size_of::<ShmSlotHeader>());
            let copy_len = pixels
                .len()
                .min(self.slot_size - std::mem::size_of::<ShmSlotHeader>());
            std::ptr::copy_nonoverlapping(pixels.as_ptr(), pixel_ptr, copy_len);
            std::sync::atomic::fence(Ordering::Release);
            std::ptr::copy_nonoverlapping(
                &header as *const ShmSlotHeader as *const u8,
                slot_ptr,
                std::mem::size_of::<ShmSlotHeader>(),
            );
        }
        // Publish only after the pixels are written, so readers never see a torn frame.
        self.publish();

        Ok(ShmFrame {
            slot: target_slot,
            header,
            offset,
            length: self.slot_size,
        })
    }

    /// Acknowledges release from a reader for a slot.
    pub fn release_slot(&self, slot: usize) {
        if slot < POOL_SLOT_COUNT {
            let mut readers = self.in_flight_readers.lock();
            if readers[slot] > 0 {
                readers[slot] -= 1;
            }
        }
    }

    /// Marks slot acquired by a reader.
    pub fn acquire_slot(&self, slot: usize) {
        if slot < POOL_SLOT_COUNT {
            let mut readers = self.in_flight_readers.lock();
            readers[slot] += 1;
        }
    }

    /// Describes the most recently written frame, if the pool has been written.
    pub fn latest_frame(&self) -> Option<ShmFrame> {
        let slot = *self.active_slot.lock();
        let offset = CONTROL_BYTES + slot * self.slot_size;
        let header = unsafe { *(self.mmap_ptr.add(offset) as *const ShmSlotHeader) };
        if header.size_bytes == 0 {
            return None;
        }
        Some(ShmFrame {
            slot,
            header,
            offset,
            length: self.slot_size,
        })
    }

    /// Copies the pixel payload of `slot` into `out`, replacing its contents.
    ///
    /// Callers that must not race the producer should bracket this with
    /// [`ShmFramePool::acquire_slot`] and [`ShmFramePool::release_slot`].
    pub fn read_pixels(
        &self,
        slot: usize,
        out: &mut Vec<u8>,
    ) -> Result<ShmSlotHeader, DesktopError> {
        if slot >= POOL_SLOT_COUNT {
            return Err(DesktopError::Shm(format!("slot {slot} out of range")));
        }
        let offset = CONTROL_BYTES + slot * self.slot_size;
        let header = unsafe { *(self.mmap_ptr.add(offset) as *const ShmSlotHeader) };
        let len = header.size_bytes as usize;
        if len > self.slot_size - std::mem::size_of::<ShmSlotHeader>() {
            return Err(DesktopError::Shm(format!(
                "slot {slot} header claims {len} bytes, larger than the slot payload"
            )));
        }
        out.clear();
        out.resize(len, 0);
        unsafe {
            std::ptr::copy_nonoverlapping(
                self.mmap_ptr
                    .add(offset + std::mem::size_of::<ShmSlotHeader>()),
                out.as_mut_ptr(),
                len,
            );
        }
        Ok(header)
    }
}

impl Drop for ShmFramePool {
    fn drop(&mut self) {
        unsafe {
            if !self.mmap_ptr.is_null() {
                libc::munmap(self.mmap_ptr as *mut libc::c_void, self.size);
            }
        }
    }
}

/// Helper wrapper for reading a passed sealed memfd pool in a local viewer process.
pub struct ShmReader {
    fd: RawFd,
    size: usize,
    slot_size: usize,
    mmap_ptr: *const u8,
    pool_id: String,
    /// Notified when a slot is reserved or released, so a producer in another
    /// process can honour the reservation.
    reservation: Option<Box<dyn Fn(usize, bool) + Send + Sync>>,
}

unsafe impl Send for ShmReader {}
unsafe impl Sync for ShmReader {}

impl ShmReader {
    pub fn attach(fd: RawFd, size: usize) -> Result<Self, DesktopError> {
        Self::attach_named(fd, size, String::new(), None)
    }

    /// Attaches with an identity and a reservation callback.
    pub fn attach_named(
        fd: RawFd,
        size: usize,
        pool_id: String,
        reservation: Option<Box<dyn Fn(usize, bool) + Send + Sync>>,
    ) -> Result<Self, DesktopError> {
        let mmap_ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                size,
                libc::PROT_READ,
                libc::MAP_SHARED,
                fd,
                0,
            )
        };
        if mmap_ptr == libc::MAP_FAILED {
            return Err(DesktopError::Shm(format!(
                "reader mmap failed: {}",
                std::io::Error::last_os_error()
            )));
        }

        let slot_size = match validate_layout(mmap_ptr as *const u8, size) {
            Ok(slot_size) => slot_size,
            Err(err) => {
                unsafe { libc::munmap(mmap_ptr, size) };
                return Err(err);
            }
        };

        Ok(Self {
            fd,
            size,
            slot_size,
            mmap_ptr: mmap_ptr as *const u8,
            pool_id,
            reservation,
        })
    }

    /// The pool's identity.
    pub fn pool_id(&self) -> &str {
        &self.pool_id
    }

    /// The descriptor backing the mapping (read-only for viewers).
    pub fn raw_fd(&self) -> RawFd {
        self.fd
    }

    /// Total mapped size in bytes.
    pub fn size(&self) -> usize {
        self.size
    }

    /// Reserves a slot so a producer does not overwrite it while it is read.
    pub fn acquire_slot(&self, slot: usize) {
        if let Some(reservation) = &self.reservation {
            reservation(slot, true);
        }
    }

    /// Releases a previously reserved slot.
    pub fn release_slot(&self, slot: usize) {
        if let Some(reservation) = &self.reservation {
            reservation(slot, false);
        }
    }

    /// Byte size of one slot in the mapped pool.
    pub fn slot_size(&self) -> usize {
        self.slot_size
    }

    /// The producer's publish sequence; it changes once per published frame.
    pub fn publish_seq(&self) -> u32 {
        self.control().publish_seq.load(Ordering::Acquire)
    }

    /// Blocks until the producer publishes a frame whose sequence differs from
    /// `after`, or until `timeout` elapses. Returns at once when it already
    /// differs, so starting from `0` also returns a frame published earlier.
    pub fn wait_for_frame(&self, after: u32, timeout: Duration) -> Result<FrameWait, DesktopError> {
        let deadline = Instant::now() + timeout;
        let word = &self.control().publish_seq;
        loop {
            let current = word.load(Ordering::Acquire);
            if current != after {
                return Ok(FrameWait::Published(current));
            }
            let Some(remaining) = deadline
                .checked_duration_since(Instant::now())
                .filter(|left| !left.is_zero())
            else {
                return Ok(FrameWait::TimedOut);
            };
            match futex_wait(word, after, remaining) {
                Ok(()) => {}
                Err(err)
                    if matches!(
                        err.raw_os_error(),
                        Some(libc::EAGAIN | libc::EINTR | libc::ETIMEDOUT)
                    ) => {}
                Err(err) => return Err(DesktopError::Shm(format!("futex wait failed: {err}"))),
            }
        }
    }

    fn control(&self) -> &ShmControl {
        // SAFETY: `attach_named` validated the control block at this base.
        unsafe { &*(self.mmap_ptr as *const ShmControl) }
    }

    /// The most recently written slot, chosen by the highest frame id.
    ///
    /// A reader has no producer state, so it scans the fixed slot set rather
    /// than trusting a shared cursor.
    pub fn latest_frame(&self) -> Result<Option<ShmFrame>, DesktopError> {
        let mut best: Option<ShmFrame> = None;
        for slot in 0..POOL_SLOT_COUNT {
            let (offset, header) = self.checked_slot(slot, self.slot_size)?;
            if header.size_bytes == 0 {
                continue;
            }
            let candidate = ShmFrame {
                slot,
                header,
                offset,
                length: self.slot_size,
            };
            if best
                .as_ref()
                .map(|current| candidate.header.frame_id > current.header.frame_id)
                .unwrap_or(true)
            {
                best = Some(candidate);
            }
        }
        // Pairs with the producer's release fence: the header was observed
        // above, so the pixels of the chosen slot are complete.
        std::sync::atomic::fence(Ordering::Acquire);
        Ok(best)
    }

    /// Validates slot `slot` of `slot_size` bytes and returns its byte offset and
    /// header. The producer writes the header into shared memory, so `size_bytes`
    /// is untrusted until the payload is checked to lie inside the slot.
    fn checked_slot(
        &self,
        slot: usize,
        slot_size: usize,
    ) -> Result<(usize, ShmSlotHeader), DesktopError> {
        let header_size = std::mem::size_of::<ShmSlotHeader>();
        let offset = slot_offset(slot, slot_size)
            .filter(|offset| {
                offset
                    .checked_add(slot_size)
                    .is_some_and(|end| end <= self.size)
            })
            .ok_or_else(|| DesktopError::Shm("slot offset out of bounds".to_string()))?;
        if slot_size < header_size {
            return Err(DesktopError::Shm(
                "slot smaller than its header".to_string(),
            ));
        }
        let header =
            unsafe { std::ptr::read_unaligned(self.mmap_ptr.add(offset) as *const ShmSlotHeader) };
        if header.size_bytes as usize > slot_size - header_size {
            return Err(DesktopError::Shm(format!(
                "slot {slot} payload of {} bytes exceeds the slot",
                header.size_bytes
            )));
        }
        Ok((offset, header))
    }

    pub fn read_slot(
        &self,
        slot: usize,
        slot_size: usize,
    ) -> Result<(ShmSlotHeader, &[u8]), DesktopError> {
        let (offset, header) = self.checked_slot(slot, slot_size)?;
        let header_size = std::mem::size_of::<ShmSlotHeader>();
        let pixel_slice = unsafe {
            std::slice::from_raw_parts(
                self.mmap_ptr.add(offset + header_size),
                header.size_bytes as usize,
            )
        };
        Ok((header, pixel_slice))
    }
}

impl Drop for ShmReader {
    fn drop(&mut self) {
        unsafe {
            if !self.mmap_ptr.is_null() {
                libc::munmap(self.mmap_ptr as *mut libc::c_void, self.size);
            }
            libc::close(self.fd);
        }
    }
}

pub type ShmWriter = Arc<ShmFramePool>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reader_refuses_slot_header_whose_payload_exceeds_slot() {
        let pool = ShmFramePool::create_unsealed("corrupt-header-test", 4, 4).unwrap();
        // Claim a 4 GiB payload in slot 0 through the still-writable memfd.
        let claimed = u32::MAX.to_ne_bytes();
        let written = unsafe {
            libc::pwrite(
                pool.raw_fd(),
                claimed.as_ptr() as *const libc::c_void,
                claimed.len(),
                (CONTROL_BYTES + std::mem::offset_of!(ShmSlotHeader, size_bytes)) as libc::off_t,
            )
        };
        assert_eq!(written, claimed.len() as isize);
        // ShmReader closes the fd it owns, so hand it a duplicate.
        let reader = ShmReader::attach(unsafe { libc::dup(pool.raw_fd()) }, pool.size()).unwrap();
        let slot_size = reader.slot_size();
        assert!(matches!(
            reader.read_slot(0, slot_size),
            Err(DesktopError::Shm(_))
        ));
        assert!(matches!(reader.latest_frame(), Err(DesktopError::Shm(_))));
    }

    #[test]
    fn reader_reads_pixels_past_the_control_block() {
        let pool = ShmFramePool::create_unsealed("pixel-offset-test", 4, 4).unwrap();
        let pixels: Vec<u8> = (0..64).collect();
        pool.write_frame(1, 0, 0, 4, 4, &pixels).unwrap();
        let reader = ShmReader::attach(unsafe { libc::dup(pool.raw_fd()) }, pool.size()).unwrap();
        let frame = reader.latest_frame().unwrap().expect("a published frame");
        assert_eq!(frame.header.frame_id, 1);
        let (_, slice) = reader.read_slot(frame.slot, reader.slot_size()).unwrap();
        assert_eq!(slice, pixels.as_slice());
    }

    #[test]
    fn reader_wakes_when_producer_publishes() {
        let pool = ShmFramePool::create_unsealed("wake-test", 4, 4).unwrap();
        let reader = ShmReader::attach(unsafe { libc::dup(pool.raw_fd()) }, pool.size()).unwrap();
        let pixels = vec![7u8; 64];
        let started = Instant::now();
        let outcome = std::thread::scope(|scope| {
            scope.spawn(|| {
                std::thread::sleep(Duration::from_millis(50));
                pool.write_frame(1, 0, 0, 4, 4, &pixels).unwrap();
            });
            reader.wait_for_frame(0, Duration::from_secs(10)).unwrap()
        });
        assert_eq!(outcome, FrameWait::Published(1));
        assert!(started.elapsed() < Duration::from_secs(5));
    }

    #[test]
    fn reader_wait_times_out_without_publish() {
        let pool = ShmFramePool::create_unsealed("timeout-test", 4, 4).unwrap();
        let reader = ShmReader::attach(unsafe { libc::dup(pool.raw_fd()) }, pool.size()).unwrap();
        assert_eq!(
            reader.wait_for_frame(0, Duration::from_millis(40)).unwrap(),
            FrameWait::TimedOut
        );
    }
}
