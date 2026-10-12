use crate::error::DesktopError;
use parking_lot::Mutex;
use std::fs::File;
use std::os::fd::{FromRawFd, RawFd};
use std::sync::Arc;

/// Number of buffers in an attachment-private triple-buffered pool.
pub const POOL_SLOT_COUNT: usize = 3;

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
        let pool_id = pool_id.into();
        let stride = width * 4;
        let pixel_bytes = (stride * height) as usize;
        let header_bytes = std::mem::size_of::<ShmSlotHeader>();
        // Align slot size to page boundary (4096 bytes)
        let slot_size = ((header_bytes + pixel_bytes) + 4095) & !4095;
        let total_size = slot_size * POOL_SLOT_COUNT;

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

        // 4. Apply write and size seals: F_SEAL_GROW | F_SEAL_SHRINK | F_SEAL_FUTURE_WRITE | F_SEAL_SEAL
        // This guarantees that any viewer receiving the passed descriptor cannot create new writable mappings,
        // while the producer's existing mapping remains valid.
        const F_SEAL_SEAL: libc::c_int = 0x0001;
        const F_SEAL_SHRINK: libc::c_int = 0x0002;
        const F_SEAL_GROW: libc::c_int = 0x0004;
        const F_SEAL_FUTURE_WRITE: libc::c_int = 0x0010;

        let seals = F_SEAL_SEAL | F_SEAL_SHRINK | F_SEAL_GROW | F_SEAL_FUTURE_WRITE;
        let seal_res = unsafe { libc::fcntl(fd, libc::F_ADD_SEALS, seals) };
        if seal_res != 0 {
            unsafe {
                libc::munmap(mmap_ptr, total_size);
                libc::close(fd);
            }
            return Err(DesktopError::Shm(format!(
                "F_ADD_SEALS failed: {}",
                std::io::Error::last_os_error()
            )));
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

        // Round-robin selection of next slot not held by slow readers
        let mut target_slot = (*active + 1) % POOL_SLOT_COUNT;
        let mut attempts = 0;
        while readers[target_slot] > 0 && attempts < POOL_SLOT_COUNT {
            target_slot = (target_slot + 1) % POOL_SLOT_COUNT;
            attempts += 1;
        }

        if readers[target_slot] > 0 {
            // All slots held by readers: backpressure skip (drop frame export, never block seat)
            return Err(DesktopError::LimitExceeded(
                "all SHM pool slots held by slow readers; dropped frame".to_string(),
            ));
        }

        *active = target_slot;
        let offset = target_slot * self.slot_size;
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
            // Write slot header
            std::ptr::copy_nonoverlapping(
                &header as *const ShmSlotHeader as *const u8,
                slot_ptr,
                std::mem::size_of::<ShmSlotHeader>(),
            );
            // Write pixel payload
            let pixel_ptr = slot_ptr.add(std::mem::size_of::<ShmSlotHeader>());
            let copy_len = pixels
                .len()
                .min(self.slot_size - std::mem::size_of::<ShmSlotHeader>());
            std::ptr::copy_nonoverlapping(pixels.as_ptr(), pixel_ptr, copy_len);
        }

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
        let offset = slot * self.slot_size;
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
        let offset = slot * self.slot_size;
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
    mmap_ptr: *const u8,
}

unsafe impl Send for ShmReader {}
unsafe impl Sync for ShmReader {}

impl ShmReader {
    pub fn attach(fd: RawFd, size: usize) -> Result<Self, DesktopError> {
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

        Ok(Self {
            fd,
            size,
            mmap_ptr: mmap_ptr as *const u8,
        })
    }

    pub fn read_slot(
        &self,
        slot: usize,
        slot_size: usize,
    ) -> Result<(ShmSlotHeader, &[u8]), DesktopError> {
        let offset = slot * slot_size;
        if offset + slot_size > self.size {
            return Err(DesktopError::Shm("slot offset out of bounds".to_string()));
        }

        let header_size = std::mem::size_of::<ShmSlotHeader>();
        unsafe {
            let ptr = self.mmap_ptr.add(offset);
            let header = *(ptr as *const ShmSlotHeader);
            let pixel_ptr = ptr.add(header_size);
            let pixel_slice = std::slice::from_raw_parts(pixel_ptr, header.size_bytes as usize);
            Ok((header, pixel_slice))
        }
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
