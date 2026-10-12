//! Unix descriptor passing (`SCM_RIGHTS`).
//!
//! Descriptors are transferred as ancillary data on the control socket; an
//! integer is never serialized as a transferable resource
//! (`docs/agent-compositor.md` section 7).

use std::io;
use std::os::fd::{AsRawFd, BorrowedFd, FromRawFd, OwnedFd, RawFd};
use std::os::unix::net::UnixStream;

/// Control-message storage aligned for `cmsghdr`.
#[repr(align(8))]
struct CmsgBuffer([u8; 64]);

/// Sends `data` followed by `fd` as ancillary data.
pub fn send_with_fd(stream: &UnixStream, data: &[u8], fd: RawFd) -> io::Result<()> {
    let mut cmsg = CmsgBuffer([0u8; 64]);
    let iov = [libc::iovec {
        iov_base: data.as_ptr() as *mut libc::c_void,
        iov_len: data.len(),
    }];
    let mut msg: libc::msghdr = unsafe { std::mem::zeroed() };
    msg.msg_iov = iov.as_ptr() as *mut libc::iovec;
    msg.msg_iovlen = 1;
    msg.msg_control = cmsg.0.as_mut_ptr() as *mut libc::c_void;
    msg.msg_controllen = cmsg.0.len();

    unsafe {
        let header = libc::CMSG_FIRSTHDR(&msg);
        if header.is_null() {
            return Err(io::Error::other("no control message header"));
        }
        (*header).cmsg_level = libc::SOL_SOCKET;
        (*header).cmsg_type = libc::SCM_RIGHTS;
        (*header).cmsg_len = libc::CMSG_LEN(std::mem::size_of::<RawFd>() as u32) as _;
        std::ptr::copy_nonoverlapping(
            &fd as *const RawFd as *const u8,
            libc::CMSG_DATA(header),
            std::mem::size_of::<RawFd>(),
        );
        msg.msg_controllen = libc::CMSG_SPACE(std::mem::size_of::<RawFd>() as u32) as _;

        let sent = libc::sendmsg(stream.as_raw_fd(), &msg, 0);
        if sent < 0 {
            return Err(io::Error::last_os_error());
        }
    }
    Ok(())
}

/// Receives up to `data.len()` bytes and one descriptor, if present.
pub fn recv_with_fd(stream: &UnixStream, data: &mut [u8]) -> io::Result<(usize, Option<OwnedFd>)> {
    let mut cmsg = CmsgBuffer([0u8; 64]);
    let iov = [libc::iovec {
        iov_base: data.as_mut_ptr() as *mut libc::c_void,
        iov_len: data.len(),
    }];
    let mut msg: libc::msghdr = unsafe { std::mem::zeroed() };
    msg.msg_iov = iov.as_ptr() as *mut libc::iovec;
    msg.msg_iovlen = 1;
    msg.msg_control = cmsg.0.as_mut_ptr() as *mut libc::c_void;
    msg.msg_controllen = cmsg.0.len();

    let received = unsafe { libc::recvmsg(stream.as_raw_fd(), &mut msg, 0) };
    if received < 0 {
        return Err(io::Error::last_os_error());
    }

    let mut fd = None;
    unsafe {
        let mut header = libc::CMSG_FIRSTHDR(&msg);
        while !header.is_null() {
            if (*header).cmsg_level == libc::SOL_SOCKET && (*header).cmsg_type == libc::SCM_RIGHTS {
                let mut raw: RawFd = -1;
                std::ptr::copy_nonoverlapping(
                    libc::CMSG_DATA(header),
                    &mut raw as *mut RawFd as *mut u8,
                    std::mem::size_of::<RawFd>(),
                );
                if raw >= 0 {
                    fd = Some(OwnedFd::from_raw_fd(raw));
                }
            }
            header = libc::CMSG_NXTHDR(&msg, header);
        }
    }
    Ok((received as usize, fd))
}

/// The peer's credentials on a connected Unix socket.
pub fn peer_credentials(stream: &UnixStream) -> io::Result<libc::ucred> {
    let fd: BorrowedFd<'_> = stream.as_fd();
    let mut creds: libc::ucred = unsafe { std::mem::zeroed() };
    let mut len = std::mem::size_of::<libc::ucred>() as libc::socklen_t;
    let result = unsafe {
        libc::getsockopt(
            fd.as_raw_fd(),
            libc::SOL_SOCKET,
            libc::SO_PEERCRED,
            &mut creds as *mut libc::ucred as *mut libc::c_void,
            &mut len,
        )
    };
    if result != 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(creds)
}

use std::os::fd::AsFd as _;
