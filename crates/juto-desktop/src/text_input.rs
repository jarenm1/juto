//! `zwp_text_input_manager_v3` support.
//!
//! The compositor is the input method for its own clients, so it advertises the
//! text-input global itself rather than routing through an external IME. Focus
//! changes send `enter`/`leave`; committed text and pre-edit strings are
//! delivered only to instances that enabled themselves for the focused surface
//! (`docs/agent-compositor.md` section 8).
//!
//! The protocol handlers are implemented for the compositor state in
//! `compositor.rs`; this module owns the state and the delivery helpers.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

use parking_lot::Mutex;
use smithay::reexports::wayland_protocols::wp::text_input::zv3::server::zwp_text_input_v3::ZwpTextInputV3;
use smithay::reexports::wayland_server::Resource;
use smithay::reexports::wayland_server::protocol::wl_surface::WlSurface;

/// Per-instance text-input state.
#[derive(Debug, Default)]
pub struct TextInputUserData {
    enabled: AtomicBool,
    serial: AtomicU32,
}

impl TextInputUserData {
    /// Whether the instance enabled itself for the focused surface.
    pub fn enabled(&self) -> bool {
        self.enabled.load(Ordering::SeqCst)
    }

    /// Records the enable/disable request.
    pub fn set_enabled(&self, enabled: bool) {
        self.enabled.store(enabled, Ordering::SeqCst);
    }

    /// Advances and returns the commit serial.
    pub fn next_serial(&self) -> u32 {
        self.serial.fetch_add(1, Ordering::SeqCst).wrapping_add(1)
    }

    /// The current commit serial.
    pub fn serial(&self) -> u32 {
        self.serial.load(Ordering::SeqCst)
    }
}

/// The compositor's text-input global state.
#[derive(Debug, Default)]
pub struct TextInputManagerState {
    instances: Arc<Mutex<Vec<ZwpTextInputV3>>>,
    focus: Arc<Mutex<Option<WlSurface>>>,
}

impl TextInputManagerState {
    /// Registers a text-input instance created by a client.
    pub fn add_instance(&self, instance: ZwpTextInputV3) {
        self.instances.lock().push(instance);
    }

    /// Drops instances whose client has gone away.
    pub fn retain_live(&self) {
        self.instances.lock().retain(|instance| instance.is_alive());
    }

    /// Records the focused surface and notifies text-input instances.
    pub fn set_focus(&self, surface: Option<WlSurface>) {
        let previous = self.focus.lock().clone();
        if previous.as_ref() == surface.as_ref() {
            return;
        }
        {
            let instances = self.instances.lock();
            if let Some(previous) = previous.as_ref() {
                for instance in instances.iter() {
                    if previous.id().same_client_as(&instance.id()) {
                        instance.leave(previous);
                        instance.done(serial_of(instance));
                    }
                }
            }
            if let Some(surface) = surface.as_ref() {
                for instance in instances.iter() {
                    if surface.id().same_client_as(&instance.id()) {
                        instance.enter(surface);
                        instance.done(serial_of(instance));
                    }
                }
            }
        }
        *self.focus.lock() = surface;
    }

    /// Sends committed text to enabled instances for the focused surface.
    pub fn commit_string(&self, text: &str) -> bool {
        self.send(|instance| instance.commit_string(Some(text.to_string())))
    }

    /// Sends a pre-edit string to enabled instances for the focused surface.
    pub fn preedit_string(&self, text: &str, cursor_begin: i32, cursor_end: i32) -> bool {
        self.send(|instance| {
            instance.preedit_string(Some(text.to_string()), cursor_begin, cursor_end)
        })
    }

    fn send(&self, send: impl Fn(&ZwpTextInputV3)) -> bool {
        let focus = self.focus.lock().clone();
        let Some(focus) = focus else {
            return false;
        };
        let instances = self.instances.lock();
        let mut delivered = false;
        for instance in instances.iter() {
            if !focus.id().same_client_as(&instance.id()) {
                continue;
            }
            let enabled = instance
                .data::<TextInputUserData>()
                .map(|data| data.enabled())
                .unwrap_or(false);
            if !enabled {
                continue;
            }
            send(instance);
            instance.done(serial_of(instance));
            delivered = true;
        }
        delivered
    }
}

fn serial_of(instance: &ZwpTextInputV3) -> u32 {
    instance
        .data::<TextInputUserData>()
        .map(|data| data.serial())
        .unwrap_or(0)
}
