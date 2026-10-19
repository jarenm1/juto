//! Embedded live desktop viewer.
//!
//! Renders the composed frames of a desktop environment inside the harness
//! window and forwards pointer/keyboard input to the environment's virtual
//! seat. Frames are read from the environment's sealed shared-memory pool; no
//! video codec is involved on this path (`docs/agent-compositor.md` section 7).

use std::cell::Cell;
use std::rc::Rc;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::time::Duration;

use gpui::{
    Bounds, Context, FocusHandle, KeyDownEvent, KeyUpEvent, MouseButton, MouseDownEvent,
    MouseMoveEvent, MouseUpEvent, ObjectFit, Pixels, Render, RenderImage, ScrollWheelEvent, Task,
    Window, canvas, div, img, prelude::*, px, rgb,
};
use image::{Frame, ImageBuffer, Rgba};
use juto_daemon::client::Client;
use juto_desktop::protocol::InputEvent;
use juto_desktop::shm::ShmSlotHeader;
use juto_desktop::{DesktopError, ViewerClient};

/// Frame polling interval (~30 fps).
const FRAME_INTERVAL: Duration = Duration::from_millis(33);
/// Control lease renewal interval (spec section 8: heartbeat 2s, expiry 6s).
const LEASE_HEARTBEAT: Duration = Duration::from_secs(2);

/// An embedded viewer for one agent-owned desktop environment.
pub struct DesktopViewer {
    client: Client,
    env_id: String,
    title: String,
    /// The attached viewer, shared with the frame poller and the lease heartbeat.
    viewer: Option<Arc<Mutex<ViewerClient>>>,
    frame: Option<Arc<RenderImage>>,
    frame_size: (u32, u32),
    frame_generation: u64,
    surface_bounds: Rc<Cell<Bounds<Pixels>>>,
    focus_handle: FocusHandle,
    status: String,
    error: Option<String>,
    has_control: bool,
    poll_task: Option<Task<()>>,
    heartbeat_task: Option<Task<()>>,
}

impl DesktopViewer {
    /// Creates a viewer using the desktop socket advertised by the daemon.
    /// Frame dimensions remain unknown until the first composed frame arrives.
    pub fn new(
        client: Client,
        env_id: String,
        title: String,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Self {
        let mut viewer = Self {
            client,
            env_id,
            title,
            viewer: None,
            frame: None,
            frame_size: (0, 0),
            frame_generation: 0,
            surface_bounds: Rc::new(Cell::new(Bounds::default())),
            focus_handle: cx.focus_handle(),
            status: "connecting".to_string(),
            error: None,
            has_control: false,
            poll_task: None,
            heartbeat_task: None,
        };
        viewer.start_attach(window, cx);
        viewer
    }

    /// Attaches to the environment, then polls its composed frames.
    fn start_attach(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let Some(hello) = self.client.hello() else {
            self.status = "unavailable".to_string();
            self.error = Some("cannot attach to desktop: daemon is disconnected".to_string());
            cx.notify();
            return;
        };
        let socket = hello.desktop_socket;
        let env_id = self.env_id.clone();
        self.poll_task = Some(cx.spawn_in(window, async move |this, cx| {
            let attached = cx
                .background_executor()
                .spawn(async move { ViewerClient::attach(&socket, &env_id) })
                .await;
            let viewer = match attached {
                Ok(viewer) => Arc::new(Mutex::new(viewer)),
                Err(err) => {
                    let _ = this.update(cx, |this, cx| {
                        this.status = "unavailable".to_string();
                        this.error = Some(format!("cannot attach to desktop: {err}"));
                        cx.notify();
                    });
                    return;
                }
            };
            if this
                .update(cx, |this, _| this.viewer = Some(viewer.clone()))
                .is_err()
            {
                return;
            }
            loop {
                cx.background_executor().timer(FRAME_INTERVAL).await;
                let reader = viewer.clone();
                let read = cx
                    .background_executor()
                    .spawn(async move {
                        let guard = lock_viewer(&reader);
                        let (header, pixels) = guard.latest_frame().ok()??;
                        Some((header, pixels))
                    })
                    .await;
                let Some((header, pixels)) = read else {
                    continue;
                };
                let updated = this.update_in(cx, |this, window, cx| {
                    this.set_frame(header, pixels, window, cx);
                });
                if updated.is_err() {
                    break;
                }
            }
        }));
    }

    fn set_frame(
        &mut self,
        header: ShmSlotHeader,
        pixels: Vec<u8>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let image = render_image_from_bgra(&pixels, header.width, header.height);
        let previous = self.frame.replace(image);
        if let Some(previous) = previous {
            cx.drop_image(previous, Some(window));
        }
        self.frame_size = (header.width, header.height);
        self.frame_generation = header.generation;
        if self.error.is_none() {
            self.status = format!("live {}x{}", header.width, header.height);
        }
        cx.notify();
    }

    /// The attached viewer. Before attachment completes, reports that and returns `None`.
    fn attached_viewer(&mut self, cx: &mut Context<Self>) -> Option<Arc<Mutex<ViewerClient>>> {
        if self.viewer.is_none() {
            self.error = Some("desktop is still connecting".to_string());
            cx.notify();
        }
        self.viewer.clone()
    }

    /// Acquires (or renews) the human control lease.
    pub fn take_control(&mut self, cx: &mut Context<Self>) {
        let Some(viewer) = self.attached_viewer(cx) else {
            return;
        };
        let result = lock_viewer(&viewer).take_control();
        match result {
            Ok(epoch) => {
                self.has_control = true;
                self.error = None;
                self.status = format!("control held (epoch {epoch})");
                self.start_heartbeat(viewer, cx);
            }
            Err(DesktopError::ControlBusy { .. }) => {
                self.has_control = false;
                self.status = "no control".to_string();
            }
            Err(err) => {
                self.has_control = false;
                self.error = Some(err.to_string());
            }
        }
        cx.notify();
    }

    /// Releases the human control lease.
    pub fn release_control(&mut self, cx: &mut Context<Self>) {
        self.heartbeat_task = None;
        let Some(viewer) = self.attached_viewer(cx) else {
            return;
        };
        let result = lock_viewer(&viewer).release_control();
        match result {
            Ok(()) => {
                self.has_control = false;
                self.status = "control released".to_string();
                self.error = None;
            }
            Err(err) => self.error = Some(err.to_string()),
        }
        cx.notify();
    }

    fn start_heartbeat(&mut self, viewer: Arc<Mutex<ViewerClient>>, cx: &mut Context<Self>) {
        if self.heartbeat_task.is_some() {
            return;
        }
        self.heartbeat_task = Some(cx.spawn(async move |this, cx| {
            loop {
                cx.background_executor().timer(LEASE_HEARTBEAT).await;
                let lease = viewer.clone();
                let renewed = cx
                    .background_executor()
                    .spawn(async move {
                        let mut guard = lock_viewer(&lease);
                        // The viewer is the human owner while it is open, so the
                        // environment is not retired as abandoned.
                        let _ = guard.heartbeat_owner();
                        guard.take_control().is_ok()
                    })
                    .await;
                if this
                    .update(cx, |this, cx| {
                        this.has_control = renewed;
                        if !renewed {
                            this.error = Some("control lease lost".to_string());
                        }
                        cx.notify();
                    })
                    .is_err()
                {
                    break;
                }
                if !renewed {
                    break;
                }
            }
        }));
    }

    fn send_events(&mut self, events: Vec<InputEvent>, cx: &mut Context<Self>) {
        if events.is_empty() {
            return;
        }
        let Some(viewer) = self.attached_viewer(cx) else {
            return;
        };
        if !self.has_control {
            self.take_control(cx);
            if !self.has_control {
                return;
            }
        }
        let result = lock_viewer(&viewer).send_input(self.frame_generation, events);
        match result {
            Ok(()) => self.error = None,
            Err(err) => self.error = Some(err.to_string()),
        }
        cx.notify();
    }

    /// Geometry of the letterboxed image inside the surface container:
    /// `(origin_x, origin_y, scale)` in window coordinates.
    fn fitted_geometry(&self) -> Option<(f32, f32, f32)> {
        let bounds = self.surface_bounds.get();
        let (image_width, image_height) = self.frame_size;
        if image_width == 0 || image_height == 0 {
            return None;
        }
        let box_width = f32::from(bounds.size.width);
        let box_height = f32::from(bounds.size.height);
        if box_width <= 0.0 || box_height <= 0.0 {
            return None;
        }
        let scale = (box_width / image_width as f32).min(box_height / image_height as f32);
        let drawn_width = image_width as f32 * scale;
        let drawn_height = image_height as f32 * scale;
        let origin_x = f32::from(bounds.origin.x) + (box_width - drawn_width) / 2.0;
        let origin_y = f32::from(bounds.origin.y) + (box_height - drawn_height) / 2.0;
        Some((origin_x, origin_y, scale))
    }

    fn pointer_position(&self, position: &gpui::Point<Pixels>) -> Option<(u32, u32)> {
        let (origin_x, origin_y, scale) = self.fitted_geometry()?;
        let (image_width, image_height) = self.frame_size;
        let x = f32::from(position.x) - origin_x;
        let y = f32::from(position.y) - origin_y;
        if x < 0.0 || y < 0.0 || x >= image_width as f32 * scale || y >= image_height as f32 * scale
        {
            return None;
        }
        Some((
            (x / scale).clamp(0.0, image_width as f32 - 1.0) as u32,
            (y / scale).clamp(0.0, image_height as f32 - 1.0) as u32,
        ))
    }

    fn on_key(&mut self, event: &KeyDownEvent, pressed: bool, cx: &mut Context<Self>) {
        let Some(keycode) = evdev_keycode(&event.keystroke.key) else {
            return;
        };
        self.send_events(vec![InputEvent::Key { keycode, pressed }], cx);
    }

    fn on_key_up(&mut self, event: &KeyUpEvent, cx: &mut Context<Self>) {
        self.on_key(
            &KeyDownEvent {
                keystroke: event.keystroke.clone(),
                is_held: false,
            },
            false,
            cx,
        );
    }
}

impl Render for DesktopViewer {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl gpui::IntoElement {
        let frame = self.frame.clone();
        let bounds_cell = self.surface_bounds.clone();
        let status = self.status.clone();
        let error = self.error.clone();
        let has_control = self.has_control;
        let title = self.title.clone();

        let surface = match frame {
            Some(frame) => img(frame)
                .size_full()
                .object_fit(ObjectFit::Contain)
                .into_any_element(),
            None => div()
                .size_full()
                .flex()
                .items_center()
                .justify_center()
                .text_color(rgb(0x9aa0a6))
                .child("waiting for the first frame…")
                .into_any_element(),
        };

        div()
            .id("desktop-viewer")
            .track_focus(&self.focus_handle)
            .on_key_down(cx.listener(|this, event: &KeyDownEvent, _window, cx| {
                this.on_key(event, true, cx);
            }))
            .on_key_up(cx.listener(|this, event: &KeyUpEvent, _window, cx| {
                this.on_key_up(event, cx);
            }))
            .on_mouse_move(cx.listener(|this, event: &MouseMoveEvent, _window, cx| {
                if let Some((x, y)) = this.pointer_position(&event.position) {
                    this.send_events(vec![InputEvent::PointerMoveAbs { x, y }], cx);
                }
            }))
            .on_mouse_down(
                MouseButton::Left,
                cx.listener(|this, event: &MouseDownEvent, window, cx| {
                    window.focus(&this.focus_handle);
                    if let Some((x, y)) = this.pointer_position(&event.position) {
                        this.send_events(
                            vec![
                                InputEvent::PointerMoveAbs { x, y },
                                InputEvent::PointerButton {
                                    button: 272,
                                    pressed: true,
                                },
                            ],
                            cx,
                        );
                    }
                }),
            )
            .on_mouse_up(
                MouseButton::Left,
                cx.listener(|this, event: &MouseUpEvent, _window, cx| {
                    if let Some((x, y)) = this.pointer_position(&event.position) {
                        this.send_events(
                            vec![
                                InputEvent::PointerMoveAbs { x, y },
                                InputEvent::PointerButton {
                                    button: 272,
                                    pressed: false,
                                },
                            ],
                            cx,
                        );
                    }
                }),
            )
            .on_mouse_down(
                MouseButton::Right,
                cx.listener(|this, event: &MouseDownEvent, window, cx| {
                    window.focus(&this.focus_handle);
                    if let Some((x, y)) = this.pointer_position(&event.position) {
                        this.send_events(
                            vec![
                                InputEvent::PointerMoveAbs { x, y },
                                InputEvent::PointerButton {
                                    button: 273,
                                    pressed: true,
                                },
                            ],
                            cx,
                        );
                    }
                }),
            )
            .on_mouse_up(
                MouseButton::Right,
                cx.listener(|this, event: &MouseUpEvent, _window, cx| {
                    if let Some((x, y)) = this.pointer_position(&event.position) {
                        this.send_events(
                            vec![
                                InputEvent::PointerMoveAbs { x, y },
                                InputEvent::PointerButton {
                                    button: 273,
                                    pressed: false,
                                },
                            ],
                            cx,
                        );
                    }
                }),
            )
            .on_scroll_wheel(cx.listener(|this, event: &ScrollWheelEvent, window, cx| {
                let delta = event.delta.pixel_delta(window.line_height());
                let dy = f32::from(delta.y);
                if dy != 0.0 {
                    this.send_events(
                        vec![InputEvent::PointerScroll {
                            axis: 1,
                            value: -dy as i32,
                        }],
                        cx,
                    );
                }
            }))
            .size_full()
            .w(px(900.))
            .h(px(620.))
            .flex()
            .flex_col()
            .gap_2()
            .child(
                div()
                    .flex()
                    .items_center()
                    .justify_between()
                    .child(div().text_color(rgb(0xe8eaed)).child(title))
                    .child(
                        div()
                            .flex()
                            .gap_2()
                            .child(
                                div()
                                    .id("desktop-take-control")
                                    .px_2()
                                    .py_1()
                                    .rounded_md()
                                    .bg(rgb(0x2d2f34))
                                    .text_color(rgb(0xe8eaed))
                                    .cursor_pointer()
                                    .child(if has_control {
                                        "Renew control"
                                    } else {
                                        "Take control"
                                    })
                                    .on_click(cx.listener(|this, _, _window, cx| {
                                        this.take_control(cx);
                                    })),
                            )
                            .child(
                                div()
                                    .id("desktop-release-control")
                                    .px_2()
                                    .py_1()
                                    .rounded_md()
                                    .bg(rgb(0x2d2f34))
                                    .text_color(rgb(0xe8eaed))
                                    .cursor_pointer()
                                    .child("Release control")
                                    .on_click(cx.listener(|this, _, _window, cx| {
                                        this.release_control(cx);
                                    })),
                            ),
                    ),
            )
            .child(
                div()
                    .flex_1()
                    .min_h_0()
                    .bg(rgb(0x000000))
                    .child(
                        canvas(
                            move |bounds, _window, _cx| {
                                bounds_cell.set(bounds);
                            },
                            move |_bounds, _state, _window, _cx| {},
                        )
                        .absolute()
                        .size_full(),
                    )
                    .child(surface),
            )
            .child(div().text_color(rgb(0x9aa0a6)).child(match &error {
                Some(error) => format!("{status} — {error}"),
                None => status,
            }))
    }
}

/// Locks the shared viewer, recovering from poisoning so frame polling keeps running.
fn lock_viewer(viewer: &Mutex<ViewerClient>) -> MutexGuard<'_, ViewerClient> {
    viewer.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Builds a gpui `RenderImage` from BGRA8 pixels with straight alpha.
///
/// gpui's texture atlas is `Bgra8Unorm`, so the bytes are handed over
/// unchanged; only the row stride is normalized.
fn render_image_from_bgra(bgra: &[u8], width: u32, height: u32) -> Arc<RenderImage> {
    let row = width as usize * 4;
    let mut packed = Vec::with_capacity(row * height as usize);
    packed.extend_from_slice(&bgra[..row * height as usize]);
    let buffer: ImageBuffer<Rgba<u8>, Vec<u8>> =
        ImageBuffer::from_raw(width, height, packed).expect("pixel buffer size mismatch");
    Arc::new(RenderImage::new(vec![Frame::new(buffer)]))
}

/// Maps a gpui key name to a Linux evdev keycode.
///
/// Only the keys a desktop session needs for ordinary interaction are mapped;
/// unmapped keys are dropped rather than synthesized as something else.
fn evdev_keycode(key: &str) -> Option<u32> {
    let code = match key {
        "escape" => 1,
        "1" => 2,
        "2" => 3,
        "3" => 4,
        "4" => 5,
        "5" => 6,
        "6" => 7,
        "7" => 8,
        "8" => 9,
        "9" => 10,
        "0" => 11,
        "minus" => 12,
        "equal" => 13,
        "backspace" => 14,
        "tab" => 15,
        "q" => 16,
        "w" => 17,
        "e" => 18,
        "r" => 19,
        "t" => 20,
        "y" => 21,
        "u" => 22,
        "i" => 23,
        "o" => 24,
        "p" => 25,
        "[" => 26,
        "]" => 27,
        "enter" => 28,
        "control" => 29,
        "a" => 30,
        "s" => 31,
        "d" => 32,
        "f" => 33,
        "g" => 34,
        "h" => 35,
        "j" => 36,
        "k" => 37,
        "l" => 38,
        ";" => 39,
        "'" => 40,
        "`" => 41,
        "shift" => 42,
        "\\" => 43,
        "z" => 44,
        "x" => 45,
        "c" => 46,
        "v" => 47,
        "b" => 48,
        "n" => 49,
        "m" => 50,
        "," => 51,
        "." => 52,
        "/" => 53,
        "alt" => 56,
        "space" => 57,
        "capslock" => 58,
        "f1" => 59,
        "f2" => 60,
        "f3" => 61,
        "f4" => 62,
        "f5" => 63,
        "f6" => 64,
        "f7" => 65,
        "f8" => 66,
        "f9" => 67,
        "f10" => 68,
        "f11" => 87,
        "f12" => 88,
        "home" => 102,
        "up" => 103,
        "pageup" => 104,
        "left" => 105,
        "right" => 106,
        "end" => 107,
        "down" => 108,
        "pagedown" => 109,
        "insert" => 110,
        "delete" => 111,
        "super" | "meta" | "cmd" => 125,
        _ => return None,
    };
    Some(code)
}
