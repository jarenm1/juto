# Agent-owned headless desktop compositor

**Status: proposed design, not an implemented capability.** This document owns
compositor design end to end, including environment supervision and presentation
interfaces. It does not implement the compositor, runtime tools, GPUI viewer,
browser viewer, or deployment provisioning. Existing Juto capabilities are
reported in [README](../README.md); the [runtime inventory](omp-runtime.md) is
historical port context, not a current implementation inventory.

## 1. Decision and scope

Build a Linux Rust executable, `juto-desktop`, using Smithay for a headless
Wayland compositor with one virtual output and one virtual seat per environment.
An agent owns a desktop environment, launches actual applications into it,
observes its rendered output, and controls it. A human can attach to that same
live environment and play the running game inside the harness. No recording,
physical display, nested host desktop, or TypeScript agent is required.

Local and remote presentation deliberately differ:

- **Same-host native viewer:** Unix-socket control plus shared-memory frames;
  negotiate DMA-BUF import with explicit synchronization when the viewer supports
  it. Avoid video encode/decode on this path.
- **Remote laptop/phone/native viewer:** authenticated signaling plus WebRTC
  video/audio and input. A native viewer on the same machine can explicitly use
  this path, but a failed local DMA-BUF import normally falls back to local shared
  memory, not to network streaming.
- Both paths consume the same composed desktop and share input semantics,
  geometry generations, authorization, and exclusive control leases.

The desktop module is deep: callers learn environment creation, application
launch, observation, attachment, control, reconfiguration, and destruction;
Wayland, X11 window management, rendering, pacing, codecs, and process cleanup
stay behind that interface. Local and remote presentation are two real adapters
at an internal seam, not separate desktops.

### Requirements and design locations

| Requirement | Design |
|---|---|
| Headless desktop | Sections 2 and 5: virtual output, no KMS/physical input |
| Different resolutions and FPS caps | Sections 4 and 6: admitted modes, atomic generations, virtual clock |
| System limits | Sections 3 and 6: process-tree and aggregate admission limits |
| Agent owns environment and launches apps | Sections 3 and 4: owner capability, argv launch, supervised lifetime |
| Computer use or other automation | Section 8: snapshots, virtual seat, explicitly granted app interfaces |
| Remote streaming and user play | Sections 7–9: WebRTC, input handoff, game scenario |
| Efficient local streaming | Section 7: shared memory/DMA-BUF, separate from remote codec pipeline |
| Laptop and phone interaction | Sections 7 and 8: browser interoperability, touch mapping, browser restrictions |

### Explicit exclusions

This is not a replacement host login compositor or full GNOME/KDE session.
Multi-monitor, HDR, VR, physical device passthrough, automatic package installation,
recording, and unrestricted public sharing are not part of this contract. Gamepad
and rumble are not baseline input; keyboard/mouse games are the acceptance target.
A desktop contains a minimal window manager, not a shell panel or application
store. Games requiring anti-cheat drivers, DRM display ownership, privileged
launchers, or unsupported graphics drivers are not guaranteed compatible.

## 2. Process model and seams

```text
Juto runtime                         GPUI / browser / remote native viewer
     | authenticated commands             | attach + lease + input
     v                                    v
Desktop supervisor: admission, authorization, lifecycle, app launch
     |
     +-- environment cgroup (aggregate CPU/memory/tasks limits)
         +-- compositor worker: Wayland, Xwayland WM, seat, virtual clock
         +-- presentation worker: local frame broker, WebRTC, encode/audio
         +-- private PipeWire + PulseAudio compatibility server
         +-- application supervisor and application descendants
```

The desktop supervisor remains outside individual environment cgroups so one
application OOM cannot remove the cleanup authority. The host also caps the
supervisor and the shared agent slice. Per-environment worker and application
subgroups reserve a modest control budget so a game cannot consume every task
slot or starve teardown. Placement must complete before any application executes.

Smithay's `calloop` thread owns Wayland and seat state sequentially. Commands
enter through a bounded queue and wakeup FD; encoding, network, filesystem work,
and process supervision never block it. The presentation worker consumes
completed buffers, not live Smithay objects. GPUI only consumes frames/events
and submits commands; it does not own the compositor or application lifetime.

Proposed code placement when implementing: desktop executable and library under
`crates/juto-desktop`, a thin runtime desktop-tool adapter, and viewer adapters in
the harness. Create no empty crates now. Reuse the governance policy in
[`workspace.rs`](../crates/juto-runtime/src/workspace.rs) and
[`agent-scope.sh`](../scripts/agent-scope.sh), but do not treat their existing
one-command scope launcher as a persistent environment supervisor. Any shared
governance extraction must migrate both callers rather than duplicate defaults.

## 3. Ownership, isolation, launch, and lifetime

An environment belongs to a runtime task/session identity, not to a viewer.
The owner can launch/terminate applications, set output policy, observe, grant
view/control access, and destroy the desktop. Viewers cannot launch applications
or raise limits merely because they can send input. Human administrative override
is a separately authenticated privilege.

Creation allocates an opaque environment ID, private runtime directory, supervised
process tree, and lease deadline. Private Wayland, X11, D-Bus, and audio sockets
are never the host's sockets. Set `WAYLAND_DISPLAY`, environment-private
`XDG_RUNTIME_DIR`, `DISPLAY`/Xauthority when enabled, and private audio/session-bus
addresses at launch. Do not inherit the host display, SSH agent, provider secrets,
host session bus, or host audio devices. Preserve only an approved environment
allowlist plus explicit application configuration; secret values stay out of
command/event logs. Use an argv vector, never implicit shell interpolation.
An explicitly launched shell remains an application with the same grants.

The launch contract accepts executable, argv, workspace-relative cwd, approved
environment overrides, and filesystem/network grants. Resolve cwd and executable
inside the configured filesystem view; symlinks cannot expand a grant. Mount the
assigned workspace and approved dependencies, provide private home/config/cache,
and writable scratch/artifact directories. Installing dependencies is a runtime
or provisioning decision; missing programs return `ExecutableUnavailable`.
Support approved wrappers such as Nix-provided launch commands without granting
them the user's credentials or unrestricted host paths.

**Resource isolation is not a security sandbox.** Local trusted development mode
uses user-systemd governance and private sockets but cannot claim hostile same-UID
isolation. Running untrusted games or offering remote multi-tenant desktops
requires dedicated environment identities and a launcher that supplies mount,
PID, IPC, and network isolation with restricted devices and no privilege
escalation. A container/VM-backed host launcher may satisfy this contract; the
compositor must refuse that security profile if the launcher is unavailable.
Do not silently downgrade it to trusted mode. Same-UID Unix credentials are
necessary for local attachment but are not protection against malicious peers
with that UID. X11 clients within one Xwayland desktop also share its trust domain.

Only authorized render nodes may be exposed; no `/dev/input`, `/dev/uinput`,
primary DRM nodes, host audio sockets, or host display sockets. Network access
is an explicit profile (`none` or approved egress), independent of desktop view
access. Hard network allowlists require a host launcher capable of enforcing them.
Each application-specific automation socket is also an explicit grant.

### Lifecycle

```text
Creating -> Ready -> Stopping -> Stopped
    |         |
    +---------+-> Failed -> Stopping -> Stopped
```

- `Ready` means socket listeners, virtual output, renderer, and enforced limits
  are operational. It does not mean an application has drawn a frame.
- Application lifecycle is independent: `Starting -> Running -> Exited`, with
  spawn errors separate from exit status. `WindowMapped` and `FrameAvailable`
  are observations, not inferred from successful process launch. Launchers may
  spawn several processes/windows; an application handle tracks its subtree.
- Owner heartbeat interval is 10 seconds; expiry after 60 seconds destroys the
  environment unless the runtime has explicitly transferred retention ownership
  to a human session. Task cancellation stops the desktop by default. A viewer
  alone never keeps an abandoned desktop alive.
- `Destroy` first revokes grants/leases, releases held input, closes viewers,
  requests application termination, then stops the entire environment cgroup.
  Default grace period: 5 seconds, followed by forced termination. Destroy is
  idempotent; `Stopped` is emitted only after process and buffer cleanup.
- Compositor crash fails the environment and tears down applications; no fake
  successful resume. Presentation failure detaches viewers and releases input
  but need not kill a healthy desktop; reattachment starts fresh buffer/peer state.
- After supervisor restart, reconcile persisted non-secret environment/unit IDs;
  kill orphan units and stale private directories. Do not recover old capability
  secrets or active input leases. Files explicitly saved in the workspace survive;
  windows and live processes are not session-persistence promises.

## 4. Caller interface

The following is a proposed version-1 protocol, not existing Rust symbols.
Local control uses length-prefixed JSON messages over a private Unix socket;
remote management/signaling uses authenticated HTTPS/WSS at the host gateway.
Protocol version, request ID, and authenticated caller identity are mandatory;
environment ID is required except on `Create`.
Capabilities travel outside durable request logs. Reject unknown required fields,
unsupported versions, messages over 64 KiB, and requests above admitted rates.

| Command | Result / invariant |
|---|---|
| `Create(spec, request_id)` | Environment ID, effective limits/output, capabilities; same owner/request ID is idempotent |
| `Launch(env, argv, cwd, grants, request_id)` | Application handle; repeated request cannot launch a second copy |
| `Terminate(env, app)` | Subtree termination and eventual exit event |
| `Describe(env, after_event_seq?)` | Authoritative state, windows, apps, output, grants, effective limits |
| `ConfigureOutput(env, expected_generation, mode)` | New geometry generation (unchanged for FPS-only) or explicit rejection; owner only |
| `Observe(env, generation, after_frame?, timeout)` | Lossless composed image reference + metadata, or timeout/stale/error |
| `Attach(env, transport, capabilities)` | Viewer ID, negotiated transport/codec, output metadata; grants checked |
| `Grant/Revoke(env, principal, roles, audience, expiry)` | Owner issues/renews recipient-bound view/control grants; revoke also closes active peers |
| `Heartbeat(env, owner_or_lease, epoch?)` | Return renewed deadline/TTL; expired/revoked identities fail, never reacquire control |
| `Retain(env, human_owner, deadline)` | Explicit human acceptance transfers lifetime responsibility with a bounded deadline |
| `Window(env, id, focus/close/move/resize)` | Owner WM action; mutation requires the active control lease |
| `Control(env, acquire/release, viewer_or_agent)` | Exclusive input lease or `ControlBusy`; human can preempt |
| `Input(env, lease, epoch, seq, generation, events)` | Applied sequence watermark or explicit rejection |
| `GeometryReady(env, viewer, generation, track_or_pool_id)` | Resume pointer input only after identified new-generation presentation |
| `AutomationQuiesced(env, old_epoch, barrier_id)` | Registered runtime/application authority confirms drained mutations; cannot itself grant control |
| `Destroy(env)` | Accepted stop, followed by terminal event; idempotent |

Owner heartbeat renews only the environment lifetime; control heartbeat must
match the current holder and lease epoch and renews only that lease. Healthy
controllers heartbeat even while holding a key without new input. Input traffic
does not implicitly renew either deadline. A grant renewal keeps its grant ID
but cannot revive an expired lease; `Control(acquire)` is still required.
Retention requires authenticated human acceptance and a host-approved deadline
(default maximum 30 minutes), not a unilateral owner nomination of another user.

Events carry monotonic sequence numbers and include `Ready`, `ApplicationExited`,
`WindowMapped/Unmapped`, `OutputChanged`, `ControlChanged`, `ViewerStateChanged`,
`TextFocusChanged`, `LimitPressure`, `Failed`, and `Stopped`.
Use a bounded 256-event replay ring;
a lagging consumer gets `ResyncRequired` and must call `Describe`, not silently
lose exits. Images/media are not JSON events. Terminal state remains queryable
for a bounded retention period (default 10 minutes).

Output spec: integer pixel width/height, integer scale (1 or 2), refresh/FPS cap,
render profile (`software` or a granted render node), and allowed stream bitrate.
Logical coordinates are pixels divided by scale; wire pointer coordinates use
physical output pixels. Refresh is virtual output cadence, not client game logic
rate. Effective settings and requested settings are both returned.

Errors include `Unauthorized`, `UnknownEnvironment`, `InvalidMode`,
`LimitExceeded`, `RendererUnavailable`, `IsolationUnavailable`,
`ExecutableUnavailable`, `ControlBusy`, `AutomationBusy`, `TextUnavailable`,
`StaleGeneration`, `StaleLease`, `ObservationTimeout`, `TransportUnavailable`,
and `EnvironmentFailed`.
Never substitute a black image, uncapped execution, or pretend launched process.

## 5. Headless rendering and desktop policy

Select **Smithay 0.7.0** as the research baseline; pin an exact tested release
when implementing. Smithay is a framework, not a ready-made desktop: virtual
outputs, WM policy, headless clock, and frame export are owned here [S1–S4].
No host seat, KMS modesetting, libinput, or physical vblank is needed.

Use `Gles2Renderer` with render-node/GBM EGL and offscreen targets for the hardware
profile. The software profile uses surfaceless EGL with Mesa software GL; expose
software Vulkan only when the launcher verifies lavapipe support. Software
compositing does not make arbitrary Vulkan games compatible. Pixman can support
SHM-only desktop clients but is not the primary game rendering profile [S2].
Probe imports, formats/modifiers, synchronization, and export at startup; fail an
explicit hardware request rather than silently switching the game to CPU rendering.

Required protocol policy: compositor/subcompositor, SHM, xdg-shell, output/xdg-output,
seat, data-device, viewporter, decorations, relative-pointer, pointer-constraints,
and text-input-v3 version 1 (phone text).
Advertise linux-dmabuf and explicit sync only when supported by the chosen renderer
and device. Presentation feedback describes composition completion on the virtual
output, **not** delivery to a remote screen. Honor supported FIFO/commit-timing
semantics; do not advertise globals without their implementation [S1–S3].
Computer-use capture/input is private compositor control, not globally exposed
virtual-keyboard/screencopy authority to every application.

Minimal WM policy: new normal windows cascade within output bounds; dialogs stay
above their parent; click-to-focus and explicit owner focus; maximize/fullscreen
fill the logical output; popups follow xdg positioning and grabs; window list
exposes opaque ID, title, app ID, geometry, and focus state. Closing a window is
not proof its process exited. Apply server decorations to clients requesting them;
do not double-decorate client-side windows. Keep ordinary drag/resize and text
selection usable. The harness chrome is outside the streamed desktop.

Xwayland is an optional creation-time compatibility choice, supervised per
environment. Implement Smithay's X11 WM handling, surface association, focus,
fullscreen, configure/resize, selections, and pointer grabs; merely spawning
Xwayland is insufficient [S4]. Run its sockets privately with Xauthority and
no TCP listener. Validate native Wayland and X11 games separately. Wine/Proton
compatibility depends on game, launcher, driver, and grants, not just Xwayland.

## 6. Modes, pacing, and system limits

Initial host policy (configurable downward; raising ceilings requires host approval):

| Budget | Software profile | Hardware profile |
|---|---|---|
| Default output | 1280×720, scale 1, 30 FPS | 1920×1080, scale 1, 60 FPS |
| Admitted dimensions | 320–3840 per axis, up to 8,294,400 pixels | Same |
| Admitted FPS | 1–60; default pixel-rate ceiling 62,208,000 pixels/s | 1–120; default ceiling 497,664,000 pixels/s |
| Per-environment CPU / RAM / swap | 400% / 8 GiB / 0 | Same |
| Tasks / output buffers / active viewers | 512 / 3 per negotiated frame pool / 2 | Same |
| Per-environment remote bitrate ceiling | 12 Mbit/s total media | Same |

These are policy ceilings, not performance claims. Admission also checks host
aggregate pixel rate, compositor/encoder memory estimates, viewers, CPU/RAM/task
budgets, and GPU lock availability. The existing shared `agents.slice` policy is
600% CPU/16 GiB; preserve stricter human-set limits [S5]. CPUQuota percentages
refer to one CPU, so 400% is four CPUs' bandwidth, not four dedicated cores [S6].
MemoryHigh/MemoryMax, MemorySwapMax, and TasksMax apply to the full environment
including encoder and audio. Verify required controllers and effective properties;
unavailable enforcement makes creation fail. Disk/scratch quotas require host
filesystem support; report whether enforced rather than confusing MemoryMax
with disk limits. Cap temporary images (two pending observations) and reclaim
unconsumed images after 30 seconds. Bound sockets, descriptors, SHM pool bytes,
client buffers, and event/input queues independently of process RAM ceilings.

Generic GPU execution/VRAM ceilings are **not promised**. Hold the repository's
shared advisory GPU lock for the hardware environment lifetime, including its
encoder. Serialize cooperating hardware desktops; return busy or a bounded wait
rather than silently sharing the device. This cannot prevent hostile GPU use or
protect against driver failure. Strict GPU denial uses device isolation; stronger
hardware isolation requires host-specific partitioning or a VM. Local viewer
import/display is not another exclusively locked agent GPU workload.

### Virtual clock and backpressure

Maintain monotonic deadlines with interval `1 / effective_fps`. Never render twice
inside one interval or catch up missed ticks with a burst. Process input and
Wayland requests independently of rendering. Issue frame callbacks for eligible
visible clients on virtual ticks even with no attached viewer, so applications do
not stall when humans disconnect. Frame callbacks throttle cooperative clients,
not arbitrary unthrottled game simulation; cgroup CPU limits govern that work.
Render damage/cursor changes or pending observations; skip redundant composition
when idle without starving clients waiting for callbacks. Fence completion, not
submission, governs buffer release and presentation feedback [S2, S3].

Raw-frame mailboxes hold at most one pending latest frame per consumer. Drop an
older pending frame, never mutate an in-flight buffer. Keep the composition target
separate from attachment-private export pools and the remote encoder mailbox:
one stalled local reader cannot hold the render target or another viewer's pool.
Three-buffer export pools are fixed-size; when a pool is held, skip that consumer's
export rather than block the seat or another consumer. Do not grow pools for slow
viewers. Sustained stalls detach the viewer and retire its pool under the local
revocation rules below. Encoding runs without B-frames/lookahead; congestion
reduces bitrate and stream FPS within ceilings. Stream-only downscaling may reduce
network cost but must not silently resize the application's desktop.

### Reconfiguration

Validate generation and budgets; allocate a new bounded pool while accounting
for old and new memory; then atomically publish output mode/scale and a new
geometry generation. Send Wayland configuration/X11 size updates, reset codec
state and request a keyframe, and mark old snapshots stale. Clients can retain
old-size buffers until they acknowledge/repaint; compose those safely rather
than claiming every app resizes synchronously. Reclaim old pools on release;
reject further resize requests while a previous pool cannot retire. Failure
before publication leaves the previous mode intact. Release held input; accept
new-generation pointer input only after `GeometryReady(viewer, generation)`,
which asserts the viewer has displayed an identified new-generation frame.
FPS-only changes do not increment geometry generation or replace media tracks.

## 7. Local and remote presentation

### Same-host path

Authenticate attachment at the private Unix socket using peer credentials and
a short-lived environment view grant. Negotiate version, pixel format, color
space, dimensions, strides, plane count, modifiers, and synchronization abilities.
Use `SCM_RIGHTS` for descriptors; never serialize FD integers as transferable
resources. Validate descriptors, sizes, plane ranges, and integer overflow [S7].

The portable native path is a fixed-size memfd pool. The compositor owns writable
mappings; viewers receive read-only descriptors/mappings and frame notifications
with pool ID, slot, frame ID, timestamp, and geometry generation. Do not reuse a
slot until all admitted readers release it. Create the producer mapping first,
then set `F_SEAL_GROW`, `F_SEAL_SHRINK`, `F_SEAL_FUTURE_WRITE`, and
`F_SEAL_SEAL` before sending descriptors. Future shared writable mappings and
write syscalls are denied; the producer's existing mapping remains writable.
Do not substitute `F_SEAL_WRITE`, which forbids the reusable producer mapping.
Snapshots use a separate immutable image or copy, not a live reusable slot.
This avoids codecs but still requires readback/upload where the renderer and
viewer cannot share textures. Memory copies and bandwidth must be measured.

Each local attachment receives its own pool; never export the shared composition
target. On disconnect, expiry, or revocation, stop all future writes of desktop
content into every allocation exported to that attachment, even if it previously
released a slot. A retained mapping/FD cannot be remotely revoked. Retire those
allocations permanently; no other viewer receives new frames through them.
This applies to SHM and DMA-BUF alike. Outstanding producer writes are fenced
before revocation completes. Retained/quarantined pools count against bounded
host admission until safely released; refuse replacement when the budget is
exhausted. At environment shutdown, abandon rather than recycle exported storage;
kernel references preserve its lifetime without retaining live producer processes.

The optimized adapter transfers DMA-BUF plane FDs, format/modifier/stride/offset,
an acquire sync-file fence, and pool/frame metadata. Viewer imports only supported
combinations, waits on acquire, displays, and releases with a completion fence
(or acknowledges after GPU completion). Producer waits for every reader's release
before reusing storage. Disconnect is not proof GPU access has completed.
Quarantine the disconnected viewer's pool until its outstanding GPU work is
known complete; do not kill the user's harness to retire a frame. Quarantined
pools remain charged to the environment/host budget; reject new DMA-BUF
attachments when that bounded budget is exhausted. Never synchronize with a
busy-wait on the GPUI thread [S2, S7, S8].

DMA-BUF zero-copy is conditional on renderer/export/import compatibility and the
viewer renderer exposing a foreign-image interface. GPUI 0.2.2 support is **not
established by this design**. Shared-memory import/upload is the required baseline;
a viewer implementation must explicitly negotiate DMA-BUF only after proving
its texture/fence integration. Different GPUs/drivers or unsupported modifiers
fall back to SHM without relaunching the game. Local audio travels as bounded PCM
blocks from the private sink to the viewer's playback adapter; no host microphone
capture is involved.

### Remote path

Choose **GStreamer through its Rust bindings**, using `webrtcbin` in the
presentation worker, for encoding, RTP, DTLS-SRTP, ICE, congestion feedback,
audio, and native receive/decode. Keep its event loop off the compositor and
GPUI threads. This accepts native dependencies in exchange for one established
media pipeline; pure-Rust WebRTC transports alone do not supply codecs [S9].
The packaging contract includes required plugins, TLS/ICE dependencies, and
runtime encoder discovery; missing required plugins fail stream attachment.

`webrtcbin` does not choose codecs or automatically tune the encoder. The
presentation implementation explicitly builds conversion/encoder/RTP elements,
connects transport-wide congestion feedback to a GStreamer congestion estimator,
and applies its target to encoder bitrate and the frame gate, under the aggregate
media ceiling. Use the official `webrtcsink` implementation as a reference for
this wiring, not its unauthenticated default signaling server [S9]. Admission
accounts for each viewer's encoder/peer independently; no SFU is required.

Negotiate VP8 and H.264 Constrained Baseline; prefer verified low-latency hardware
H.264 when available, otherwise software VP8. Only expose profiles/levels the
encoder and receiver actually support. Do not assume H.264 negotiation alone
admits 1080p60. HDR/HEVC/AV1 are not baseline. Video uses SDR sRGB content with
explicit YUV conversion/range metadata; snapshots remain lossless RGB, since
4:2:0 video can blur small text [S10]. Validate CPU encode against admitted
limits rather than promising 60 FPS on every software host.

The baseline sender consumes bounded CPU frames; optional DMA-BUF import into a
hardware encoder is another probed optimization, never a universal zero-copy
claim. A private PipeWire sink captures only environment applications; provide
private PulseAudio compatibility for games that need it. Encode Opus, use a
shared monotonic A/V clock, and bound audio buffering. No host microphone,
camera, or default audio graph is exposed [S11]. Local and remote viewers can
coexist; their backpressure is independent and their memory/bitrate budgets
are admitted together.

Signaling exchanges offer/answer and trickled ICE candidates over authenticated
HTTPS/WSS. Support configured STUN and TURN, including relay-only policy for
restricted networks. TURN credentials are ephemeral and supplied by the host,
not the agent. A usable TURN deployment is a remote-connectivity prerequisite,
not something the compositor invents. Fail connectivity after a 15-second
negotiation deadline; permit explicit fresh attachment/ICE negotiation [S12].
Remote access requires the provisioned gateway, trusted TLS, and reachable relay
configuration. No unauthenticated port or world-readable attach URL is created.

For geometry changes, pause pointer input and retire the old video track. Create
a replacement track containing only new-generation frames and signal
`{viewer, generation, track_id, SDP_mid, desktop_rect, encoded_size, scale}` over
the authenticated control path. Bind metadata to the browser/native receiver's
new transceiver/track identity, not to whichever video element is currently visible.
Start with a keyframe; the viewer sends `GeometryReady` only after presenting the
first frame of that identified track (browser `requestVideoFrameCallback` or
native presentation completion). Hide the stale old frame during the switch.
Bound simultaneous tracks/pools to old + new and charge their peak memory.
Negotiation/display failure keeps pointer input disabled and detaches that viewer
after the deadline; the desktop remains on its accepted mode. Ordinary
stream-only downscaling retains track/generation and uses the known desktop
rectangle, so equal encoded dimensions cannot disguise a desktop resize.

Use one reliable ordered data channel for all baseline pointer motion, key/button
transitions, text, and control acknowledgements. No second unordered motion
channel: a tap's move and button transitions form one ordered batch, so a click
cannot overtake its coordinates. Coalesce only adjacent unsent motion events,
never across a button/key/text transition; sum relative deltas rather than dropping
them. Cap input batches at 64 events and 1,000 events/second per controller.
Excess/stalled input detaches control and releases held keys/buttons instead of
retaining seconds of future clicks.

PLI/keyframe requests recover decoders; bounded retransmission follows transport
feedback. Never drop arbitrary already-encoded dependent frames as a raw-frame
backpressure policy. A disconnected stream is explicitly stale, not live visual
evidence. The viewer displays connection state and frame age; reconnection creates
a new input epoch, not replayed clicks. Network quality cannot increase host caps.

### Harness integration contract (not implemented here)

The viewer owns rendering decoded/local images, letterboxing, audio playback,
focus, pointer lock, touch interpretation, and connection/control indicators.
It submits authenticated commands and maps input; it cannot infer authority from
UI focus. Remote native viewing uses receive/decode into bounded images on a
background worker; GPU decode/import is optional. Browser viewing uses WebRTC
media elements/data channels without asking for microphone/camera permission.
A laptop on another machine always uses remote transport, even if on the same LAN.

Phones expose tap-to-click, drag, scroll, on-screen keys, and keyboard text input.
Touch gesture interpretation maps to the same mouse/keyboard interface; native
multi-touch games and gamepad controls need a separate negotiated capability.
Audio playback and browser pointer lock require user gestures; Escape/lock loss
and tab blur release held input. Pointer-lock availability varies across mobile
browsers: a phone can use desktop controls without a promise of FPS mouse-look
parity. Browser-reserved shortcuts cannot all be forwarded [S13].

### Chat artifact seam and current integration limit

Agreed descriptor for the separate runtime/chat implementation:

```text
DesktopSessionArtifact {
    version: 1,
    artifact_id,
    title,
    host_id,
    environment_id,
    grant_id
}
```

Persist only typed opaque IDs and display text in the conversation. Artifact
creation is **not** environment creation, grant issuance, or proof that a desktop
exists. A `desktop_artifact` tool may register durable details and mount/replay
the descriptor; its success must not be reported as streaming success.
No credentials, bearer URLs, Unix paths, SDP, or media belong in the descriptor.

The runtime owns a resolver at the desktop-backend seam. Given authenticated
principal plus descriptor, a real configured adapter validates host/environment/
grant and returns an authenticated browser viewer route with expiry and roles,
or explicit `Unsupported`, `Unauthorized`, `Expired`, `Stopped`, or `Unavailable`.
Host IDs resolve through trusted configuration, not an arbitrary agent-provided
URL. Re-resolve on every open; a replayed artifact never restores an old lease.
Authentication uses the viewer's principal/session or an out-of-band one-time
exchange, not a transcript token. Environment readiness is not viewer playback:
only received/presented frames establish a live stream.

**No actual desktop/media backend or viewer is introduced by this design PR.**
The separately agreed chat slice can ship descriptor/tool registration and
mount/replay with explicitly unsupported opening until a real backend exists.
It must not fabricate URLs, grants, live state, or production mock streams.
Native WebRTC receive is not required for that artifact slice. A future real
browser-viewer open is valid, but external-browser play must be labeled external,
not presented as inline GPUI interaction. Live game play inside the harness
still requires the compositor and an actual embedded/native viewer implementation.

## 8. Input, observation, and alternate automation

One virtual seat, one active control lease. The owner agent can acquire control
by default; view grants are read-only. A human with both view and control grants
can request control and preempt the agent. Control alone is not lifecycle authority.
Transition atomically emits synthetic key/button releases, increments lease epoch,
clears buffered input, and changes focus/control state. Other viewers remain
read-only. Lease heartbeat: 2 seconds, expiry: 6 seconds; disconnect/blur triggers
immediate release. Human release/expiry returns to **no controller**, not automatic
agent replay; the owner must acquire a fresh agent lease. This makes disconnect
safe. The administrative stop/revoke route remains available outside the seat.

Input includes sequence, lease epoch, geometry generation, physical key code
(Linux evdev mapping negotiated from browser `KeyboardEvent.code`), key/button
state, absolute pixel position or relative deltas, scroll units, and modifiers.
Explicit key down/up goes through xkb state; text/IME composition uses a distinct
text-input capability and must not pretend Unicode text is physical game keys.
Reject duplicates, stale epochs/generations, nonfinite/out-of-range coordinates,
and unnegotiated kinds. Acknowledgement means delivered to the seat, not that the
application consumed it. Clicks are never automatically retried after ambiguity.

Phone text uses negotiated `text_v3`: `TextPreedit(utf8, cursor_begin,
cursor_end, focus_epoch)` and `TextCommit(utf8, focus_epoch)`, maximum 4,000 UTF-8
bytes each, indices at byte/code-point boundaries. The compositor implements
`zwp_text_input_manager_v3` and acts as the input method for the active focused
text-input instance using Smithay's text-input handle. Send preedit/commit events
followed by the protocol `done` with the current client commit serial. Focus
changes, disable, lease changes, and disconnect clear preedit and increment the
text focus epoch; stale commits are rejected, not delivered to the next window.
Do not echo surrounding/password text into telemetry or expose it to observers.
Native/Xwayland clients without enabled text-input-v3 receive explicit
`TextUnavailable`; the phone still offers physical on-screen keys/modifiers for
game control. Unicode/IME text in X11 is not claimed via fake keycode synthesis.
Mobile acceptance must exercise an actual text-input-v3 client and the visible
unsupported-text behavior for X11 [S14].

The viewer removes letterbox offsets and maps to physical desktop pixels. A
scaled stream supplies its encoded dimensions plus the full desktop rectangle;
input still targets desktop coordinates. Relative input is used only when both
viewer pointer lock and application pointer constraints authorize it. No `/dev/uinput`
is needed: inject authorized events directly into Smithay's virtual seat [S1].

`Observe` returns frame ID, monotonic capture timestamp, generation, dimensions,
scale, focused window, and a lossless composed image reference. With `after_frame`,
wait for a newer frame up to the requested timeout; do not relabel an old frame.
Computer-use actions reference the observation generation and obey the same lease
as humans. Observation rate/size are capped independently of stream FPS.

Alternate automation (application RPC, browser automation, accessibility, test
hooks) is allowed through explicitly granted sockets in the environment. The
compositor does not pretend Wayland supplies a universal accessibility tree.
Before human control, stop mutation admission and require a runtime/application
`AutomationQuiesced(env, old_epoch)` acknowledgement proving all outstanding
mutations drained/cancelled, or application-side rejection of the old epoch at
execution. Submission-time lease checking alone is insufficient. The supervisor
grants the human lease only after this barrier; timeout, failed runtime, or an
interface that cannot prove quiescence returns `AutomationBusy` and grants no
controller. Mark unmediated app RPC non-exclusive and refuse handoff until it is
stopped and drained. Releasing a seat lease cannot revoke direct game RPC access.
Read-only observations may continue while the human plays.

## 9. Concrete live-game flow

1. Agent calls `Create` with workspace grant, hardware/software profile, output
   dimensions, FPS and budgets. It receives effective configuration or a concrete
   admission error; it does not get a host desktop.
2. Agent calls `Launch` for the actual built game. Runtime waits for mapped window
   and a new rendered frame, and reports launch failure/exit without fake evidence.
3. Agent acquires input, observes, and exercises the game. Application RPC may be
   used if governed by the same lease.
4. Agent issues recipient-bound **view and control** grants to the human and sends
   an environment reference through the runtime. The harness chooses local
   SHM/DMA-BUF on the host or remote WebRTC on a laptop/phone; it presents the
   running game, not a prerecorded clip.
5. Human selects **Take control**. The supervisor releases agent input and completes
   the alternate-automation quiescence barrier before granting the human lease.
   User plays; agent may observe. Harness chrome includes release control;
   destroy is offered only to an authenticated owner/administrator.
6. A resize changes output generation; viewer presents an identified new-generation
   frame and sends `GeometryReady` before more pointer input. Network downscaling
   alone does not resize the game. Phone orientation changes viewer layout unless
   the owner explicitly requests desktop reconfiguration.
7. Human releases control; agent explicitly reacquires. Task completion destroys
   the environment unless a human accepts bounded retention ownership. Saved
   workspace artifacts persist; the live desktop is ephemeral.

## 10. Security and failure contract

Capabilities are environment-scoped, role-scoped (`owner`, `view`, `control`),
audience-bound, revocable, and bound to an authenticated principal. Role grants
have a host-bounded lifetime (default 30 minutes); one-use attach tokens expire
after 60 seconds if not consumed. Token consumption binds the live connection
to the grant ID/principal, not to a reusable bearer URL. Grant expiry/renewal and
revocation apply to existing peers, independently of the consumed token.
Secrets are not embedded in journaled URLs, logs, screenshots, or media.
Authorization is checked at attach, lease acquisition, and every command/input
batch; revocation releases input, closes peers, and retires local exported pools
without future writes. Gate browser origins and prevent cross-origin authenticated
signaling/management requests. DTLS encryption is transport protection, not
application authorization [S12].

| Failure | Required observable result |
|---|---|
| Missing systemd/controllers/isolation/render node | Creation rejected before apps execute |
| CPU throttling / memory pressure | Effective limits + pressure events; no uncapped fallback |
| Application OOM / fork limit / crash | Exit/resource failure with app identity; healthy control worker remains if possible |
| Compositor OOM / GPU reset / worker crash | Environment failure, app teardown, viewers marked disconnected |
| Encoder/plugin failure | Stream attach fails or viewer detaches; desktop remains observable locally |
| Slow local importer / decoder | Bounded buffers, dropped pending raw frames, eventual detach; no seat stall |
| ICE/TURN failure / packet loss | Explicit connection state/deadline; keyframe recovery where possible |
| Lease expiry / blur / disconnect / revocation | Keys/buttons released; no controller and no replay |
| Stale observation/resize/input | Explicit generation error; resync before actions |
| Destroy during launch or resize | Stop wins; no new app/pool escapes teardown |
| Viewer disconnect with GPU buffer in use | Safe fence retirement; descriptor lifetime is not GPU completion |

Desktop pixels and window titles are potentially sensitive. Grants authorize the
whole environment, not a redacted per-window view. Clipboard is environment-local;
no automatic host clipboard or drag/drop bridging. Any future file/clipboard
transfer requires explicit user grants. Media is live, not recorded by default.

## 11. Implementation acceptance and evidence gates

These are required future acceptance scenarios, **not verification claims for this
document**. Use real applications and protocol flow, deterministic regression
coverage for state/lease/generation invariants, and screenshots plus live interaction
for viewer work. Do not call compilation or a mocked media track proof of a desktop.

| Scenario | Evidence required |
|---|---|
| Software headless desktop | Launch without host DISPLAY/input devices; Wayland app frame; 1280×720@30 and 1920×1080@30; no physical monitor |
| Native and X11 game | Real keyboard/mouse game in both modes; focus, fullscreen, resize, relative mouse; report driver/game versions |
| Hardware profile | Scoped/locked render-node run; actual frame import/export; unsupported device rejects explicitly |
| Agent computer use | Snapshot -> authorized click/key -> observable changed application state; stale generation/lease rejection |
| Alternate automation handoff | Delay an in-flight mutation during takeover; grant only after acknowledged quiescence; failed authority rejects handoff |
| Same-host baseline | Native viewer consumes SHM without video codec pipeline; plays game; slow consumer cannot grow buffers |
| Same-host optimization | Actual DMA-BUF import and fence release, mismatch fallback, disconnect with in-flight GPU work; measure copies rather than assert zero-copy |
| Remote laptop | Real browser and native receive/decode; play game with audio; direct ICE and forced TURN paths; reconnect never replays input |
| Remote phone | At least Android Chrome and iOS Safari; tap/drag/scroll/text, orientation/letterbox mapping, autoplay/lock limitations visibly handled |
| Local + remote together | Same frame generation, independent backpressure, exactly one controller; viewer count/bitrate admission |
| Limits and pacing | Measure composition/export rate at 15/30/60 caps, including stalls/no viewer; no catch-up burst; stress CPU/RAM/tasks under effective cgroup caps |
| Resize under load | Change mode/FPS while streaming; display new-track frame before geometry acknowledgement; old-frame input rejected even at identical encoded size |
| Cancellation/cleanup | Launch grandchildren, cancel task/destroy during startup; all environment processes/sockets/fences retired; files preserved |
| Authorization/isolation | Wrong owner/token/origin fails; revoked local viewer retains old FD but receives no new content; host credentials absent; sandbox downgrade refused |

Suggested implementation order: supervised Wayland/software environment and
private observation/input; native SHM viewer; Xwayland/game and hardware renderer;
remote A/V/input with forced TURN; DMA-BUF optimization once viewer imports are
proven. These are dependency gates, not permission to call partial implementation
complete. Runtime and viewer owners must implement their integration contracts
before the live-in-harness game scenario can ship.

Performance targets to measure, not guarantees: same-host composition-to-display
p95 below two frame intervals; remote input-to-visible response p95 below 150 ms
on a specified 30 ms RTT test link at 720p60; no sustained raw-frame queue growth.
Report requested/effective FPS, frame age, render/encode time, drops, control holder,
transport and effective limits. Do not log input text or pixel contents as metrics.

## 12. Primary-source basis and unresolved implementation proofs

Research checked on 2026-10-11. This design copies no upstream code/assets.
Implementation must retain upstream notices when adapting examples. Smithay and
GStreamer/plugin dependency licenses and codec distribution obligations must be
reviewed for the actual selected package set, not inferred from Rust bindings.

- **S1:** [Smithay 0.7.0 framework and module inventory](https://docs.rs/smithay/0.7.0/smithay/),
  including calloop state ownership and protocol helpers.
- **S2:** [Smithay renderer interfaces](https://docs.rs/smithay/0.7.0/smithay/backend/renderer/index.html),
  [EGL](https://docs.rs/smithay/0.7.0/smithay/backend/egl/index.html), and
  [offscreen maintainer guidance](https://github.com/Smithay/smithay/issues/344).
- **S3:** [Virtual output interface](https://docs.rs/smithay/0.7.0/smithay/output/struct.Output.html)
  and [presentation feedback](https://docs.rs/smithay/0.7.0/smithay/wayland/presentation/index.html).
- **S4:** [Smithay Xwayland helpers and compositor WM responsibilities](https://docs.rs/smithay/0.7.0/smithay/xwayland/index.html).
- **S5:** [Existing Juto governance](../crates/juto-runtime/src/workspace.rs),
  [resource wrapper](../scripts/agent-scope.sh), and [runtime design](omp-runtime.md).
- **S6:** [systemd resource control](https://www.freedesktop.org/software/systemd/man/latest/systemd.resource-control.html),
  [process-tree termination](https://www.freedesktop.org/software/systemd/man/latest/systemd.kill.html),
  and [kernel cgroup v2](https://www.kernel.org/doc/html/latest/admin-guide/cgroup-v2.html).
- **S7:** [Unix descriptor passing](https://man7.org/linux/man-pages/man7/unix.7.html)
  and [memfd/sealing semantics](https://man7.org/linux/man-pages/man2/memfd_create.2.html).
- **S8:** [Kernel DMA-BUF and synchronization](https://docs.kernel.org/driver-api/dma-buf.html).
- **S9:** [GStreamer webrtcbin](https://gstreamer.freedesktop.org/documentation/webrtc/index.html),
  [webrtcsink encoding/congestion implementation](https://gstreamer.freedesktop.org/documentation/rswebrtc/webrtcsink.html),
  and [official Rust bindings](https://gitlab.freedesktop.org/gstreamer/gstreamer-rs).
- **S10:** [RFC 7742 video codec/resize/screen-content requirements](https://www.rfc-editor.org/rfc/rfc7742).
- **S11:** [PipeWire access control](https://docs.pipewire.org/page_access.html)
  and [RFC 7874 audio requirements](https://www.rfc-editor.org/rfc/rfc7874).
- **S12:** [RFC 8445 ICE](https://www.rfc-editor.org/rfc/rfc8445),
  [RFC 8656 TURN](https://www.rfc-editor.org/rfc/rfc8656),
  and [RFC 8827 WebRTC security architecture](https://www.rfc-editor.org/rfc/rfc8827).
- **S13:** [W3C Pointer Lock](https://www.w3.org/TR/pointerlock-2/),
  [keyboard physical-code specification](https://www.w3.org/TR/uievents-code/),
  and [WebRTC specification](https://www.w3.org/TR/webrtc/).
- **S14:** [Smithay text-input delivery handle](https://docs.rs/smithay/0.7.0/smithay/wayland/text_input/struct.TextInputHandle.html)
  and [Wayland text-input-v3 protocol](https://gitlab.freedesktop.org/wayland/wayland-protocols/-/blob/main/unstable/text-input/text-input-unstable-v3.xml).

Before implementation can claim the corresponding capability: prove GPUI foreign
texture/fence support, tested renderer/encoder format compatibility, mobile-browser
interop, private audio routing, enforceable host isolation profile, and deployed
TLS/TURN reachability. These are concrete validation dependencies, not reasons to
replace live game interaction with a recording.
