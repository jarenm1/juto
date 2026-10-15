//! Shared palette for the chat shell and focus-thread sidebar: neutral gray
//! surfaces with a single blue accent and two signal colors.

/// App and sidebar background.
pub const BG: u32 = 0x181818;
/// Hovered row or control surface.
pub const BG_HOVER: u32 = 0x212121;
/// Selected row, inset fields, and raised controls.
pub const BG_RAISED: u32 = 0x262626;
/// Hairline dividers between regions.
pub const BORDER: u32 = 0x2a2a2a;
/// Control outlines and progress-bar tracks.
pub const BORDER_STRONG: u32 = 0x3a3a3a;

pub const TEXT: u32 = 0xe6e6e6;
pub const TEXT_MUTED: u32 = 0x9a9a9a;
pub const TEXT_FAINT: u32 = 0x6b6b6b;

/// Primary accent: selection, active stage fill, primary actions.
pub const ACCENT: u32 = 0x7aa2f7;
/// Text on an accent-filled surface.
pub const ON_ACCENT: u32 = 0x10141c;
/// Needs attention: blocked threads.
pub const WARNING: u32 = 0xe0af68;
/// Failures and destructive actions.
pub const DANGER: u32 = 0xf7768e;
/// Added diff lines.
pub const DIFF_ADD: u32 = 0x9ece6a;
pub const DIFF_ADD_BG: u32 = 0x1d2620;
pub const DIFF_REMOVE_BG: u32 = 0x2b1e22;

/// Height shared by the sidebar brand row and the chat header so their
/// bottom borders line up.
pub const HEADER_HEIGHT: f32 = 56.0;
