//! Frame pacing for the compositor loop.
//!
//! The compositor composes only when the desktop changed or a snapshot was
//! requested, and never more than once per interval of the effective FPS.
//! Frame callbacks run on the same cadence even when nothing changed, so idle
//! clients keep their schedule without triggering a composition.

use std::time::{Duration, Instant};

/// What one pacing pass should do.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PacingDecision {
    /// Compose and export a frame now.
    pub compose: bool,
    /// Send frame callbacks to clients now.
    pub callbacks: bool,
}

/// Remembers when the compositor last composed and last sent callbacks.
#[derive(Debug)]
pub(crate) struct Pacer {
    interval: Duration,
    last_compose: Option<Instant>,
    last_callbacks: Option<Instant>,
}

impl Pacer {
    pub(crate) fn new(fps: u32) -> Self {
        Self {
            interval: interval_for(fps),
            last_compose: None,
            last_callbacks: None,
        }
    }

    /// Changes the cadence; it applies from the next decision.
    pub(crate) fn set_fps(&mut self, fps: u32) {
        self.interval = interval_for(fps);
    }

    /// Decides what to do at `now`. `want_compose` says the desktop changed or
    /// a snapshot was requested. Composition never repeats inside one interval,
    /// and callbacks go out at least once per interval.
    pub(crate) fn decide(&mut self, now: Instant, want_compose: bool) -> PacingDecision {
        let compose = want_compose && due(self.last_compose, now, self.interval);
        let callbacks = compose || due(self.last_callbacks, now, self.interval);
        if compose {
            self.last_compose = Some(now);
        }
        if callbacks {
            self.last_callbacks = Some(now);
        }
        PacingDecision { compose, callbacks }
    }

    /// How long the loop may sleep before `decide` could return something new.
    pub(crate) fn next_wake(&self, now: Instant, want_compose: bool) -> Duration {
        let mut wake = until_due(self.last_callbacks, now, self.interval);
        if want_compose {
            wake = wake.min(until_due(self.last_compose, now, self.interval));
        }
        wake
    }
}

fn interval_for(fps: u32) -> Duration {
    Duration::from_micros(1_000_000 / u64::from(fps.max(1)))
}

fn due(last: Option<Instant>, now: Instant, interval: Duration) -> bool {
    last.is_none_or(|last| now.saturating_duration_since(last) >= interval)
}

fn until_due(last: Option<Instant>, now: Instant, interval: Duration) -> Duration {
    last.map_or(Duration::ZERO, |last| {
        (last + interval).saturating_duration_since(now)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn at(base: Instant, ms: u64) -> Instant {
        base + Duration::from_millis(ms)
    }

    #[test]
    fn first_wanted_frame_composes_immediately() {
        let base = Instant::now();
        let mut pacer = Pacer::new(10);
        let decision = pacer.decide(base, true);
        assert!(decision.compose);
        assert!(decision.callbacks);
    }

    #[test]
    fn continuous_damage_composes_at_most_once_per_interval() {
        let base = Instant::now();
        let mut pacer = Pacer::new(10);
        let composes: Vec<u64> = (0..1000u64)
            .filter(|ms| pacer.decide(at(base, *ms), true).compose)
            .collect();
        assert_eq!(composes.len(), 10);
        assert!(composes.windows(2).all(|pair| pair[1] - pair[0] >= 100));
    }

    #[test]
    fn deferred_compose_waits_for_the_next_interval() {
        let base = Instant::now();
        let mut pacer = Pacer::new(10);
        assert!(pacer.decide(base, true).compose);
        assert!(!pacer.decide(at(base, 30), true).compose);
        assert_eq!(
            pacer.next_wake(at(base, 30), true),
            Duration::from_millis(70)
        );
    }

    #[test]
    fn idle_clients_get_callbacks_without_composition() {
        let base = Instant::now();
        let mut pacer = Pacer::new(5);
        let mut callbacks = Vec::new();
        for ms in 0..1000u64 {
            let decision = pacer.decide(at(base, ms), false);
            assert!(!decision.compose);
            if decision.callbacks {
                callbacks.push(ms);
            }
        }
        assert_eq!(callbacks, vec![0, 200, 400, 600, 800]);
    }

    #[test]
    fn set_fps_changes_the_interval() {
        let base = Instant::now();
        let mut pacer = Pacer::new(10);
        assert!(pacer.decide(base, true).compose);
        pacer.set_fps(5);
        assert!(!pacer.decide(at(base, 150), true).compose);
        assert!(pacer.decide(at(base, 200), true).compose);
    }
}
