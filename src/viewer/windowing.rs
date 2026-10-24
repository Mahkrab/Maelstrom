//! Window transport is separate from graphics and simulation selection. An
//! explicit choice lets WSLg use X11 without changing other applications' displays.

use crate::render::RenderError;
use winit::event_loop::EventLoop;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Backend {
    Auto,
    X11,
    Wayland,
}
impl Backend {
    fn parse(value: Option<&std::ffi::OsStr>) -> Result<Self, RenderError> {
        match value.and_then(std::ffi::OsStr::to_str) {
            None if value.is_none() => Ok(Self::Auto),
            Some("auto") => Ok(Self::Auto),
            Some("x11") => Ok(Self::X11),
            Some("wayland") => Ok(Self::Wayland),
            _ => Err(RenderError(
                "MAELSTROM_WINDOW_BACKEND must be auto, x11 or wayland".into(),
            )),
        }
    }
    pub(super) fn read() -> Result<Self, RenderError> {
        Self::parse(std::env::var_os("MAELSTROM_WINDOW_BACKEND").as_deref())
    }
}

pub(crate) fn create() -> Result<EventLoop<()>, RenderError> {
    let mut builder = EventLoop::builder();
    match Backend::read()? {
        Backend::Auto => {}
        #[cfg(target_os = "linux")]
        Backend::X11 => {
            use winit::platform::x11::EventLoopBuilderExtX11;
            builder.with_x11();
        }
        #[cfg(target_os = "linux")]
        Backend::Wayland => {
            use winit::platform::wayland::EventLoopBuilderExtWayland;
            builder.with_wayland();
        }
        #[cfg(not(target_os = "linux"))]
        _ => {
            return Err(RenderError(
                "explicit window transport requires Linux".into(),
            ));
        }
    }
    builder
        .build()
        .map_err(|e| super::window_error("window event loop unavailable", e))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn window_transport_is_explicit_and_rejects_invalid_names() {
        assert_eq!(Backend::parse(None).unwrap(), Backend::Auto);
        for (name, expected) in [
            ("auto", Backend::Auto),
            ("x11", Backend::X11),
            ("wayland", Backend::Wayland),
        ] {
            assert_eq!(Backend::parse(Some(name.as_ref())).unwrap(), expected);
        }
        for value in ["", "X11", "cuda", "software"] {
            assert!(Backend::parse(Some(value.as_ref())).is_err());
        }
        #[cfg(unix)]
        {
            use std::os::unix::ffi::OsStrExt;
            assert!(Backend::parse(Some(std::ffi::OsStr::from_bytes(b"\xff"))).is_err());
        }
    }
}
