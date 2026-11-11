//! Large manifest arrays stream from temporary logs. The public schema stays
//! unchanged; command-line exports return only their small completion summary.
use super::{ExportError, FrameSchedule, fail, frame_index::FrameIndex};
use crate::{storage::JsonLog, workflow::RecordedInput};
use serde::{
    Serialize, Serializer,
    ser::{Error, SerializeMap},
};
use serde_json::Value;
use std::{
    io::{self, BufWriter, Write},
    path::Path,
};

pub(super) struct Manifest<'a> {
    pub metadata: &'a Value,
    pub frames: &'a FrameIndex,
    pub input: &'a RecordedInput,
    pub diagnostics: Option<&'a JsonLog>,
    pub request: Option<(u64, &'a FrameSchedule)>,
}
impl Serialize for Manifest<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let metadata = self
            .metadata
            .as_object()
            .ok_or_else(|| S::Error::custom("invalid manifest metadata"))?;
        let mut map = serializer.serialize_map(None)?;
        for (key, value) in metadata {
            map.serialize_entry(key, value)?;
        }
        map.serialize_entry("frames", self.frames)?;
        map.serialize_entry("input", self.input)?;
        if let Some(diagnostics) = self.diagnostics {
            map.serialize_entry("accepted_diagnostics", diagnostics)?;
        }
        if let Some((steps, schedule)) = self.request {
            #[derive(Serialize)]
            struct Request<'a> {
                steps: String,
                checkpoints: Checkpoints<'a>,
            }
            struct Checkpoints<'a>(&'a FrameSchedule);
            impl Serialize for Checkpoints<'_> {
                fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
                    serializer.collect_seq(self.0.steps.iter().map(|step| step.to_string()))
                }
            }
            map.serialize_entry(
                "request",
                &Request {
                    steps: steps.to_string(),
                    checkpoints: Checkpoints(schedule),
                },
            )?;
        }
        map.end()
    }
}
pub(super) fn publish(path: &Path, value: &impl Serialize) -> Result<(), ExportError> {
    crate::workflow::output::publish_with(path, |file| {
        let mut writer = BufWriter::with_capacity(64 * 1024, file);
        serde_json::to_writer_pretty(&mut writer, value).map_err(io::Error::other)?;
        writer.flush()
    })
    .map_err(fail)
}
pub(super) fn finish(
    path: &Path,
    metadata: &Value,
    document: &impl Serialize,
    retain: bool,
) -> Result<Value, ExportError> {
    if retain {
        // Legacy callers request an owned document. Build it before publishing
        // so a result-assembly error cannot follow a completion marker.
        let value = serde_json::to_value(document).map_err(fail)?;
        publish(path, &value)?;
        Ok(value)
    } else {
        publish(path, document)?;
        Ok(metadata.clone())
    }
}
pub(super) fn progress(
    path: &Path,
    metadata: &Value,
    frames: &FrameIndex,
) -> Result<(), ExportError> {
    struct Progress<'a>(&'a Value, &'a FrameIndex);
    impl Serialize for Progress<'_> {
        fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            let fields = self
                .0
                .as_object()
                .ok_or_else(|| S::Error::custom("invalid progress metadata"))?;
            let mut map = serializer.serialize_map(None)?;
            for (key, value) in fields {
                if key != "frames" {
                    map.serialize_entry(key, value)?;
                }
            }
            map.serialize_entry("frames", self.1)?;
            map.end()
        }
    }
    let temp = path.join(".progress.tmp");
    publish(&temp, &Progress(metadata, frames))?;
    std::fs::rename(temp, path.join("progress.json")).map_err(fail)
}
