//! Discovery and diagnostics do not start a fluid run or modify application setup.
use super::{
    CliError,
    arguments::{Cli, Doctor},
    storage::Storage,
};
use clap::CommandFactory;
use serde_json::json;

pub(super) fn paths(cli: &Cli, storage: &Storage) -> Result<u8, CliError> {
    super::print_document(
        cli.format,
        format!(
            "Scenes: {}\nRuns: {}\nPreferences: {}",
            storage.scenes.display(),
            storage.runs.display(),
            storage.preferences.display()
        ),
        json!({"schema_version":1,"scenes":storage.scenes,"runs":storage.runs,"preferences":storage.preferences}),
    )?;
    Ok(0)
}
pub(super) fn config(cli: &Cli, storage: &Storage) -> Result<u8, CliError> {
    let p = storage.preferences()?;
    let value = json!({"schema_version":1,"path":storage.preferences,"exists":storage.preferences.exists(),
        "numerical_mode":p.numerical_mode.as_deref().unwrap_or("fast"),
        "backend":p.backend.as_deref().unwrap_or("auto"),"threads":p.threads,"device":p.device,"block_size":p.block_size,
        "precedence":"explicit command selection > application preferences > built-in auto policy; scene physics is separate"});
    super::print_document(
        cli.format,
        serde_json::to_string_pretty(&value).map_err(|e| CliError::output(e.to_string()))?,
        value,
    )?;
    Ok(0)
}
pub(super) fn doctor(cli: &Cli, command: Option<&Doctor>) -> Result<u8, CliError> {
    match command {
        None => {
            let value = json!({"schema_version":1,"features":{"cuda":cfg!(feature="cuda"),"render":cfg!(feature="render"),"viewer":cfg!(feature="viewer")},
                "guidance":"Use doctor cuda or doctor graphics to exercise available resources. Install with --features viewer for compile/play; add cuda for NVIDIA simulation."});
            super::print_document(
                cli.format,
                serde_json::to_string_pretty(&value)
                    .map_err(|e| CliError::output(e.to_string()))?,
                value,
            )?;
        }
        Some(Doctor::Cuda {
            device,
            repetitions,
        }) => {
            let count = std::num::NonZeroU32::new(*repetitions)
                .ok_or_else(|| CliError::usage("repetitions must be positive".into()))?;
            let report = maelstrom::cuda::verify_roundtrip(*device, count).map_err(|e| {
                CliError::input(format!(
                    "{e}; CUDA builds require --features cuda and a compatible device"
                ))
            })?;
            super::print_document(
                cli.format,
                format!("CUDA verified: {}", report.device.name),
                json!({"schema_version":1,"operation":"cuda_check","report":report}),
            )?;
        }
        Some(Doctor::Graphics {
            graphics_adapter,
            software,
        }) => graphics(cli, *graphics_adapter, *software)?,
    }
    Ok(0)
}
#[cfg(feature = "render")]
fn graphics(cli: &Cli, index: usize, software: bool) -> Result<(), CliError> {
    use maelstrom::render::{Graphics, GraphicsSelection};
    let instance = Graphics::instance();
    let graphics = pollster::block_on(Graphics::new(
        &instance,
        GraphicsSelection { index, software },
        None,
        wgpu::TextureFormat::Rgba8UnormSrgb,
    ))
    .map_err(|e| CliError::input(e.to_string()))?;
    graphics
        .check()
        .map_err(|e| CliError::input(e.to_string()))?;
    super::print_document(
        cli.format,
        format!("Graphics verified: {}", graphics.adapter_info.name),
        json!({"schema_version":1,"backend":format!("{:?}",graphics.adapter_info.backend),"device_type":format!("{:?}",graphics.adapter_info.device_type),"name":graphics.adapter_info.name,"driver":graphics.adapter_info.driver,"software_requested":software,"window_checked":false}),
    )
}
#[cfg(not(feature = "render"))]
fn graphics(_: &Cli, _: usize, _: bool) -> Result<(), CliError> {
    Err(CliError::input("graphics support is not installed; reinstall with --features viewer (or render for windowless compilation)".into()))
}
pub(super) fn completions(shell: &str) -> Result<u8, CliError> {
    let shell: clap_complete::Shell = shell
        .parse()
        .map_err(|_| CliError::usage("unsupported completion shell".into()))?;
    let mut command = Cli::command();
    let mut buffer = Vec::new();
    clap_complete::generate(shell, &mut command, "maelstrom", &mut buffer);
    super::print_stdout(
        std::str::from_utf8(&buffer).map_err(|e| CliError::output(e.to_string()))?,
    )?;
    Ok(0)
}
