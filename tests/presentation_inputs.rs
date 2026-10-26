//! Presentation parsing is available in headless builds. Strict field ownership
//! and version checks do not depend on an installed graphics driver.

use maelstrom::presentation::{CameraMode, ParticleColours, Presentation};

#[test]
fn presentation_versions_camera_modes_and_roundtrip_are_explicit() {
    assert_eq!(Presentation::default().camera_mode, CameraMode::Fit);
    assert_eq!(
        Presentation::parse("schema_version=1").unwrap().camera_mode,
        CameraMode::Fixed
    );
    assert_eq!(
        Presentation::parse("schema_version=2").unwrap().camera_mode,
        CameraMode::Fit
    );
    let fixed = Presentation::parse(
        "schema_version=2\n[camera]\neye_m=[1.0,2.0,3.0]\ntarget_m=[0.0,0.0,0.0]",
    )
    .unwrap();
    assert_eq!(fixed.camera_mode, CameraMode::Fixed);
    let reread = Presentation::parse(&fixed.to_toml().unwrap()).unwrap();
    assert_eq!(reread.settings.camera, fixed.settings.camera);
    assert_eq!(reread.camera_mode, CameraMode::Fixed);
    for source in [
        "schema_version=0",
        "schema_version=3",
        "width=10",
        "schema_version=1\ncamera_mode='fit'",
        "schema_version=2\ncamera_mode='fixed'",
        "schema_version=2\ncamera_mode='fit'\n[camera]\neye_m=[1.0,2.0,3.0]",
        "schema_version=2\nunknown=1",
        "schema_version=2\nwidth=0",
        "schema_version=2\nwidth=8193",
        "schema_version=2\nstep_stride=0",
        "schema_version=2\nheight=0",
        "schema_version=2\nformat='jpeg'",
        "schema_version=2\n[camera]\neye_m=[nan,0.0,0.0]",
    ] {
        assert!(Presentation::parse(source).is_err(), "{source}");
    }
}

#[test]
fn colour_domains_and_supported_drawing_roundtrip() {
    let prefix = "schema_version=2\n[particles]\n";
    let fixed = Presentation::parse(&format!("{prefix}colour='#2B9FD9'")).unwrap();
    assert!(fixed.drawing_supported());
    let energy = Presentation::parse(&format!(
        "{prefix}colour_mode='kinetic_energy'\nmin_energy_j=0.0\nmax_energy_j=1.0"
    ))
    .unwrap();
    assert_eq!(
        energy.settings.particles,
        ParticleColours::KineticEnergy {
            min_energy_j: 0.0,
            max_energy_j: 1.0
        }
    );
    assert!(energy.drawing_supported());
    assert_eq!(
        Presentation::parse(&energy.to_toml().unwrap())
            .unwrap()
            .settings
            .particles,
        energy.settings.particles
    );
    for invalid in [
        "colour='blue'",
        "colour='#12345G'",
        "colour_mode='temperature'",
        "colour_mode='fixed'\nmin_energy_j=0.0",
        "colour_mode='kinetic_energy'",
        "colour_mode='kinetic_energy'\nmin_energy_j=0.0\nmax_energy_j=0.0",
        "colour_mode='kinetic_energy'\nmin_energy_j=-1.0\nmax_energy_j=2.0",
        "colour_mode='kinetic_energy'\nmin_energy_j=0.0\nmax_energy_j=inf",
        "colour_mode='kinetic_energy'\nmin_energy_j=nan\nmax_energy_j=2.0",
        "colour_mode='kinetic_energy'\nmin_energy_j=0.0\nmax_energy_j=2.0\ncolour='#ff0000'",
    ] {
        assert!(
            Presentation::parse(&format!("{prefix}{invalid}")).is_err(),
            "{invalid}"
        );
    }
}
