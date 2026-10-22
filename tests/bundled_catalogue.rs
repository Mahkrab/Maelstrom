//! Build discovery is checked against tiny independent directories, including
//! additions and removals that previously required edits to the Rust registry.

#[path = "../build_support/bundled.rs"]
mod bundled;

use std::{
    fs,
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
};

static NEXT: AtomicU64 = AtomicU64::new(0);

struct Inputs(PathBuf);

impl Inputs {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!(
            "maelstrom-bundled-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&path).unwrap();
        fs::create_dir(path.join("scenes")).unwrap();
        fs::create_dir(path.join("config")).unwrap();
        Self(path)
    }
}

impl Drop for Inputs {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

#[test]
fn catalogue_is_sorted_and_tracks_added_and_removed_files() {
    let inputs = Inputs::new();
    for name in ["zeta", "alpha"] {
        fs::write(inputs.0.join(format!("scenes/{name}.toml")), "scene").unwrap();
    }
    fs::write(inputs.0.join("scenes/notes.md"), "not an input").unwrap();
    fs::write(inputs.0.join("config/water.toml"), "configuration").unwrap();
    let first = bundled::generate(&inputs.0).unwrap();
    assert_eq!(first, bundled::generate(&inputs.0).unwrap());
    let rows: Vec<_> = first
        .lines()
        .filter(|line| line.starts_with("    ("))
        .collect();
    assert_eq!(rows.len(), 3);
    for (row, key, file) in [
        (rows[0], "alpha", "scenes/alpha.toml"),
        (rows[1], "zeta", "scenes/zeta.toml"),
        (rows[2], "config/water.toml", "config/water.toml"),
    ] {
        assert_eq!(
            row,
            format!("    ({key:?}, include_str!({:?})),", inputs.0.join(file))
        );
    }
    fs::write(inputs.0.join("scenes/middle.toml"), "new scene").unwrap();
    let added = bundled::generate(&inputs.0).unwrap();
    assert!(added.contains("\"middle\""));
    assert!(added.find("\"alpha\"").unwrap() < added.find("\"middle\"").unwrap());
    assert!(added.find("\"middle\"").unwrap() < added.find("\"zeta\"").unwrap());
    fs::remove_file(inputs.0.join("scenes/middle.toml")).unwrap();
    assert_eq!(first, bundled::generate(&inputs.0).unwrap());
}

#[test]
fn filenames_must_be_selectable_through_the_cli() {
    let inputs = Inputs::new();
    fs::write(inputs.0.join("scenes/not selectable.toml"), "scene").unwrap();
    assert_eq!(
        bundled::generate(&inputs.0).unwrap_err().kind(),
        std::io::ErrorKind::InvalidInput
    );
}

#[cfg(unix)]
#[test]
fn symbolic_links_cannot_pull_external_inputs_into_the_binary() {
    let inputs = Inputs::new();
    fs::write(inputs.0.join("config/water.toml"), "configuration").unwrap();
    std::os::unix::fs::symlink("../config/water.toml", inputs.0.join("scenes/water.toml")).unwrap();
    assert_eq!(
        bundled::generate(&inputs.0).unwrap_err().kind(),
        std::io::ErrorKind::InvalidInput
    );
}
