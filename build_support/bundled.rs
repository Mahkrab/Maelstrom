//! The shipped TOML files own the bundled catalogue. Generate sorted includes so
//! adding an example needs no matching edit to Rust and no runtime checkout.

use std::{fs, io, path::Path};

pub fn generate(root: &Path) -> io::Result<String> {
    let mut scenes = Vec::new();
    let mut configurations = Vec::new();
    let mut renders = Vec::new();
    for entry in fs::read_dir(root.join("config"))? {
        let entry = entry?;
        if entry.path().extension().is_some_and(|e| e == "toml") {
            regular(&entry.path())?;
            configurations.push((
                format!("config/{}.toml", stem(&entry.path())?),
                entry.path(),
            ));
        }
    }
    for entry in fs::read_dir(root.join("scenes"))? {
        let entry = entry?;
        let path = entry.path();
        if entry.file_type()?.is_dir() {
            let name = stem(&path)?;
            for file in ["scene.toml", "config.toml", "render.toml"] {
                regular(&path.join(file))?;
            }
            scenes.push((name.clone(), path.join("scene.toml")));
            configurations.push((
                format!("scenes/{name}/config.toml"),
                path.join("config.toml"),
            ));
            renders.push((name, path.join("render.toml")));
        } else if path.extension().is_some_and(|e| e == "toml") {
            regular(&path)?;
            scenes.push((stem(&path)?, path));
        }
    }
    scenes.sort();
    configurations.sort();
    renders.sort();
    if scenes.windows(2).any(|v| v[0].0 == v[1].0) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "duplicate bundled selector",
        ));
    }
    let mut output = String::new();
    for (table, entries) in [
        ("SCENES", scenes),
        ("CONFIGURATIONS", configurations),
        ("RENDERS", renders),
    ] {
        output.push_str(&format!("pub(super) const {table}: &[(&str, &str)] = &[\n"));
        for (key, path) in entries {
            output.push_str(&format!("    ({key:?}, include_str!({path:?})),\n"));
        }
        output.push_str("];\n");
    }
    Ok(output)
}
fn regular(path: &Path) -> io::Result<()> {
    if !fs::symlink_metadata(path)?.file_type().is_file() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("bundled input must be a regular file: {}", path.display()),
        ));
    }
    Ok(())
}
fn stem(path: &Path) -> io::Result<String> {
    let name = if path.is_dir() {
        path.file_name()
    } else {
        path.file_stem()
    }
    .and_then(|s| s.to_str())
    .unwrap_or("");
    if name.is_empty()
        || name.len() > 120
        || !name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "bundled filename requires a CLI-safe stem",
        ));
    }
    Ok(name.into())
}
