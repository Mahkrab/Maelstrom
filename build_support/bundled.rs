//! The shipped TOML files own the bundled catalogue. Generate sorted includes so
//! adding an example needs no matching edit to Rust and no runtime checkout.

use std::{fs, io, path::Path};

pub fn generate(root: &Path) -> io::Result<String> {
    let mut output = String::new();
    for (directory, table) in [("scenes", "SCENES"), ("config", "CONFIGURATIONS")] {
        let mut files = Vec::new();
        for entry in fs::read_dir(root.join(directory))? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().is_none_or(|extension| extension != "toml") {
                continue;
            }
            if !entry.file_type()?.is_file() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("bundled input must be a regular file: {}", path.display()),
                ));
            }
            files.push(path);
        }
        files.sort();
        output.push_str(&format!("pub(super) const {table}: &[(&str, &str)] = &[\n"));
        for path in files {
            let stem = path.file_stem().and_then(|s| s.to_str()).ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "bundled filename must be UTF-8",
                )
            })?;
            if stem.is_empty()
                || stem.len() > 120
                || !stem
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
            {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("bundled filename requires a CLI-safe stem: {stem:?}"),
                ));
            }
            let key = if directory == "scenes" {
                stem.to_owned()
            } else {
                format!("{directory}/{stem}.toml")
            };
            output.push_str(&format!("    ({key:?}, include_str!({path:?})),\n"));
        }
        output.push_str("];\n");
    }
    Ok(output)
}
