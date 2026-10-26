//! Camera saves edit only camera fields and publish a complete replacement file.
//! The selected generation owns its target; unapplied draft edits cannot redirect it.

use super::{Camera, Presentation, RenderError};
use std::{
    fs::{self, OpenOptions},
    io::{self, Read, Write},
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};
use toml_edit::{DocumentMut, Item, TableLike, value};

static NEXT: AtomicU64 = AtomicU64::new(0);
const LIMIT: u64 = 1024 * 1024;

/// An explicit camera-save destination captured with a resolved viewer selection.
#[derive(Clone, Debug)]
pub struct CameraSaveTarget {
    path: PathBuf,
}

/// A camera save rejected before publication, leaving the target intact.
#[derive(Debug)]
pub enum CameraSaveError {
    /// An input document or camera is invalid.
    Invalid(RenderError),
    /// The target is not an ordinary writable file or changed during preparation.
    Changed,
    /// Filesystem work failed at the named operation.
    Io {
        operation: &'static str,
        kind: io::ErrorKind,
    },
}
impl std::fmt::Display for CameraSaveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Invalid(error) => write!(f, "camera save: {error}"),
            Self::Changed => write!(
                f,
                "camera save target must be an unchanged regular writable file"
            ),
            Self::Io { operation, kind } => write!(f, "camera save could not {operation}: {kind}"),
        }
    }
}
impl std::error::Error for CameraSaveError {}

impl CameraSaveTarget {
    /// Captures a destination independently of subsequent working-directory changes.
    ///
    /// # Errors
    /// Returns a filesystem error when the destination's parent cannot be located.
    pub fn new(path: impl AsRef<Path>) -> Result<Self, CameraSaveError> {
        let path = path.as_ref();
        let name = path.file_name().ok_or(CameraSaveError::Changed)?;
        let parent = path
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or(Path::new("."));
        let parent =
            fs::canonicalize(parent).map_err(|e| io_error("locate destination directory", e))?;
        Ok(Self {
            path: parent.join(name),
        })
    }

    /// Saves a fixed camera while preserving other settings and their comments.
    ///
    /// # Errors
    /// Rejects invalid documents, symlinks, read-only files, detected concurrent edits,
    /// or a failed write/publication. No fallible work follows successful publication.
    pub fn save(&self, camera: &Camera, aspect: f64) -> Result<(), CameraSaveError> {
        self.save_using(camera, aspect, |_, _| Ok(()))
    }

    fn save_using(
        &self,
        camera: &Camera,
        aspect: f64,
        before_publish: impl FnOnce(&Path, &Path) -> io::Result<()>,
    ) -> Result<(), CameraSaveError> {
        camera.validate(aspect).map_err(CameraSaveError::Invalid)?;
        let original = read(&self.path)?;
        let permissions = original
            .as_ref()
            .map(|_| fs::metadata(&self.path).map(|m| m.permissions()))
            .transpose()
            .map_err(|e| io_error("read permissions", e))?;
        let source = original.as_deref().unwrap_or("schema_version = 2\n");
        Presentation::parse(source).map_err(CameraSaveError::Invalid)?;
        let mut document: DocumentMut = source
            .parse()
            .map_err(|e| CameraSaveError::Invalid(RenderError(format!("render document: {e}"))))?;
        replace(document.as_table_mut(), "schema_version", value(2));
        replace(document.as_table_mut(), "camera_mode", value("fixed"));
        if document.get("camera").is_none() {
            document["camera"] = Item::Table(toml_edit::Table::new());
        }
        let encoded = toml::to_string(camera)
            .map_err(|e| CameraSaveError::Invalid(RenderError(e.to_string())))?;
        let encoded: DocumentMut = encoded
            .parse()
            .map_err(|e| CameraSaveError::Invalid(RenderError(format!("camera encoding: {e}"))))?;
        let table = document["camera"]
            .as_table_like_mut()
            .ok_or(CameraSaveError::Changed)?;
        for (key, item) in encoded.iter() {
            replace(table, key, item.clone());
        }
        let output = document.to_string();
        Presentation::parse(&output).map_err(CameraSaveError::Invalid)?;
        if output.len() as u64 > LIMIT {
            return Err(CameraSaveError::Invalid(RenderError(
                "render document exceeds 1 MiB".into(),
            )));
        }
        let parent = self.path.parent().ok_or(CameraSaveError::Changed)?;
        let mut staged = None;
        for _ in 0..1024 {
            let sequence = NEXT.fetch_add(1, Ordering::Relaxed);
            let path = parent.join(format!(
                ".maelstrom-camera-{}-{sequence}.tmp",
                std::process::id()
            ));
            let mut options = OpenOptions::new();
            options.write(true).create_new(true);
            #[cfg(unix)]
            {
                use std::os::unix::fs::OpenOptionsExt;
                options.mode(0o600);
            }
            match options.open(&path) {
                Ok(file) => {
                    staged = Some((Staged(path), file));
                    break;
                }
                Err(e) if e.kind() == io::ErrorKind::AlreadyExists => continue,
                Err(e) => return Err(io_error("create staged file", e)),
            }
        }
        let (staged, mut file) = staged.ok_or(CameraSaveError::Changed)?;
        if let Some(permissions) = permissions {
            file.set_permissions(permissions)
                .map_err(|e| io_error("preserve permissions", e))?;
        }
        file.write_all(output.as_bytes())
            .and_then(|_| file.sync_all())
            .map_err(|e| io_error("write staged file", e))?;
        drop(file);
        before_publish(&staged.0, &self.path).map_err(|e| io_error("publish staged file", e))?;
        if read(&self.path)? != original {
            return Err(CameraSaveError::Changed);
        }
        if original.is_some() {
            fs::rename(&staged.0, &self.path)
        } else {
            fs::hard_link(&staged.0, &self.path)
        }
        .map_err(|e| io_error("publish staged file", e))
    }
}

fn replace(table: &mut dyn TableLike, key: &str, mut item: Item) {
    if let Some(existing) = table.get_mut(key) {
        if let (Some(old), Some(next)) = (existing.as_value(), item.as_value_mut()) {
            *next.decor_mut() = old.decor().clone();
        }
        // Replacing the item in place also retains comments attached to its key.
        *existing = item;
    } else {
        table.insert(key, item);
    }
}

fn read(path: &Path) -> Result<Option<String>, CameraSaveError> {
    let metadata = match fs::symlink_metadata(path) {
        Ok(metadata) => metadata,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(io_error("inspect target", e)),
    };
    if !metadata.is_file() || metadata.permissions().readonly() {
        return Err(CameraSaveError::Changed);
    }
    let mut source = String::new();
    fs::File::open(path)
        .and_then(|file| file.take(LIMIT + 1).read_to_string(&mut source))
        .map_err(|e| io_error("read target", e))?;
    if source.len() as u64 > LIMIT {
        return Err(CameraSaveError::Invalid(RenderError(
            "render document exceeds 1 MiB".into(),
        )));
    }
    Ok(Some(source))
}
fn io_error(operation: &'static str, error: io::Error) -> CameraSaveError {
    CameraSaveError::Io {
        operation,
        kind: error.kind(),
    }
}
struct Staged(PathBuf);
impl Drop for Staged {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    struct Directory(PathBuf);
    impl Directory {
        fn new() -> Self {
            let directory = std::env::temp_dir().join(format!(
                "maelstrom-camera-test-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir(&directory).unwrap();
            Self(directory)
        }
        fn target(&self) -> CameraSaveTarget {
            CameraSaveTarget::new(self.0.join("render.toml")).unwrap()
        }
        fn only_target(&self) {
            assert_eq!(fs::read_dir(&self.0).unwrap().count(), 1);
        }
    }
    impl Drop for Directory {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn save_preserves_unrelated_values_comments_and_camera_field_comments() {
        for camera in [
            "[camera]\neye_m = [4.0,3.0,2.0] # eye comment\n",
            "camera = { eye_m = [4.0,3.0,2.0] } # camera comment\n",
            "camera.eye_m = [4.0,3.0,2.0] # eye comment\n",
        ] {
            let directory = Directory::new();
            let target = directory.target();
            let source = format!(
                "# keep this heading\nschema_version = 1 # old version\nwidth = 321 # width comment\n{camera}"
            );
            fs::write(&target.path, source).unwrap();
            let selected = Camera {
                eye_m: [5., 4., 3.],
                ..Camera::default()
            };
            target.save(&selected, 1.).unwrap();
            let saved = fs::read_to_string(&target.path).unwrap();
            assert!(saved.contains("# keep this heading"));
            assert!(saved.contains("width = 321 # width comment"));
            assert!(saved.contains("# old version"));
            assert!(saved.contains(if camera.contains("# eye") {
                "# eye comment"
            } else {
                "# camera comment"
            }));
            let presentation = Presentation::parse(&saved).unwrap();
            assert_eq!(presentation.schema_version, 2);
            assert_eq!(presentation.camera_mode, super::super::CameraMode::Fixed);
            assert_eq!(presentation.settings.camera, selected);
            directory.only_target();
        }
    }

    #[test]
    fn new_save_roundtrips_and_energy_settings_survive_replacement() {
        let directory = Directory::new();
        let target = directory.target();
        assert!(!target.path.exists());
        target.save(&Camera::default(), 1.).unwrap();
        assert_eq!(
            Presentation::parse(&fs::read_to_string(&target.path).unwrap())
                .unwrap()
                .settings
                .camera,
            Camera::default()
        );
        let source = "schema_version=2\nwidth=123\n[particles]\ncolour_mode='kinetic_energy' # keep energy\nmin_energy_j=0.0\nmax_energy_j=2.0\n";
        fs::write(&target.path, source).unwrap();
        target.save(&Camera::default(), 1.).unwrap();
        let saved = fs::read_to_string(&target.path).unwrap();
        assert!(saved.contains("colour_mode='kinetic_energy' # keep energy"));
        assert_eq!(
            Presentation::parse(&saved).unwrap().settings.particles,
            Presentation::parse(source).unwrap().settings.particles
        );
        directory.only_target();
    }

    #[test]
    fn publication_failure_and_detected_concurrent_edit_preserve_target() {
        let directory = Directory::new();
        let target = directory.target();
        let original = "schema_version=2\nwidth=456\n";
        fs::write(&target.path, original).unwrap();
        let result = target.save_using(&Camera::default(), 1., |_, _| {
            Err(io::Error::other("injected publication failure"))
        });
        assert!(result.is_err());
        assert_eq!(fs::read_to_string(&target.path).unwrap(), original);
        directory.only_target();
        let concurrent = "schema_version=2\nwidth=789\n";
        let result = target.save_using(&Camera::default(), 1., |_, path| {
            fs::write(path, concurrent)
        });
        assert!(matches!(result, Err(CameraSaveError::Changed)));
        assert_eq!(fs::read_to_string(&target.path).unwrap(), concurrent);
        directory.only_target();
        fs::remove_file(&target.path).unwrap();
        let result = target.save_using(&Camera::default(), 1., |_, path| fs::write(path, original));
        assert!(result.is_err());
        assert_eq!(fs::read_to_string(&target.path).unwrap(), original);
        directory.only_target();
    }

    #[test]
    fn invalid_documents_and_camera_are_rejected_before_writing() {
        let directory = Directory::new();
        let target = directory.target();
        for source in [
            "schema_version=99",
            "schema_version=2\nunknown=1",
            "schema_version=2\nwidth=0",
        ] {
            fs::write(&target.path, source).unwrap();
            assert!(target.save(&Camera::default(), 1.).is_err());
            assert_eq!(fs::read_to_string(&target.path).unwrap(), source);
            directory.only_target();
        }
        fs::remove_file(&target.path).unwrap();
        let camera = Camera {
            near_m: 0.,
            ..Camera::default()
        };
        assert!(target.save(&camera, 1.).is_err());
        assert!(!target.path.exists());
    }

    #[cfg(unix)]
    #[test]
    fn symlinks_and_readonly_files_are_rejected_and_permissions_retained() {
        use std::os::unix::fs::{PermissionsExt, symlink};
        let directory = Directory::new();
        let target = directory.target();
        let original = directory.0.join("original.toml");
        fs::write(&original, "schema_version=2").unwrap();
        symlink(&original, &target.path).unwrap();
        assert!(target.save(&Camera::default(), 1.).is_err());
        assert_eq!(fs::read_to_string(&original).unwrap(), "schema_version=2");
        fs::remove_file(&target.path).unwrap();
        fs::rename(original, &target.path).unwrap();
        fs::set_permissions(&target.path, fs::Permissions::from_mode(0o444)).unwrap();
        assert!(target.save(&Camera::default(), 1.).is_err());
        fs::set_permissions(&target.path, fs::Permissions::from_mode(0o640)).unwrap();
        target.save(&Camera::default(), 1.).unwrap();
        assert_eq!(
            fs::metadata(&target.path).unwrap().permissions().mode() & 0o777,
            0o640
        );
        directory.only_target();
    }
}
