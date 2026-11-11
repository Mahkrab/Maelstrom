//! Temporary indexes keep validation memory independent of sequence length.
//! Files are private, exclusively created and removed on release; they are derived
//! scratch data, never part of a cache's identity or publication protocol.
use std::{
    fs::{self, File, OpenOptions},
    io::{self, BufWriter, Read, Seek, SeekFrom, Write},
    path::PathBuf,
    sync::{
        Mutex,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
};

static NEXT: AtomicU64 = AtomicU64::new(0);

pub(crate) struct Scratch {
    file: Mutex<Option<BufWriter<File>>>,
    path: PathBuf,
    read_cursor: AtomicBool,
}
impl Scratch {
    pub(crate) fn new() -> io::Result<Self> {
        for _ in 0..1024 {
            let path = std::env::temp_dir().join(format!(
                ".maelstrom-index-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            let mut options = OpenOptions::new();
            options.read(true).write(true).create_new(true);
            #[cfg(unix)]
            {
                use std::os::unix::fs::OpenOptionsExt;
                options.mode(0o600);
            }
            match options.open(&path) {
                Ok(file) => {
                    // Unix can unlink an open file, including on error or abrupt exit.
                    #[cfg(unix)]
                    if let Err(error) = fs::remove_file(&path) {
                        drop(file);
                        let _ = fs::remove_file(&path);
                        return Err(error);
                    }
                    return Ok(Self {
                        file: Mutex::new(Some(BufWriter::with_capacity(64 * 1024, file))),
                        path,
                        read_cursor: AtomicBool::new(false),
                    });
                }
                Err(e) if e.kind() == io::ErrorKind::AlreadyExists => continue,
                Err(e) => return Err(e),
            }
        }
        Err(io::Error::new(
            io::ErrorKind::AlreadyExists,
            "no unused temporary index name",
        ))
    }
    pub(crate) fn append(&mut self, bytes: &[u8]) -> io::Result<()> {
        let writer = self
            .file
            .get_mut()
            .map_err(|_| io::Error::other("temporary index lock poisoned"))?
            .as_mut()
            .ok_or_else(|| io::Error::other("temporary index closed"))?;
        if *self.read_cursor.get_mut() {
            writer.seek(SeekFrom::End(0))?;
            *self.read_cursor.get_mut() = false;
        }
        writer.write_all(bytes)
    }
    pub(crate) fn read_at(&self, offset: u64, bytes: &mut [u8]) -> io::Result<()> {
        let mut guard = self
            .file
            .lock()
            .map_err(|_| io::Error::other("temporary index lock poisoned"))?;
        let writer = guard
            .as_mut()
            .ok_or_else(|| io::Error::other("temporary index closed"))?;
        writer.flush()?;
        self.read_cursor.store(true, Ordering::Relaxed);
        let file = writer.get_mut();
        file.seek(SeekFrom::Start(offset))?;
        file.read_exact(bytes)
    }
}
impl Drop for Scratch {
    fn drop(&mut self) {
        if let Ok(file) = self.file.get_mut() {
            file.take();
        }
        // On Unix the original name was already unlinked and may have been
        // reused by someone else. Never remove it a second time.
        #[cfg(not(unix))]
        let _ = fs::remove_file(&self.path);
        #[cfg(unix)]
        let _ = &self.path;
    }
}

#[cfg(feature = "render")]
impl Scratch {
    pub(crate) fn with_reader<T>(&self, read: impl FnOnce(&mut File) -> T) -> io::Result<T> {
        let mut guard = self
            .file
            .lock()
            .map_err(|_| io::Error::other("temporary index lock poisoned"))?;
        let writer = guard
            .as_mut()
            .ok_or_else(|| io::Error::other("temporary index closed"))?;
        writer.flush()?;
        self.read_cursor.store(true, Ordering::Relaxed);
        writer.get_mut().seek(SeekFrom::Start(0))?;
        Ok(read(writer.get_mut()))
    }
}

/// Append-only small records are serialised as an array without retaining history.
#[cfg(feature = "render")]
pub(crate) struct JsonLog {
    file: Scratch,
    count: u64,
}
#[cfg(feature = "render")]
impl JsonLog {
    pub(crate) fn new() -> io::Result<Self> {
        Ok(Self {
            file: Scratch::new()?,
            count: 0,
        })
    }
    pub(crate) fn push(&mut self, value: &impl serde::Serialize) -> io::Result<()> {
        let count = self
            .count
            .checked_add(1)
            .ok_or_else(|| io::Error::other("record count overflow"))?;
        let mut bytes = serde_json::to_vec(value).map_err(io::Error::other)?;
        bytes.push(b'\n');
        self.file.append(&bytes)?;
        self.count = count;
        Ok(())
    }
}
#[cfg(feature = "render")]
impl serde::Serialize for JsonLog {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::{Error, SerializeSeq};
        let mut sequence = serializer.serialize_seq(usize::try_from(self.count).ok())?;
        self.file
            .with_reader(|file| {
                let mut count = 0;
                for value in serde_json::Deserializer::from_reader(std::io::BufReader::new(file))
                    .into_iter::<serde_json::Value>()
                {
                    sequence.serialize_element(&value.map_err(S::Error::custom)?)?;
                    count += 1;
                }
                if count != self.count {
                    return Err(S::Error::custom("temporary record index is incomplete"));
                }
                sequence.end()
            })
            .map_err(S::Error::custom)?
    }
}

pub(crate) fn vector<T>(count: usize) -> Result<Vec<T>, std::collections::TryReserveError> {
    let mut values = Vec::new();
    values.try_reserve_exact(count)?;
    Ok(values)
}
pub(crate) fn collect<T>(
    items: impl ExactSizeIterator<Item = T>,
) -> Result<Vec<T>, std::collections::TryReserveError> {
    let mut values = vector(items.len())?;
    values.extend(items);
    Ok(values)
}
pub(crate) fn deserialize_vec<'de, D, T>(deserializer: D) -> Result<Vec<T>, D::Error>
where
    D: serde::Deserializer<'de>,
    T: serde::Deserialize<'de>,
{
    struct Values<T>(std::marker::PhantomData<T>);
    impl<'de, T: serde::Deserialize<'de>> serde::de::Visitor<'de> for Values<T> {
        type Value = Vec<T>;
        fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("an array")
        }
        fn visit_seq<A: serde::de::SeqAccess<'de>>(self, mut seq: A) -> Result<Vec<T>, A::Error> {
            use serde::de::Error;
            let mut values = Vec::new();
            while let Some(value) = seq.next_element()? {
                values.try_reserve(1).map_err(A::Error::custom)?;
                values.push(value);
            }
            Ok(values)
        }
    }
    deserializer.deserialize_seq(Values(std::marker::PhantomData))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn capacity_overflow_is_recoverable_and_scratch_reads_are_exact() {
        assert!(vector::<u64>(usize::MAX).is_err());
        let mut file = Scratch::new().unwrap();
        file.append(&[7; 32]).unwrap();
        file.append(&[9; 32]).unwrap();
        let mut bytes = [0; 32];
        file.read_at(32, &mut bytes).unwrap();
        assert_eq!(bytes, [9; 32]);
        file.read_at(0, &mut bytes).unwrap();
        assert_eq!(bytes, [7; 32]);
        file.append(&[11; 32]).unwrap();
        file.read_at(64, &mut bytes).unwrap();
        assert_eq!(bytes, [11; 32]);
        assert_eq!(
            file.read_at(96, &mut bytes).unwrap_err().kind(),
            io::ErrorKind::UnexpectedEof
        );
    }
    #[cfg(feature = "render")]
    #[test]
    fn streaming_log_preserves_order_and_handles_more_appends_after_serialisation() {
        let mut log = JsonLog::new().unwrap();
        for i in 0..5 {
            log.push(&i).unwrap();
        }
        assert_eq!(serde_json::to_string(&log).unwrap(), "[0,1,2,3,4]");
        log.push(&5).unwrap();
        assert_eq!(serde_json::to_string(&log).unwrap(), "[0,1,2,3,4,5]");
    }
}
