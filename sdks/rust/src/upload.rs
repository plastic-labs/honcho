//! File source abstraction for uploads.

use std::path::{Path, PathBuf};
use std::pin::Pin;

use tokio::io::AsyncRead;

/// A source for file data that will be uploaded.
///
/// Construct with [`FileSource::bytes`], [`FileSource::path`], or
/// [`FileSource::stream`], or convert from [`PathBuf`]/[`std::path::Path`] via the
/// `From` impls.
pub enum FileSource {
    /// Raw bytes with explicit filename and content type.
    Bytes {
        /// File name to send.
        filename: String,
        /// Raw file data.
        bytes: Vec<u8>,
        /// MIME content type.
        content_type: String,
    },
    /// A filesystem path. Resolved at upload time.
    Path(PathBuf),
    /// A streaming reader — fully buffered into memory before uploading.
    Stream {
        /// File name to send.
        filename: String,
        /// Async reader producing the file data.
        reader: Pin<Box<dyn AsyncRead + Send>>,
        /// MIME content type.
        content_type: String,
    },
}

impl std::fmt::Debug for FileSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Bytes {
                filename,
                bytes,
                content_type,
            } => f
                .debug_struct("Bytes")
                .field("filename", filename)
                .field("bytes", &bytes.len())
                .field("content_type", content_type)
                .finish(),
            Self::Path(p) => f.debug_tuple("Path").field(p).finish(),
            Self::Stream {
                filename,
                content_type,
                ..
            } => f
                .debug_struct("Stream")
                .field("filename", filename)
                .field("content_type", content_type)
                .finish_non_exhaustive(),
        }
    }
}

impl FileSource {
    /// Create a `Bytes` variant from explicit parts.
    pub fn bytes(
        filename: impl Into<String>,
        data: impl Into<Vec<u8>>,
        content_type: impl Into<String>,
    ) -> Self {
        Self::Bytes {
            filename: filename.into(),
            bytes: data.into(),
            content_type: content_type.into(),
        }
    }

    /// Create a `Path` variant.
    pub fn path(path: impl Into<PathBuf>) -> Self {
        Self::Path(path.into())
    }

    /// Create a `Stream` variant from an [`AsyncRead`] source.
    ///
    /// The reader is fully consumed with [`tokio::io::AsyncReadExt::read_to_end`]
    /// and buffered into a `Vec<u8>` before the upload begins. This is **not**
    /// true streaming — the entire payload resides in memory during the request.
    ///
    /// For files on disk, prefer [`FileSource::path`] which streams from the
    /// filesystem without buffering.
    ///
    /// # Examples
    ///
    /// ```
    /// use honcho_ai::FileSource;
    ///
    /// let cursor = std::io::Cursor::new(b"hello".to_vec());
    /// let src = FileSource::stream("out.txt", cursor, "text/plain");
    /// ```
    pub fn stream(
        filename: impl Into<String>,
        reader: impl AsyncRead + Send + 'static,
        content_type: impl Into<String>,
    ) -> Self {
        Self::Stream {
            filename: filename.into(),
            reader: Box::pin(reader),
            content_type: content_type.into(),
        }
    }
}

impl From<PathBuf> for FileSource {
    fn from(p: PathBuf) -> Self {
        Self::Path(p)
    }
}

impl From<&Path> for FileSource {
    fn from(p: &Path) -> Self {
        Self::Path(p.to_path_buf())
    }
}

/// Resolve a [`FileSource`] into `(filename, bytes, content_type)`.
///
/// For the `Bytes` variant the fields are returned directly.
/// For the `Path` variant the file is read, the filename is extracted from
/// the final path component, and the MIME type is guessed from the extension
/// (falling back to `application/octet-stream`).
#[cfg(test)]
pub(crate) async fn resolve_to_bytes(
    src: FileSource,
) -> std::io::Result<(String, Vec<u8>, String)> {
    match src {
        FileSource::Bytes {
            filename,
            bytes,
            content_type,
        } => Ok((filename, bytes, content_type)),
        FileSource::Path(p) => {
            let data = tokio::fs::read(&p).await?;
            let filename = p
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default();
            let content_type = mime_guess::from_path(&p)
                .first_or_octet_stream()
                .to_string();
            Ok((filename, data, content_type))
        }
        FileSource::Stream { .. } => Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "cannot buffer a stream source — use the streaming upload path",
        )),
    }
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::unnecessary_wraps,
    clippy::needless_pass_by_value,
    clippy::unused_async
)]
mod tests {
    use static_assertions::assert_impl_all;

    use super::*;
    use std::io::Write;

    assert_impl_all!(FileSource: Send);

    #[tokio::test]
    async fn file_source_path_resolves_filename_and_content_type() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("report.pdf");
        {
            let mut f = std::fs::File::create(&file_path).unwrap();
            f.write_all(b"%PDF-1.4 fake").unwrap();
        }

        let src = FileSource::path(&file_path);
        let (name, data, ctype) = resolve_to_bytes(src).await.unwrap();

        assert_eq!(name, "report.pdf");
        assert_eq!(data, b"%PDF-1.4 fake".as_slice());
        assert_eq!(ctype, "application/pdf");
    }

    #[tokio::test]
    async fn file_source_unknown_extension_falls_back_to_octet_stream() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("data.unknownext");
        {
            let mut f = std::fs::File::create(&file_path).unwrap();
            f.write_all(b"hello").unwrap();
        }

        let src = FileSource::path(&file_path);
        let (_, _, ctype) = resolve_to_bytes(src).await.unwrap();

        assert_eq!(ctype, "application/octet-stream");
    }

    #[tokio::test]
    async fn file_source_path_nonexistent_returns_io_error() {
        let src = FileSource::path("/tmp/honcho_test_nonexistent_42deadbeef.pdf");
        let result = resolve_to_bytes(src).await;
        assert!(result.is_err());
    }
}
