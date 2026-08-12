//! Shared input loading and content-first format detection.

use std::io::{Cursor, Read};
use std::path::Path;

use docling::InputFormat;
use thiserror::Error;
use zip::ZipArchive;

#[derive(Debug, Clone)]
pub struct DetectedSource {
    pub name: String,
    pub bytes: Vec<u8>,
    pub format: InputFormat,
    /// Present when a strong content signature overrode the filename extension.
    pub warning: Option<String>,
}

#[derive(Debug, Error)]
pub enum InputError {
    #[error("reading {name}: {source}")]
    Io {
        name: String,
        #[source]
        source: std::io::Error,
    },
    #[error("unsupported input format for {name:?}")]
    Unsupported { name: String },
}

impl DetectedSource {
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self, InputError> {
        let path = path.as_ref();
        let bytes = std::fs::read(path).map_err(|source| InputError::Io {
            name: path.display().to_string(),
            source,
        })?;
        let name = path
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| path.display().to_string());
        Self::from_bytes(name, bytes)
    }

    pub fn from_bytes(name: impl Into<String>, bytes: Vec<u8>) -> Result<Self, InputError> {
        let name = name.into();
        let extension = extension_format(&name);
        let content = content_format(&bytes);
        let format = content
            .or(extension)
            .ok_or_else(|| InputError::Unsupported { name: name.clone() })?;
        let warning = match (content, extension) {
            (Some(actual), Some(hint)) if actual != hint => Some(format!(
                "content identifies {} but filename suggests {}; using content",
                actual.as_str(), hint.as_str()
            )),
            _ => None,
        };
        Ok(Self {
            name,
            bytes,
            format,
            warning,
        })
    }

    pub fn into_docling(self) -> docling::SourceDocument {
        docling::SourceDocument::from_bytes(self.name, self.format, self.bytes)
    }
}

fn extension_format(name: &str) -> Option<InputFormat> {
    let ext = Path::new(name).extension()?.to_str()?;
    InputFormat::from_extension(ext)
}

fn content_format(bytes: &[u8]) -> Option<InputFormat> {
    let head = bytes.strip_prefix(&[0xEF, 0xBB, 0xBF]).unwrap_or(bytes);
    let first = head
        .iter()
        .position(|byte| !byte.is_ascii_whitespace())
        .unwrap_or(head.len());
    if head[first..].starts_with(br"{\rtf") {
        return Some(InputFormat::Rtf);
    }
    if bytes.starts_with(b"%PDF-") {
        return Some(InputFormat::Pdf);
    }
    if is_image(bytes) {
        return Some(InputFormat::Image);
    }
    if bytes.starts_with(b"PK\x03\x04")
        || bytes.starts_with(b"PK\x05\x06")
        || bytes.starts_with(b"PK\x07\x08")
    {
        return detect_zip(bytes);
    }
    None
}

fn is_image(bytes: &[u8]) -> bool {
    bytes.starts_with(b"\x89PNG\r\n\x1a\n")
        || bytes.starts_with(b"\xff\xd8\xff")
        || bytes.starts_with(b"GIF87a")
        || bytes.starts_with(b"GIF89a")
        || bytes.starts_with(b"II*\0")
        || bytes.starts_with(b"MM\0*")
        || bytes.starts_with(b"BM")
        || (bytes.len() >= 12 && &bytes[..4] == b"RIFF" && &bytes[8..12] == b"WEBP")
}

fn detect_zip(bytes: &[u8]) -> Option<InputFormat> {
    let mut zip = ZipArchive::new(Cursor::new(bytes)).ok()?;
    if zip.len() > 10_000 {
        return None;
    }
    let mut names = std::collections::HashSet::with_capacity(zip.len());
    for i in 0..zip.len() {
        let file = zip.by_index(i).ok()?;
        names.insert(file.name().replace('\\', "/").to_ascii_lowercase());
    }
    if names.contains("xl/workbook.bin") && names.contains("[content_types].xml") {
        return Some(InputFormat::Xlsx);
    }
    if names.contains("word/document.xml") {
        return Some(InputFormat::Docx);
    }
    if names.contains("ppt/presentation.xml") {
        return Some(InputFormat::Pptx);
    }
    if names.contains("xl/workbook.xml") {
        return Some(InputFormat::Xlsx);
    }
    if names.contains("document.xml") && names.contains("[content_types].xml") {
        return Some(InputFormat::Dclx);
    }
    if names.contains("meta-inf/container.xml") {
        return Some(InputFormat::Epub);
    }
    if names.contains("mimetype") {
        let mut mimetype = String::new();
        zip.by_name("mimetype")
            .ok()?
            .take(256)
            .read_to_string(&mut mimetype)
            .ok()?;
        return match mimetype.trim() {
            "application/vnd.oasis.opendocument.text" => Some(InputFormat::Odt),
            "application/vnd.oasis.opendocument.spreadsheet" => Some(InputFormat::Ods),
            "application/vnd.oasis.opendocument.presentation" => Some(InputFormat::Odp),
            "application/epub+zip" => Some(InputFormat::Epub),
            _ => None,
        };
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn content_wins_over_extension() {
        let src = DetectedSource::from_bytes("wrong.pdf", br"{\rtf1 hello}".to_vec()).unwrap();
        assert_eq!(src.format, InputFormat::Rtf);
        assert!(src.warning.is_some());
    }

    #[test]
    fn extension_fallback_handles_new_formats() {
        assert_eq!(
            DetectedSource::from_bytes("empty.RTF", Vec::new())
                .unwrap()
                .format,
            InputFormat::Rtf
        );
        assert_eq!(
            DetectedSource::from_bytes("empty.xlsb", Vec::new())
                .unwrap()
                .format,
            InputFormat::Xlsx
        );
    }

    #[test]
    fn image_magic_needs_no_extension() {
        let src = DetectedSource::from_bytes("upload", b"\x89PNG\r\n\x1a\nrest".to_vec()).unwrap();
        assert_eq!(src.format, InputFormat::Image);
    }

    #[test]
    fn opc_main_parts_are_detected_without_extensions() {
        let mut out = Cursor::new(Vec::new());
        {
            let mut writer = zip::ZipWriter::new(&mut out);
            let options: zip::write::FileOptions<'_, ()> = zip::write::FileOptions::default();
            writer.start_file("[Content_Types].xml", options).unwrap();
            writer.write_all(b"<Types/>").unwrap();
            writer.start_file("xl/workbook.bin", options).unwrap();
            writer.write_all(b"workbook").unwrap();
            writer.finish().unwrap();
        }
        let source = DetectedSource::from_bytes("upload", out.into_inner()).unwrap();
        assert_eq!(source.format, InputFormat::Xlsx);
    }
}
