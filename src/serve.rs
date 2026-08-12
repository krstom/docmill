//! `docmill serve` — a small local HTTP conversion service.
//!
//! Deliberately minimal, in the spirit of the CLI: `tiny_http` (synchronous,
//! like the converter — an async runtime would only add moving parts), one
//! warm [`OcrRunner`] shared across requests (models load once, the disk
//! cache dedupes repeats), and a conversion mutex so concurrent uploads
//! queue instead of fighting over CPU. Intended to listen on localhost
//! behind an nginx proxy — TLS, auth, and rate limiting are nginx's job
//! (see the README's nginx snippet).
//!
//! Routes:
//! - `GET /`         a tiny HTML upload form for manual testing
//! - `GET /health`   `ok` (never blocks behind a running conversion)
//! - `POST /convert` the file as `multipart/form-data` (field `file`) or as
//!                   the raw request body with `?filename=doc.docx`.
//!                   Query/form options (all optional): `to=md|json`,
//!                   `mode=fence|markers|text|quote|placeholder`,
//!                   `pages=A-B`, `strict=1`, `force_full_page_ocr=1`,
//!                   `no_text_panels=1`. The OCR engine chain is fixed
//!                   at server start (sessions stay warm); per-request
//!                   engine switching is deliberately not offered.

use std::io::Read;
use std::sync::{Arc, Mutex};

use docling::{DocumentConverter, InputFormat, SourceDocument};
use tiny_http::{Header, Method, Response, Server};

use crate::config::ImgOcrConfig;
use crate::engine::OcrRunner;
use crate::input::{DetectedSource, InputError};
use crate::postprocess::{self, one_picture_document, OutputMode, PostOptions};

/// Uploads beyond this are rejected with 413 — a local conversion service
/// has no business buffering gigabytes.
const MAX_BODY: usize = 200 * 1024 * 1024;

pub struct ServeConfig {
    pub addr: String,
    pub cfg: ImgOcrConfig,
}

pub fn serve(sc: ServeConfig) -> Result<(), String> {
    let runner = if sc.cfg.mode == OutputMode::Placeholder {
        None
    } else {
        Some(sc.cfg.build_runner()?)
    };
    let server = Server::http(&sc.addr).map_err(|e| format!("bind {}: {e}", sc.addr))?;
    eprintln!(
        "docmill serve: listening on http://{} (mode {:?}, engines {:?})",
        sc.addr, sc.cfg.mode, sc.cfg.engines
    );
    let state = Arc::new(State {
        cfg: sc.cfg,
        runner: Mutex::new(runner),
    });
    // Thread-per-request: /health and the form stay responsive while a long
    // conversion holds the runner lock.
    for request in server.incoming_requests() {
        let state = state.clone();
        std::thread::spawn(move || handle(request, &state));
    }
    Ok(())
}

struct State {
    cfg: ImgOcrConfig,
    runner: Mutex<Option<OcrRunner>>,
}

fn handle(mut request: tiny_http::Request, state: &State) {
    let url = request.url().to_string();
    let (path, query) = match url.split_once('?') {
        Some((p, q)) => (p.to_string(), q.to_string()),
        None => (url, String::new()),
    };
    let response = match (request.method(), path.as_str()) {
        (Method::Get, "/") => html(FORM_HTML),
        (Method::Get, "/health") => text(200, "ok\n"),
        (Method::Post, "/convert") => match convert(&mut request, &query, state) {
            Ok(resp) => resp,
            Err((status, msg)) => text(status, &format!("error: {msg}\n")),
        },
        _ => text(404, "not found (try GET /, GET /health, POST /convert)\n"),
    };
    let _ = request.respond(response);
}

type Resp = Response<std::io::Cursor<Vec<u8>>>;

fn text(status: u16, body: &str) -> Resp {
    with_type(
        status,
        body.as_bytes().to_vec(),
        "text/plain; charset=utf-8",
    )
}

fn html(body: &str) -> Resp {
    with_type(200, body.as_bytes().to_vec(), "text/html; charset=utf-8")
}

fn with_type(status: u16, body: Vec<u8>, ctype: &str) -> Resp {
    Response::from_data(body)
        .with_status_code(status)
        .with_header(
            Header::from_bytes(&b"Content-Type"[..], ctype.as_bytes()).expect("static header"),
        )
}

fn convert(
    request: &mut tiny_http::Request,
    query: &str,
    state: &State,
) -> Result<Resp, (u16, String)> {
    let mut opts = parse_query(query);
    let content_type = request
        .headers()
        .iter()
        .find(|h| h.field.equiv("Content-Type"))
        .map(|h| h.value.as_str().to_string())
        .unwrap_or_default();

    let mut body = Vec::new();
    request
        .as_reader()
        .take(MAX_BODY as u64 + 1)
        .read_to_end(&mut body)
        .map_err(|e| (400, format!("read body: {e}")))?;
    if body.len() > MAX_BODY {
        return Err((413, format!("body exceeds {} MiB", MAX_BODY / 1024 / 1024)));
    }

    // multipart/form-data (the HTML form, `curl -F file=@…`) or a raw body
    // with ?filename= (`curl --data-binary @doc.docx '…?filename=doc.docx'`).
    let (filename, bytes) = if let Some(boundary) = multipart_boundary(&content_type) {
        let (filename, bytes, fields) = parse_multipart(&body, &boundary)
            .ok_or((400, "multipart body has no file part".to_string()))?;
        // Form fields complement (but never override) query params.
        for (k, v) in fields {
            opts.entry(k).or_insert(v);
        }
        (filename, bytes)
    } else {
        let filename = opts
            .get("filename")
            .cloned()
            .ok_or((400, "raw uploads need ?filename=<name.ext>".to_string()))?;
        (filename, body)
    };

    let source = DetectedSource::from_bytes(filename.clone(), bytes)
        .map_err(|error| (input_error_status(&error), error.to_string()))?;
    if let Some(warning) = &source.warning {
        eprintln!("convert {filename}: warning: {warning}");
    }

    let to = opts.get("to").map(String::as_str).unwrap_or("md");
    if !matches!(to, "md" | "markdown" | "json") {
        return Err((400, format!("to={to:?} is not md|json")));
    }
    let strict = matches!(opts.get("strict").map(String::as_str), Some("1" | "true"));
    let force_full_page_ocr = matches!(
        opts.get("force_full_page_ocr").map(String::as_str),
        Some("1" | "true")
    );
    let no_text_panels = matches!(
        opts.get("no_text_panels").map(String::as_str),
        Some("1" | "true")
    );
    let mode = match opts.get("mode").map(String::as_str) {
        None => state.cfg.mode,
        Some("fence") => OutputMode::Fence,
        Some("markers") => OutputMode::Markers,
        Some("text") => OutputMode::Text,
        Some("quote") => OutputMode::Quote,
        Some("placeholder") => OutputMode::Placeholder,
        Some(other) => return Err((400, format!("mode={other:?} is unknown"))),
    };
    let pages = match opts.get("pages") {
        Some(p) => Some(docling::parse_page_range(p).map_err(|e| (400, format!("pages: {e}")))?),
        None => None,
    };

    let run_ocr = mode != OutputMode::Placeholder;
    let mut document = {
        let format = source.format;
        let source = SourceDocument::from_bytes(source.name, format, source.bytes);
        if run_ocr && state.cfg.remote_first() && format == InputFormat::Image {
            one_picture_document(&filename, source.bytes, strict)
        } else {
            let mut converter = DocumentConverter::new()
                .strict(strict)
                .force_full_page_ocr(force_full_page_ocr)
                .no_text_panels(no_text_panels);
            if let Some((first, last)) = pages {
                converter = converter.page_range(first, last);
            }
            converter
                .convert(source)
                .map(|r| r.document)
                .map_err(|e| (422, format!("convert {filename}: {e}")))?
        }
    };

    if run_ocr {
        let mut guard = state.runner.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(runner) = guard.as_mut() {
            let post = PostOptions {
                mode,
                min_pixels: state.cfg.min_pixels,
                keep_picture: false,
                ocr_hidden: false,
            };
            let stats = postprocess::apply(&mut document, runner, &post);
            eprintln!("convert {filename}: {stats}");
        }
    }

    Ok(if to == "json" {
        with_type(
            200,
            document.export_to_json().into_bytes(),
            "application/json",
        )
    } else {
        with_type(
            200,
            document.export_to_markdown().into_bytes(),
            "text/markdown; charset=utf-8",
        )
    })
}

fn input_error_status(error: &InputError) -> u16 {
    match error {
        InputError::Unsupported { .. } => 415,
        InputError::Io { .. } => 400,
    }
}

/// Extract the boundary from a `multipart/form-data; boundary=…` header.
fn multipart_boundary(content_type: &str) -> Option<String> {
    if !content_type
        .to_ascii_lowercase()
        .starts_with("multipart/form-data")
    {
        return None;
    }
    let b = content_type.split("boundary=").nth(1)?;
    Some(b.trim().trim_matches('"').to_string())
}

/// Minimal multipart/form-data parser: returns the first part carrying a
/// `filename` (its name + bytes) and every simple text field. Enough for the
/// HTML form and `curl -F`; not a general MIME implementation.
fn parse_multipart(
    body: &[u8],
    boundary: &str,
) -> Option<(String, Vec<u8>, Vec<(String, String)>)> {
    let delim = format!("--{boundary}");
    let mut file: Option<(String, Vec<u8>)> = None;
    let mut fields = Vec::new();
    for part in split_bytes(body, delim.as_bytes()) {
        // Each part: CRLF headers CRLF CRLF content CRLF; the last chunk is
        // the closing "--".
        let part = strip_crlf(part);
        if part.is_empty() || part == b"--" {
            continue;
        }
        let header_end = find_bytes(part, b"\r\n\r\n")?;
        let headers = String::from_utf8_lossy(&part[..header_end]);
        let content = strip_crlf(&part[header_end + 4..]);
        let disposition = headers
            .lines()
            .find(|l| l.to_ascii_lowercase().starts_with("content-disposition"))?;
        let attr = |key: &str| -> Option<String> {
            disposition
                .split(&format!("{key}=\""))
                .nth(1)
                .and_then(|r| r.split('"').next())
                .map(str::to_string)
        };
        match attr("filename") {
            Some(filename) if !filename.is_empty() => {
                if file.is_none() {
                    file = Some((filename, content.to_vec()));
                }
            }
            _ => {
                if let Some(name) = attr("name") {
                    fields.push((name, String::from_utf8_lossy(content).trim().to_string()));
                }
            }
        }
    }
    file.map(|(name, bytes)| (name, bytes, fields))
}

fn split_bytes<'a>(haystack: &'a [u8], sep: &[u8]) -> Vec<&'a [u8]> {
    let mut out = Vec::new();
    let mut start = 0;
    while let Some(pos) = find_bytes(&haystack[start..], sep) {
        out.push(&haystack[start..start + pos]);
        start += pos + sep.len();
    }
    out.push(&haystack[start..]);
    out
}

fn find_bytes(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack.windows(needle.len()).position(|w| w == needle)
}

fn strip_crlf(mut part: &[u8]) -> &[u8] {
    while part.starts_with(b"\r\n") {
        part = &part[2..];
    }
    while part.ends_with(b"\r\n") {
        part = &part[..part.len() - 2];
    }
    part
}

/// Parse a query string into a map, with minimal percent-decoding.
fn parse_query(query: &str) -> std::collections::HashMap<String, String> {
    query
        .split('&')
        .filter(|kv| !kv.is_empty())
        .filter_map(|kv| {
            let (k, v) = kv.split_once('=')?;
            Some((percent_decode(k), percent_decode(v)))
        })
        .collect()
}

fn percent_decode(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'+' => out.push(b' '),
            b'%' if i + 2 < bytes.len() => {
                let hex = std::str::from_utf8(&bytes[i + 1..i + 3]).ok();
                match hex.and_then(|h| u8::from_str_radix(h, 16).ok()) {
                    Some(b) => {
                        out.push(b);
                        i += 2;
                    }
                    None => out.push(b'%'),
                }
            }
            b => out.push(b),
        }
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

const FORM_HTML: &str = r#"<!doctype html>
<meta charset="utf-8">
<title>docmill</title>
<style>body{font:15px system-ui;margin:3em auto;max-width:36em;padding:0 1em}
label{display:block;margin:.8em 0 .2em}</style>
<h1>docmill</h1>
<p>Convert a document or picture to Markdown, including OCR over embedded images.</p>
<form method="post" action="convert" enctype="multipart/form-data">
  <label>File</label><input type="file" name="file" required>
  <label>Output</label>
  <select name="to"><option value="md">markdown</option><option value="json">json</option></select>
  <label>Picture-OCR mode</label>
  <select name="mode">
    <option value="fence">fence (layout-preserving, default)</option>
    <option value="markers">markers</option>
    <option value="text">text</option>
    <option value="quote">quote</option>
    <option value="placeholder">placeholder (no OCR)</option>
  </select>
  <label></label><button>Convert</button>
</form>
<p><small>POST /convert also accepts a raw body with ?filename=doc.docx (the extension may be omitted when content is recognizable);
GET /health for liveness.</small></p>
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn query_parsing_decodes() {
        let q = parse_query("to=json&filename=my%20doc.docx&mode=fence&strict=1");
        assert_eq!(q["to"], "json");
        assert_eq!(q["filename"], "my doc.docx");
        assert_eq!(q["strict"], "1");
        assert!(parse_query("").is_empty());
    }

    #[test]
    fn multipart_extracts_file_and_fields() {
        let boundary = "XBOUND";
        let body = format!(
            "--XBOUND\r\n\
             Content-Disposition: form-data; name=\"mode\"\r\n\r\n\
             quote\r\n\
             --XBOUND\r\n\
             Content-Disposition: form-data; name=\"file\"; filename=\"a b.docx\"\r\n\
             Content-Type: application/octet-stream\r\n\r\n\
             BYTES\x00HERE\r\n\
             --XBOUND--\r\n"
        );
        let (name, bytes, fields) = parse_multipart(body.as_bytes(), boundary).unwrap();
        assert_eq!(name, "a b.docx");
        assert_eq!(bytes, b"BYTES\x00HERE");
        assert_eq!(fields, vec![("mode".to_string(), "quote".to_string())]);
    }

    #[test]
    fn multipart_without_file_is_none() {
        let body = "--B\r\nContent-Disposition: form-data; name=\"x\"\r\n\r\n1\r\n--B--\r\n";
        assert!(parse_multipart(body.as_bytes(), "B").is_none());
    }

    #[test]
    fn boundary_extraction() {
        assert_eq!(
            multipart_boundary("multipart/form-data; boundary=----abc123"),
            Some("----abc123".to_string())
        );
        assert_eq!(multipart_boundary("application/json"), None);
    }
}
