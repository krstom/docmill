//! `docmill serve` — a small local HTTP conversion service.
//!
//! Deliberately minimal, in the spirit of the CLI: `tiny_http` (synchronous,
//! like the converter — an async runtime would only add moving parts), one
//! warm OCR runner and PDF pipeline owned by one conversion worker. Four
//! requests may wait in the queue, before their bodies are buffered. Health
//! checks are handled independently. Bind behind nginx for TLS/auth.
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
use std::sync::mpsc::{self, SyncSender};
use std::time::Duration;

use crate::conversion::{
    normalize_ocr_lang, parse_ocr_mode, parse_ocr_scale, ConversionOptions, WarmPipeline,
};
use docling::{ImageMode, InputFormat, SourceDocument};
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
    pub conversion: ConversionOptions,
}

pub fn serve(sc: ServeConfig) -> Result<(), String> {
    let server = Server::http(&sc.addr).map_err(|e| format!("bind {}: {e}", sc.addr))?;
    eprintln!(
        "docmill serve: listening on http://{} (mode {:?}, engines {:?})",
        sc.addr, sc.cfg.mode, sc.cfg.engines
    );
    let state = State {
        cfg: sc.cfg,
        conversion: sc.conversion,
    };
    let mut worker = WorkerState::default();
    if state.cfg.mode != OutputMode::Placeholder {
        worker.runner = Some(
            state
                .cfg
                .build_runner()?
                .with_cooldown(Duration::from_secs(30)),
        );
    }
    let (tx, rx) = mpsc::sync_channel::<tiny_http::Request>(4);
    let worker_thread = std::thread::spawn(move || {
        for mut request in rx {
            let query = request
                .url()
                .split_once('?')
                .map(|(_, q)| q.to_string())
                .unwrap_or_default();
            let response = recover(&mut worker, |worker| {
                convert(&mut request, &query, &state, worker)
            });
            let _ = request.respond(response);
        }
    });
    for request in server.incoming_requests() {
        dispatch(request, &tx);
    }
    drop(tx);
    worker_thread
        .join()
        .map_err(|_| "conversion worker stopped unexpectedly".to_string())?;
    Ok(())
}

struct State {
    cfg: ImgOcrConfig,
    conversion: ConversionOptions,
}

#[derive(Default)]
struct WorkerState {
    runner: Option<OcrRunner>,
    pipeline: WarmPipeline,
}

fn recover(
    worker: &mut WorkerState,
    convert: impl FnOnce(&mut WorkerState) -> Result<Resp, (u16, String)>,
) -> Resp {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| convert(worker))) {
        Ok(Ok(response)) => response,
        Ok(Err((status, message))) => text(status, &format!("error: {message}\n")),
        Err(_) => {
            *worker = WorkerState::default();
            eprintln!("docmill serve: conversion panicked; discarded worker models");
            text(500, "error: conversion failed unexpectedly; worker reset\n")
        }
    }
}

fn dispatch(request: tiny_http::Request, tx: &SyncSender<tiny_http::Request>) {
    let path = request.url().split('?').next().unwrap_or_default();
    let response = match (request.method(), path) {
        (Method::Get, "/") => html(FORM_HTML),
        (Method::Get, "/health") => text(200, "ok\n"),
        (Method::Post, "/convert") => {
            if request.body_length().is_some_and(|len| len > MAX_BODY) {
                text(413, "error: body exceeds 200 MiB\n")
            } else {
                match tx.try_send(request) {
                    Ok(()) => return,
                    Err(error) => {
                        let request = match error {
                            mpsc::TrySendError::Full(request)
                            | mpsc::TrySendError::Disconnected(request) => request,
                        };
                        let _ = request
                            .respond(text(503, "error: conversion queue is full; retry later\n"));
                        return;
                    }
                }
            }
        }
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
    worker: &mut WorkerState,
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
    let conversion = request_conversion(&opts, &state.conversion).map_err(|e| (400, e))?;
    let images = match opts
        .get("images")
        .map(String::as_str)
        .unwrap_or("placeholder")
    {
        "placeholder" => ImageMode::Placeholder,
        "embedded" => ImageMode::Embedded,
        _ => {
            return Err((
                400,
                "images must be placeholder|embedded; referenced artifacts are not served".into(),
            ))
        }
    };
    let mode = match opts.get("mode").map(String::as_str) {
        None => state.cfg.mode,
        Some("fence") => OutputMode::Fence,
        Some("markers") => OutputMode::Markers,
        Some("text") => OutputMode::Text,
        Some("quote") => OutputMode::Quote,
        Some("placeholder") => OutputMode::Placeholder,
        Some(other) => return Err((400, format!("mode={other:?} is unknown"))),
    };
    let run_ocr = mode != OutputMode::Placeholder;
    let mut document = {
        let format = source.format;
        let source = SourceDocument::from_bytes(source.name, format, source.bytes);
        if run_ocr && state.cfg.remote_first() && format == InputFormat::Image {
            one_picture_document(&filename, source.bytes, conversion.strict)
        } else {
            worker
                .pipeline
                .convert(source, &conversion)
                .map_err(|e| (422, format!("convert {filename}: {e}")))?
        }
    };

    conversion.finish_document(&mut document);
    if run_ocr {
        if worker.runner.is_none() {
            worker.runner = Some(
                state
                    .cfg
                    .build_runner()
                    .map_err(|e| (500, e))?
                    .with_cooldown(Duration::from_secs(30)),
            );
        }
        if let Some(runner) = worker.runner.as_mut() {
            let post = PostOptions {
                mode,
                min_pixels: state.cfg.min_pixels,
                keep_picture: images != ImageMode::Placeholder,
                target: postprocess::OutputTarget::from_format(&to),
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
            document
                .export_to_markdown_with_images(images, "artifacts")
                .0
                .into_bytes(),
            "text/markdown; charset=utf-8",
        )
    })
}

fn request_conversion(
    opts: &std::collections::HashMap<String, String>,
    defaults: &ConversionOptions,
) -> Result<ConversionOptions, String> {
    let mut conversion = defaults.clone();
    let flag = |name: &str, default: bool| match opts.get(name).map(String::as_str) {
        None => Ok(default),
        Some("1" | "true") => Ok(true),
        Some("0" | "false") => Ok(false),
        Some(_) => Err(format!("{name} must be 0|1|false|true")),
    };
    conversion.strict = flag("strict", conversion.strict)?;
    conversion.no_ocr = flag("no_ocr", conversion.no_ocr)?;
    conversion.no_table_former = flag("no_table_former", conversion.no_table_former)?;
    conversion.skip_ocr = flag("skip_ocr", conversion.skip_ocr)?;
    conversion.force_full_page_ocr = flag("force_full_page_ocr", conversion.force_full_page_ocr)?;
    conversion.no_text_panels = flag("no_text_panels", conversion.no_text_panels)?;
    conversion.heading_hierarchy = flag("heading_hierarchy", conversion.heading_hierarchy)?;
    if let Some(pages) = opts.get("pages") {
        conversion.pages = Some(docling::parse_page_range(pages).map_err(|e| e.to_string())?);
    }
    if let Some(lang) = opts.get("ocr_lang") {
        conversion.ocr_lang = Some(normalize_ocr_lang(lang)?);
    }
    if let Some(mode) = opts.get("ocr_mode") {
        conversion.ocr_mode = Some(parse_ocr_mode(mode)?);
    }
    if let Some(scale) = opts.get("ocr_scale") {
        conversion.ocr_scale = Some(parse_ocr_scale(scale)?);
    }
    if let Some(encoding) = opts.get("encoding") {
        conversion.encoding = Some(encoding.clone());
    }
    if let Some(placeholder) = opts.get("page_break_placeholder") {
        conversion.page_break_placeholder = Some(placeholder.clone());
    }
    Ok(conversion)
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
    #[test]
    fn per_request_options_reset_to_startup_defaults() {
        let defaults = ConversionOptions::default();
        let custom = request_conversion(&parse_query("skip_ocr=1&ocr_mode=full_page&ocr_scale=3&heading_hierarchy=1&ocr_lang=zh-Hant&pages=2-3&encoding=windows-1251&page_break_placeholder=PAGE"), &defaults).unwrap();
        assert!(custom.skip_ocr && custom.heading_hierarchy);
        assert_eq!(custom.ocr_lang.as_deref(), Some("ch"));
        assert_eq!(custom.pages, Some((2, 3)));
        let reset = request_conversion(&parse_query(""), &defaults).unwrap();
        assert!(!reset.skip_ocr && !reset.heading_hierarchy);
        assert_eq!(reset.ocr_scale, None);
        assert_eq!(reset.pages, None);
        assert_eq!(reset.encoding, None);
        assert_eq!(reset.page_break_placeholder, None);
        assert!(request_conversion(&parse_query("skip_ocr=maybe"), &defaults).is_err());
        assert!(request_conversion(&parse_query("ocr_scale=NaN"), &defaults).is_err());
    }

    #[test]
    fn a_panicking_conversion_resets_worker_and_returns_500() {
        let (engine, _) = crate::engine::tests::MockEngine::new("mock", vec![]);
        let mut worker = WorkerState {
            runner: Some(OcrRunner::new(
                vec![Box::new(engine)],
                crate::cache::OcrCache::disabled(),
            )),
            ..Default::default()
        };
        let response = recover(&mut worker, |_| panic!("simulated backend panic"));
        assert_eq!(response.status_code().0, 500);
        assert!(worker.runner.is_none());
        assert_eq!(
            recover(&mut worker, |_| Ok(text(200, "recovered")))
                .status_code()
                .0,
            200
        );
    }

    #[test]
    fn queue_is_bounded_and_health_does_not_wait_for_conversion() {
        use std::io::Write;
        use std::net::TcpStream;
        let server = Server::http("127.0.0.1:0").unwrap();
        let addr = server.server_addr().to_ip().unwrap();
        let (tx, rx) = mpsc::sync_channel(4);
        let submit = |path: &str, method: &str| {
            let mut client = TcpStream::connect(addr).unwrap();
            client
                .set_read_timeout(Some(Duration::from_secs(2)))
                .unwrap();
            write!(client, "{method} {path} HTTP/1.1\r\nHost: localhost\r\nContent-Length: 0\r\nConnection: close\r\n\r\n").unwrap();
            let request = server
                .recv_timeout(Duration::from_secs(2))
                .unwrap()
                .unwrap();
            dispatch(request, &tx);
            client
        };
        let queued: Vec<_> = (0..4)
            .map(|_| submit("/convert?filename=test.md", "POST"))
            .collect();
        let mut rejected = submit("/convert?filename=test.md", "POST");
        let mut response = String::new();
        rejected.read_to_string(&mut response).unwrap();
        assert!(response.starts_with("HTTP/1.1 503"), "{response}");
        let mut health = submit("/health", "GET");
        response.clear();
        health.read_to_string(&mut response).unwrap();
        assert!(response.starts_with("HTTP/1.1 200"), "{response}");
        for request in rx.try_iter() {
            request.respond(text(200, "done")).unwrap();
        }
        drop(queued);
    }

    #[test]
    fn raw_upload_uses_encoding_and_rejects_referenced_images() {
        use std::io::Write;
        use std::net::TcpStream;
        let state = State {
            cfg: ImgOcrConfig::resolve(crate::config::CliOverrides {
                engine: Some("local".into()),
                mode: Some("placeholder".into()),
                ..Default::default()
            })
            .unwrap(),
            conversion: ConversionOptions::default(),
        };
        let server = Server::http("127.0.0.1:0").unwrap();
        for (query, expected) in [
            ("filename=test.txt&encoding=windows-1251", 200),
            ("filename=test.txt&images=referenced", 400),
        ] {
            let mut client = TcpStream::connect(server.server_addr().to_ip().unwrap()).unwrap();
            write!(client, "POST /convert?{query} HTTP/1.1\r\nHost: localhost\r\nContent-Length: 6\r\nConnection: close\r\n\r\n").unwrap();
            client
                .write_all(&[0xcf, 0xf0, 0xe8, 0xe2, 0xe5, 0xf2])
                .unwrap();
            let mut request = server
                .recv_timeout(Duration::from_secs(2))
                .unwrap()
                .unwrap();
            let response = recover(&mut WorkerState::default(), |worker| {
                convert(&mut request, query, &state, worker)
            });
            assert_eq!(response.status_code().0, expected);
            request.respond(response).unwrap();
            let mut body = String::new();
            client.read_to_string(&mut body).unwrap();
            if expected == 200 {
                assert!(body.contains("Привет"), "{body}");
            }
        }
    }
}
