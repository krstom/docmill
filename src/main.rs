//! docmill — docling.rs conversion plus custom OCR over embedded
//! pictures (DOCX drawings, PDF figure regions, standalone images).
//!
//! Usage: docmill [conversion flags] [--img-ocr-* flags] <input-file>
//!
//! Conversion flags (same semantics as `docling-rs`, buffered path):
//!   --to md|json|dclx   output format (default: md)
//!   --output FILE, -o   write output to FILE instead of stdout (dclx always
//!                       writes a file; this overrides its default name)
//!   --strict            cleaner Markdown instead of docling-legacy output
//!   --pages A-B         PDF page window (1-based, inclusive)
//!   --images MODE       placeholder (default) | embedded | referenced
//!   --fetch-images      resolve external <img src> for HTML/EPUB
//!   --no-table-former   skip the TableFormer model for PDF/image input
//!   --no-ocr            skip the whole ML stack for PDF input (text layer
//!                       only — produces no picture crops, so nothing to OCR)
//!   --force-full-page-ocr  OCR every PDF page even when it has a text layer
//!   --no-text-panels    keep every detected picture as a picture (disable
//!                       the text-panel-to-paragraphs demotion)
//!   --ocr-lang en|ch    the PDF pipeline's own page-OCR language
//!   --enrich-picture-classes / --enrich-code / --enrich-formula
//!
//! Picture-OCR flags (flag > DOCMILL_* env var > default):
//!   --img-ocr-engine local|vlm|paddle[,fallback…]   engine chain (default: local)
//!   --img-ocr-mode fence|markers|text|quote|placeholder   (default: fence)
//!       fence        a fenced code block; engines with box geometry (local
//!                    v5, paddle) render the text with the source image's
//!                    approximate spacing (sidebar left, content right)
//!       markers      <!-- ocr:begin engine=… --> text <!-- ocr:end -->
//!       text         just the recognized text
//!       quote        the text as a blockquote box (`> …` per line)
//!       placeholder  no OCR at all — byte-identical to `docling-rs`
//!   --img-ocr-lang en|ch          local engine language (default: en)
//!   --img-ocr-endpoint URL        vlm: OpenAI-compatible base or full URL;
//!                                 paddle: the predict URL
//!   --img-ocr-model NAME          vlm model name
//!   --img-ocr-api-key KEY         vlm bearer token
//!   --img-ocr-prompt TEXT         vlm transcription prompt override
//!   --img-ocr-min-px N            skip pictures below N pixels area (default 2500)
//!   --img-ocr-cache-dir DIR       OCR cache (default ~/.cache/docmill)
//!   --no-img-ocr-cache            disable the cache for this run
//!   --img-ocr-timeout SECS        remote request timeout (default 120)
//!   --img-ocr-models-dir DIR      local models dir (default ./models; point it
//!                                 at your docling.rs checkout's models/)
//!
//! Standalone image inputs: with a local-first chain the image runs through
//! docling's full ML pipeline (layout + OCR + tables) and embedded figure
//! regions are post-processed like any PDF; with a remote-first chain the ML
//! pipeline is bypassed entirely — the image goes straight to the remote
//! engine, so no local models are needed at all.

use std::io::Write;
use std::path::Path;
use std::process::ExitCode;

use docling::{DocumentConverter, ImageMode, InputFormat, SourceDocument};
use docling_core::DoclingDocument;
use docmill::config::{CliOverrides, ImgOcrConfig};
use docmill::postprocess::{self, one_picture_document, OutputMode, PostOptions};

/// The complete flag reference, printed by `--help` — kept as one constant so
/// the help text and the parser can be reviewed side by side.
const USAGE: &str = "\
docmill — document conversion (docling.rs) + OCR over embedded pictures

usage: docmill [OPTIONS] <input-file>

Subcommands:
  serve [--addr HOST:PORT] [--img-ocr-* flags]
                              local HTTP conversion service (default
                              127.0.0.1:8877; POST /convert, GET /health,
                              GET / for a test form) — put nginx in front

Conversion (same semantics as docling-rs):
  --to md|json|dclx|chunks    output format (default: md)
  -o, --output FILE           write output to FILE instead of stdout
  --strict                    cleaner Markdown instead of docling-legacy output
  --pages A-B                 PDF page window (1-based, inclusive)
  --images MODE               placeholder (default) | embedded | referenced
  --fetch-images              resolve external <img src> for HTML/EPUB
  --no-stream                 (placeholder mode only) buffer Markdown instead
                              of streaming it page by page; any picture-OCR
                              mode implies buffering
  --no-table-former           skip the TableFormer model for PDF/image input
  --no-ocr                    PDF text-layer only (no ML; produces no picture
                              crops, so nothing to picture-OCR)
  --force-full-page-ocr       OCR every PDF page even when it has a text layer
                              (for text layers that lie)
  --no-text-panels            keep every detected picture as a picture —
                              disable the text-panel-to-paragraphs demotion
  --ocr-lang en|ch            the PDF pipeline's own page-OCR language
  --asr-model PRESET          Whisper preset for audio inputs
  --video-frames N            max frames sampled from a video input
  --use-web-browser           pre-render HTML in Chromium (web-browser feature)
  --pipeline standard|vlm     vlm = convert pages via a remote OpenAI-
                              compatible vision endpoint (docling-rs #77)
  --vlm-endpoint URL          vlm pipeline endpoint
  --vlm-model NAME            vlm pipeline model
  --enrich-picture-classes    classify pictures (needs the classifier model)
  --enrich-code               rewrite code blocks with CodeFormula
  --enrich-formula            decode formulas to LaTeX with CodeFormula

Picture OCR (flag > DOCMILL_* env var > default):
  --img-ocr-engine local|vlm|paddle[,fallback...]
                              engine chain (default: local)
  --img-ocr-mode fence|markers|text|quote|placeholder
                              text insertion (default: fence)
                                fence:  code block; engines with box geometry
                                        (local v5, paddle) keep the image's
                                        approximate spatial layout
                                markers: <!-- ocr:begin --> text <!-- ocr:end -->
                                text:   just the recognized text
                                quote:  blockquote box (> ... per line)
                                placeholder: no OCR; byte-identical to docling-rs
  --img-ocr-lang en|ch        local v3 fallback language (v5 is multilingual)
  --img-ocr-endpoint URL      vlm: OpenAI-compatible base or full URL;
                              paddle: the predict URL
  --img-ocr-model NAME        vlm model name
  --img-ocr-api-key KEY       vlm bearer token
  --img-ocr-prompt TEXT       vlm transcription prompt override
  --img-ocr-min-px N          skip pictures below N pixels area (default 2500)
  --img-ocr-cache-dir DIR     OCR result cache (default ~/.cache/docmill)
  --no-img-ocr-cache          disable the cache for this run
  --img-ocr-timeout SECS      remote request timeout (default 120)
  --img-ocr-models-dir DIR    local models dir (default: ./models, else next to
                              the installed binary)

  -h, --help                  print this help
  -V, --version               print version

Extras: DOCMILL_EXTRA_BODY merges a JSON object into every vlm
request; DOCMILL_{DET_ONNX,REC_ONNX,DICT} pin an explicit PP-OCRv5
model set. See README.md for engines, cache, and install details.
";

fn main() -> ExitCode {
    // `docmill serve …` — the local HTTP conversion service.
    {
        let mut args = std::env::args().skip(1);
        if args.next().as_deref() == Some("serve") {
            return run_serve(args.collect());
        }
    }

    let mut strict = false;
    let mut to = "md".to_string();
    let mut output: Option<String> = None;
    let mut images = "placeholder".to_string();
    let mut fetch_images = false;
    let mut no_stream = false;
    let mut no_table_former = false;
    let mut no_ocr = false;
    let mut force_full_page_ocr = false;
    let mut no_text_panels = false;
    let mut use_web_browser = false;
    let mut asr_model: Option<String> = None;
    let mut video_frames: Option<usize> = None;
    let mut enrich_picture_classes = false;
    let mut enrich_code = false;
    let mut enrich_formula = false;
    let mut bench_warm: Option<usize> = None;
    let mut pages: Option<(usize, usize)> = None;
    let mut ocr_lang: Option<String> = None;
    let mut pipeline: Option<String> = None;
    let mut vlm_endpoint: Option<String> = None;
    let mut vlm_model: Option<String> = None;
    let mut cli = CliOverrides::default();
    let mut path: Option<String> = None;

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "-h" | "--help" => {
                print!("{USAGE}");
                return ExitCode::SUCCESS;
            }
            "-V" | "--version" => {
                println!("docmill {}", env!("CARGO_PKG_VERSION"));
                return ExitCode::SUCCESS;
            }
            "--strict" => strict = true,
            "--fetch-images" => fetch_images = true,
            "--no-stream" => no_stream = true,
            "--no-table-former" => no_table_former = true,
            "--no-ocr" => no_ocr = true,
            "--force-full-page-ocr" => force_full_page_ocr = true,
            "--no-text-panels" => no_text_panels = true,
            "--use-web-browser" => use_web_browser = true,
            "--enrich-picture-classes" => enrich_picture_classes = true,
            "--enrich-code" => enrich_code = true,
            "--enrich-formula" => enrich_formula = true,
            "--to" => to = args.next().unwrap_or_default(),
            "-o" | "--output" => output = args.next(),
            "--asr-model" => asr_model = args.next(),
            "--video-frames" => video_frames = args.next().and_then(|v| v.parse().ok()),
            "--images" => images = args.next().unwrap_or_default(),
            "--pipeline" => match args.next() {
                Some(v) if matches!(v.trim(), "standard" | "vlm") => pipeline = Some(v),
                Some(v) => return usage_error(&format!("--pipeline {v:?} is not standard|vlm")),
                None => return usage_error("--pipeline needs a value (standard|vlm)"),
            },
            "--vlm-endpoint" => vlm_endpoint = args.next(),
            "--vlm-model" => vlm_model = args.next(),
            "--bench-warm" => {
                bench_warm = args.next().and_then(|n| n.parse::<usize>().ok());
                if bench_warm.is_none() {
                    return usage_error("--bench-warm needs a positive run count");
                }
            }
            "--pages" => match args.next().as_deref().map(docling::parse_page_range) {
                Some(Ok(range)) => pages = Some(range),
                Some(Err(e)) => return usage_error(&format!("--pages: {e}")),
                None => return usage_error("--pages needs a range like 1-10"),
            },
            "--ocr-lang" => match args.next() {
                Some(v) if matches!(v.trim(), "en" | "ch") => ocr_lang = Some(v),
                Some(v) => return usage_error(&format!("--ocr-lang {v:?} is not en|ch")),
                None => return usage_error("--ocr-lang needs a value (en|ch)"),
            },
            "--img-ocr-engine" => cli.engine = args.next(),
            "--img-ocr-mode" => cli.mode = args.next(),
            "--img-ocr-lang" => cli.lang = args.next(),
            "--img-ocr-endpoint" => cli.endpoint = args.next(),
            "--img-ocr-model" => cli.model = args.next(),
            "--img-ocr-api-key" => cli.api_key = args.next(),
            "--img-ocr-prompt" => cli.prompt = args.next(),
            "--img-ocr-min-px" => cli.min_px = args.next(),
            "--img-ocr-cache-dir" => cli.cache_dir = args.next(),
            "--no-img-ocr-cache" => cli.no_cache = true,
            "--img-ocr-timeout" => cli.timeout = args.next(),
            "--img-ocr-models-dir" => cli.models_dir = args.next(),
            _ if arg.starts_with("--") => {
                return usage_error(&format!("unknown flag '{arg}' (try --help)"))
            }
            _ => path = Some(arg),
        }
    }

    if !matches!(to.as_str(), "md" | "markdown" | "json" | "dclx" | "chunks") {
        return usage_error(&format!("unknown --to '{to}' (expected: md, json, dclx, chunks)"));
    }
    let image_mode = match images.as_str() {
        "placeholder" => ImageMode::Placeholder,
        "embedded" => ImageMode::Embedded,
        "referenced" => ImageMode::Referenced,
        other => {
            return usage_error(&format!(
                "unknown --images '{other}' (expected: placeholder, embedded, referenced)"
            ))
        }
    };
    let Some(path) = path else {
        eprint!("{USAGE}");
        return ExitCode::from(2);
    };

    let cfg = match ImgOcrConfig::resolve(cli) {
        Ok(cfg) => cfg,
        Err(e) => return usage_error(&e),
    };

    let source = match SourceDocument::from_file(&path) {
        Ok(src) => src,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::FAILURE;
        }
    };

    if let Some(runs) = bench_warm {
        return bench_warm_conversion(&source, runs, no_table_former, no_ocr);
    }

    let run_ocr = cfg.mode != OutputMode::Placeholder;

    // The remote-VLM page pipeline (docling-rs --pipeline vlm): the whole ML
    // stack is replaced; picture OCR still post-processes the result (VLM
    // pictures carry no bytes, so it is a no-op unless a backend adds them).
    if pipeline.as_deref() == Some("vlm") {
        return run_vlm_pipeline(
            &source, vlm_endpoint, vlm_model, pages, strict, &cfg, run_ocr, &to, image_mode,
            &path, output.as_deref(),
        );
    }

    // Standalone image + remote-first chain: skip the ML pipeline, build a
    // one-picture document and let the post-processor do all the work.
    let bypass = run_ocr && cfg.remote_first() && source.format == InputFormat::Image;
    let mut document = if bypass {
        one_picture_document(&source.name, source.bytes, strict)
    } else {
        let mut converter = DocumentConverter::new()
            .strict(strict)
            .asr_model(asr_model.clone())
            .fetch_images(fetch_images)
            .no_table_former(no_table_former)
            .no_ocr(no_ocr)
            .force_full_page_ocr(force_full_page_ocr)
            .no_text_panels(no_text_panels)
            .use_web_browser(use_web_browser)
            .do_picture_classification(enrich_picture_classes)
            .do_code_enrichment(enrich_code)
            .do_formula_enrichment(enrich_formula);
        if let Some(max) = video_frames {
            converter = converter.video_frames(max);
        }
        if let Some((first, last)) = pages {
            converter = converter.page_range(first, last);
        }
        if let Some(lang) = &ocr_lang {
            converter = converter.ocr_lang(lang.clone());
        }
        // docling-rs parity: Markdown streams page by page by default. That
        // only composes with picture OCR disabled (post-processing needs the
        // full tree), so the streaming path is placeholder-mode-only. Without
        // the pdf feature docling has no streaming API at all; the buffered
        // path below produces byte-identical Markdown, just less
        // incrementally.
        #[cfg(feature = "pdf")]
        {
            let is_markdown = matches!(to.as_str(), "md" | "markdown");
            if !run_ocr && is_markdown && !no_stream && output.is_none() {
                return stream_markdown(converter, source, image_mode);
            }
        }
        #[cfg(not(feature = "pdf"))]
        let _ = no_stream;
        match converter.convert(source) {
            Ok(result) => result.document,
            Err(e) => {
                eprintln!("error: {e}");
                return ExitCode::FAILURE;
            }
        }
    };

    if run_ocr {
        let mut runner = match cfg.build_runner() {
            Ok(r) => r,
            Err(e) => return usage_error(&e),
        };
        let opts = PostOptions {
            mode: cfg.mode,
            min_pixels: cfg.min_pixels,
            keep_picture: image_mode != ImageMode::Placeholder,
            ocr_hidden: to == "dclx",
        };
        let stats = postprocess::apply(&mut document, &mut runner, &opts);
        eprintln!("{stats}");
    }

    output_document(document, &to, image_mode, &path, output.as_deref())
}

fn usage_error(msg: &str) -> ExitCode {
    eprintln!("error: {msg}");
    ExitCode::from(2)
}

/// docling-rs's default Markdown path: print each chunk as the converter
/// produces it (page by page for PDF). Placeholder-mode only — picture OCR
/// needs the whole tree.
#[cfg(feature = "pdf")]
fn stream_markdown(
    converter: DocumentConverter,
    source: SourceDocument,
    image_mode: ImageMode,
) -> ExitCode {
    let stream = match converter.convert_streaming_images(source, image_mode) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::FAILURE;
        }
    };
    let stdout = std::io::stdout();
    let mut out = std::io::BufWriter::new(stdout.lock());
    for chunk in stream {
        match chunk {
            Ok(s) => {
                if let Err(e) = out.write_all(s.as_bytes()) {
                    eprintln!("error: writing output: {e}");
                    return ExitCode::FAILURE;
                }
            }
            Err(e) => {
                let _ = out.flush();
                eprintln!("error: {e}");
                return ExitCode::FAILURE;
            }
        }
    }
    if let Err(e) = out.flush() {
        eprintln!("error: writing output: {e}");
        return ExitCode::FAILURE;
    }
    if image_mode == ImageMode::Referenced {
        eprintln!("referenced images (if any) written to ./artifacts/ as pages completed");
    }
    ExitCode::SUCCESS
}

/// docling-rs's `--pipeline vlm`: pages through a remote OpenAI-compatible
/// vision endpoint instead of the local ML stack.
#[allow(clippy::too_many_arguments)] // mirrors the flag surface it forwards
fn run_vlm_pipeline(
    source: &SourceDocument,
    vlm_endpoint: Option<String>,
    vlm_model: Option<String>,
    pages: Option<(usize, usize)>,
    strict: bool,
    cfg: &ImgOcrConfig,
    run_ocr: bool,
    to: &str,
    image_mode: ImageMode,
    path: &str,
    output: Option<&str>,
) -> ExitCode {
    #[cfg(feature = "vlm")]
    {
        let mut opts = match docling::vlm::VlmOptions::resolve(vlm_endpoint, vlm_model) {
            Ok(o) => o,
            Err(e) => return usage_error(&e.to_string()),
        };
        opts.page_range = pages;
        let mut document = match docling::vlm::convert_vlm(source, &opts) {
            Ok(doc) => doc,
            Err(e) => {
                eprintln!("error: {e}");
                return ExitCode::FAILURE;
            }
        };
        document.strict_markdown = strict;
        if run_ocr {
            match cfg.build_runner() {
                Ok(mut runner) => {
                    let opts = PostOptions {
                        mode: cfg.mode,
                        min_pixels: cfg.min_pixels,
                        keep_picture: image_mode != ImageMode::Placeholder,
                        ocr_hidden: to == "dclx",
                    };
                    let stats = postprocess::apply(&mut document, &mut runner, &opts);
                    eprintln!("{stats}");
                }
                Err(e) => return usage_error(&e),
            }
        }
        output_document(document, to, image_mode, path, output)
    }
    #[cfg(not(feature = "vlm"))]
    {
        let _ = (source, vlm_endpoint, vlm_model, pages, strict, cfg, run_ocr, to, image_mode, path, output);
        usage_error("this binary was built without the vlm feature (rebuild with --features vlm)")
    }
}

/// docling-rs's hidden `--bench-warm N`: average warm PDF/image conversion
/// time with models pre-loaded (startup excluded).
fn bench_warm_conversion(
    source: &SourceDocument,
    runs: usize,
    no_table_former: bool,
    no_ocr: bool,
) -> ExitCode {
    #[cfg(feature = "pdf")]
    {
        let result = (|| -> Result<f64, String> {
            let mut pipeline = docling::Pipeline::new()
                .map_err(|e| e.to_string())?
                .no_table_former(no_table_former)
                .no_ocr(no_ocr);
            let once = |p: &mut docling::Pipeline| -> Result<(), String> {
                match source.format {
                    InputFormat::Pdf => p
                        .convert(&source.bytes, None, &source.name)
                        .map(|_| ())
                        .map_err(|e| e.to_string()),
                    InputFormat::Image => p
                        .convert_image(&source.bytes, &source.name)
                        .map(|_| ())
                        .map_err(|e| e.to_string()),
                    other => Err(format!("--bench-warm supports PDF/image only, not {other:?}")),
                }
            };
            once(&mut pipeline)?; // warm-up: load models, prime caches
            let mut total = 0.0f64;
            for _ in 0..runs {
                let t = std::time::Instant::now();
                once(&mut pipeline)?;
                total += t.elapsed().as_secs_f64();
            }
            Ok(total / runs as f64)
        })();
        match result {
            Ok(avg) => {
                println!("{avg:.6}");
                eprintln!("warm conversion: {avg:.4}s/doc over {runs} runs (startup excluded)");
                ExitCode::SUCCESS
            }
            Err(e) => {
                eprintln!("error: {e}");
                ExitCode::FAILURE
            }
        }
    }
    #[cfg(not(feature = "pdf"))]
    {
        let _ = (source, runs, no_table_former, no_ocr);
        usage_error("--bench-warm needs the pdf feature")
    }
}

/// `--to chunks`: the chunk-record dump docling-rs prints (see
/// `docling::chunks::chunk_records` for the tokenizer resolution rules).
fn chunks_json(document: &DoclingDocument) -> String {
    let mut warn = |msg: String| eprintln!("warning: {msg}");
    let out = docling::chunks::chunk_records(document, &mut warn);
    format!(
        "{}\n",
        serde_json::to_string_pretty(&out).expect("chunks are serializable")
    )
}

/// `docmill serve …`: parse the serve flags and run the HTTP service.
fn run_serve(args: Vec<String>) -> ExitCode {
    #[cfg(feature = "serve")]
    {
        let mut addr = "127.0.0.1:8877".to_string();
        let mut cli = CliOverrides::default();
        let mut it = args.into_iter();
        while let Some(arg) = it.next() {
            match arg.as_str() {
                "--addr" => match it.next() {
                    Some(v) => addr = v,
                    None => return usage_error("--addr needs HOST:PORT"),
                },
                "--img-ocr-engine" => cli.engine = it.next(),
                "--img-ocr-mode" => cli.mode = it.next(),
                "--img-ocr-lang" => cli.lang = it.next(),
                "--img-ocr-endpoint" => cli.endpoint = it.next(),
                "--img-ocr-model" => cli.model = it.next(),
                "--img-ocr-api-key" => cli.api_key = it.next(),
                "--img-ocr-prompt" => cli.prompt = it.next(),
                "--img-ocr-min-px" => cli.min_px = it.next(),
                "--img-ocr-cache-dir" => cli.cache_dir = it.next(),
                "--no-img-ocr-cache" => cli.no_cache = true,
                "--img-ocr-timeout" => cli.timeout = it.next(),
                "--img-ocr-models-dir" => cli.models_dir = it.next(),
                other => {
                    eprintln!("error: unknown serve argument '{other}'");
                    eprintln!(
                        "usage: docmill serve [--addr HOST:PORT] [--img-ocr-* flags]"
                    );
                    return ExitCode::from(2);
                }
            }
        }
        let cfg = match ImgOcrConfig::resolve(cli) {
            Ok(cfg) => cfg,
            Err(e) => return usage_error(&e),
        };
        match docmill::serve::serve(docmill::serve::ServeConfig { addr, cfg }) {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("error: {e}");
                ExitCode::FAILURE
            }
        }
    }
    #[cfg(not(feature = "serve"))]
    {
        let _ = args;
        usage_error("this binary was built without the serve feature (rebuild with --features serve)")
    }
}

/// Buffered output tail, mirroring docling-cli's: `--to` selection, image
/// sidecars, optional output file.
fn output_document(
    document: DoclingDocument,
    to: &str,
    image_mode: ImageMode,
    input_path: &str,
    output: Option<&str>,
) -> ExitCode {
    if to == "dclx" {
        let out = match output {
            Some(p) => std::path::PathBuf::from(p),
            None => {
                let stem = Path::new(input_path)
                    .file_stem()
                    .map(|s| s.to_string_lossy().into_owned())
                    .unwrap_or_else(|| "document".into());
                std::path::PathBuf::from(format!("{stem}.dclx"))
            }
        };
        if let Err(e) = docling::dclx::save_as_dclx(&document, &out) {
            eprintln!("error: dclx: {e}");
            return ExitCode::FAILURE;
        }
        println!("{}", out.display());
        return ExitCode::SUCCESS;
    }

    let body = if to == "json" {
        format!("{}\n", document.export_to_json())
    } else if to == "chunks" {
        chunks_json(&document)
    } else if image_mode == ImageMode::Placeholder {
        document.export_to_markdown()
    } else {
        let (md, artifacts) = document.export_to_markdown_with_images(image_mode, "artifacts");
        for (rel, bytes) in &artifacts {
            let rel = Path::new(rel);
            if let Some(dir) = rel.parent() {
                if let Err(e) = std::fs::create_dir_all(dir) {
                    eprintln!("error: creating {}: {e}", dir.display());
                    return ExitCode::FAILURE;
                }
            }
            if let Err(e) = std::fs::write(rel, bytes) {
                eprintln!("error: writing {}: {e}", rel.display());
                return ExitCode::FAILURE;
            }
        }
        if !artifacts.is_empty() {
            eprintln!("wrote {} image(s) to ./artifacts/", artifacts.len());
        }
        md
    };

    match output {
        Some(p) => {
            if let Err(e) = std::fs::write(p, body.as_bytes()) {
                eprintln!("error: writing {p}: {e}");
                return ExitCode::FAILURE;
            }
            eprintln!("wrote {p}");
        }
        None => {
            let stdout = std::io::stdout();
            let mut out = stdout.lock();
            if let Err(e) = out.write_all(body.as_bytes()) {
                eprintln!("error: writing output: {e}");
                return ExitCode::FAILURE;
            }
        }
    }
    ExitCode::SUCCESS
}
