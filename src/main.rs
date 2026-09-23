//! docmill — docling.rs conversion plus custom OCR over embedded
//! pictures (DOCX drawings, PDF figure regions, standalone images).
//!
//! Usage: docmill [conversion flags] [--img-ocr-* flags] <input-file>
//!        docmill --input GLOB|DIR --output DIR [--jobs N] [flags]
//!
//! Conversion flags (same semantics as `docling-rs`, buffered path):
//!   --to md|json|dclx   output format (default: md)
//!   --output PATH, -o   single input: write to FILE; batch: output DIR
//!   --input GLOB|DIR    batch-convert matching/supported files
//!   --jobs N            batch workers (default: 1)
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
//!   --asr-model PRESET / --asr-lang CODE|auto
//!                       Whisper preset and language for audio/video
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
//!   --img-ocr-models-dir DIR      local models dir (default ./.models; point it
//!                                 at your docling.rs checkout's .models/)
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
use docmill::conversion::{normalize_ocr_lang, ConversionOptions, WarmPipeline};
use docmill::input::DetectedSource;
use docmill::postprocess::{self, one_picture_document, OutputMode, PostOptions};

/// The complete flag reference, printed by `--help` — kept as one constant so
/// the help text and the parser can be reviewed side by side.
const USAGE: &str = "\
docmill — document conversion (docling.rs) + OCR over embedded pictures

usage: docmill [OPTIONS] <input-file>
       docmill --input GLOB|DIR --output DIR [--jobs N] [OPTIONS]

Subcommands:
  serve [--addr HOST:PORT] [--img-ocr-* flags]
                              local HTTP conversion service (default
                              127.0.0.1:8877; POST /convert, GET /health,
                              GET / for a test form) — put nginx in front

Conversion (same semantics as docling-rs):
  --to md|json|dclx|chunks    output format (default: md)
  --input GLOB|DIR            batch-convert a glob or supported files under DIR
  -o, --output PATH           single input: output FILE; batch: required DIR
  --jobs N                    batch workers (default: 1; requires --input)
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
  --skip-ocr                  keep layout/tables/pictures; skip page recognition
  --ocr-mode MODE             default|full_page|layout_regions|pdf_aware_layout_regions
  --ocr-scale SCALE           positive OCR pixels per PDF point
  --heading-hierarchy         infer PDF heading levels
  --encoding NAME             explicit encoding for text input
  --page-break-placeholder TEXT  text inserted between Markdown pages
  --ocr-lang TAG              English/Chinese page-OCR language (e.g. en-US, zh-Hant)
  --asr-model PRESET          Whisper preset for audio inputs
  --asr-lang CODE             Whisper language code or auto (default)
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
  --img-ocr-max-retries N     retries after transient errors (default 3)
  --img-ocr-models-dir DIR    local models dir (default: ./.models, else next to
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

    let mut conversion = ConversionOptions::default();
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
    let mut asr_lang: Option<String> = None;
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
    let mut input: Option<String> = None;
    let mut jobs = 1usize;
    let mut jobs_set = false;

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
            "--skip-ocr"
            | "--heading-hierarchy"
            | "--ocr-mode"
            | "--ocr-scale"
            | "--encoding"
            | "--page-break-placeholder" => {
                if let Err(e) = conversion.parse_flag(&arg, &mut args) {
                    return usage_error(&e);
                }
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
            "--input" => match args.next() {
                Some(v) => input = Some(v),
                None => return usage_error("--input needs a glob pattern or directory"),
            },
            "-o" | "--output" => match args.next() {
                Some(v) => output = Some(v),
                None => return usage_error("--output needs a path"),
            },
            "--jobs" => {
                jobs_set = true;
                jobs = match args.next().and_then(|v| v.parse().ok()) {
                    Some(n) if n >= 1 => n,
                    _ => return usage_error("--jobs needs a positive integer"),
                };
            }
            "--asr-model" => asr_model = args.next(),
            "--asr-lang" => asr_lang = args.next(),
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
                Some(v) => match normalize_ocr_lang(&v) {
                    Ok(lang) => ocr_lang = Some(lang),
                    Err(e) => return usage_error(&e),
                },
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
            "--img-ocr-max-retries" => {
                cli.max_retries = Some(match args.next() {
                    Some(v) => v,
                    None => return usage_error("--img-ocr-max-retries needs a value"),
                })
            }
            "--img-ocr-models-dir" => cli.models_dir = args.next(),
            _ if arg.starts_with("--") => {
                return usage_error(&format!("unknown flag '{arg}' (try --help)"))
            }
            _ => path = Some(arg),
        }
    }

    if !matches!(to.as_str(), "md" | "markdown" | "json" | "dclx" | "chunks") {
        return usage_error(&format!(
            "unknown --to '{to}' (expected: md, json, dclx, chunks)"
        ));
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

    let conversion = ConversionOptions {
        strict,
        fetch_images,
        no_table_former,
        no_ocr,
        force_full_page_ocr,
        no_text_panels,
        use_web_browser,
        enrich_picture_classes,
        enrich_code,
        enrich_formula,
        asr_model: asr_model.clone(),
        asr_lang: asr_lang.clone(),
        video_frames,
        pages,
        ocr_lang: ocr_lang.clone(),
        ..conversion
    };

    if let Some(pattern) = input {
        if path.is_some() {
            return usage_error("--input and a positional input file are mutually exclusive");
        }
        if bench_warm.is_some() {
            return usage_error("--bench-warm is a single-file mode; drop --input");
        }
        let Some(outdir) = output else {
            return usage_error("--input needs --output DIR for the converted files");
        };
        let (files, base) = match expand_glob(&pattern) {
            Ok(found) => found,
            Err(e) => return usage_error(&e),
        };
        let cfg = match ImgOcrConfig::resolve(cli) {
            Ok(cfg) => cfg,
            Err(e) => return usage_error(&e),
        };
        #[cfg(feature = "vlm")]
        let vlm = if pipeline.as_deref() == Some("vlm") {
            let mut opts = match docling::vlm::VlmOptions::resolve(vlm_endpoint, vlm_model) {
                Ok(opts) => opts,
                Err(e) => return usage_error(&e.to_string()),
            };
            opts.page_range = pages;
            Some(opts)
        } else {
            None
        };
        #[cfg(not(feature = "vlm"))]
        if pipeline.as_deref() == Some("vlm") {
            return usage_error(
                "this binary was built without the vlm feature (rebuild with --features vlm)",
            );
        }
        let batch = BatchCfg {
            to,
            image_mode,
            conversion,
            picture_ocr: cfg,
            #[cfg(feature = "vlm")]
            vlm,
        };
        return run_batch(files, &base, Path::new(&outdir), jobs, &batch);
    }
    if jobs_set {
        return usage_error("--jobs requires --input GLOB|DIR");
    }
    let Some(path) = path else {
        eprint!("{USAGE}");
        return ExitCode::from(2);
    };

    let cfg = match ImgOcrConfig::resolve(cli) {
        Ok(cfg) => cfg,
        Err(e) => return usage_error(&e),
    };

    let source = match DetectedSource::from_path(&path) {
        Ok(src) => src,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::FAILURE;
        }
    };
    if let Some(warning) = &source.warning {
        eprintln!("warning: {warning}");
    }

    if let Some(runs) = bench_warm {
        let source = SourceDocument::from_bytes(source.name, source.format, source.bytes);
        return bench_warm_conversion(&source, runs, &conversion);
    }

    let run_ocr = cfg.mode != OutputMode::Placeholder;

    // The remote-VLM page pipeline (docling-rs --pipeline vlm): the whole ML
    // stack is replaced; picture OCR still post-processes the result (VLM
    // pictures carry no bytes, so it is a no-op unless a backend adds them).
    if pipeline.as_deref() == Some("vlm") {
        let format = source.format;
        if !matches!(format, InputFormat::Pdf | InputFormat::Image) {
            return usage_error("--pipeline vlm supports docling PDF/image inputs only");
        }
        let source = SourceDocument::from_bytes(source.name, format, source.bytes);
        return run_vlm_pipeline(
            &source,
            vlm_endpoint,
            vlm_model,
            pages,
            &conversion,
            &cfg,
            run_ocr,
            &to,
            image_mode,
            &path,
            output.as_deref(),
        );
    }

    let mut document = {
        let format = source.format;
        let source = SourceDocument::from_bytes(source.name, format, source.bytes);
        // Standalone image + remote-first chain: skip the ML pipeline,
        // build one Picture and let the post-processor do the work.
        let bypass = run_ocr && cfg.remote_first() && format == InputFormat::Image;
        if bypass {
            one_picture_document(&source.name, source.bytes, strict)
        } else {
            let converter = conversion.converter();
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
        }
    };

    conversion.finish_document(&mut document);
    if run_ocr {
        let mut runner = match cfg.build_runner() {
            Ok(r) => r,
            Err(e) => return usage_error(&e),
        };
        let opts = PostOptions {
            mode: cfg.mode,
            min_pixels: cfg.min_pixels,
            keep_picture: image_mode != ImageMode::Placeholder,
            target: postprocess::OutputTarget::from_format(&to),
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
    conversion: &ConversionOptions,
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
        conversion.finish_document(&mut document);
        if run_ocr {
            match cfg.build_runner() {
                Ok(mut runner) => {
                    let opts = PostOptions {
                        mode: cfg.mode,
                        min_pixels: cfg.min_pixels,
                        keep_picture: image_mode != ImageMode::Placeholder,
                        target: postprocess::OutputTarget::from_format(&to),
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
        let _ = (
            source,
            vlm_endpoint,
            vlm_model,
            pages,
            conversion,
            cfg,
            run_ocr,
            to,
            image_mode,
            path,
            output,
        );
        usage_error("this binary was built without the vlm feature (rebuild with --features vlm)")
    }
}

/// docling-rs's hidden `--bench-warm N`: average warm PDF/image conversion
/// time with models pre-loaded (startup excluded).
fn bench_warm_conversion(
    source: &SourceDocument,
    runs: usize,
    options: &ConversionOptions,
) -> ExitCode {
    #[cfg(feature = "pdf")]
    {
        let result = (|| -> Result<f64, String> {
            if !matches!(source.format, InputFormat::Pdf | InputFormat::Image) {
                return Err(format!(
                    "--bench-warm supports PDF/image only, not {:?}",
                    source.format
                ));
            }
            let mut pipeline = WarmPipeline::default();
            let once = |p: &mut WarmPipeline| -> Result<(), String> {
                p.convert(
                    SourceDocument::from_bytes(
                        source.name.clone(),
                        source.format,
                        source.bytes.clone(),
                    ),
                    options,
                )
                .map(|_| ())
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
        let _ = (source, runs, options);
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

/// Conversion and picture-OCR settings frozen for every file in a batch.
struct BatchCfg {
    to: String,
    image_mode: ImageMode,
    conversion: ConversionOptions,
    picture_ocr: ImgOcrConfig,
    #[cfg(feature = "vlm")]
    vlm: Option<docling::vlm::VlmOptions>,
}

/// Expand a glob or recursively sweep a directory, returning the files and
/// the static base path used to preserve their relative output layout.
fn expand_glob(pattern: &str) -> Result<(Vec<std::path::PathBuf>, std::path::PathBuf), String> {
    let dir = Path::new(pattern);
    if dir.is_dir() {
        let mut files = Vec::new();
        let mut stack = vec![dir.to_path_buf()];
        while let Some(current) = stack.pop() {
            let entries = std::fs::read_dir(&current)
                .map_err(|e| format!("--input '{}': {e}", current.display()))?;
            for entry in entries {
                let path = entry.map_err(|e| format!("--input: {e}"))?.path();
                if path.is_dir() {
                    stack.push(path);
                } else if path
                    .extension()
                    .and_then(|ext| ext.to_str())
                    .is_some_and(|ext| InputFormat::from_extension(ext).is_some())
                {
                    files.push(path);
                }
            }
        }
        if files.is_empty() {
            return Err(format!(
                "--input '{pattern}' contains no files with a convertible extension"
            ));
        }
        files.sort();
        return Ok((files, dir.to_path_buf()));
    }

    let base = glob_base(pattern);
    let mut files = Vec::new();
    for entry in glob::glob(pattern).map_err(|e| format!("--input: {e}"))? {
        match entry {
            Ok(path) if path.is_file() => files.push(path),
            Ok(_) => {}
            Err(e) => eprintln!("warning: {e}"),
        }
    }
    if files.is_empty() {
        return Err(format!("--input '{pattern}' matches no files"));
    }
    files.sort();
    Ok((files, base))
}

fn glob_base(pattern: &str) -> std::path::PathBuf {
    let mut base = std::path::PathBuf::new();
    for component in Path::new(pattern).components() {
        let text = component.as_os_str().to_string_lossy();
        if text.contains(['*', '?', '[']) {
            break;
        }
        base.push(component);
    }
    if base == Path::new(pattern) {
        base.pop();
    }
    base
}

fn batch_out_path(file: &Path, base: &Path, output: &Path, to: &str) -> std::path::PathBuf {
    let relative = file
        .strip_prefix(base)
        .map(Path::to_path_buf)
        .unwrap_or_else(|_| Path::new(file.file_name().unwrap_or_default()).to_path_buf());
    let extension = match to {
        "json" => "json",
        "dclx" => "dclx",
        "chunks" => "chunks.json",
        _ => "md",
    };
    output.join(relative).with_extension(extension)
}

fn batch_converter(cfg: &BatchCfg) -> DocumentConverter {
    cfg.conversion.converter()
}

type SharedBatchPipeline = std::sync::Mutex<WarmPipeline>;

fn batch_convert_one(
    file: &Path,
    base: &Path,
    output: &Path,
    cfg: &BatchCfg,
    converter: &DocumentConverter,
    runner: &mut Option<docmill::engine::OcrRunner>,
    shared_pipeline: &SharedBatchPipeline,
) -> Result<(std::path::PathBuf, f64, Option<usize>), String> {
    let source = DetectedSource::from_path(file).map_err(|e| e.to_string())?;
    if let Some(warning) = &source.warning {
        eprintln!("warning: {}: {warning}", file.display());
    }
    let page_count = batch_page_count(&source, cfg.conversion.pages);
    match page_count {
        Some(1) => eprintln!("start: {} (1 page)", file.display()),
        Some(n) => eprintln!("start: {} ({n} pages)", file.display()),
        None => eprintln!("start: {}", file.display()),
    }

    let started = std::time::Instant::now();
    let format = source.format;
    let mut document = if !batch_uses_vlm(cfg)
        && cfg.picture_ocr.mode != OutputMode::Placeholder
        && cfg.picture_ocr.remote_first()
        && format == InputFormat::Image
    {
        one_picture_document(&source.name, source.bytes, cfg.conversion.strict)
    } else {
        batch_convert_source(source, converter, cfg, shared_pipeline)?
    };
    cfg.conversion.finish_document(&mut document);

    if let Some(runner) = runner {
        let options = PostOptions {
            mode: cfg.picture_ocr.mode,
            min_pixels: cfg.picture_ocr.min_pixels,
            keep_picture: cfg.image_mode != ImageMode::Placeholder,
            target: postprocess::OutputTarget::from_format(&cfg.to),
        };
        let stats = postprocess::apply(&mut document, runner, &options);
        eprintln!("{}: {stats}", file.display());
    }

    let out = batch_out_path(file, base, output, &cfg.to);
    write_batch_document(document, cfg, &out)?;
    Ok((out, started.elapsed().as_secs_f64(), page_count))
}

fn batch_uses_vlm(cfg: &BatchCfg) -> bool {
    #[cfg(feature = "vlm")]
    {
        cfg.vlm.is_some()
    }
    #[cfg(not(feature = "vlm"))]
    {
        let _ = cfg;
        false
    }
}

fn batch_page_count(source: &DetectedSource, pages: Option<(usize, usize)>) -> Option<usize> {
    #[cfg(feature = "pdf")]
    {
        if source.format == InputFormat::Pdf {
            return docling::pdf_page_count(&source.bytes, None)
                .ok()
                .map(|total| match pages {
                    Some((first, last)) => (last.min(total) + 1).saturating_sub(first).min(total),
                    None => total,
                });
        }
    }
    #[cfg(not(feature = "pdf"))]
    let _ = (source, pages);
    None
}

fn batch_convert_source(
    source: DetectedSource,
    converter: &DocumentConverter,
    cfg: &BatchCfg,
    shared_pipeline: &SharedBatchPipeline,
) -> Result<DoclingDocument, String> {
    #[cfg(not(any(feature = "pdf", feature = "vlm")))]
    let _ = cfg;
    #[cfg(feature = "vlm")]
    if let Some(vlm) = &cfg.vlm {
        return docling::vlm::convert_vlm(&source.into_docling(), vlm).map_err(|e| e.to_string());
    }

    #[cfg(feature = "pdf")]
    if matches!(source.format, InputFormat::Pdf | InputFormat::Image) {
        let mut guard = shared_pipeline.lock().unwrap_or_else(|p| p.into_inner());
        return guard.convert(source.into_docling(), &cfg.conversion);
    }
    #[cfg(not(feature = "pdf"))]
    let _ = shared_pipeline;
    converter
        .convert(source.into_docling())
        .map(|result| result.document)
        .map_err(|e| e.to_string())
}

fn write_batch_document(
    document: DoclingDocument,
    cfg: &BatchCfg,
    out: &Path,
) -> Result<(), String> {
    if let Some(dir) = out.parent() {
        std::fs::create_dir_all(dir).map_err(|e| format!("creating {}: {e}", dir.display()))?;
    }
    match cfg.to.as_str() {
        "json" => std::fs::write(out, document.export_to_json())
            .map_err(|e| format!("writing {}: {e}", out.display())),
        "chunks" => std::fs::write(out, chunks_json(&document))
            .map_err(|e| format!("writing {}: {e}", out.display())),
        "dclx" => docling::dclx::save_as_dclx(&document, out).map_err(|e| e.to_string()),
        _ if cfg.image_mode == ImageMode::Placeholder => {
            std::fs::write(out, document.export_to_markdown())
                .map_err(|e| format!("writing {}: {e}", out.display()))
        }
        _ => {
            let stem = out
                .file_stem()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_else(|| "document".into());
            let artifact_dir = format!("{stem}_artifacts");
            let (markdown, artifacts) =
                document.export_to_markdown_with_images(cfg.image_mode, &artifact_dir);
            let parent = out.parent().unwrap_or(Path::new(""));
            for (relative, bytes) in artifacts {
                let target = parent.join(relative);
                if let Some(dir) = target.parent() {
                    std::fs::create_dir_all(dir)
                        .map_err(|e| format!("creating {}: {e}", dir.display()))?;
                }
                std::fs::write(&target, bytes)
                    .map_err(|e| format!("writing {}: {e}", target.display()))?;
            }
            std::fs::write(out, markdown).map_err(|e| format!("writing {}: {e}", out.display()))
        }
    }
}

fn run_batch(
    files: Vec<std::path::PathBuf>,
    base: &Path,
    output: &Path,
    jobs: usize,
    cfg: &BatchCfg,
) -> ExitCode {
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    let run_picture_ocr = cfg.picture_ocr.mode != OutputMode::Placeholder;
    if run_picture_ocr {
        if let Err(e) = cfg.picture_ocr.build_runner() {
            return usage_error(&e);
        }
    }

    let next = AtomicUsize::new(0);
    let failed = AtomicUsize::new(0);
    let succeeded = AtomicUsize::new(0);
    let abort = AtomicBool::new(false);
    #[cfg(feature = "pdf")]
    let shared_pipeline = std::sync::Mutex::new(WarmPipeline::default());
    #[cfg(not(feature = "pdf"))]
    let shared_pipeline = SharedBatchPipeline::default();
    let workers = jobs.min(files.len()).max(1);

    std::thread::scope(|scope| {
        for _ in 0..workers {
            scope.spawn(|| {
                let converter = batch_converter(cfg);
                let mut runner = run_picture_ocr
                    .then(|| cfg.picture_ocr.build_runner().expect("runner prevalidated"));
                loop {
                    if abort.load(Ordering::Relaxed) {
                        break;
                    }
                    let index = next.fetch_add(1, Ordering::Relaxed);
                    let Some(file) = files.get(index) else {
                        break;
                    };
                    match batch_convert_one(
                        file,
                        base,
                        output,
                        cfg,
                        &converter,
                        &mut runner,
                        &shared_pipeline,
                    ) {
                        Ok((out, seconds, pages)) => {
                            match pages {
                                Some(n) if n > 0 => eprintln!(
                                    "ok: {} -> {} ({seconds:.1}s, {:.0} ms/page)",
                                    file.display(),
                                    out.display(),
                                    seconds * 1000.0 / n as f64
                                ),
                                _ => eprintln!(
                                    "ok: {} -> {} ({seconds:.1}s)",
                                    file.display(),
                                    out.display()
                                ),
                            }
                            println!("{}", out.display());
                            succeeded.fetch_add(1, Ordering::Relaxed);
                        }
                        Err(error) => {
                            failed.fetch_add(1, Ordering::Relaxed);
                            eprintln!("error: {}: {error}", file.display());
                            if global_batch_failure(&error) {
                                abort.store(true, Ordering::Relaxed);
                                eprintln!(
                                    "fatal: shared conversion runtime is unavailable; aborting batch"
                                );
                            }
                        }
                    }
                }
            });
        }
    });

    let failures = failed.load(Ordering::Relaxed);
    let successes = succeeded.load(Ordering::Relaxed);
    let skipped = files.len() - successes - failures;
    if skipped > 0 {
        eprintln!("batch: {successes} converted, {failures} failed, {skipped} skipped");
    } else {
        eprintln!("batch: {successes} converted, {failures} failed");
    }
    if failures > 0 {
        ExitCode::FAILURE
    } else {
        ExitCode::SUCCESS
    }
}

fn global_batch_failure(error: &str) -> bool {
    error.contains("execution provider")
        || error.contains("pdfium library is not installed")
        || error.contains("model not found at")
}

/// `docmill serve …`: parse the serve flags and run the HTTP service.
fn run_serve(args: Vec<String>) -> ExitCode {
    #[cfg(feature = "serve")]
    {
        let mut addr = "127.0.0.1:8877".to_string();
        let mut cli = CliOverrides::default();
        let mut conversion = ConversionOptions::default();
        let mut it = args.into_iter();
        while let Some(arg) = it.next() {
            match arg.as_str() {
                "--skip-ocr"
                | "--heading-hierarchy"
                | "--ocr-mode"
                | "--ocr-scale"
                | "--encoding"
                | "--page-break-placeholder" => {
                    if let Err(e) = conversion.parse_flag(&arg, &mut it) {
                        return usage_error(&e);
                    }
                }
                "--no-ocr" => conversion.no_ocr = true,
                "--no-table-former" => conversion.no_table_former = true,
                "--force-full-page-ocr" => conversion.force_full_page_ocr = true,
                "--no-text-panels" => conversion.no_text_panels = true,
                "--strict" => conversion.strict = true,
                "--ocr-lang" => match it.next().map(|v| normalize_ocr_lang(&v)) {
                    Some(Ok(lang)) => conversion.ocr_lang = Some(lang),
                    Some(Err(e)) => return usage_error(&e),
                    None => return usage_error("--ocr-lang needs a language tag"),
                },
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
                "--img-ocr-max-retries" => {
                    cli.max_retries = Some(match it.next() {
                        Some(v) => v,
                        None => return usage_error("--img-ocr-max-retries needs a value"),
                    })
                }
                "--img-ocr-models-dir" => cli.models_dir = it.next(),
                other => {
                    eprintln!("error: unknown serve argument '{other}'");
                    eprintln!("usage: docmill serve [--addr HOST:PORT] [--img-ocr-* flags]");
                    return ExitCode::from(2);
                }
            }
        }
        let cfg = match ImgOcrConfig::resolve(cli) {
            Ok(cfg) => cfg,
            Err(e) => return usage_error(&e),
        };
        match docmill::serve::serve(docmill::serve::ServeConfig {
            addr,
            cfg,
            conversion,
        }) {
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
        usage_error(
            "this binary was built without the serve feature (rebuild with --features serve)",
        )
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn glob_base_is_the_static_prefix() {
        assert_eq!(glob_base("reports/**/*.pdf"), Path::new("reports"));
        assert_eq!(glob_base("reports/one.pdf"), Path::new("reports"));
    }

    #[test]
    fn batch_output_extensions_match_the_target() {
        let file = Path::new("input/nested/report.pdf");
        let base = Path::new("input");
        let output = Path::new("converted");
        assert_eq!(
            batch_out_path(file, base, output, "md"),
            Path::new("converted/nested/report.md")
        );
        assert_eq!(
            batch_out_path(file, base, output, "json"),
            Path::new("converted/nested/report.json")
        );
        assert_eq!(
            batch_out_path(file, base, output, "dclx"),
            Path::new("converted/nested/report.dclx")
        );
        assert_eq!(
            batch_out_path(file, base, output, "chunks"),
            Path::new("converted/nested/report.chunks.json")
        );
    }

    #[test]
    fn directory_batch_discovery_is_sorted_and_filters_extensions() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("nested")).unwrap();
        std::fs::write(dir.path().join("z.md"), "z").unwrap();
        std::fs::write(dir.path().join("nested/a.md"), "a").unwrap();
        std::fs::write(dir.path().join("ignored.log"), "ignored").unwrap();

        let (files, base) = expand_glob(dir.path().to_str().unwrap()).unwrap();
        assert_eq!(base, dir.path());
        assert_eq!(
            files,
            vec![dir.path().join("nested/a.md"), dir.path().join("z.md")]
        );
    }
}
