//! Post-conversion document transform: find every `Node::Picture` that
//! carries extracted image bytes, OCR it through the engine chain, and splice
//! the recognized text into the node tree in place of (or after) the picture.
//!
//! Operating on the converted [`DoclingDocument`] — rather than inside the
//! backends — is what keeps the upstream crates untouched: the inserted nodes
//! are ordinary `Paragraph`s, which every serializer (Markdown/JSON/DocLang)
//! already renders, and Markdown passes their text through verbatim
//! (docling-core's `strict_text` does no HTML escaping), so `<!-- ocr:… -->`
//! markers and `> ` quote prefixes survive as written.
//!
//! Output-aware traversal preserves wrappers, content layers, captions and
//! provenance. JSON uses the backend's item tree when present; other outputs
//! use nodes, including rich table cells for DocLang.

use docling_core::{ContentLayer, DoclingDocument, Node};
use std::collections::HashMap;

use crate::engine::OcrRunner;

mod tree;

/// The representation and content layers the requested exporter consumes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputTarget {
    Markdown,
    Json,
    Doclang,
    Chunks,
}

impl OutputTarget {
    pub fn from_format(format: &str) -> Self {
        match format {
            "json" => Self::Json,
            "dclx" => Self::Doclang,
            "chunks" => Self::Chunks,
            _ => Self::Markdown,
        }
    }
}

/// How OCR text lands in the document, per picture.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputMode {
    /// A fenced code block holding the layout-preserving grid rendering when
    /// the engine provided geometry (local v5, paddle), else the flat text —
    /// monospace, so the source image's spacing survives Markdown (default).
    Fence,
    /// `<!-- ocr:begin engine=… -->` / text / `<!-- ocr:end -->`.
    Markers,
    /// Just the text.
    Text,
    /// The text as a Markdown blockquote (`> ` per line) — a visual box.
    Quote,
    /// Upstream behavior: post-processing is skipped entirely (callers never
    /// invoke [`apply`] in this mode; the variant exists for flag parsing).
    Placeholder,
}

#[derive(Debug, Clone, Copy)]
pub struct PostOptions {
    pub mode: OutputMode,
    /// Pictures below this pixel area (width × height) are skipped — icons
    /// and logos aren't worth an engine call. 0 disables the check, and
    /// unknown dimensions (0×0) are never skipped.
    pub min_pixels: u32,
    /// Keep the Picture node and append the text after it (used with
    /// `--images embedded|referenced`, where the image itself still renders).
    pub keep_picture: bool,
    /// Export target; determines the authoritative tree and visible layers.
    pub target: OutputTarget,
}

/// Counters for the end-of-run summary line.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct OcrStats {
    /// Every `Node::Picture` encountered (with or without bytes).
    pub pictures: usize,
    /// Pictures whose OCR text was spliced in.
    pub ocred: usize,
    /// Of `ocred` + `empty`, how many came from the disk cache.
    pub cached: usize,
    pub skipped_small: usize,
    /// Picture OCR skipped because a structured table from the same page is
    /// substantially contained by the picture region.
    pub skipped_structured: usize,
    /// OCR ran and legitimately found no text: the picture is dropped from
    /// the output entirely (no placeholder) — unless the image itself renders
    /// (`keep_picture`), in which case the node stays.
    pub empty: usize,
    /// Every engine failed for this picture (placeholder kept).
    pub failed: usize,
}

impl std::fmt::Display for OcrStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "docmill: {} picture(s), {} ocr'd ({} cached), {} skipped (small), {} skipped (structured table), {} empty, {} failed",
            self.pictures,
            self.ocred,
            self.cached,
            self.skipped_small,
            self.skipped_structured,
            self.empty,
            self.failed
        )
    }
}

/// Wrap a raw image file as a document holding one picture — the remote-first
/// fast path for standalone image inputs (no ML pipeline, the post-processor
/// does all the work). `width`/`height` are 0 when unknown, which the
/// min-pixels check treats as "never skip"; the mimetype comes from the file
/// extension.
pub fn one_picture_document(name: &str, bytes: Vec<u8>, strict: bool) -> DoclingDocument {
    #[cfg(feature = "local-ocr")]
    let (width, height) = image::load_from_memory(&bytes)
        .map(|i| (i.width(), i.height()))
        .unwrap_or((0, 0));
    #[cfg(not(feature = "local-ocr"))]
    let (width, height) = (0u32, 0u32);
    let ext = std::path::Path::new(name)
        .extension()
        .map(|e| e.to_string_lossy().to_ascii_lowercase())
        .unwrap_or_default();
    let mimetype = if bytes.starts_with(b"\xff\xd8\xff") {
        "image/jpeg"
    } else if bytes.starts_with(b"GIF87a") || bytes.starts_with(b"GIF89a") {
        "image/gif"
    } else if bytes.starts_with(b"BM") {
        "image/bmp"
    } else if bytes.starts_with(b"II*\0") || bytes.starts_with(b"MM\0*") {
        "image/tiff"
    } else if bytes.len() >= 12 && &bytes[..4] == b"RIFF" && &bytes[8..12] == b"WEBP" {
        "image/webp"
    } else {
        match ext.as_str() {
            "jpg" | "jpeg" => "image/jpeg",
            "gif" => "image/gif",
            "bmp" => "image/bmp",
            "tif" | "tiff" => "image/tiff",
            "webp" => "image/webp",
            _ => "image/png",
        }
    };
    let stem = std::path::Path::new(name)
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| name.to_string());
    DoclingDocument {
        name: stem,
        nodes: vec![Node::Picture {
            caption: None,
            image: Some(docling_core::PictureImage {
                mimetype: mimetype.to_string(),
                width,
                height,
                data: bytes,
            }),
            classification: None,
            caption_href: None,
            caption_parent: Default::default(),
        }],
        strict_markdown: strict,
        compact_tables: false,
        links: Vec::new(),
        confidence: None,
        page_break_placeholder: None,
        tree: None,
    }
}

/// Transform the representation consumed by `opts.target` in place. Other
/// representations are left intact: export using this target after calling.
pub fn apply(doc: &mut DoclingDocument, runner: &mut OcrRunner, opts: &PostOptions) -> OcrStats {
    if opts.mode == OutputMode::Placeholder {
        return OcrStats::default();
    }
    if opts.target == OutputTarget::Json {
        if let Some(tree) = &mut doc.tree {
            return tree::apply(tree, runner, opts);
        }
    }
    let mut stats = OcrStats::default();
    let table_locations = table_locations(&doc.nodes, opts.target);
    let mut page = 0usize;
    walk(
        &mut doc.nodes,
        false,
        &mut page,
        &table_locations,
        runner,
        opts,
        &mut stats,
    );
    stats
}

type TableLocations = HashMap<(usize, bool), Vec<[f64; 4]>>;

fn location(wraps: &[Wrap]) -> Option<(bool, [f64; 4])> {
    wraps
        .iter()
        .find_map(|w| match w {
            Wrap::Prov { bbox, .. } => Some((true, bbox.map(f64::from))),
            _ => None,
        })
        .or_else(|| {
            wraps.iter().find_map(|w| match w {
                Wrap::Located(bbox) => Some((false, bbox.map(f64::from))),
                _ => None,
            })
        })
}

fn table_locations(nodes: &[Node], target: OutputTarget) -> TableLocations {
    fn collect(nodes: &[Node], target: OutputTarget, page: &mut usize, out: &mut TableLocations) {
        for node in nodes {
            let (wraps, inner) = peel(node);
            if node_hidden(&wraps, inner, target) {
                continue;
            }
            if let Node::PageInfo { page_no, .. } = inner {
                *page = *page_no;
                continue;
            }
            if let Node::Group { children, .. } = inner {
                let mut child_page = wrapped_page(&wraps, *page);
                collect(children, target, &mut child_page, out);
                continue;
            }
            let Node::Table(table) = inner else { continue };
            if table
                .rows
                .iter()
                .flatten()
                .any(|cell| !cell.trim().is_empty())
            {
                if let Some((exact, bbox)) =
                    location(&wraps).or_else(|| table.location.map(|b| (false, b.map(f64::from))))
                {
                    out.entry((wrapped_page(&wraps, *page), exact))
                        .or_default()
                        .push(bbox);
                }
            }
            if let Some(rows) = table
                .cell_blocks
                .as_ref()
                .filter(|_| target == OutputTarget::Doclang)
            {
                for cell in rows.iter().flatten() {
                    let mut child_page = wrapped_page(&wraps, *page);
                    collect(cell, target, &mut child_page, out);
                }
            }
        }
    }

    let mut out = HashMap::new();
    let mut page = 0usize;
    collect(nodes, target, &mut page, &mut out);
    out
}

/// A wrapper layer peeled off on the way down to a Picture, reapplied to
/// every node spliced in — insertions inherit the picture's layer/provenance.
#[derive(Clone)]
enum Wrap {
    Furniture(ContentLayer),
    Located([u16; 4]),
    Doclang,
    Commented(Vec<usize>),
    Prov {
        page_no: usize,
        bbox: [f32; 4],
        charspan: [usize; 2],
        seq: Option<usize>,
    },
}

impl Wrap {
    fn hidden(&self, target: OutputTarget) -> bool {
        match self {
            Self::Furniture(layer) => {
                target != OutputTarget::Doclang
                    && !(target == OutputTarget::Json && *layer == ContentLayer::Notes)
            }
            Self::Doclang => target != OutputTarget::Doclang,
            _ => false,
        }
    }
}

fn node_hidden(wraps: &[Wrap], node: &Node, target: OutputTarget) -> bool {
    wraps.iter().any(|w| w.hidden(target))
        || matches!(node, Node::Group { layer: Some(_), .. }
            if !matches!(target, OutputTarget::Json | OutputTarget::Doclang))
}

/// Peel wrapper nodes down to the innermost node, recording the chain.
fn peel(node: &Node) -> (Vec<Wrap>, &Node) {
    match node {
        Node::Furniture { layer, inner } => {
            let (mut wraps, n) = peel(inner);
            wraps.insert(0, Wrap::Furniture(*layer));
            (wraps, n)
        }
        Node::Located { location, inner } => {
            let (mut wraps, n) = peel(inner);
            wraps.insert(0, Wrap::Located(*location));
            (wraps, n)
        }
        Node::DoclangOnly(inner) => {
            let (mut wraps, n) = peel(inner);
            wraps.insert(0, Wrap::Doclang);
            (wraps, n)
        }
        Node::Commented { comments, inner } => {
            let (mut wraps, n) = peel(inner);
            wraps.insert(0, Wrap::Commented(comments.clone()));
            (wraps, n)
        }
        Node::Prov {
            page_no,
            bbox,
            charspan,
            seq,
            inner,
        } => {
            let (mut wraps, n) = peel(inner);
            wraps.insert(
                0,
                Wrap::Prov {
                    page_no: *page_no,
                    bbox: *bbox,
                    charspan: *charspan,
                    seq: *seq,
                },
            );
            (wraps, n)
        }
        other => (Vec::new(), other),
    }
}

/// An OCR line that itself starts with ``` would terminate the enclosing
/// Markdown fence early; a leading space neutralizes it without touching
/// normal content.
fn defuse_fences(body: &str) -> String {
    if !body.contains("```") {
        return body.to_string();
    }
    body.lines()
        .map(|l| {
            if l.trim_start().starts_with("```") {
                format!(" {l}")
            } else {
                l.to_string()
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Re-apply a peeled wrapper chain around a fresh node.
fn rewrap(wraps: &[Wrap], node: Node) -> Node {
    wraps.iter().rev().fold(node, |inner, wrap| match wrap {
        Wrap::Furniture(layer) => Node::Furniture {
            layer: *layer,
            inner: Box::new(inner),
        },
        Wrap::Located(location) => Node::Located {
            location: *location,
            inner: Box::new(inner),
        },
        Wrap::Doclang => Node::DoclangOnly(Box::new(inner)),
        Wrap::Commented(comments) => Node::Commented {
            comments: comments.clone(),
            inner: Box::new(inner),
        },
        Wrap::Prov {
            page_no,
            bbox,
            charspan,
            seq,
        } => Node::Prov {
            page_no: *page_no,
            bbox: *bbox,
            charspan: match peel(&inner).1 {
                Node::Paragraph { text } | Node::Code { text, .. } | Node::Caption { text, .. } => {
                    [0, text.chars().count()]
                }
                _ => *charspan,
            },
            seq: *seq,
            inner: Box::new(inner),
        },
    })
}

fn unwrapped_mut(node: &mut Node) -> &mut Node {
    match node {
        Node::Furniture { inner, .. }
        | Node::Located { inner, .. }
        | Node::Prov { inner, .. }
        | Node::Commented { inner, .. }
        | Node::DoclangOnly(inner) => unwrapped_mut(inner),
        _ => node,
    }
}

fn wrapped_page(wraps: &[Wrap], page: usize) -> usize {
    wraps
        .iter()
        .find_map(|w| match w {
            Wrap::Prov { page_no, .. } => Some(*page_no),
            _ => None,
        })
        .unwrap_or(page)
}

fn walk(
    nodes: &mut Vec<Node>,
    hidden: bool,
    page: &mut usize,
    table_locations: &TableLocations,
    runner: &mut OcrRunner,
    opts: &PostOptions,
    stats: &mut OcrStats,
) {
    let mut i = 0;
    while i < nodes.len() {
        if let Node::PageInfo { page_no, .. } = &nodes[i] {
            *page = *page_no;
            i += 1;
            continue;
        }
        let (wraps, inner) = peel(&nodes[i]);
        let effective_page = wrapped_page(&wraps, *page);
        let sub_hidden = hidden || node_hidden(&wraps, inner, opts.target);
        match unwrapped_mut(&mut nodes[i]) {
            Node::Group { children, .. } => {
                let mut child_page = effective_page;
                walk(
                    children,
                    sub_hidden,
                    &mut child_page,
                    table_locations,
                    runner,
                    opts,
                    stats,
                );
                i += 1;
                continue;
            }
            Node::Table(table) if opts.target == OutputTarget::Doclang => {
                if let Some(rows) = &mut table.cell_blocks {
                    for cell in rows.iter_mut().flatten() {
                        let mut child_page = effective_page;
                        walk(
                            cell,
                            sub_hidden,
                            &mut child_page,
                            table_locations,
                            runner,
                            opts,
                            stats,
                        );
                    }
                }
                i += 1;
                continue;
            }
            _ => {}
        }
        match replacement_for(
            &nodes[i],
            hidden,
            effective_page,
            table_locations,
            runner,
            opts,
            stats,
        ) {
            Some(mut replacement) => {
                if opts.keep_picture {
                    // Steal the original node so its (possibly large) image
                    // bytes aren't cloned; splice puts it back in front.
                    let original = std::mem::replace(&mut nodes[i], Node::PageBreak);
                    replacement.insert(0, original);
                }
                let advance = replacement.len();
                nodes.splice(i..=i, replacement);
                i += advance;
            }
            None => i += 1,
        }
    }
}

/// Decide what (if anything) replaces the node at this position. `None`
/// leaves the node untouched. When `opts.keep_picture` is set the returned
/// nodes are *appended after* the original instead of replacing it.
fn replacement_for(
    node: &Node,
    hidden: bool,
    page: usize,
    table_locations: &TableLocations,
    runner: &mut OcrRunner,
    opts: &PostOptions,
    stats: &mut OcrStats,
) -> Option<Vec<Node>> {
    let (wraps, inner) = peel(node);
    let Node::Picture {
        caption,
        caption_href,
        image,
        ..
    } = inner
    else {
        return None;
    };
    stats.pictures += 1;
    let img = image.as_ref()?;
    let hidden_here = hidden || wraps.iter().any(|w| w.hidden(opts.target));
    if hidden_here {
        return None;
    }
    if location(&wraps).is_some_and(|(exact, picture)| {
        table_locations
            .get(&(page, exact))
            .is_some_and(|tables| tables.iter().any(|table| contains_box(picture, *table)))
    }) {
        stats.skipped_structured += 1;
        return None;
    }
    let mut parts = ocr_parts(img, runner, opts, stats)?;
    if !opts.keep_picture {
        if let Some(caption) = caption.as_ref().filter(|c| !c.trim().is_empty()) {
            parts.insert(
                0,
                Node::Caption {
                    text: caption.clone(),
                    href: caption_href.clone(),
                },
            );
        }
    }
    Some(parts.into_iter().map(|n| rewrap(&wraps, n)).collect())
}

fn ocr_parts(
    img: &docling_core::PictureImage,
    runner: &mut OcrRunner,
    opts: &PostOptions,
    stats: &mut OcrStats,
) -> Option<Vec<Node>> {
    let area = img.width.saturating_mul(img.height);
    if opts.min_pixels > 0 && area > 0 && area < opts.min_pixels {
        stats.skipped_small += 1;
        return None;
    }
    let Some(outcome) = runner.run(&img.data, &img.mimetype) else {
        stats.failed += 1;
        return None;
    };
    if outcome.from_cache {
        stats.cached += 1;
    }
    let text = outcome.text.trim();
    if text.is_empty() {
        stats.empty += 1;
        return if opts.keep_picture {
            None
        } else {
            Some(Vec::new())
        };
    }
    stats.ocred += 1;
    let mut parts = Vec::new();
    match opts.mode {
        OutputMode::Fence => {
            let body = match outcome.grid.as_deref().map(str::trim_end) {
                Some(g) if !g.trim().is_empty() => g.to_string(),
                _ => text.to_string(),
            };
            parts.push(Node::Code {
                language: None,
                text: defuse_fences(&body),
                orig: None,
                pretty: None,
            });
        }
        OutputMode::Markers => {
            parts.push(Node::Paragraph {
                text: format!("<!-- ocr:begin engine={} -->", outcome.engine),
            });
            parts.push(Node::Paragraph {
                text: text.to_string(),
            });
            parts.push(Node::Paragraph {
                text: "<!-- ocr:end -->".to_string(),
            });
        }
        OutputMode::Text => parts.push(Node::Paragraph {
            text: text.to_string(),
        }),
        OutputMode::Quote => {
            let quoted: Vec<String> = text
                .lines()
                .map(|l| {
                    if l.trim().is_empty() {
                        ">".to_string()
                    } else {
                        format!("> {l}")
                    }
                })
                .collect();
            parts.push(Node::Paragraph {
                text: quoted.join("\n"),
            });
        }
        // Callers skip apply() entirely in Placeholder mode.
        OutputMode::Placeholder => return None,
    }
    Some(parts)
}

/// True when at least 80% of the structured table lies inside the picture.
#[cfg(test)]
fn contains_table(picture: [u16; 4], table: [u16; 4]) -> bool {
    contains_box(picture.map(f64::from), table.map(f64::from))
}

fn contains_box(picture: [f64; 4], table: [f64; 4]) -> bool {
    if !picture.iter().chain(table.iter()).all(|n| n.is_finite()) {
        return false;
    }
    let [px0, py0, px1, py1] = picture;
    let [tx0, ty0, tx1, ty1] = table;
    let area = (tx1 - tx0).abs() * (ty1 - ty0).abs();
    let width = (px0.max(px1).min(tx0.max(tx1)) - px0.min(px1).max(tx0.min(tx1))).max(0.0);
    let height = (py0.max(py1).min(ty0.max(ty1)) - py0.min(py1).max(ty0.min(ty1))).max(0.0);
    area > 0.0 && width * height >= 0.8 * area
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::OcrCache;
    use crate::engine::tests::MockEngine;
    use crate::engine::OcrFailure;
    use docling_core::PictureImage;

    fn picture(w: u32, h: u32) -> Node {
        Node::Picture {
            caption: None,
            image: Some(PictureImage {
                mimetype: "image/png".into(),
                width: w,
                height: h,
                data: vec![1, 2, 3, w as u8, h as u8],
            }),
            classification: None,
            caption_href: None,
            caption_parent: Default::default(),
        }
    }

    fn doc(nodes: Vec<Node>) -> DoclingDocument {
        DoclingDocument {
            name: "test".into(),
            nodes,
            strict_markdown: false,
            compact_tables: false,
            links: Vec::new(),
            confidence: None,
            page_break_placeholder: None,
            tree: None,
        }
    }

    fn runner(responses: Vec<Result<String, OcrFailure>>) -> OcrRunner {
        let (m, _) = MockEngine::new("mock", responses);
        OcrRunner::new(vec![Box::new(m)], OcrCache::disabled())
    }

    fn opts(mode: OutputMode) -> PostOptions {
        PostOptions {
            mode,
            min_pixels: 0,
            keep_picture: false,
            target: OutputTarget::Markdown,
        }
    }

    #[test]
    fn fence_mode_prefers_grid_and_falls_back_to_text() {
        // Grid-capable engine: the fence holds the spatial rendering.
        let (mut m, _) = MockEngine::new("mock", vec![Ok("Menu\nContent".into())]);
        m.grid = Some("Menu          Content".into());
        let mut r = OcrRunner::new(vec![Box::new(m)], OcrCache::disabled());
        let mut d = doc(vec![picture(100, 100)]);
        apply(&mut d, &mut r, &opts(OutputMode::Fence));
        let md = d.export_to_markdown();
        assert!(md.contains("```\nMenu          Content\n```"), "{md:?}");
        // Geometry-less engine (VLM): the fence holds the flat text.
        let mut d = doc(vec![picture(100, 100)]);
        let mut r = runner(vec![Ok("just\ntext".into())]);
        apply(&mut d, &mut r, &opts(OutputMode::Fence));
        let md = d.export_to_markdown();
        assert!(md.contains("```\njust\ntext\n```"), "{md:?}");
        assert!(!md.contains("<!-- image -->"), "{md:?}");
    }

    #[test]
    fn fence_mode_defuses_backtick_lines() {
        let mut d = doc(vec![picture(100, 100)]);
        let mut r = runner(vec![Ok("code:\n```sh\nls\n```".into())]);
        apply(&mut d, &mut r, &opts(OutputMode::Fence));
        let md = d.export_to_markdown();
        // The inner fences are neutralized, the outer fence stays balanced:
        // exactly two lines consist of a bare ``` (open + close).
        assert!(md.contains(" ```sh"), "{md:?}");
        let bare = md
            .lines()
            .filter(|l| l.trim() == "```" && !l.starts_with(' '))
            .count();
        assert_eq!(bare, 2, "outer fence only: {md:?}");
    }

    #[test]
    fn markers_mode_replaces_placeholder() {
        let mut d = doc(vec![
            Node::Paragraph {
                text: "before".into(),
            },
            picture(100, 100),
            Node::Paragraph {
                text: "after".into(),
            },
        ]);
        let mut r = runner(vec![Ok("line one\nline two".into())]);
        let stats = apply(&mut d, &mut r, &opts(OutputMode::Markers));
        assert_eq!((stats.pictures, stats.ocred), (1, 1));
        let md = d.export_to_markdown();
        assert!(
            !md.contains("<!-- image -->"),
            "placeholder replaced: {md:?}"
        );
        let expected = "before\n\n<!-- ocr:begin engine=mock -->\n\nline one  \nline two\n\n<!-- ocr:end -->\n\nafter";
        assert!(md.contains(expected), "markers block: {md:?}");
    }

    #[test]
    fn text_mode_inserts_bare_text() {
        let mut d = doc(vec![picture(100, 100)]);
        let mut r = runner(vec![Ok("just text".into())]);
        apply(&mut d, &mut r, &opts(OutputMode::Text));
        let md = d.export_to_markdown();
        assert!(md.contains("just text"), "{md:?}");
        assert!(!md.contains("ocr:begin"), "{md:?}");
        assert!(!md.contains("<!-- image -->"), "{md:?}");
    }

    #[test]
    fn quote_mode_boxes_each_line() {
        let mut d = doc(vec![picture(100, 100)]);
        let mut r = runner(vec![Ok("Invoice #1\n\nTotal: $5".into())]);
        apply(&mut d, &mut r, &opts(OutputMode::Quote));
        let md = d.export_to_markdown();
        assert!(md.contains("> Invoice #1  \n>  \n> Total: $5"), "{md:?}");
    }

    #[test]
    fn strict_mode_keeps_markers_intact() {
        let mut d = doc(vec![picture(100, 100)]);
        d.strict_markdown = true;
        let mut r = runner(vec![Ok("text".into())]);
        apply(&mut d, &mut r, &opts(OutputMode::Markers));
        let md = d.export_to_markdown();
        assert!(md.contains("<!-- ocr:begin engine=mock -->"), "{md:?}");
        assert!(md.contains("<!-- ocr:end -->"), "{md:?}");
    }

    #[test]
    fn empty_ocr_drops_picture_and_placeholder() {
        let mut d = doc(vec![
            Node::Paragraph {
                text: "before".into(),
            },
            picture(100, 100),
            Node::Paragraph {
                text: "after".into(),
            },
        ]);
        let mut r = runner(vec![Ok("   \n  ".into())]);
        let stats = apply(&mut d, &mut r, &opts(OutputMode::Markers));
        assert_eq!((stats.empty, stats.ocred), (1, 0));
        assert_eq!(d.nodes.len(), 2, "picture removed from the tree");
        let md = d.export_to_markdown();
        assert!(!md.contains("<!-- image -->"), "no placeholder: {md:?}");
        assert!(!md.contains("ocr:begin"), "no markers either: {md:?}");
    }

    #[test]
    fn empty_ocr_keeps_caption_and_keeps_rendered_image() {
        // Caption survives the drop…
        let mut d = doc(vec![Node::Picture {
            caption: Some("Figure 2. Logo".into()),
            image: Some(PictureImage {
                mimetype: "image/png".into(),
                width: 100,
                height: 100,
                data: vec![7],
            }),
            classification: None,
            caption_href: None,
            caption_parent: Default::default(),
        }]);
        let mut r = runner(vec![Ok(String::new())]);
        apply(&mut d, &mut r, &opts(OutputMode::Markers));
        let md = d.export_to_markdown();
        assert!(md.contains("Figure 2. Logo"), "{md:?}");
        assert!(!md.contains("<!-- image -->"), "{md:?}");
        // …and in embedded/referenced mode the picture node itself stays.
        let mut d = doc(vec![picture(100, 100)]);
        let mut r = runner(vec![Ok(String::new())]);
        let mut o = opts(OutputMode::Markers);
        o.keep_picture = true;
        let stats = apply(&mut d, &mut r, &o);
        assert_eq!(stats.empty, 1);
        assert!(matches!(&d.nodes[0], Node::Picture { image: Some(_), .. }));
    }

    #[test]
    fn failed_ocr_keeps_placeholder() {
        let mut d = doc(vec![picture(100, 100)]);
        let mut r = runner(vec![Err(OcrFailure::Engine("no model".into()))]);
        let stats = apply(&mut d, &mut r, &opts(OutputMode::Markers));
        assert_eq!((stats.failed, stats.ocred), (1, 0));
        assert!(d.export_to_markdown().contains("<!-- image -->"));
    }

    #[test]
    fn imageless_picture_untouched() {
        let mut d = doc(vec![Node::Picture {
            caption: None,
            image: None,
            classification: None,
            caption_href: None,
            caption_parent: Default::default(),
        }]);
        let mut r = runner(vec![]);
        let stats = apply(&mut d, &mut r, &opts(OutputMode::Markers));
        assert_eq!((stats.pictures, stats.ocred, stats.failed), (1, 0, 0));
        assert!(d.export_to_markdown().contains("<!-- image -->"));
    }

    #[test]
    fn small_picture_skipped() {
        let mut d = doc(vec![picture(16, 16), picture(200, 200)]);
        let mut r = runner(vec![Ok("big".into())]);
        let mut o = opts(OutputMode::Text);
        o.min_pixels = 2500;
        let stats = apply(&mut d, &mut r, &o);
        assert_eq!((stats.skipped_small, stats.ocred), (1, 1));
        let md = d.export_to_markdown();
        assert!(
            md.contains("<!-- image -->"),
            "small one keeps placeholder: {md:?}"
        );
        assert!(md.contains("big"), "{md:?}");
    }

    #[test]
    fn caption_survives_picture_removal() {
        let mut d = doc(vec![Node::Picture {
            caption: Some("Figure 1. Results".into()),
            image: Some(PictureImage {
                mimetype: "image/png".into(),
                width: 100,
                height: 100,
                data: vec![9],
            }),
            classification: None,
            caption_href: None,
            caption_parent: Default::default(),
        }]);
        let mut r = runner(vec![Ok("bars".into())]);
        apply(&mut d, &mut r, &opts(OutputMode::Text));
        let md = d.export_to_markdown();
        let cap = md.find("Figure 1. Results").expect("caption kept");
        let body = md.find("bars").expect("text kept");
        assert!(cap < body, "caption precedes text: {md:?}");
    }

    #[test]
    fn located_wrapper_recursed_and_preserved() {
        let mut d = doc(vec![Node::Located {
            location: [1, 2, 3, 4],
            inner: Box::new(picture(100, 100)),
        }]);
        let mut r = runner(vec![Ok("pdf text".into())]);
        let stats = apply(&mut d, &mut r, &opts(OutputMode::Text));
        assert_eq!(stats.ocred, 1);
        // The insertion carries the same provenance wrapper.
        assert!(matches!(
            &d.nodes[0],
            Node::Located { location: [1, 2, 3, 4], inner } if matches!(**inner, Node::Paragraph { .. })
        ));
        assert!(d.export_to_markdown().contains("pdf text"));
    }

    #[test]
    fn group_children_are_walked() {
        let mut d = doc(vec![Node::Group {
            label: "section".into(),
            name: None,
            layer: None,
            children: vec![picture(100, 100)],
        }]);
        let mut r = runner(vec![Ok("grouped".into())]);
        let stats = apply(&mut d, &mut r, &opts(OutputMode::Text));
        assert_eq!(stats.ocred, 1);
        assert!(d.export_to_markdown().contains("grouped"));
    }

    #[test]
    fn hidden_tables_do_not_suppress_visible_picture_ocr() {
        let table = Node::Located {
            location: [10, 10, 90, 90],
            inner: Box::new(Node::Table(docling_core::Table {
                rows: vec![vec!["hidden table".into()]],
                ..Default::default()
            })),
        };
        let mut d = doc(vec![
            Node::DoclangOnly(Box::new(table)),
            Node::Located {
                location: [0, 0, 100, 100],
                inner: Box::new(picture(100, 100)),
            },
        ]);
        let mut r = runner(vec![Ok("visible".into())]);
        let stats = apply(&mut d, &mut r, &opts(OutputMode::Text));
        assert_eq!(stats.skipped_structured, 0);
        assert_eq!(stats.ocred, 1);
        assert!(d.export_to_markdown().contains("visible"));
    }

    #[test]
    fn rich_cells_are_walked_for_doclang_and_wrappers_survive() {
        let cell = Node::Commented {
            comments: vec![7],
            inner: Box::new(Node::Prov {
                page_no: 2,
                bbox: [1., 2., 30., 40.],
                charspan: [0, 0],
                seq: Some(3),
                inner: Box::new(picture(100, 100)),
            }),
        };
        let mut d = doc(vec![Node::Table(docling_core::Table {
            rows: vec![vec![String::new()]],
            cell_blocks: Some(vec![vec![vec![cell]]]),
            ..Default::default()
        })]);
        let mut r = runner(vec![Ok("Čitljiv tekst".into())]);
        assert_eq!(apply(&mut d, &mut r, &opts(OutputMode::Text)).ocred, 0);
        let mut options = opts(OutputMode::Text);
        options.target = OutputTarget::Doclang;
        assert_eq!(apply(&mut d, &mut r, &options).ocred, 1);
        let Node::Table(table) = &d.nodes[0] else {
            panic!("table")
        };
        let Node::Commented { comments, inner } = &table.cell_blocks.as_ref().unwrap()[0][0][0]
        else {
            panic!("comments")
        };
        assert_eq!(comments, &[7]);
        assert!(matches!(
            inner.as_ref(),
            Node::Prov {
                page_no: 2,
                charspan: [0, 13],
                seq: Some(3),
                ..
            }
        ));
    }

    #[test]
    fn furniture_skipped_unless_hidden_enabled() {
        let furniture_pic = || Node::Furniture {
            layer: ContentLayer::Furniture,
            inner: Box::new(picture(100, 100)),
        };
        // Default (md/json target): no engine call at all.
        let (m, calls) = MockEngine::new("mock", vec![Ok("header".into())]);
        let mut r = OcrRunner::new(vec![Box::new(m)], OcrCache::disabled());
        let mut d = doc(vec![furniture_pic()]);
        let stats = apply(&mut d, &mut r, &opts(OutputMode::Text));
        assert_eq!((stats.pictures, stats.ocred), (1, 0));
        assert_eq!(
            calls.load(std::sync::atomic::Ordering::SeqCst),
            0,
            "no engine call for hidden layers"
        );
        // DocLang target: OCR'd, and the insertion stays furniture.
        let mut d = doc(vec![furniture_pic()]);
        let mut r = runner(vec![Ok("header".into())]);
        let mut o = opts(OutputMode::Text);
        o.target = OutputTarget::Doclang;
        let stats = apply(&mut d, &mut r, &o);
        assert_eq!(stats.ocred, 1);
        assert!(matches!(&d.nodes[0], Node::Furniture { .. }));
        // …and furniture stays out of Markdown.
        assert!(!d.export_to_markdown().contains("header"));
    }

    #[test]
    fn keep_picture_appends_after_the_image() {
        let mut d = doc(vec![picture(100, 100)]);
        let mut r = runner(vec![Ok("visible text".into())]);
        let mut o = opts(OutputMode::Text);
        o.keep_picture = true;
        apply(&mut d, &mut r, &o);
        assert!(matches!(&d.nodes[0], Node::Picture { image: Some(_), .. }));
        assert!(matches!(&d.nodes[1], Node::Paragraph { .. }));
        // Embedded image mode: both the data URI and the text render.
        let (md, _) =
            d.export_to_markdown_with_images(docling_core::ImageMode::Embedded, "artifacts");
        assert!(md.contains("![Image](data:image/png;base64,"), "{md:?}");
        assert!(md.contains("visible text"), "{md:?}");
    }

    #[test]
    fn multiple_pictures_all_processed() {
        let mut d = doc(vec![
            picture(100, 100),
            Node::Paragraph { text: "mid".into() },
            picture(90, 90),
        ]);
        let mut r = runner(vec![Ok("first".into()), Ok("second".into())]);
        let stats = apply(&mut d, &mut r, &opts(OutputMode::Markers));
        assert_eq!(stats.ocred, 2);
        let md = d.export_to_markdown();
        let (a, b, c) = (
            md.find("first").unwrap(),
            md.find("mid").unwrap(),
            md.find("second").unwrap(),
        );
        assert!(a < b && b < c, "order preserved: {md:?}");
    }

    #[test]
    fn upstream_rtf_picture_reaches_picture_ocr() {
        let png_hex = "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4890000000d49444154789c626001000000ffff03000006000557bfabd40000000049454e44ae426082";
        let rtf =
            format!(r"{{\rtf1\ansi{{\pict\pngblip\picw100\pich100 {png_hex}}}\par}}").into_bytes();
        let source =
            docling::SourceDocument::from_bytes("picture.rtf", docling::InputFormat::Rtf, rtf);
        let mut document = docling::DocumentConverter::new()
            .convert(source)
            .unwrap()
            .document;
        let mut runner = runner(vec![Ok("chosen paddle text".into())]);

        let stats = apply(&mut document, &mut runner, &opts(OutputMode::Text));

        assert_eq!(stats.ocred, 1);
        assert!(document.export_to_markdown().contains("chosen paddle text"));
    }

    #[test]
    fn structured_table_on_same_page_suppresses_picture_ocr() {
        let table = docling_core::Table {
            rows: vec![vec!["A".into(), "B".into()]],
            location: None,
            structure: None,
            cell_blocks: None,
            caption: None,
            ..Default::default()
        };
        let mut d = doc(vec![
            Node::PageInfo {
                page_no: 1,
                width: 100.0,
                height: 100.0,
            },
            Node::Located {
                location: [10, 10, 110, 110],
                inner: Box::new(Node::Table(table)),
            },
            Node::Located {
                // Exactly 80% of the table is inside this picture.
                location: [0, 0, 90, 110],
                inner: Box::new(picture(100, 100)),
            },
        ]);
        let (engine, calls) = MockEngine::new("mock", vec![Ok("duplicate".into())]);
        let mut runner = OcrRunner::new(vec![Box::new(engine)], OcrCache::disabled());
        let stats = apply(&mut d, &mut runner, &opts(OutputMode::Text));
        assert_eq!(stats.skipped_structured, 1);
        assert_eq!(stats.ocred, 0);
        assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 0);
        assert!(d.export_to_markdown().contains("<!-- image -->"));
    }

    #[test]
    fn table_overlap_threshold_is_eighty_percent() {
        assert!(contains_table([0, 0, 80, 100], [0, 0, 100, 100]));
        assert!(!contains_table([0, 0, 79, 100], [0, 0, 100, 100]));
        assert!(!contains_table([0, 0, 100, 100], [10, 10, 10, 50]));
    }
}
