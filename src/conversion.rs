//! Shared conversion settings for single-file, batch and HTTP entry points.

use docling::DocumentConverter;

#[derive(Debug, Clone, Default)]
pub struct ConversionOptions {
    pub strict: bool,
    pub fetch_images: bool,
    pub no_table_former: bool,
    pub no_ocr: bool,
    pub skip_ocr: bool,
    pub force_full_page_ocr: bool,
    pub no_text_panels: bool,
    pub heading_hierarchy: bool,
    pub use_web_browser: bool,
    pub enrich_picture_classes: bool,
    pub enrich_code: bool,
    pub enrich_formula: bool,
    pub asr_model: Option<String>,
    pub asr_lang: Option<String>,
    pub video_frames: Option<usize>,
    pub pages: Option<(usize, usize)>,
    pub ocr_lang: Option<String>,
    pub ocr_mode: Option<String>,
    pub ocr_scale: Option<f32>,
    pub encoding: Option<String>,
    pub page_break_placeholder: Option<String>,
}

impl ConversionOptions {
    pub fn converter(&self) -> DocumentConverter {
        let mut converter = DocumentConverter::new()
            .strict(self.strict)
            .fetch_images(self.fetch_images)
            .no_table_former(self.no_table_former)
            .no_ocr(self.no_ocr)
            .skip_ocr(self.skip_ocr)
            .force_full_page_ocr(self.force_full_page_ocr)
            .no_text_panels(self.no_text_panels)
            .heading_hierarchy(self.heading_hierarchy)
            .use_web_browser(self.use_web_browser)
            .do_picture_classification(self.enrich_picture_classes)
            .do_code_enrichment(self.enrich_code)
            .do_formula_enrichment(self.enrich_formula)
            .asr_model(self.asr_model.clone())
            .asr_lang(self.asr_lang.clone())
            .encoding(self.encoding.clone())
            .page_break_placeholder(self.page_break_placeholder.clone());
        if let Some(n) = self.video_frames {
            converter = converter.video_frames(n);
        }
        if let Some((first, last)) = self.pages {
            converter = converter.page_range(first, last);
        }
        if let Some(lang) = &self.ocr_lang {
            converter = converter.ocr_lang(lang.clone());
        }
        if let Some(mode) = &self.ocr_mode {
            converter = converter.ocr_mode(mode.clone());
        }
        if let Some(scale) = self.ocr_scale {
            converter = converter.ocr_scale(scale);
        }
        converter
    }

    /// Additional CLI controls, also used by the serve startup parser.
    pub fn parse_flag(
        &mut self,
        arg: &str,
        args: &mut impl Iterator<Item = String>,
    ) -> Result<(), String> {
        let value = |args: &mut dyn Iterator<Item = String>| {
            args.next().ok_or_else(|| format!("{arg} needs a value"))
        };
        match arg {
            "--skip-ocr" => self.skip_ocr = true,
            "--heading-hierarchy" => self.heading_hierarchy = true,
            "--ocr-mode" => self.ocr_mode = Some(parse_ocr_mode(&value(args)?)?),
            "--ocr-scale" => self.ocr_scale = Some(parse_ocr_scale(&value(args)?)?),
            "--encoding" => self.encoding = Some(value(args)?),
            "--page-break-placeholder" => self.page_break_placeholder = Some(value(args)?),
            _ => return Err(format!("unknown conversion option {arg}")),
        }
        Ok(())
    }

    pub fn finish_document(&self, document: &mut docling_core::DoclingDocument) {
        document.strict_markdown = self.strict;
        document.page_break_placeholder = self.page_break_placeholder.clone();
    }
}

pub fn parse_ocr_mode(value: &str) -> Result<String, String> {
    let mode = value.trim().to_ascii_lowercase();
    match mode.as_str() {
        "default" | "full_page" | "layout_regions" | "pdf_aware_layout_regions" => Ok(mode),
        _ => Err(format!("unknown OCR mode {value:?}; expected default|full_page|layout_regions|pdf_aware_layout_regions")),
    }
}

pub fn parse_ocr_scale(value: &str) -> Result<f32, String> {
    value
        .parse::<f32>()
        .ok()
        .filter(|s| s.is_finite() && *s > 0.0)
        .ok_or_else(|| format!("OCR scale {value:?} must be finite and positive"))
}

/// Same aliases as the pinned upstream OcrLang::parse, available even in
/// declarative-only builds where docling does not expose OcrLang.
pub fn normalize_ocr_lang(value: &str) -> Result<String, String> {
    let value = value.trim().to_ascii_lowercase();
    let tag = value.strip_prefix("iso:").unwrap_or(&value).trim();
    let primary = tag.split(['-', '_']).next().unwrap_or_default();
    match primary {
        "en" | "eng" | "english" => Ok("en".into()),
        "ch" | "zh" | "zho" | "chi" | "cmn" | "chinese" => Ok("ch".into()),
        _ => Err(format!(
            "unsupported page OCR language {value:?}; expected an English or Chinese language tag"
        )),
    }
}

#[cfg(feature = "pdf")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PipelineKey {
    no_table_former: bool,
    no_ocr: bool,
    skip_ocr: bool,
    no_text_panels: bool,
    picture_classes: bool,
    code: bool,
    formula: bool,
}

/// Holds a single warm PDF/image pipeline. Construction switches form its
/// key; mutable per-request settings are always applied, including defaults.
#[derive(Default)]
pub struct WarmPipeline {
    #[cfg(feature = "pdf")]
    slot: Option<(PipelineKey, docling::Pipeline)>,
}

impl WarmPipeline {
    pub fn convert(
        &mut self,
        source: docling::SourceDocument,
        opts: &ConversionOptions,
    ) -> Result<docling_core::DoclingDocument, String> {
        #[cfg(feature = "pdf")]
        if matches!(
            source.format,
            docling::InputFormat::Pdf | docling::InputFormat::Image
        ) {
            let key = PipelineKey {
                no_table_former: opts.no_table_former,
                no_ocr: opts.no_ocr,
                skip_ocr: opts.skip_ocr,
                no_text_panels: opts.no_text_panels,
                picture_classes: opts.enrich_picture_classes,
                code: opts.enrich_code,
                formula: opts.enrich_formula,
            };
            if self.slot.as_ref().is_none_or(|(old, _)| *old != key) {
                // Drop the old model pool before allocating its replacement.
                self.slot = None;
                let pipeline = docling::Pipeline::new()
                    .map_err(|e| e.to_string())?
                    .no_table_former(key.no_table_former)
                    .no_ocr(key.no_ocr)
                    .skip_ocr(key.skip_ocr)
                    .no_text_panels(key.no_text_panels)
                    .enrichments(docling::EnrichmentOptions {
                        picture_classification: key.picture_classes,
                        code: key.code,
                        formula: key.formula,
                    });
                self.slot = Some((key, pipeline));
            }
            let pipeline = &mut self.slot.as_mut().expect("initialized pipeline").1;
            pipeline.set_pages(opts.pages);
            pipeline.set_ocr_lang(opts.ocr_lang.as_deref().and_then(docling::OcrLang::parse));
            pipeline.set_force_full_page_ocr(opts.force_full_page_ocr);
            pipeline.set_ocr_mode(opts.ocr_mode.as_deref().and_then(docling::OcrMode::parse));
            pipeline.set_ocr_scale(opts.ocr_scale);
            pipeline.set_heading_hierarchy(docling::HeadingHierarchyOptions {
                enabled: opts.heading_hierarchy,
                ..Default::default()
            });
            let mut document = if source.format == docling::InputFormat::Pdf {
                pipeline.convert(&source.bytes, None, &source.name)
            } else {
                pipeline.convert_image(&source.bytes, &source.name)
            }
            .map_err(|e| e.to_string())?;
            opts.finish_document(&mut document);
            return Ok(document);
        }
        opts.converter()
            .convert(source)
            .map(|r| r.document)
            .map_err(|e| e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explicit_encoding_reaches_text_backend() {
        let opts = ConversionOptions {
            encoding: Some("windows-1251".into()),
            ..Default::default()
        };
        let doc = opts
            .converter()
            .convert(docling::SourceDocument::from_bytes(
                "text.txt",
                docling::InputFormat::Md,
                vec![0xcf, 0xf0, 0xe8, 0xe2, 0xe5, 0xf2],
            ))
            .unwrap()
            .document;
        assert!(doc.export_to_markdown().contains("Привет"));
    }

    #[test]
    fn language_aliases_and_numeric_controls_are_validated() {
        for (tag, expected) in [
            ("en-US", "en"),
            ("iso:eng", "en"),
            ("zh-Hant", "ch"),
            ("chinese_cht", "ch"),
            ("ch_sim", "ch"),
        ] {
            assert_eq!(normalize_ocr_lang(tag).unwrap(), expected);
            #[cfg(feature = "pdf")]
            assert_eq!(
                docling::OcrLang::parse(tag),
                docling::OcrLang::parse(expected)
            );
        }
        assert!(normalize_ocr_lang("sr").is_err());
        for scale in ["0", "-1", "NaN", "inf", "large"] {
            assert!(parse_ocr_scale(scale).is_err());
        }
        assert_eq!(parse_ocr_scale("3").unwrap(), 3.0);
        assert!(parse_ocr_mode("typo").is_err());
    }

    #[cfg(feature = "pdf")]
    #[test]
    fn warm_text_pipeline_resets_options_between_requests() {
        // This committed digital PDF needs neither models nor pdfium when
        // no_ocr is set. Both page windows and output settings must reset.
        let fixture = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../docling.rs/tests/data/pdf/sources/multi_page.pdf");
        let bytes = std::fs::read(fixture).unwrap();
        let mut warm = WarmPipeline::default();
        let source = || {
            docling::SourceDocument::from_bytes(
                "paper.pdf",
                docling::InputFormat::Pdf,
                bytes.clone(),
            )
        };
        let custom = ConversionOptions {
            no_ocr: true,
            pages: Some((1, 1)),
            strict: true,
            page_break_placeholder: Some("PAGE".into()),
            ..Default::default()
        };
        let first = warm.convert(source(), &custom).unwrap();
        assert!(first.strict_markdown);
        let defaults = ConversionOptions {
            no_ocr: true,
            ..Default::default()
        };
        let reset = warm.convert(source(), &defaults).unwrap();
        let fresh = defaults.converter().convert(source()).unwrap().document;
        assert_eq!(reset.export_to_markdown(), fresh.export_to_markdown());
        assert_eq!(reset.page_break_placeholder, None);
    }
}
