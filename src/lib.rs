//! docmill — convert documents with docling.rs, then OCR the pictures
//! embedded in them (DOCX drawings, PDF figure regions, standalone images)
//! with a pluggable engine chain, splicing the recognized text into the
//! document where upstream docling emits only an `<!-- image -->` placeholder.
//!
//! The library half exists so the pieces with interesting logic — the disk
//! cache, the document post-processor, and the remote-response parsers — are
//! unit-testable without models or network. The binary (`src/main.rs`) wires
//! them behind a docling-cli-style flag surface.
//!
//! Design constraints inherited from the sibling docling.rs checkout:
//! - the upstream repo is used strictly through its public API (path deps),
//!   never modified — conversion output with OCR disabled stays byte-identical
//!   to `docling-rs`;
//! - conversion is synchronous, so HTTP goes over blocking `ureq` (the same
//!   stack docling's own VLM pipeline uses);
//! - a missing optional dependency (model file, unreachable endpoint) warns
//!   and degrades — the picture keeps its placeholder — rather than failing
//!   the conversion.

pub mod cache;
pub mod config;
pub mod engine;
pub mod layout;
pub mod postprocess;
#[cfg(feature = "serve")]
pub mod serve;
