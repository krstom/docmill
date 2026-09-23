//! OCR directly on the authoritative JSON item tree. Arena IDs stay stable;
//! the upstream exporter regenerates bucket references after insertions.

use docling_core::tree::{ItemTree, TreeKind};
use docling_core::Node;

use super::{ocr_parts, OcrRunner, OcrStats, PostOptions};

pub(super) fn apply(tree: &mut ItemTree, runner: &mut OcrRunner, opts: &PostOptions) -> OcrStats {
    let mut stats = OcrStats::default();
    // Walk children, not creation order (office formats create and position
    // items in different orders). Only visit original items, once.
    let mut pending: Vec<usize> = tree.body.iter().rev().copied().collect();
    let mut seen = std::collections::HashSet::new();
    while let Some(id) = pending.pop() {
        if !seen.insert(id) || tree.items[id].deleted {
            continue;
        }
        pending.extend(tree.items[id].children.iter().rev().copied());
        let item = &tree.items[id];
        let TreeKind::Picture { image, .. } = &item.kind else {
            continue;
        };
        stats.pictures += 1;
        let Some(image) = image else { continue };
        // JSON retains all tree content layers, including furniture and notes.
        let suppressed = item.prov.as_ref().is_some_and(|picture| {
            tree.items.iter().any(|t| {
                if t.deleted {
                    return false;
                }
                let TreeKind::Table { table, .. } = &t.kind else {
                    return false;
                };
                table.rows.iter().flatten().any(|c| !c.trim().is_empty())
                    && t.prov.as_ref().is_some_and(|table| {
                        table.page_no == picture.page_no
                            && table.bottom_left == picture.bottom_left
                            && super::contains_box(picture.bbox, table.bbox)
                    })
            })
        });
        if suppressed {
            stats.skipped_structured += 1;
            continue;
        }
        let Some(parts) = ocr_parts(image, runner, opts, &mut stats) else {
            continue;
        };
        let parent = item.parent;
        let layer = item.layer;
        let prov = item.prov.clone();
        let comments = item.comments.clone();
        let source = item.source.clone();
        let mut inserted = Vec::new();
        if !opts.keep_picture {
            // Caption children (and any other children) must survive deletion.
            // Captions already parented elsewhere retain their existing place.
            for child in tree.items[id].children.clone() {
                tree.reparent(child, parent);
                inserted.push(child);
            }
        }
        for part in parts {
            let (kind, len) = match part {
                Node::Code { text, language, .. } => {
                    let len = text.chars().count();
                    (
                        TreeKind::Code {
                            text,
                            language,
                            orig: None,
                            formatting: None,
                            hyperlink: None,
                        },
                        len,
                    )
                }
                Node::Paragraph { text } => {
                    let len = text.chars().count();
                    (
                        TreeKind::Text {
                            label: "text".into(),
                            text,
                            orig: None,
                            formatting: None,
                            hyperlink: None,
                            level: None,
                            list: None,
                        },
                        len,
                    )
                }
                _ => unreachable!("OCR produces only code or text"),
            };
            let new_id = tree.add(parent, layer, kind);
            tree.items[new_id].prov = prov.clone().map(|mut p| {
                p.charspan = [0, len];
                p
            });
            tree.items[new_id].comments = comments.clone();
            tree.items[new_id].source = source.clone();
            inserted.push(new_id);
        }
        let siblings = match parent {
            Some(p) => &mut tree.items[p].children,
            None => &mut tree.body,
        };
        // add/reparent append; reposition at the original picture.
        siblings.retain(|child| !inserted.contains(child));
        let pos = siblings
            .iter()
            .position(|&child| child == id)
            .expect("tree parent contains picture");
        if opts.keep_picture {
            siblings.splice(pos + 1..pos + 1, inserted);
        } else {
            siblings.splice(pos..pos + 1, inserted);
            tree.items[id].deleted = true;
        }
    }
    stats
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::OcrCache;
    use crate::engine::{tests::MockEngine, OcrFailure};
    use crate::postprocess::{one_picture_document, OutputMode, OutputTarget};
    use docling_core::tree::TreeProv;
    use docling_core::{ContentLayer, PictureImage};

    fn text(value: &str) -> TreeKind {
        TreeKind::Text {
            label: "caption".into(),
            text: value.into(),
            orig: None,
            formatting: None,
            hyperlink: Some("https://example.com/caption".into()),
            level: None,
            list: None,
        }
    }

    fn fixture() -> (ItemTree, usize, usize, usize) {
        let mut tree = ItemTree::default();
        let heading = tree.add(
            None,
            None,
            TreeKind::Text {
                label: "section_header".into(),
                text: "Heading".into(),
                orig: None,
                formatting: None,
                hyperlink: None,
                level: Some(2),
                list: None,
            },
        );
        let pic = tree.add(
            Some(heading),
            Some(ContentLayer::Notes),
            TreeKind::Picture {
                image: Some(PictureImage {
                    mimetype: "image/png".into(),
                    width: 100,
                    height: 100,
                    data: vec![1],
                }),
                captions: vec![],
                classification: None,
                confidence: None,
                chart: None,
                dpi: Some(96),
            },
        );
        let caption = tree.add(Some(pic), Some(ContentLayer::Notes), text("Caption"));
        if let TreeKind::Picture { captions, .. } = &mut tree.items[pic].kind {
            captions.push(caption);
        }
        tree.items[pic].prov = Some(TreeProv {
            page_no: 1,
            bbox: [10.0, 20.0, 90.0, 80.0],
            bottom_left: false,
            charspan: [0, 0],
        });
        let comment = tree.add(None, Some(ContentLayer::Notes), text("Reviewer"));
        tree.items[pic].comments.push(comment);
        tree.add(Some(heading), None, text("After"));
        (tree, heading, pic, caption)
    }

    fn options(keep: bool) -> PostOptions {
        PostOptions {
            target: OutputTarget::Json,
            mode: OutputMode::Text,
            keep_picture: keep,
            min_pixels: 0,
        }
    }

    fn assert_references(value: &serde_json::Value, root: &serde_json::Value) {
        match value {
            serde_json::Value::Object(map) => {
                if let Some(reference) = map.get("$ref").and_then(|v| v.as_str()) {
                    assert!(
                        root.pointer(reference.strip_prefix('#').unwrap()).is_some(),
                        "dangling {reference}"
                    );
                }
                for value in map.values() {
                    assert_references(value, root);
                }
            }
            serde_json::Value::Array(values) => {
                for value in values {
                    assert_references(value, root);
                }
            }
            _ => {}
        }
    }

    #[test]
    fn office_item_trees_receive_ocr_and_keep_valid_references() {
        let root =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../docling.rs/tests/data");
        for fixture in [
            "docx/sources/docx_rich_cells.docx",
            "pptx/sources/powerpoint_with_image.pptx",
        ] {
            let source = crate::input::DetectedSource::from_path(root.join(fixture)).unwrap();
            let mut document = docling::DocumentConverter::new()
                .convert(source.into_docling())
                .unwrap()
                .document;
            assert!(document.tree.is_some(), "{fixture}");
            let (engine, _) = MockEngine::new("mock", vec![Ok("recognized in JSON".into())]);
            let mut runner = OcrRunner::new(vec![Box::new(engine)], OcrCache::disabled());
            let stats = crate::postprocess::apply(&mut document, &mut runner, &options(false));
            assert!(stats.ocred > 0, "{fixture}: {stats}");
            let json: serde_json::Value = serde_json::from_str(&document.export_to_json()).unwrap();
            assert!(
                json["texts"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .any(|text| text["text"] == "recognized in JSON"),
                "{fixture}"
            );
            assert_references(&json, &json);
        }
    }

    #[test]
    fn json_ocr_preserves_hierarchy_caption_links_comments_and_provenance() {
        let (tree, heading, picture, caption) = fixture();
        let mut doc = one_picture_document("fixture.png", vec![2], false);
        let original_nodes = doc.nodes.clone();
        doc.tree = Some(tree);
        let (engine, calls) = MockEngine::new("mock", vec![Ok("Čitljiv tekst".into())]);
        let mut runner = OcrRunner::new(vec![Box::new(engine)], OcrCache::disabled());
        let stats = crate::postprocess::apply(&mut doc, &mut runner, &options(false));
        assert_eq!(stats.ocred, 1);
        assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 1);
        assert_eq!(
            doc.nodes, original_nodes,
            "JSON processing must not traverse the duplicate flat picture"
        );
        let tree = doc.tree.as_ref().unwrap();
        assert!(tree.items[picture].deleted);
        assert_eq!(tree.items[caption].parent, Some(heading));
        let children = &tree.items[heading].children;
        assert_eq!(children[0], caption);
        let inserted = &tree.items[children[1]];
        assert_eq!(inserted.layer, Some(ContentLayer::Notes));
        assert_eq!(inserted.prov.as_ref().unwrap().charspan, [0, 13]);
        assert_eq!(inserted.comments.len(), 1);
        let json: serde_json::Value = serde_json::from_str(&doc.export_to_json()).unwrap();
        assert_references(&json, &json);
        assert!(json["texts"]
            .as_array()
            .unwrap()
            .iter()
            .any(|t| t["text"] == "Caption" && t["hyperlink"] == "https://example.com/caption"));
        assert!(json["texts"]
            .as_array()
            .unwrap()
            .iter()
            .any(|t| t["text"] == "Čitljiv tekst"));
    }

    #[test]
    fn retained_images_empty_results_and_failures_keep_the_right_items() {
        for (keep, result, removed, ocred) in [
            (true, Ok("OCR".into()), false, 1),
            (false, Ok(String::new()), true, 0),
            (true, Ok(String::new()), false, 0),
            (false, Err(OcrFailure::Image("bad image".into())), false, 0),
        ] {
            let (mut tree, heading, picture, caption) = fixture();
            let (engine, _) = MockEngine::new("mock", vec![result]);
            let mut runner = OcrRunner::new(vec![Box::new(engine)], OcrCache::disabled());
            let stats = apply(&mut tree, &mut runner, &options(keep));
            assert_eq!(stats.ocred, ocred);
            assert_eq!(tree.items[picture].deleted, removed);
            assert!(!tree.items[caption].deleted);
            if keep && ocred == 1 {
                assert_eq!(tree.items[heading].children[0], picture);
            }
        }
    }

    #[test]
    fn real_html_tree_receives_ocr_instead_of_only_its_flat_nodes() {
        let html = br#"<h1>Section</h1><p>Before</p><img alt="Caption" src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQImWNgYGBgAAAABQABpfZFQAAAAABJRU5ErkJggg=="><p>After</p>"#;
        let mut doc = docling::DocumentConverter::new()
            .fetch_images(true)
            .convert(docling::SourceDocument::from_bytes(
                "test.html",
                docling::InputFormat::Html,
                html.to_vec(),
            ))
            .unwrap()
            .document;
        assert!(doc.tree.is_some());
        let (engine, _) = MockEngine::new("mock", vec![Ok("Picture words".into())]);
        let mut runner = OcrRunner::new(vec![Box::new(engine)], OcrCache::disabled());
        assert_eq!(
            crate::postprocess::apply(&mut doc, &mut runner, &options(false)).ocred,
            1
        );
        let json: serde_json::Value = serde_json::from_str(&doc.export_to_json()).unwrap();
        assert_references(&json, &json);
        assert!(json["texts"]
            .as_array()
            .unwrap()
            .iter()
            .any(|t| t["text"] == "Picture words"));
    }
}
