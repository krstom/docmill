use std::fs;
use std::path::Path;
use std::process::{Command, Output};

fn docmill(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_docmill"))
        .args(args)
        .output()
        .expect("run docmill")
}

fn write(path: &Path, body: &str) {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).unwrap();
    }
    fs::write(path, body).unwrap();
}

#[test]
fn version_matches_docling_release() {
    let out = docmill(&["--version"]);
    assert!(out.status.success());
    assert_eq!(String::from_utf8_lossy(&out.stdout), "docmill 1.4.2\n");
}

#[test]
fn positional_output_remains_an_exact_file() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("one.md");
    let output = dir.path().join("chosen.output");
    write(&input, "# One\n");
    let out = docmill(&[
        "--img-ocr-mode",
        "placeholder",
        "--output",
        output.to_str().unwrap(),
        input.to_str().unwrap(),
    ]);
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    assert!(output.is_file());
}

#[test]
fn batch_recurses_and_preserves_relative_paths() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("input");
    let output = dir.path().join("output");
    write(&input.join("root.md"), "root\n");
    write(&input.join("nested/child.md"), "child\n");
    write(&input.join("ignored.log"), "not a document\n");

    let out = docmill(&[
        "--input",
        input.to_str().unwrap(),
        "--output",
        output.to_str().unwrap(),
        "--jobs",
        "2",
        "--img-ocr-mode",
        "placeholder",
    ]);
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    assert!(output.join("root.md").is_file());
    assert!(output.join("nested/child.md").is_file());
    assert!(!output.join("ignored.md").exists());
}

#[test]
fn batch_continues_after_a_file_failure() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("input");
    let output = dir.path().join("output");
    write(&input.join("good.md"), "good\n");
    write(&input.join("bad.bin"), "not recognizable\n");
    let pattern = format!("{}/*", input.display());

    let out = docmill(&[
        "--input",
        &pattern,
        "--output",
        output.to_str().unwrap(),
        "--jobs",
        "2",
        "--img-ocr-mode",
        "placeholder",
    ]);
    assert_eq!(out.status.code(), Some(1));
    assert!(output.join("good.md").is_file());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("bad.bin"), "{stderr}");
    assert!(stderr.contains("batch: 1 converted, 1 failed"), "{stderr}");
}

#[test]
fn batch_only_flags_are_validated() {
    let jobs = docmill(&["--jobs", "2", "fake.md"]);
    assert_eq!(jobs.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&jobs.stderr).contains("--jobs requires --input"));

    let zero = docmill(&["--jobs", "0", "fake.md"]);
    assert_eq!(zero.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&zero.stderr).contains("positive integer"));

    let missing_output = docmill(&["--input", "*.md"]);
    assert_eq!(missing_output.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&missing_output.stderr).contains("needs --output DIR"));
}
