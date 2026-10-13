// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Exact token-ID parity of `Tokenizer::encode` against the HuggingFace
//! `tokenizers` reference (0.22.2) for the Qwen3.5/3.6/3.8 byte-level BPE
//! tokenizer (tokenizer.json sha256 `0997f410…29b9f3`, md5
//! `085e10165f24499bb9d8fcee7e17f9ee`, the one embedded in the Qwen3.8 27B
//! hfq/mq4 header).
//!
//! The 12.8 MB tokenizer.json is not committed. The test is gated on
//! `HIPFIRE_HF_TOKENIZER_JSON=<path to that tokenizer.json>` and skips
//! (passes) without it. CPU only; no model weights, no GPU.
//!
//! The committed cases (`tests/fixtures/hf_tokenizer_parity/qwen38_synth.jsonl`)
//! cover indented Python/Rust, JSON, tool-call XML, digit runs, CJK and other
//! scripts, emoji (ZWJ, flags, VS16), combining marks and NFC cases, CRLF,
//! trailing and Unicode whitespace, contractions and chat framing with
//! special tokens. Each line is `{"id","group","text","ids"}` with `ids` from
//! `Tokenizer.from_file(tokenizer.json).encode(text, add_special_tokens=False)`.
//!
//! Extra corpora (same line format) can be checked with
//! `HIPFIRE_HF_TOKENIZER_CASES=a.jsonl:b.jsonl`; `HIPFIRE_HF_TOKENIZER_REPORT_ONLY=1`
//! prints the match rate without failing.
//!
//! ```text
//! HIPFIRE_HF_TOKENIZER_JSON=/path/tokenizer.json \
//!   cargo test --release -p hipfire-runtime --test hf_tokenizer_parity -- --nocapture
//! ```

use std::collections::BTreeMap;
use std::path::PathBuf;

use hipfire_runtime::tokenizer::Tokenizer;
use serde_json::Value;
use sha2::{Digest, Sha256};

const QWEN_TOKENIZER_SHA256: &str =
    "0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3";

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

#[derive(Default)]
struct Tally {
    cases: usize,
    exact: usize,
    count_differs: usize,
    ref_tokens: u64,
    our_tokens: u64,
}

#[test]
fn encode_matches_hf_reference_ids() {
    let Ok(tok_path) = std::env::var("HIPFIRE_HF_TOKENIZER_JSON") else {
        eprintln!("hf_tokenizer_parity: HIPFIRE_HF_TOKENIZER_JSON unset, skipping");
        return;
    };
    let tok_bytes = std::fs::read(&tok_path).expect("read HIPFIRE_HF_TOKENIZER_JSON");
    let tok_sha = sha256_hex(&tok_bytes);
    let tokenizer = Tokenizer::from_hf_json(std::str::from_utf8(&tok_bytes).expect("utf-8"))
        .expect("Tokenizer::from_hf_json");

    let default_cases = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/hf_tokenizer_parity/qwen38_synth.jsonl");
    let case_files: Vec<PathBuf> = match std::env::var("HIPFIRE_HF_TOKENIZER_CASES") {
        Ok(list) => list.split(':').filter(|s| !s.is_empty()).map(PathBuf::from).collect(),
        Err(_) => {
            // The committed IDs are only valid for the Qwen tokenizer.
            assert_eq!(
                tok_sha, QWEN_TOKENIZER_SHA256,
                "committed cases are pinned to the Qwen3.8 tokenizer.json"
            );
            vec![default_cases]
        }
    };
    let report_only = std::env::var("HIPFIRE_HF_TOKENIZER_REPORT_ONLY").is_ok_and(|v| v == "1");

    let mut groups: BTreeMap<String, Tally> = BTreeMap::new();
    let mut failures = Vec::new();
    for file in &case_files {
        let raw = std::fs::read_to_string(file)
            .unwrap_or_else(|e| panic!("read {}: {e}", file.display()));
        for (n, line) in raw.lines().enumerate() {
            if line.trim().is_empty() {
                continue;
            }
            let case: Value = serde_json::from_str(line)
                .unwrap_or_else(|e| panic!("{}:{}: {e}", file.display(), n + 1));
            let id = case["id"].as_str().expect("id");
            let group = case["group"].as_str().unwrap_or("default");
            let text = case["text"].as_str().expect("text");
            let want: Vec<u32> = case["ids"]
                .as_array()
                .expect("ids")
                .iter()
                .map(|x| x.as_u64().expect("int id") as u32)
                .collect();
            let got = tokenizer.encode(text);
            let t = groups.entry(group.to_string()).or_default();
            t.cases += 1;
            t.ref_tokens += want.len() as u64;
            t.our_tokens += got.len() as u64;
            if got == want {
                t.exact += 1;
            } else {
                if got.len() != want.len() {
                    t.count_differs += 1;
                }
                if failures.len() < 20 {
                    let at = got
                        .iter()
                        .zip(&want)
                        .position(|(a, b)| a != b)
                        .unwrap_or(got.len().min(want.len()));
                    let lo = at.saturating_sub(2);
                    let (gh, wh) = ((at + 3).min(got.len()), (at + 3).min(want.len()));
                    failures.push(format!(
                        "{id}: ours {} vs hf {} tokens, first diff at {at}: ours {:?} {:?} / hf {:?} {:?}",
                        got.len(),
                        want.len(),
                        &got[lo.min(gh)..gh],
                        tokenizer.decode(&got[lo.min(gh)..gh]),
                        &want[lo.min(wh)..wh],
                        tokenizer.decode(&want[lo.min(wh)..wh]),
                    ));
                }
            }
        }
    }

    let (mut cases, mut exact) = (0, 0);
    for (name, t) in &groups {
        cases += t.cases;
        exact += t.exact;
        eprintln!(
            "hf_tokenizer_parity {name}: {}/{} exact ({:.1}%), {} count-differ, tokens ours {} / hf {}",
            t.exact,
            t.cases,
            100.0 * t.exact as f64 / t.cases.max(1) as f64,
            t.count_differs,
            t.our_tokens,
            t.ref_tokens
        );
    }
    for f in &failures {
        eprintln!("  {f}");
    }
    assert!(cases > 0, "no cases loaded");
    if !report_only {
        assert_eq!(exact, cases, "{} of {cases} cases differ from HF", cases - exact);
    }
}
