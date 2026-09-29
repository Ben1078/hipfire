//! claude r2: FastMTP-style compressed draft vocabulary for the Qwen3.5/3.8 MTP
//! head, built WITHOUT the original HF safetensors.
//!
//! `mtp_extract --vocab-sidecar` needs the HF `lm_head.weight` (bf16) and
//! re-quantizes the selected rows to MQ4G256. This tool instead copies the
//! selected rows BYTE-FOR-BYTE out of the trunk's already-quantized lm_head
//! (MQ4G256V2 / qt=44 in the production `.mq4-xts`, 2720 B per 5120-wide row),
//! so the draft logits of every in-slice token are exactly the logits the
//! full-vocab draft computes today (same bytes, same `gemv_mq4g256v2`
//! kernel, per-row independent): the compressed argmax equals the full-vocab
//! argmax whenever the latter lies in the slice. The rows are stored in
//! ascending token-id order, so exact-tie breaks (lowest index wins) also
//! match the full-vocab kernels.
//!
//! Subcommands
//!   rank   --tokenizer <trunk.hfq> --corpus <corpus.jsonl> --top-k 32768,49152
//!          --out-prefix <path> [--weights outputs=2,prompts=1,code=1,docs=1]
//!          [--holdout-cat outputs_holdout] [--prior 0.05]
//!     Tokenizes every `{"cat","text"}` JSONL document with the runtime's own
//!     BPE tokenizer, scores each id by the weighted per-category frequency
//!     (plus a small BPE-merge-rank prior so unseen ids are ordered by the
//!     tokenizer's own training frequency), force-includes the special/added
//!     tokens and the 256 byte tokens, and writes `<prefix>-<K>.json` in the
//!     `build_mtp_vocab_sidecar.py` schema (`draft_to_full`, sorted ascending).
//!     Held-out coverage (categories not in --weights) is reported per K.
//!   build  --mtp <base.mtp> --trunk <trunk.hfq> --sidecar <S.json> --output <out.mtp>
//!     Copies every tensor of the base `.mtp` unchanged, appends
//!     `lm_head_draft.weight` (qt of the trunk lm_head, rows byte-sliced) and
//!     `lm_head_draft.vocab_map` (u32 LE, qt=2 like mtp_extract), sets
//!     `has_compressed_lm_head_draft` / `compressed_vocab_size`, then re-opens
//!     the output and checks every slice row against the trunk bytes.
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::tokenizer::Tokenizer;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};

fn arg(args: &[String], k: &str) -> Option<String> {
    args.iter().position(|a| a == k).map(|i| args[i + 1].clone())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(|s| s.as_str()) {
        Some("rank") => rank(&args),
        Some("build") => build(&args),
        _ => {
            eprintln!("usage: claude_mtp_draft_vocab rank|build ... (see the module docs)");
            std::process::exit(2);
        }
    }
}

// ─── rank ────────────────────────────────────────────────────────────────

fn rank(args: &[String]) {
    let tok_path = PathBuf::from(arg(args, "--tokenizer").expect("--tokenizer <trunk.hfq>"));
    let corpus = PathBuf::from(arg(args, "--corpus").expect("--corpus <corpus.jsonl>"));
    let prefix = arg(args, "--out-prefix").expect("--out-prefix");
    let ks: Vec<usize> = arg(args, "--top-k")
        .unwrap_or_else(|| "32768".into())
        .split(',')
        .map(|s| s.trim().parse().expect("--top-k"))
        .collect();
    let weights: BTreeMap<String, f64> = arg(args, "--weights")
        .unwrap_or_else(|| "outputs=2,prompts=1,code=1,docs=1".into())
        .split(',')
        .map(|kv| {
            let (k, v) = kv.split_once('=').expect("--weights cat=w,...");
            (k.to_string(), v.parse::<f64>().expect("weight"))
        })
        .collect();
    let prior_w: f64 = arg(args, "--prior").map(|v| v.parse().unwrap()).unwrap_or(0.05);

    let hfq = HfqFile::open(&tok_path).expect("open tokenizer hfq");
    let tok = Tokenizer::from_hfq_metadata(&hfq.metadata_json).expect("tokenizer from metadata");
    let lm_rows = hfq
        .tensor_infos()
        .iter()
        .find(|t| t.name == "lm_head.weight")
        .map(|t| t.shape[0] as usize)
        .expect("trunk has lm_head.weight");
    let tv = tok.vocab_size();
    eprintln!("tokenizer vocab {tv}, lm_head rows {lm_rows}");

    // Per-category token counts.
    let mut counts: BTreeMap<String, Vec<u64>> = BTreeMap::new();
    let mut totals: BTreeMap<String, u64> = BTreeMap::new();
    let f = std::fs::File::open(&corpus).expect("open corpus");
    let t0 = std::time::Instant::now();
    let mut ndocs = 0usize;
    for line in std::io::BufReader::new(f).lines() {
        let line = line.expect("read corpus line");
        if line.trim().is_empty() {
            continue;
        }
        let v: serde_json::Value = serde_json::from_str(&line).expect("corpus JSONL");
        let cat = v["cat"].as_str().expect("cat").to_string();
        let text = v["text"].as_str().expect("text");
        let ids = tok.encode(text);
        let c = counts.entry(cat.clone()).or_insert_with(|| vec![0u64; lm_rows]);
        for &id in &ids {
            if (id as usize) < lm_rows {
                c[id as usize] += 1;
            }
        }
        *totals.entry(cat).or_insert(0) += ids.len() as u64;
        ndocs += 1;
    }
    eprintln!(
        "tokenized {ndocs} docs in {:.1}s: {:?}",
        t0.elapsed().as_secs_f64(),
        totals
    );

    // BPE merge rank: lower = merged earlier = more frequent in the tokenizer's
    // own training data. Byte tokens rank 0; specials/unknown rank last.
    let table = tok.build_merge_rank_table();
    let n_merges = table.len().max(1);
    let rank_of = |id: usize| -> Option<usize> {
        if let Some(&r) = table.get(&(id as u32)) {
            return Some(r + 1);
        }
        let s = tok.vocab().get(id)?;
        if s.chars().count() <= 1 {
            Some(0)
        } else {
            None
        }
    };

    // Forced: special / added tokens (chatml frame, <think>, tool markers) and
    // the byte-level base symbols (any byte sequence stays spellable).
    let mut forced: HashSet<usize> = HashSet::new();
    for (_, id) in tok.special_tokens() {
        if (*id as usize) < lm_rows {
            forced.insert(*id as usize);
        }
    }
    for id in 0..tv.min(lm_rows) {
        if rank_of(id) == Some(0) {
            forced.insert(id);
        }
    }
    eprintln!("forced ids: {} (specials + byte tokens)", forced.len());

    // Score = sum_c w_c * n_c(t) / N_c + prior_w * zipf(rank(t)).
    let wsum: f64 = weights.values().sum::<f64>().max(1e-12);
    let zipf_h: f64 = (1..=n_merges + 1).map(|r| 1.0 / (r as f64 + 10.0)).sum();
    let mut score = vec![0f64; lm_rows];
    for (cat, w) in &weights {
        if let (Some(c), Some(&n)) = (counts.get(cat), totals.get(cat)) {
            if n == 0 {
                continue;
            }
            for id in 0..lm_rows {
                score[id] += (w / wsum) * c[id] as f64 / n as f64;
            }
        } else {
            eprintln!("warning: weighted category {cat} has no documents");
        }
    }
    // Script class of the decoded token: the merge-rank prior is a proxy for
    // the tokenizer's (multilingual) training frequency; scale it down for
    // non-Latin scripts so it mostly promotes common English words / code
    // fragments the corpus happened to miss (--prior-other sets the factor).
    let prior_other: f64 = arg(args, "--prior-other").map(|v| v.parse().unwrap()).unwrap_or(0.1);
    let typo = ['\u{2014}', '\u{2013}', '\u{2019}', '\u{2018}', '\u{201c}', '\u{201d}', '\u{2026}', '\u{2022}', '\u{2192}'];
    let script_factor = |id: usize| -> f64 {
        let s = tok.decode(&[id as u32]);
        if s.chars().all(|c| c.is_ascii()) {
            1.0
        } else if s.chars().all(|c| (c as u32) < 0x250 || c.is_whitespace() || typo.contains(&c)) {
            0.5 // Latin-1 / Latin Extended-A/B + common typographic punctuation
        } else {
            prior_other
        }
    };
    for id in 0..lm_rows {
        if id >= tv {
            score[id] = -1.0; // padded lm_head rows: never real tokens
            continue;
        }
        if let Some(r) = rank_of(id) {
            score[id] += prior_w * script_factor(id) * (1.0 / (r as f64 + 10.0)) / zipf_h;
        }
    }
    let mut order: Vec<usize> = (0..lm_rows).collect();
    order.sort_by(|&a, &b| {
        let fa = forced.contains(&a);
        let fb = forced.contains(&b);
        fb.cmp(&fa)
            .then(score[b].partial_cmp(&score[a]).unwrap())
            .then(rank_of(a).unwrap_or(usize::MAX).cmp(&rank_of(b).unwrap_or(usize::MAX)))
            .then(a.cmp(&b))
    });

    // Held-out evaluation categories = those present but not weighted.
    let eval_cats: Vec<String> = counts
        .keys()
        .filter(|c| !weights.contains_key(*c))
        .cloned()
        .collect();
    let all_ks: Vec<usize> = {
        let mut v: Vec<usize> = vec![8192, 16384, 24576, 32768, 49152, 65536];
        v.extend(ks.iter().copied());
        v.sort();
        v.dedup();
        v
    };
    let mut report = serde_json::Map::new();
    for &k in &all_ks {
        let sel: HashSet<usize> = order[..k.min(lm_rows)].iter().copied().collect();
        let mut row = serde_json::Map::new();
        for cat in counts.keys() {
            let c = &counts[cat];
            let n = totals[cat].max(1);
            let covered: u64 = sel.iter().map(|&id| c[id]).sum();
            let miss_ids = (0..lm_rows).filter(|&i| c[i] > 0 && !sel.contains(&i)).count();
            row.insert(
                cat.clone(),
                serde_json::json!({"coverage": covered as f64 / n as f64, "missing_distinct_ids": miss_ids}),
            );
        }
        eprintln!(
            "K={k:6}: {}",
            counts
                .keys()
                .map(|cat| format!(
                    "{cat} {:.3}%",
                    100.0 * row[cat]["coverage"].as_f64().unwrap()
                ))
                .collect::<Vec<_>>()
                .join("  ")
        );
        report.insert(k.to_string(), serde_json::Value::Object(row));
    }
    let distinct: BTreeMap<String, usize> = counts
        .iter()
        .map(|(c, v)| (c.clone(), v.iter().filter(|&&x| x > 0).count()))
        .collect();
    eprintln!("distinct ids per category: {distinct:?}; eval (held-out) categories: {eval_cats:?}");

    if let Some(cat) = arg(args, "--dump-missing") {
        // Diagnostics: the most frequent ids of `cat` that fall outside the first K.
        let k = ks[0];
        let sel: HashSet<usize> = order[..k].iter().copied().collect();
        if let Some(c) = counts.get(&cat) {
            let mut miss: Vec<(u64, usize)> =
                (0..lm_rows).filter(|&i| c[i] > 0 && !sel.contains(&i)).map(|i| (c[i], i)).collect();
            miss.sort_by(|a, b| b.cmp(a));
            for (n, id) in miss.iter().take(60) {
                let s = tok.decode(&[*id as u32]);
                eprintln!("  missing@{k} {cat}: id {id:6} x{n:4} rank {:?} {:?}", rank_of(*id), s);
            }
        }
    }
    for &k in &ks {
        let mut ids: Vec<u32> = order[..k].iter().map(|&i| i as u32).collect();
        ids.sort_unstable();
        let out = serde_json::json!({
            "draft_to_full": ids,
            "compressed_vocab_size": k,
            "full_vocab_size": lm_rows,
            "stats": {
                "generator": "claude_mtp_draft_vocab rank (hipfire tokenizer from trunk metadata)",
                "tokenizer_source": tok_path.display().to_string(),
                "corpus": corpus.display().to_string(),
                "weights": weights,
                "bpe_rank_prior_weight": prior_w,
                "tokens_per_category": totals,
                "distinct_ids_per_category": distinct,
                "forced_ids": forced.len(),
                "coverage_by_k": report,
                "order": "draft_to_full ascending by token id (tie-break parity with the full-vocab kernels)",
            }
        });
        let path = format!("{prefix}-{k}.json");
        std::fs::write(&path, serde_json::to_string(&out).unwrap()).expect("write sidecar");
        eprintln!("wrote {path}");
    }
}

// ─── build ───────────────────────────────────────────────────────────────

struct OutTensor {
    name: String,
    quant_type: u8,
    shape: Vec<u32>,
    group_size: u32,
    data: Vec<u8>,
}

fn write_hfq(path: &Path, arch: u32, metadata_json: &str, tensors: &[OutTensor]) -> std::io::Result<()> {
    // Same container layout as mtp_extract::write_hfq (HFQM v1, 4 KiB data alignment).
    let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
    let meta = metadata_json.as_bytes();
    let metadata_offset = 32u64;
    let mut index = Vec::new();
    index.extend_from_slice(&(tensors.len() as u32).to_le_bytes());
    for t in tensors {
        index.extend_from_slice(&(t.name.len() as u16).to_le_bytes());
        index.extend_from_slice(t.name.as_bytes());
        index.push(t.quant_type);
        index.push(t.shape.len() as u8);
        for &d in &t.shape {
            index.extend_from_slice(&d.to_le_bytes());
        }
        index.extend_from_slice(&t.group_size.to_le_bytes());
        index.extend_from_slice(&(t.data.len() as u64).to_le_bytes());
    }
    let data_start = metadata_offset + meta.len() as u64 + index.len() as u64;
    let data_offset = (data_start + 4095) & !4095;
    f.write_all(b"HFQM")?;
    f.write_all(&1u32.to_le_bytes())?;
    f.write_all(&arch.to_le_bytes())?;
    f.write_all(&(tensors.len() as u32).to_le_bytes())?;
    f.write_all(&metadata_offset.to_le_bytes())?;
    f.write_all(&data_offset.to_le_bytes())?;
    f.write_all(meta)?;
    f.write_all(&index)?;
    f.write_all(&vec![0u8; (data_offset - data_start) as usize])?;
    for t in tensors {
        f.write_all(&t.data)?;
    }
    f.flush()
}

fn md5_hex(bytes: &[u8]) -> String {
    // Small FNV-1a-64 fingerprint (no md5 crate in this package); labelled as such.
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    format!("fnv1a64:{h:016x}")
}

fn build(args: &[String]) {
    let mtp_path = PathBuf::from(arg(args, "--mtp").expect("--mtp <base.mtp>"));
    let trunk_path = PathBuf::from(arg(args, "--trunk").expect("--trunk <trunk.hfq>"));
    let sidecar_path = PathBuf::from(arg(args, "--sidecar").expect("--sidecar <S.json>"));
    let out_path = PathBuf::from(arg(args, "--output").expect("--output <out.mtp>"));
    let lm_name = arg(args, "--lm-head-name").unwrap_or_else(|| "lm_head.weight".into());

    let sc: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&sidecar_path).expect("read sidecar"))
            .expect("sidecar JSON");
    let ids: Vec<u32> = sc["draft_to_full"]
        .as_array()
        .expect("draft_to_full")
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect();
    let k = ids.len();
    assert_eq!(k as u64, sc["compressed_vocab_size"].as_u64().unwrap(), "sidecar K mismatch");
    assert!(ids.windows(2).all(|w| w[0] < w[1]), "draft_to_full must be strictly ascending");

    let base = HfqFile::open(&mtp_path).expect("open base .mtp");
    assert_eq!(base.arch_id, 21, "base is not an MTP head (arch_id 21)");
    let mut meta: serde_json::Value =
        serde_json::from_str(&base.metadata_json).expect("base metadata");
    assert!(
        !meta["has_compressed_lm_head_draft"].as_bool().unwrap_or(false),
        "base .mtp already carries a compressed draft head"
    );
    let n_embd = meta["n_embd"].as_u64().expect("n_embd") as usize;
    let vocab = meta["vocab_size"].as_u64().expect("vocab_size") as usize;

    let trunk = HfqFile::open(&trunk_path).expect("open trunk");
    let (lm_info, lm_bytes) = trunk.tensor_data(&lm_name).expect("trunk lm_head tensor");
    let (rows, cols) = (lm_info.shape[0] as usize, lm_info.shape[1] as usize);
    assert_eq!(cols, n_embd, "trunk lm_head K != MTP n_embd");
    assert_eq!(rows, vocab, "trunk lm_head rows != MTP vocab_size");
    assert!(ids.iter().all(|&i| (i as usize) < rows), "sidecar id out of range");
    // Only row-contiguous group formats can be sliced by bytes.
    let row_bytes = match lm_info.quant_type {
        44 | 13 => (cols / 256) * 136, // MQ4G256V2 / MQ4G256: 136 B per 256-group
        3 => (cols / 32) * 34,         // Q8_F16: 34 B per 32-group
        other => panic!("unsupported trunk lm_head quant_type {other} for byte slicing"),
    };
    assert_eq!(row_bytes * rows, lm_bytes.len(), "lm_head byte size != rows * row_bytes");
    eprintln!(
        "trunk {} {lm_name}: qt={} [{rows} x {cols}], {row_bytes} B/row; slicing {k} rows",
        trunk_path.display(),
        lm_info.quant_type
    );

    let mut tensors: Vec<OutTensor> = Vec::new();
    for info in base.tensor_infos() {
        let (i2, data) = base.tensor_data_vec(&info.name).expect("base tensor");
        tensors.push(OutTensor {
            name: i2.name.clone(),
            quant_type: i2.quant_type,
            shape: i2.shape.clone(),
            group_size: i2.group_size,
            data,
        });
    }
    let mut slice = Vec::with_capacity(k * row_bytes);
    for &id in &ids {
        let o = id as usize * row_bytes;
        slice.extend_from_slice(&lm_bytes[o..o + row_bytes]);
    }
    let slice_fp = md5_hex(&slice);
    tensors.push(OutTensor {
        name: "lm_head_draft.weight".into(),
        quant_type: lm_info.quant_type,
        shape: vec![k as u32, cols as u32],
        group_size: lm_info.group_size,
        data: slice,
    });
    let map_bytes: Vec<u8> = ids.iter().flat_map(|i| i.to_le_bytes()).collect();
    tensors.push(OutTensor {
        name: "lm_head_draft.vocab_map".into(),
        quant_type: 2,
        shape: vec![k as u32],
        group_size: 0,
        data: map_bytes,
    });

    meta["has_compressed_lm_head_draft"] = serde_json::json!(true);
    meta["compressed_vocab_size"] = serde_json::json!(k);
    meta["claude_lm_head_draft"] = serde_json::json!({
        "source": format!("byte slice of trunk {} ({lm_name}, qt {})", trunk_path.display(), lm_info.quant_type),
        "rows": k,
        "row_order": "ascending token id",
        "sidecar": sidecar_path.display().to_string(),
        "slice_fingerprint": slice_fp,
        "base_mtp": mtp_path.display().to_string(),
        "tool": "claude_mtp_draft_vocab build",
    });
    let meta_json = serde_json::to_string(&meta).unwrap();
    if let Some(p) = out_path.parent() {
        std::fs::create_dir_all(p).ok();
    }
    write_hfq(&out_path, 21, &meta_json, &tensors).expect("write output");

    // Re-open and check: base tensors identical, every slice row equals the trunk row.
    let out = HfqFile::open(&out_path).expect("re-open output");
    for t in &tensors[..tensors.len() - 2] {
        let (_, d) = out.tensor_data(&t.name).expect("output tensor");
        assert!(d == t.data.as_slice(), "base tensor {} changed", t.name);
    }
    let (oi, od) = out.tensor_data("lm_head_draft.weight").expect("lm_head_draft.weight");
    assert_eq!(oi.shape, vec![k as u32, cols as u32]);
    for (j, &id) in ids.iter().enumerate() {
        let a = &od[j * row_bytes..(j + 1) * row_bytes];
        let o = id as usize * row_bytes;
        assert!(a == &lm_bytes[o..o + row_bytes], "slice row {j} (token {id}) differs from trunk");
    }
    let (_, vm) = out.tensor_data("lm_head_draft.vocab_map").expect("vocab_map");
    let back: Vec<u32> = vm.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    assert_eq!(back, ids, "vocab_map round trip");
    let om: serde_json::Value = serde_json::from_str(&out.metadata_json).unwrap();
    assert_eq!(om["has_compressed_lm_head_draft"], serde_json::json!(true));
    let size = std::fs::metadata(&out_path).unwrap().len();
    eprintln!(
        "wrote {} ({:.1} MiB, {} tensors); verified {k} slice rows byte-identical to the trunk, base tensors unchanged",
        out_path.display(),
        size as f64 / 1048576.0,
        tensors.len()
    );
    let _ = HashMap::<u8, u8>::new();
}
