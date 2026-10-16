// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! qwen4_disagg_handoff — eager offline prefill→decode handoff gate (lab).
//!
//! Explicit developer gate; never routed from the daemon and reads no
//! environment switch. A full-model producer bundle on `--prefill-device`
//! (gfx1151) prefills the prompt with the native MTP prompt-head fill
//! (`Qwen4MtpDrafter::mtp_prefill`; the greedy first token is the argmax of the
//! producer's final logits). The pinned `HandoffState` is exported, saved to
//! `--saved-state PATH` (manifest JSON at `PATH.manifest.json`) and imported
//! into a separate consumer bundle on `--decode-device` (gfx1100). The consumer
//! never runs the final prompt token.
//!
//! Checks (any failure: JSON `"pass": false` and exit 1):
//!   1. state bytes and manifest re-exported from the imported consumer equal
//!      the producer's export exactly;
//!   2. the saved files read back equal the export;
//!   3. greedy AR decode of `--tokens` ids on the consumer (the carried first
//!      token first);
//!   4. a fresh consumer bundle sharing the same immutable portable owner
//!      (`DisaggPair::reload_consumer`) imports the saved bytes, re-exports
//!      identical bytes and decodes identical greedy ids;
//!   5. the saved state is imported again and native MTP (`mtp_step` windows,
//!      same greedy request policy) decodes ids equal to AR.
//! A refusal (unsupported cards, admission, API error) prints one line
//! `qwen4_disagg_handoff: refused: ...` to stderr and exits 2.
//!
//! MTP policy: the first window is the production calibration AR step on the
//! carried token; the AR floor is then pinned off (`lab_set_floor_enabled`), a
//! lab correctness policy, so every later window drafts and verifies natively.
//!
//! Policy scopes: every producer and consumer call runs through
//! `DisaggLoaded::run`, which binds the device and applies the side's host-exec
//! policy. The pair is always unloaded in order, on error paths too.
//!
//! Usage:
//!   qwen4_disagg_handoff --model MODEL.mq4 --prefill-device N --decode-device N
//!       (--prompt TEXT | --prompt-file PATH) --tokens N --max-seq N --saved-state PATH

use hipfire_arch_qwen4::bundle::Qwen4Bundle;
use hipfire_arch_qwen4::disagg::HandoffState;
use hipfire_arch_qwen4::disagg_load::{DisaggLoaded, DisaggPair};
use hipfire_arch_qwen4::disagg_manifest::HandoffManifest;
use hipfire_arch_qwen4::mtp_spec::Qwen4MtpDrafter;
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::spec::{MtpDrafter, SpecRequestConfig};
use hipfire_runtime::tokenizer::Tokenizer;
use rdna_compute::{DType, Gpu, GpuTensor};
use serde_json::json;
use std::path::{Path, PathBuf};
use std::time::Instant;

type R<T> = Result<T, String>;

const USAGE: &str = "usage: qwen4_disagg_handoff --model PATH --prefill-device N --decode-device N (--prompt TEXT | --prompt-file PATH) --tokens N --max-seq N --saved-state PATH";

/// Frozen native-MTP window depth for the decode-policy leg.
const MTP_K: usize = 3;

fn e<E: std::fmt::Display>(x: E) -> String {
    x.to_string()
}

struct Args {
    model: PathBuf,
    prefill_device: i32,
    decode_device: i32,
    prompt: String,
    tokens: usize,
    max_seq: usize,
    saved_state: PathBuf,
}

fn parse_args(raw: Vec<String>) -> R<Args> {
    let mut model = None;
    let mut prefill_device = None;
    let mut decode_device = None;
    let mut prompt = None;
    let mut prompt_file = None;
    let mut tokens = None;
    let mut max_seq = None;
    let mut saved_state = None;
    let mut it = raw.into_iter();
    while let Some(flag) = it.next() {
        let mut value = |name: &str| it.next().ok_or_else(|| format!("{name} needs a value"));
        match flag.as_str() {
            "--model" => model = Some(PathBuf::from(value("--model")?)),
            "--prefill-device" => {
                prefill_device = Some(value("--prefill-device")?.parse::<i32>().map_err(|x| format!("--prefill-device: {x}"))?)
            }
            "--decode-device" => {
                decode_device = Some(value("--decode-device")?.parse::<i32>().map_err(|x| format!("--decode-device: {x}"))?)
            }
            "--prompt" => prompt = Some(value("--prompt")?),
            "--prompt-file" => prompt_file = Some(PathBuf::from(value("--prompt-file")?)),
            "--tokens" => tokens = Some(value("--tokens")?.parse::<usize>().map_err(|x| format!("--tokens: {x}"))?),
            "--max-seq" => max_seq = Some(value("--max-seq")?.parse::<usize>().map_err(|x| format!("--max-seq: {x}"))?),
            "--saved-state" => saved_state = Some(PathBuf::from(value("--saved-state")?)),
            other => return Err(format!("unknown argument '{other}'")),
        }
    }
    let prompt = match (prompt, prompt_file) {
        (Some(_), Some(_)) => return Err("give exactly one of --prompt / --prompt-file".into()),
        (Some(text), None) => text,
        (None, Some(path)) => std::fs::read_to_string(&path).map_err(|x| format!("read {}: {x}", path.display()))?,
        (None, None) => return Err("one of --prompt / --prompt-file is required".into()),
    };
    let need = |what: &str| format!("{what} is required");
    let args = Args {
        model: model.ok_or_else(|| need("--model"))?,
        prefill_device: prefill_device.ok_or_else(|| need("--prefill-device"))?,
        decode_device: decode_device.ok_or_else(|| need("--decode-device"))?,
        prompt,
        tokens: tokens.ok_or_else(|| need("--tokens"))?,
        max_seq: max_seq.ok_or_else(|| need("--max-seq"))?,
        saved_state: saved_state.ok_or_else(|| need("--saved-state"))?,
    };
    if args.tokens < 4 {
        return Err("--tokens must be at least 4 so native MTP drafts after its calibration window".into());
    }
    if args.prefill_device == args.decode_device {
        return Err("--prefill-device and --decode-device must name different cards".into());
    }
    Ok(args)
}

fn manifest_path(saved: &Path) -> PathBuf {
    let mut name = saved.as_os_str().to_owned();
    name.push(".manifest.json");
    PathBuf::from(name)
}

fn first_difference(a: &[u8], b: &[u8]) -> Option<usize> {
    if a.len() != b.len() {
        return Some(a.iter().zip(b).position(|(x, y)| x != y).unwrap_or(a.len().min(b.len())));
    }
    a.iter().zip(b).position(|(x, y)| x != y)
}

/// First byte where the file at `path` differs from `bytes` (or its length
/// differs), reading in bounded chunks.
fn file_difference(path: &Path, bytes: &[u8]) -> R<Option<usize>> {
    use std::io::Read;
    let mut file = std::fs::File::open(path).map_err(|x| format!("open {}: {x}", path.display()))?;
    let mut chunk = vec![0u8; 1 << 20];
    let mut at = 0usize;
    loop {
        let n = file.read(&mut chunk).map_err(|x| format!("read {}: {x}", path.display()))?;
        if n == 0 {
            return Ok((at != bytes.len()).then_some(at));
        }
        if at + n > bytes.len() {
            return Ok(Some(bytes.len().max(at)));
        }
        if let Some(i) = first_difference(&chunk[..n], &bytes[at..at + n]) {
            return Ok(Some(at + i));
        }
        at += n;
    }
}

fn first_id_difference(a: &[u32], b: &[u32]) -> Option<usize> {
    if a.len() != b.len() {
        return Some(a.iter().zip(b).position(|(x, y)| x != y).unwrap_or(a.len().min(b.len())));
    }
    a.iter().zip(b).position(|(x, y)| x != y)
}

/// Greedy AR decode of `n` ids on a bundle holding an imported state: the
/// carried `first` token is fed first (and is ids[0]); every later id is the
/// GPU argmax of the previous forward.
fn ar_decode(gpu: &mut Gpu, bundle: &mut Qwen4Bundle, row: &GpuTensor, first: u32, n: usize) -> R<Vec<u32>> {
    gpu.bind_thread().map_err(e)?;
    let mut ids = Vec::with_capacity(n);
    for step in 0..n {
        let token = (step == 0).then_some(first);
        let id = bundle
            .forward_token_or_argmax(gpu, token, row)
            .map_err(|x| format!("AR decode step {step}: {x}"))?;
        ids.push(id);
    }
    gpu.hip.device_synchronize().map_err(e)?;
    if ids[0] != first {
        return Err(format!("AR decode fed {} as the first token, expected the carried {first}", ids[0]));
    }
    Ok(ids)
}

struct MtpRun {
    ids: Vec<u32>,
    windows: usize,
    accepted: usize,
    drafted: usize,
    /// Drafter request counters at the end of the decode.
    floor_ar_windows: usize,
    ar_windows: usize,
    retired: bool,
}

/// Native MTP decode of at most `n` ids from an imported state: a fresh
/// greedy `Qwen4MtpDrafter`, `mtp_step` windows of depth `MTP_K`, the carried
/// first token as the pending seed. The first window is the production
/// floor-calibration AR step; the floor is then pinned off. Stops early only
/// after an EOS id.
fn mtp_decode(gpu: &mut Gpu, bundle: &mut Qwen4Bundle, first: u32, n: usize, max_seq: usize) -> R<MtpRun> {
    gpu.bind_thread().map_err(e)?;
    let mut drafter = Qwen4MtpDrafter::new(MTP_K, max_seq, None);
    drafter.configure_request(SpecRequestConfig::default());
    let result = (|| -> R<MtpRun> {
        let eos = bundle.config.eos_token_id;
        let mut run = MtpRun {
            ids: vec![first],
            windows: 0,
            accepted: 0,
            drafted: 0,
            floor_ar_windows: 0,
            ar_windows: 0,
            retired: false,
        };
        let mut pending = first;
        while run.ids.len() < n && pending != eos {
            let remaining = n - run.ids.len();
            let budget = MTP_K.min(remaining - 1);
            let position = bundle.state.position;
            let window = drafter
                .mtp_step(gpu, &mut *bundle, position, pending, &run.ids, budget, eos, None)
                .map_err(|x| format!("MTP window {} at position {position}: {x}", run.windows))?;
            if window.committed.is_empty() || window.committed.len() > remaining {
                return Err(format!(
                    "MTP window {} committed {} ids with {remaining} wanted",
                    run.windows,
                    window.committed.len()
                ));
            }
            run.windows += 1;
            if run.windows == 1 {
                // The request's first window is the production calibration AR
                // step on the carried token (it rebuilds the pending hidden and
                // the head row). From here the floor is pinned off, the lab
                // correctness policy: no timing-dependent retirement to AR.
                drafter.lab_set_floor_enabled(false);
            }
            run.accepted += window.accepted;
            run.drafted += window.drafts_generated;
            for &id in &window.committed {
                run.ids.push(id);
                if id == eos {
                    break;
                }
            }
            pending = *run.ids.last().expect("non-empty");
        }
        gpu.hip.device_synchronize().map_err(e)?;
        let stats = drafter.request_stats();
        run.floor_ar_windows = stats.floor_ar_windows;
        run.ar_windows = stats.ar_windows;
        run.retired = stats.mtp_retired;
        Ok(run)
    })();
    Box::new(drafter).mtp_free(gpu);
    result
}

fn real_main() -> R<bool> {
    let raw: Vec<String> = std::env::args().skip(1).collect();
    if raw.iter().any(|a| a == "--help" || a == "-h") {
        println!("{USAGE}");
        return Ok(true);
    }
    let args = parse_args(raw).map_err(|x| format!("{x}; {USAGE}"))?;
    let tokenizer = {
        let hfq = HfqFile::open(&args.model).map_err(e)?;
        Tokenizer::from_hfq_metadata(&hfq.metadata_json).map_err(e)?
    };
    let prompt = tokenizer.encode(&args.prompt);
    if prompt.is_empty() {
        return Err("the prompt encodes to no tokens".into());
    }
    let need = prompt.len() + args.tokens + MTP_K + 1;
    if need > args.max_seq {
        return Err(format!(
            "prompt {} + tokens {} + MTP window {} exceeds --max-seq {}",
            prompt.len(),
            args.tokens,
            MTP_K + 1,
            args.max_seq
        ));
    }

    let started = Instant::now();
    let mut pair = DisaggPair::load(&args.model, args.prefill_device, args.decode_device, args.max_seq)?;
    let load_s = started.elapsed().as_secs_f64();
    // The pair is always unloaded in order (Drop frees no device state), on
    // the error path too.
    let outcome = gate(&mut pair, &args, &prompt, load_s);
    let unloaded = pair.unload();
    match (outcome, unloaded) {
        (Ok((pass, mut report)), unloaded) => {
            let mut pass = pass;
            if let Err(error) = unloaded {
                pass = false;
                report["pass"] = json!(false);
                report["unload_error"] = json!(error);
            }
            println!("{}", serde_json::to_string_pretty(&report).map_err(e)?);
            Ok(pass)
        }
        (Err(error), Ok(())) => Err(error),
        (Err(error), Err(unload)) => Err(format!("{error}; unload: {unload}")),
    }
}

/// Run one side's call with its device bound and its role's host-exec policy
/// in scope (`DisaggLoaded::run`).
fn on<T>(side: &mut DisaggLoaded, f: impl FnOnce(&mut Gpu, &mut Qwen4Bundle) -> R<T>) -> R<T> {
    side.run(f)?
}

/// The gate with the consumer's reusable logits row allocated and freed around it.
fn gate(pair: &mut DisaggPair, args: &Args, prompt: &[u32], load_s: f64) -> R<(bool, serde_json::Value)> {
    let vocab = pair.consumer()?.bundle.config.vocab_size;
    let row = on(pair.consumer_mut()?, |gpu, _| gpu.zeros(&[vocab], DType::F32).map_err(e))?;
    let result = gate_with_row(pair, args, prompt, &row, load_s);
    // After a failed reload the consumer is gone with its Gpu bundle state.
    if let Ok(consumer) = pair.consumer_mut() {
        let _ = on(consumer, |gpu, _| gpu.free_tensor(row).map_err(e));
    }
    result
}

fn gate_with_row(
    pair: &mut DisaggPair,
    args: &Args,
    prompt: &[u32],
    row: &GpuTensor,
    load_s: f64,
) -> R<(bool, serde_json::Value)> {
    let mut mismatches: Vec<String> = Vec::new();
    let max_seq = args.max_seq;
    let tokens = args.tokens;

    // ── producer: native MTP prompt prefill, export ───────────────────────
    let t = Instant::now();
    let (state, seed) = {
        let domain = pair.domain.clone();
        on(&mut pair.producer, |gpu, bundle| {
            let mut drafter = Qwen4MtpDrafter::new(MTP_K, max_seq, None);
            drafter.configure_request(SpecRequestConfig::default());
            let prefilled = drafter.mtp_prefill(gpu, &mut *bundle, prompt, prompt, 0, false, &|| false);
            let exported = prefilled
                .and_then(|seed| bundle.export_handoff(gpu, &domain, seed).map(|state| (state, seed)));
            Box::new(drafter).mtp_free(gpu);
            exported
        })?
    };
    let prefill_s = t.elapsed().as_secs_f64();
    let position = state.manifest.position;
    if position != prompt.len() {
        mismatches.push(format!("exported position {position} != prompt length {}", prompt.len()));
    }
    if state.manifest.first_token != seed {
        mismatches.push(format!("manifest first token {} != producer greedy {seed}", state.manifest.first_token));
    }
    let state_bytes = state.as_bytes().len();
    let payload_digest = state.manifest.payload_digest.clone();
    let manifest = state.manifest.clone();

    // Admission charges two payload-sized buffers; at most two are live at any
    // point below (the original pinned state, the saved-bytes Vec and a pinned
    // `HandoffState` are never all alive together).

    // ── save; verify the files by bounded chunked reads ───────────────────
    let manifest_file = manifest_path(&args.saved_state);
    std::fs::write(&args.saved_state, state.as_bytes())
        .map_err(|x| format!("write {}: {x}", args.saved_state.display()))?;
    std::fs::write(&manifest_file, serde_json::to_vec_pretty(&manifest).map_err(e)?)
        .map_err(|x| format!("write {}: {x}", manifest_file.display()))?;
    if let Some(at) = file_difference(&args.saved_state, state.as_bytes())? {
        mismatches.push(format!("saved state file differs from the export at byte {at}"));
    }
    let saved_manifest: HandoffManifest = serde_json::from_slice(
        &std::fs::read(&manifest_file).map_err(|x| format!("read {}: {x}", manifest_file.display()))?,
    )
    .map_err(|x| format!("parse {}: {x}", manifest_file.display()))?;
    if saved_manifest != manifest {
        mismatches.push("saved manifest JSON differs from the export manifest".into());
    }

    // ── consumer: import, exact byte capture, AR decode ───────────────────
    let t = Instant::now();
    let domain = pair.domain.clone();
    let (first_import, import_exact, ar_ids) = on(pair.consumer_mut()?, |gpu, bundle| {
        bundle.reset(gpu).map_err(e)?;
        let first = bundle.import_handoff(gpu, &state, &domain)?;
        let at = bundle.state.position;
        if at != position {
            return Err(format!("import left the consumer at position {at}, expected {position}"));
        }
        let back = bundle.export_handoff(gpu, &domain, first)?;
        let exact = (first_difference(back.as_bytes(), state.as_bytes()), back.manifest == manifest);
        drop(back);
        bundle.reset(gpu).map_err(e)?;
        bundle.import_handoff(gpu, &state, &domain)?;
        let ids = ar_decode(gpu, bundle, row, first, tokens)?;
        Ok((first, exact, ids))
    })?;
    // The original pinned state is no longer needed: the saved file (verified
    // equal above) and the retained manifest are the reference from here on.
    drop(state);
    let ar_s = t.elapsed().as_secs_f64();
    if first_import != seed {
        mismatches.push(format!("imported first token {first_import} != producer greedy {seed}"));
    }
    if let Some(at) = import_exact.0 {
        mismatches.push(format!("bytes exported after import differ from the producer export at byte {at}"));
    }
    if !import_exact.1 {
        mismatches.push("manifest exported after import differs from the producer manifest".into());
    }

    // ── fresh consumer sharing the owner: saved bytes, reimport, decode ───
    let t = Instant::now();
    pair.reload_consumer()?;
    let saved_bytes =
        std::fs::read(&args.saved_state).map_err(|x| format!("read {}: {x}", args.saved_state.display()))?;
    let (reload_exact, reload_ids) = on(pair.consumer_mut()?, |gpu, bundle| {
        let loaded = HandoffState::from_bytes(gpu, saved_manifest.clone(), &saved_bytes)?;
        let first = bundle.import_handoff(gpu, &loaded, &domain)?;
        drop(loaded);
        if first != seed {
            return Err(format!("reloaded import first token {first} != producer greedy {seed}"));
        }
        let back = bundle.export_handoff(gpu, &domain, first)?;
        let exact = (first_difference(back.as_bytes(), &saved_bytes), back.manifest == saved_manifest);
        drop(back);
        bundle.reset(gpu).map_err(e)?;
        let loaded = HandoffState::from_bytes(gpu, saved_manifest.clone(), &saved_bytes)?;
        bundle.import_handoff(gpu, &loaded, &domain)?;
        drop(loaded);
        let ids = ar_decode(gpu, bundle, row, first, tokens)?;
        Ok((exact, ids))
    })?;
    let reload_s = t.elapsed().as_secs_f64();
    if let Some(at) = reload_exact.0 {
        mismatches.push(format!("bytes exported after reloaded import differ from the saved state at byte {at}"));
    }
    if !reload_exact.1 {
        mismatches.push("manifest exported after reloaded import differs from the saved manifest".into());
    }
    if let Some(at) = first_id_difference(&reload_ids, &ar_ids) {
        mismatches.push(format!("reloaded consumer greedy ids differ from the first consumer at index {at}"));
    }

    // ── native MTP from the same imported state, same greedy policy ───────
    let t = Instant::now();
    let (mtp, eos) = on(pair.consumer_mut()?, |gpu, bundle| {
        let loaded = HandoffState::from_bytes(gpu, saved_manifest.clone(), &saved_bytes)?;
        bundle.reset(gpu).map_err(e)?;
        let first = bundle.import_handoff(gpu, &loaded, &domain)?;
        drop(loaded);
        let eos = bundle.config.eos_token_id;
        mtp_decode(gpu, bundle, first, tokens, max_seq).map(|run| (run, eos))
    })?;
    drop(saved_bytes);
    let mtp_s = t.elapsed().as_secs_f64();
    let compared = mtp.ids.len().min(ar_ids.len());
    let ended_on_eos = mtp.ids.last() == Some(&eos);
    if mtp.ids.len() != tokens && !ended_on_eos {
        mismatches.push(format!("MTP produced {} ids, expected {tokens}", mtp.ids.len()));
    }
    if let Some(at) = first_id_difference(&mtp.ids[..compared], &ar_ids[..compared]) {
        mismatches.push(format!(
            "MTP greedy ids differ from AR at index {at} (mtp {:?}, ar {:?})",
            mtp.ids.get(at),
            ar_ids.get(at)
        ));
    }
    // The same greedy request must have run native draft+verify windows, not
    // only the production floor's calibration/AR window.
    if mtp.drafted == 0 && mtp.ids.len() == tokens {
        mismatches.push(format!(
            "MTP never drafted ({} windows, {} floor AR, {} retired AR): raise --tokens",
            mtp.windows, mtp.floor_ar_windows, mtp.ar_windows
        ));
    }

    let pass = mismatches.is_empty();
    let report = json!({
        "pass": pass,
        "mismatches": mismatches,
        "model": args.model.display().to_string(),
        "prefill_device": args.prefill_device,
        "decode_device": args.decode_device,
        "max_seq": max_seq,
        "prompt_tokens": prompt.len(),
        "tokens": tokens,
        "first_token": seed,
        "position": position,
        "state_bytes": state_bytes,
        "payload_digest": payload_digest,
        "saved_state": args.saved_state.display().to_string(),
        "saved_manifest": manifest_file.display().to_string(),
        "state_bytes_exact_after_import": import_exact.0.is_none() && import_exact.1,
        "state_bytes_exact_after_reload_import": reload_exact.0.is_none() && reload_exact.1,
        "ar_ids": ar_ids,
        "reload_ids_equal_ar": reload_ids == ar_ids,
        "mtp": {
            "k": MTP_K,
            "ids": mtp.ids,
            "equal_ar": mtp.ids[..compared] == ar_ids[..compared],
            "windows": mtp.windows,
            "accepted_drafts": mtp.accepted,
            "drafts_generated": mtp.drafted,
            "floor_ar_windows": mtp.floor_ar_windows,
            "retired_ar_windows": mtp.ar_windows,
            "retired": mtp.retired,
        },
        "admission": pair.admission.lines(),
        "seconds": {
            "load": load_s,
            "producer_prefill_export": prefill_s,
            "consumer_import_ar": ar_s,
            "reload_import_ar": reload_s,
            "mtp": mtp_s,
        },
    });
    Ok((pass, report))
}

fn main() {
    match real_main() {
        Ok(true) => {}
        Ok(false) => std::process::exit(1),
        Err(message) => {
            let flat = message.replace('\r', "\n");
            let one_line = flat.lines().map(str::trim).filter(|l| !l.is_empty()).collect::<Vec<_>>().join(" ");
            println!("{}", json!({ "pass": false, "refused": one_line }));
            eprintln!("qwen4_disagg_handoff: refused: {one_line}");
            std::process::exit(2);
        }
    }
}
