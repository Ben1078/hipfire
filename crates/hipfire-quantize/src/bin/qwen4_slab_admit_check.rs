//! CPU-only end-to-end check of a Flash-Next expert-slab artifact against the
//! qwen4 loader: parse `qwen4.expert_slabs`, build the span table from the real
//! index, validate every W3/W2 payload through the shared codec, then run the
//! production `admit_hfqm_artifact` and require its fail-closed
//! `ExpertSlabKernelsUnavailable` refusal. Never touches a GPU.
//!
//! ```text
//! qwen4_slab_admit_check mixed.hfq > admit.json
//! ```

use std::collections::HashMap;
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use clap::Parser;
use hipfire_arch_qwen4::artifact::{admit_hfqm_artifact, Qwen4ArtifactError};
use hipfire_arch_qwen4::expert_slab_table::{build_expert_slab_table_with_keep, validate_expert_slab_scales};
use hipfire_arch_qwen4::ExpertKeep;
use hipfire_arch_qwen4::expert_slabs;
use hipfire_runtime::hfq::HfqFile;

#[derive(Parser, Debug)]
#[command(
    name = "qwen4_slab_admit_check",
    about = "Validate a Flash-Next expert-slab HFQ with the qwen4 loader (CPU only)"
)]
struct Args {
    /// Expert-slab HFQM artifact.
    path: PathBuf,
}

fn fail(stage: &str, error: impl std::fmt::Display) -> ExitCode {
    eprintln!("qwen4_slab_admit_check: {stage}: {error}");
    ExitCode::FAILURE
}

fn main() -> ExitCode {
    let args = Args::parse();
    let start = Instant::now();
    let hfq = match HfqFile::open(&args.path) {
        Ok(hfq) => hfq,
        Err(error) => return fail("open", error),
    };
    let metadata = match expert_slabs::parse(&hfq.metadata_json) {
        Ok(Some(metadata)) => metadata,
        Ok(None) => return fail("metadata", "no qwen4.expert_slabs key"),
        Err(error) => return fail("metadata", error),
    };
    let json: serde_json::Value = match serde_json::from_str(&hfq.metadata_json) {
        Ok(v) => v,
        Err(error) => return fail("keep metadata", error),
    };
    let keep = match ExpertKeep::from_metadata(&json) {
        Ok(keep) => keep,
        Err(error) => return fail("keep metadata", error),
    };
    for info in hfq.tensors() {
        if let Some((layer, expert, _)) = expert_slabs::parse_expert_entry_name(&info.name) {
            if keep.as_ref().is_some_and(|k| !k.is_kept(layer, expert)) {
                return fail("absent expert", format!("unexpected tensor {}", info.name));
            }
        }
    }
    let routed_bytes: usize = hfq.tensors().iter()
        .filter(|t| expert_slabs::parse_expert_entry_name(&t.name).is_some())
        .map(|t| t.data_size).sum();
    let non_expert_bytes: usize = hfq.tensors().iter()
        .filter(|t| expert_slabs::parse_expert_entry_name(&t.name).is_none())
        .map(|t| t.data_size).sum();
    let table = match build_expert_slab_table_with_keep(&metadata, hfq.tensors(), keep.as_ref()) {
        Ok(table) => table,
        Err(error) => return fail("span table", error),
    };
    let table_s = start.elapsed().as_secs_f64();
    let slab_entries: HashMap<usize, &str> = hfq
        .tensors()
        .iter()
        .filter(|info| expert_slabs::parse_expert_entry_name(&info.name).is_some())
        .map(|info| (info.data_offset, info.name.as_str()))
        .collect();
    if let Err(error) = validate_expert_slab_scales(&table, |span| {
        let name = slab_entries.get(&span.offset)?;
        let (info, bytes) = hfq.tensor_data(name)?;
        (info.data_size == span.len).then_some(bytes)
    }) {
        return fail("payload validation", error);
    }
    let validate_s = start.elapsed().as_secs_f64() - table_s;
    let counts = table.counts();
    if keep.as_ref().is_some_and(|k| counts.w3 + counts.w2 + counts.mq4 != k.total_kept()) {
        return fail("keep counts", "table count differs from keep total");
    }
    match admit_hfqm_artifact(&hfq) {
        Err(Qwen4ArtifactError::ExpertSlabKernelsUnavailable { w3, w2, mq4 })
            if (w3, w2, mq4) == (counts.w3, counts.w2, counts.mq4) =>
        {
            let refusal = Qwen4ArtifactError::ExpertSlabKernelsUnavailable { w3, w2, mq4 };
            let report = serde_json::json!({
                "artifact": args.path.display().to_string(),
                "tensors": hfq.tensors().len(),
                "expert_slabs": metadata.to_json_value(),
                "counts": {"w3": w3, "w2": w2, "mq4": mq4},
                "expert_keep": keep.as_ref().map(ExpertKeep::to_metadata_value),
                "kept_experts": keep.as_ref().map_or(48 * 512, ExpertKeep::total_kept),
                "bytes": {"routed_experts": routed_bytes, "non_expert": non_expert_bytes, "total_base": hfq.file_len()},
                "payload_validation": "every W3/W2 projection decoded: exact length, codes <= L, finite non-negative f16 scales, no -0, zero-scale groups at the midpoint code",
                "admission": "refused (fail-closed)",
                "refusal": refusal.to_string(),
                "timings_s": {"open_and_table": table_s, "payload_validation": validate_s},
            });
            println!("{}", serde_json::to_string_pretty(&report).expect("report serializes"));
            ExitCode::SUCCESS
        }
        Err(error) => fail("admission returned a different refusal", error),
        Ok(_) => fail("admission", "artifact was ADMITTED; expected the fail-closed kernel refusal"),
    }
}
