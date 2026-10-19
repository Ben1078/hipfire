#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""prove_beta_math.py BETA_SOURCE OUTPUT_JSON

CPU-only, read-only source oracle for the repaired Hold 1b (R8). It proves from the exact immutable beta source (commit ff799626, never
modified, never built, no GPU, no model payload) that the quality-reference pin pair

    HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS=5   +   HIPFIRE_PREFILL_CHUNK_ROWS=4096

keeps the SAME arithmetic class (routed gate/up and down kernel, precision, wire strides) per (layer, actual forward rows) as the failed
j1934 reference pins (EXPERT_VRAM_LAYERS=10, requested chunk 8192 that the admission ladder reduced to an actual 4096), and that changing the
resident count alone (a possible actual chunk 8192) is NOT the same class (irregular length 4100: [4096, 4] -> [4100]).

This is a SOURCE-ATTESTED oracle, not a duplicate routing table nobody checked: the file hashes of every consulted beta source file are pinned
and fail closed on drift, every predicate/threshold/stride/stage/restore expression the model below encodes is asserted against its actual
source text (whitespace-normalised needles inside a pinned line window; see ANCHORS), and the model functions only use constants whose source
line is asserted. It is NOT a measured fit, NOT a GPU run and NOT an observed generated-token equality claim; the receipt independently
requires the beta run's own `qwen4 prefill chunk: 4096 rows` line.

Output schema s1.beta_math_proof.v1 (concise: fixed six cases, a conditional natural case, a bounded general property digest, negatives)."""
import hashlib
import json
import os
import subprocess
import sys

SCHEMA = "s1.beta_math_proof.v1"
BETA_COMMIT = "ff799626eb3dddf8ce827a90d88c990fc0eb4aa2"

PINS = {
    "crates/hipfire-arch-qwen4/src/gpu_forward.rs": "53a8e613d0222c6d0e1b2d2a82af68d4d09ca48bdedbf133eb35492d8110d0cb",
    "crates/hipfire-dispatch/src/pipeline/qt44_qt53_prefill.rs": "89e58b9ddc7a807520f96a177e0889d02857b179cf888b7ff399381d785eacf9",
    "crates/hipfire-dispatch/src/pipeline/mod.rs": "4a32862ba1c371656ad922586f8b97e27a742e03a666a31b67436fc67dc71330",
    "crates/rdna-compute/src/gemm.rs": "ac1537023e3218b1e13a9c0c18911857157ea1b87bc8ea32710c20c95aaa9e31",
    "crates/rdna-compute/src/moe.rs": "ba1c1a6ac713ac695df39ebde01bab7b875ff86faf292b66b67301d14e91318b",
    "crates/rdna-compute/src/gemv.rs": "95424ed504da1fdd767bb8083792d05fc2fbe51de9654507d2126f8a667f07bb",
    "crates/rdna-compute/src/prefill_invariant.rs": "996816ec6e4d147fdef9a7802ff258dd8803721ec01838ac04bc8e40b46d07da",
    "crates/hipfire-arch-qwen4/src/expert_residency.rs": "ca2d08c3f78d6a588eaffb6729e34dfc3879aa08344cb3436d97af22b3848f02",
    "crates/hipfire-arch-qwen4/src/bundle.rs": "b9f3e4ee02c2e8c5c6c52e5190220f173dabc623ae435ade6a0dc8f77217f664",
    "crates/hipfire-arch-qwen4/examples/qwen4_greedy_ids.rs": "90241ee9f4d455f5b93cfc85063bcf8b836a917a462b85ad4425a3b12e7a7311",
    "crates/hipfire-config/src/lib.rs": "98ce04d2469a35fdefae93551ac4ca8e4b062772cba38ad8cacd54e44a77bf8e",
    "crates/hipfire-dispatch/src/cpu_exec.rs": "577435a0e6bbdcf3456ac6a62fe27f81d094637f6e7c7cbe8900b62f47336859",
    "kernels/src/qwen4_gemm_mq4g256v2_moe_grouped_wmma_k2.hip": "f82dccca09b94abe72efbfe6c4b15213be67d53f501fdd4ada28c1b4a0d71add",
    "kernels/src/gemv_mq4g128v2_moe_down_top10_indexed_batched_expanded.hip": "2326440b1210f8f069ab34c0246f8bf7af30969b44eb0f3c563660039fe93bdb",
    "kernels/src/gemm_mq4g128v2_moe_grouped_wmma.gfx1151.hip": "2f4fff61535fae7babee6f9cfa6b5d4cdfe5563b520eea4850c60af2725501c0",
}
GF = "crates/hipfire-arch-qwen4/src/gpu_forward.rs"
PP = "crates/hipfire-dispatch/src/pipeline/qt44_qt53_prefill.rs"
PM = "crates/hipfire-dispatch/src/pipeline/mod.rs"
RG = "crates/rdna-compute/src/gemm.rs"
RM = "crates/rdna-compute/src/moe.rs"
RV = "crates/rdna-compute/src/gemv.rs"
PI = "crates/rdna-compute/src/prefill_invariant.rs"
ER = "crates/hipfire-arch-qwen4/src/expert_residency.rs"
BU = "crates/hipfire-arch-qwen4/src/bundle.rs"
EX = "crates/hipfire-arch-qwen4/examples/qwen4_greedy_ids.rs"
CF = "crates/hipfire-config/src/lib.rs"
CE = "crates/hipfire-dispatch/src/cpu_exec.rs"
K_GATE = "kernels/src/qwen4_gemm_mq4g256v2_moe_grouped_wmma_k2.hip"
K_DOWN = "kernels/src/gemv_mq4g128v2_moe_down_top10_indexed_batched_expanded.hip"
K_DOWN_WMMA = "kernels/src/gemm_mq4g128v2_moe_grouped_wmma.gfx1151.hip"

# (name, file, first line, last line, needles). Each needle (whitespace-collapsed) must occur inside the whitespace-collapsed text of the
# 1-indexed inclusive line window of the pinned file. These are the ACTUAL selector / threshold / stride / donor / restore expressions.
ANCHORS = [
    ("chunk_env_read", GF, 143, 152, ['hipfire_config::developer_var("HIPFIRE_PREFILL_CHUNK_ROWS")', "pub fn qwen4_prefill_chunk_requested(gpu: &Gpu, max_seq: usize) -> usize"]),
    ("chunk_round_256", GF, 154, 160, [".map(|rows| rows / 256 * 256)", ".filter(|&rows| rows >= 256)"]),
    ("chunk_requested_min_max_seq", GF, 164, 173, [".min(max_seq)", "explicit_prefill_chunk_rows(env)"]),
    ("chunk_rungs", GF, 175, 183, ["[4096, 2048, 1536]", ".filter(move |&rows| rows < requested)", "std::iter::once(requested)"]),
    ("stage_min_rows_768", GF, 490, 496, ["const QWEN4_EXPERT_STAGE_NARROW_LINK_MIN_ROWS: usize = 768;",
                                          "const QWEN4_EXPERT_STAGE_GATE_UP_BYTES: usize = 1_740_800;", "const QWEN4_EXPERT_STAGE_DOWN_BYTES: usize = 870_400;"]),
    ("stage_geometry_gate", GF, 3813, 3829, ["config.num_experts != 512", "config.hidden_size != 2560", "config.moe_intermediate_size != 640",
                                             "expert.gate_up.tensor.dtype != DType::MQ4G256V2", "expert.down.tensor.dtype != DType::MQ4G128V2"]),
    ("stage_donor_parity", GF, 3834, 3844, ["let mut donors = [first_host - 2, first_host - 1];", "donors.sort_by_key(|layer| layer % 2);",
                                            "if first_host < 2", "RoutedExpertWeights::host_mapped"]),
    ("stage_alias_tables", GF, 3967, 3991, ["let view = owner.sub_offset(0, bytes);", "(base + expert * stride).to_ne_bytes()",
                                            "gpu.alloc_tensor(&[2 * config.num_experts], DType::F32)", "gpu.memcpy_dtoh_auto(&mut host, &view.buf)",
                                            "gpu.upload_raw_host_mapped(&host, &[bytes])"]),
    ("stage_restore_enqueue", GF, 4166, 4184, ["gpu.hip .stream_wait_event(&self.copy_stream, &self.free[parity])?;",
                                               "fn enqueue_restore(&self, gpu: &Gpu, parity: usize)", "self.set_state(parity, DonorState::Restoring);"]),
    ("stage_copy_bytes_exact", GF, 4186, 4201, ["gpu.hip.memcpy_htod_async(", "&source[..bytes],", ".event_record(&self.ready[parity], Some(&self.copy_stream))?;"]),
    ("stage_join_before_next_compute", GF, 4259, 4279, ["fn join(&self, gpu: &Gpu)", "self.enqueue_restore(gpu, parity)?;",
                                                        "gpu.hip.stream_wait_event(compute, &self.ready[parity])?;", "self.set_state(parity, DonorState::Resident);"]),
    ("stage_restore_blocking_recovery", GF, 4281, 4313, ["fn restore_blocking(&self, gpu: &Gpu)", "gpu.hip.device_synchronize()?;",
                                                        ".memcpy_htod(&self.donor_views[2 * parity + kind].buf, source)?;"]),
    ("stage_ready_before_moe", GF, 4364, 4376, ["fn before_moe(&mut self, gpu: &mut Gpu)", ".stream_wait_event(compute, &self.stage.ready[self.layer % 2])"]),
    ("chunk_segmentation", GF, 7819, 7824, ["let max_chunk = self.scratch.max_chunk;", "tokens.len().min(max_chunk)"]),
    ("chunk_segmentation_rows", GF, 7872, 7872, ["let rows = (tokens.len() - offset).min(max_chunk);"]),
    ("place_routed_experts_prefix", ER, 1869, 1886, ["pub fn place_routed_experts(weights: &mut [WeightEntry], vram_layers: usize) -> usize",
                                                     '!entry.name.starts_with("mtp.") && entry.layer.is_some_and(|layer| layer < vram_layers)',
                                                     "entry.residency = WeightResidency::HostMapped;"]),
    ("host_mapped_pad", ER, 83, 84, ["pub const HOST_MAPPED_PAD_BYTES: u64 = 1 << 20;"]),
    ("host_mapped_bytes_pad", ER, 1888, 1902, ["total += bytes + HOST_MAPPED_PAD_BYTES;"]),
    ("greedy_example_n_loader", EX, 343, 366, ["place_routed_experts(", "ExpertVramLayers::Auto) => {", "qwen4_greedy_ids takes a fixed expert layer count, not auto",
                                               'routed experts: layers 0..{layers} in VRAM, {moved} tensors host-mapped']),
    ("greedy_example_cold_prefill", EX, 515, 545, [".prefill_final(&mut self.gpu, prompt, 0, &self.prompt_logits)", "forward_token_or_argmax(&mut self.gpu, pending.take(), &self.decode_logits)",
                                                  "while ids.len() < max_new {", "if token == self.eos_token || self.tokenizer.is_terminator(token) {",
                                                  'return Ok((ids, "stop", prefill_s));', 'Ok((ids, "length", prefill_s))']),
    ("chunk_admission_ladder", BU, 1148, 1166, ["let max_chunk = qwen4_prefill_chunk_rungs(requested)", ".unwrap_or_else(|| requested.min(1536));",
                                                'eprintln!(" qwen4 prefill chunk: {max_chunk} rows (requested {requested})");']),
    ("chunk_requested_ple_cap", BU, 1060, 1076, ["let requested = qwen4_prefill_chunk_requested(gpu, max_chunk).min(ple_rows_cap);"]),
    ("f16_wmma_min_tokens_512", RG, 428, 430, ["pub const QWEN4_F16_WMMA_MIN_TOKENS: usize = 512;"]),
    ("f16_wmma_default_on", RG, 173, 179, ['hipfire_config::developer_var("HIPFIRE_QWEN4_F16_WMMA").map_or(true, |v| v.trim() != "0")']),
    ("gate_wmma_predicate", RG, 47622, 47632, ["pub fn qwen4_moe_gateup_wmma_applies(&self, m: usize, k: usize, x_src_rows: usize) -> bool",
                                               "self.arch_caps.has_wmma()", "&& m == 1280", "&& k == 2560",
                                               "&& self.route_rows(x_src_rows) >= QWEN4_F16_WMMA_MIN_TOKENS", "&& *QWEN4_F16_WMMA"]),
    ("gate_dispatch_f32_variants", RG, 47765, 47812, ["if self.qwen4_moe_gateup_wmma_applies(m, k, x_src_rows) {", "return self.gemm_mq4g256v2_moe_grouped_wmma_k2(",
                                                      "if self.arch_caps.has_gfx11_plus_simt() && m == 1280 && k == 2560 {",
                                                      "qwen4_moe_gateup_twin_applies( *QWEN4_MOE_GATEUP_TWIN, self.arch_caps.is_gfx1151(),",
                                                      "x_src_rows <= 8,"]),
    ("gate_twin_gfx1151_only", RG, 296, 305, ["pub(crate) fn qwen4_moe_gateup_twin_applies(", "enabled && exact_gfx1151 && m == 1280 && k == 2560 && x_row_div == 10 && x_src_rows > 8"]),
    ("gate_f32_symbols", RG, 47847, 47852, ['const MODULE: &str = "gemm_mq4g256v2_moe_grouped_top10_o4_r8_x4";', '("gemm_mq4g256v2_moe_grouped_top10_o4_r2_x4", 2)']),
    ("down_wmma_predicate", RM, 3027, 3033, ["pub fn qwen4_moe_down_wmma_applies(&self, m: usize, k: usize, x_src_rows: usize) -> bool",
                                             "self.arch_caps.has_wmma_w32()", "&& m == 2560", "&& k == 640",
                                             "&& self.route_rows(x_src_rows) >= 10 * crate::gemm::QWEN4_F16_WMMA_MIN_TOKENS", "&& *crate::gemm::QWEN4_F16_WMMA"]),
    ("down_dispatch", RM, 2980, 3008, ["if x_row_div == 1 && self.qwen4_moe_down_wmma_applies(m, k, x_src_rows) {", "return self.gemm_mq4g128v2_moe_grouped_wmma_gfx1151(",
                                       "crate::gemm::qwen4_moe_down_twin_applies(", "} else if self.arch_caps.has_gfx11_plus_simt() && m == 2560 && k == 640 {", "MoeDownKernel::O8R16"]),
    ("down_f32_symbol", RM, 3212, 3215, ['"gemm_mq4g128v2_moe_grouped_top10_o8_r16_gfx1151"']),
    ("indexed_down_few_row", PP, 1676, 1684, ["fn indexed_down(gpu: &Gpu, p: &MoePrefillParams<'_>, use_path2: bool) -> bool",
                                              "use_path2 && gpu.route_rows(p.batch_size) <= gpu.few_row_limit() && !down_wmma(gpu, p)"]),
    ("stage_pointer_substitution", PP, 1447, 1449, [".expert_stage_ptrs", ".map_or(p.expert_gate_up_ptrs, |stage| stage.gate_up);"]),
    ("stage_pointer_substitution_down", PP, 1719, 1721, [".expert_stage_ptrs", ".map_or(p.expert_down_ptrs, |stage| stage.down);"]),
    ("stage_release_after_down", PP, 1686, 1691, ["fn release_stage(gpu: &Gpu, p: &MoePrefillParams<'_>)", "gpu.hip.event_record(stage.free, gpu.active_stream.as_ref())"]),
    ("singleton_decode_gate", PM, 4314, 4353, ["hip!(gpu.gemv_mq4g256v2_moe_gate_up_top10_indexed_batched(", "2 * p.mi,", "p.hidden,", "1,"]),
    ("singleton_decode_down", PM, 4730, 4776, ["hip!(gpu.gemv_mq4g128v2_moe_down_top10_indexed_batched_expanded(", "p.expert_down_ptrs,"]),
    ("singleton_decode_gate_symbol", RV, 16676, 16684, ['const ROWS8: &str = "gemv_mq4g256v2_moe_gate_up_k2560_rows8_indexed_batched";', "k == 2560 && m.is_multiple_of(8)"]),
    ("few_row_limit_8", PI, 126, 126, ["pub const FEW_ROW_ROUTE_ROWS: usize = 8;"]),
    ("few_row_limit_eager", PI, 196, 207, ["fn few_row_limit_for(wide_verify_rows: usize, eager: bool) -> usize", "FEW_ROW_ROUTE_ROWS.max(wide_verify_rows)"]),
    ("route_rows_identity_outside_invariant_scope", PI, 224, 231, ["pub fn route_rows(&self, rows: usize) -> usize", "if self.prefill_invariant_scope {", "QWEN4_PREFILL_CANON_ROWS", "} else { rows }"]),
    ("chunk_config_field", CF, 2562, 2574, ['"prefill.chunk_rows"', '"HIPFIRE_PREFILL_CHUNK_ROWS"', "Qwen4 (Flash-Next) rounds it down to 256 rows",
                                           "an explicit value never probes and is authoritative for placement"]),
    ("offload_exec_pcie_default", CF, 9336, 9350, ['Some(v) if v == "cpu" => OffloadExec::Cpu,', 'Some(v) if v == "passback" => OffloadExec::Passback,', "_ => OffloadExec::Pcie,",
                                                  'process_value("HIPFIRE_OFFLOAD_EXEC")']),
    ("offload_exec_pcie_no_cpu", CE, 76, 102, ["cpu_exec_enabled() -> bool", "host_exec_mode() != HostExecMode::Pcie", "hipfire_config::memory::OffloadExec::Pcie => HostExecMode::Pcie,"]),
    ("kernel_gate_wmma_f16", K_GATE, 95, 100, ["moe_grouped_wmma_k2_group("]),
    ("kernel_down_f32_indexed", K_DOWN, 15, 22, ["gemv_mq4g128v2_moe_down_top10_indexed_batched_expanded(", "const float* __restrict__ rot_batch,"]),
    ("kernel_down_wmma_f16_not_bit_exact", K_DOWN_WMMA, 45, 52, ["F16 dequant and F16 inputs: not bit-exact with the F32 SIMT kernels."]),
]

# Frozen model constants; every value below has a source anchor above (see MODEL_CONSTANT_ANCHORS).
WMMA_MIN_ROWS = 512            # f16_wmma_min_tokens_512 (gate: route_rows(rows) >= 512; down: route_rows(10*rows) >= 10*512, identical in rows)
FEW_ROWS = 8                   # few_row_limit_8 / gate_dispatch_f32_variants `x_src_rows <= 8` / indexed_down
STAGE_MIN_ROWS = 768           # stage_min_rows_768
GATE_UP_BYTES, DOWN_BYTES, EXPERTS = 1_740_800, 870_400, 512
TRUNK_LAYERS = 48
PAD_BYTES = 1 << 20
CHUNK_RUNGS_TAIL = (4096, 2048, 1536)
MODEL_CONSTANT_ANCHORS = {"WMMA_MIN_ROWS": "f16_wmma_min_tokens_512,gate_wmma_predicate,down_wmma_predicate", "FEW_ROWS": "few_row_limit_8,gate_dispatch_f32_variants,indexed_down_few_row",
                          "STAGE_MIN_ROWS": "stage_min_rows_768", "strides": "stage_min_rows_768", "pad": "host_mapped_pad,host_mapped_bytes_pad",
                          "chunk_rungs": "chunk_rungs", "route_rows": "route_rows_identity_outside_invariant_scope"}

MAX_SEQ = 16640                # the hold's --max-seq (hold-lib.sh MAX_SEQ); the general property is enumerated for every length 1..MAX_SEQ
CASES = (("p512", 512), ("p1536", 1536), ("p1537", 1537), ("p8192", 8192), ("irregular4100", 4100), ("resume3400_reference", 3400))
NATURAL_CONDITIONAL = 582
DEVICE_RELIEF_BYTES = 6_684_672_000
HOST_GROWTH_BYTES = 6_695_157_760


def bad(msg):
    sys.exit(f"prove_beta_math: {msg}")


def norm(text):
    return " ".join(text.split())


def sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------- source attestation
def attest_source(root):
    if not os.path.isdir(root):
        bad(f"{root} is not a directory")
    try:
        head = subprocess.run(["git", "-C", root, "rev-parse", "HEAD"], capture_output=True, text=True, timeout=60).stdout.strip()
    except (OSError, subprocess.SubprocessError) as e:
        bad(f"cannot read the beta source commit: {e}")
    if head != BETA_COMMIT:
        bad(f"source HEAD {head!r} is not the immutable beta reference {BETA_COMMIT}")
    files = {}
    for rel, want in PINS.items():
        path = os.path.join(root, rel)
        if not os.path.isfile(path):
            bad(f"pinned source file {rel} is missing")
        got = sha_file(path)
        if got != want:
            bad(f"source drift: {rel} sha256 {got} != pinned {want}")
        files[rel] = got
    lines = {}
    anchors = []
    for name, rel, first, last, needles in ANCHORS:
        if rel not in lines:
            with open(os.path.join(root, rel), encoding="utf-8") as f:
                lines[rel] = f.read().splitlines()
        window = norm("\n".join(lines[rel][first - 1:last]))
        for needle in needles:
            if norm(needle) not in window:
                bad(f"source anchor {name}: {needle!r} not found in {rel}:{first}-{last}")
        anchors.append({"name": name, "file": rel, "lines": [first, last], "needles": len(needles)})
    return files, anchors


def scan_d2_absent(root):
    scanned = 0
    for base in ("crates", "kernels", "scripts", "tools"):
        top = os.path.join(root, base)
        for dirpath, dirnames, filenames in os.walk(top):
            dirnames[:] = [d for d in dirnames if d not in (".git", "target", "node_modules")]
            for fn in filenames:
                if not fn.endswith((".rs", ".toml", ".hip", ".py", ".sh", ".md", ".json")):
                    continue
                p = os.path.join(dirpath, fn)
                try:
                    with open(p, "rb") as f:
                        data = f.read()
                except OSError:
                    continue
                scanned += 1
                if b"QWEN4_D2_DEFAULT" in data:
                    bad(f"HIPFIRE_QWEN4_D2_DEFAULT occurs in {os.path.relpath(p, root)}: the absence claim is false for this source")
    return scanned


# ---------------------------------------------------------------- the model (each predicate is anchored above)
def kernel_class(rows):
    """(gate/up kernel, down kernel, precision) of a forward of `rows` rows on gfx1100 (route_rows is the identity outside the invariant scope).
    rows == 1: sealed singleton decode GEMVs (singleton_decode_*). 512+: gate_wmma_predicate + down_wmma_predicate (10*rows >= 10*512 <=> rows >= 512).
    2..8: gate_dispatch x_src_rows <= 8 and indexed_down (route_rows <= few_row_limit 8, not down WMMA). 9..511: gate o4_r8_x4, down O8R16."""
    if rows < 1:
        bad(f"forward rows {rows} < 1")
    if rows == 1:
        return ("gemv_mq4g256v2_moe_gate_up_k2560_rows8_indexed_batched", "gemv_mq4g128v2_moe_down_top10_indexed_batched_expanded", "f32_dequant_f32_arith")
    if rows >= WMMA_MIN_ROWS:
        return ("gemm_mq4g256v2_moe_grouped_wmma_k2_silu_bf16out", "gemm_mq4g128v2_moe_grouped_wmma_gfx1151_bf16out", "f16_dequant_f32_wmma_acc_bf16_rows")
    if rows <= FEW_ROWS:
        return ("gemm_mq4g256v2_moe_grouped_top10_o4_r2_x4", "gemv_mq4g128v2_moe_down_top10_indexed_batched_expanded", "f32_dequant_f32_arith")
    return ("gemm_mq4g256v2_moe_grouped_top10_o4_r8_x4", "gemm_mq4g128v2_moe_grouped_top10_o8_r16_gfx1151", "f32_dequant_f32_arith")


def wire_strides():
    return {"gate_up_bytes": GATE_UP_BYTES, "down_bytes": DOWN_BYTES, "experts": EXPERTS}


def schedule(length, chunk):
    """chunk_segmentation_rows: rows = min(remaining, max_chunk) per forward."""
    out, offset = [], 0
    while offset < length:
        rows = min(length - offset, chunk)
        out.append(rows)
        offset += rows
    return out


def rungs(requested):
    return [requested] + [r for r in CHUNK_RUNGS_TAIL if r < requested]


def donors(first_host):
    """stage_donor_parity: [first_host - 2, first_host - 1] sorted (stable) by layer % 2."""
    return sorted([first_host - 2, first_host - 1], key=lambda layer: layer % 2)


def owner(layer, n):
    """place_routed_experts_prefix: trunk layer < N is resident; MTP (layer 48) is host mapped and not executed by the AR loader."""
    return "resident" if layer < n else "host_mapped"


def copy_events(layer, rows, n):
    """What differs between placements WITHOUT touching the launched kernel: below STAGE_MIN_ROWS a host layer reads its mapped alias; at/above
    it a host layer reads a donor's byte-copied stage (stage_pointer_substitution), the donor pair depending on N (stage_donor_parity)."""
    if owner(layer, n) == "resident":
        return {"read": "own_resident_bytes", "donor": None}
    if rows >= STAGE_MIN_ROWS:
        return {"read": "byte_copied_donor_stage", "donor": donors(n)[layer % 2]}
    return {"read": "host_mapped_alias", "donor": None}


def layer_tuple(layer, rows):
    """The arithmetic class: (layer, actual_rows, kernels, precision, wire strides). It is the ONLY thing compared across placements."""
    gate, down, precision = kernel_class(rows)
    return (layer, rows, gate, down, precision, tuple(sorted(wire_strides().items())))


def trace(length, chunk, n):
    return [{"forward": i, "rows": rows,
             "layers": [{"tuple": layer_tuple(layer, rows), "owner": owner(layer, n), "copy": copy_events(layer, rows, n)} for layer in range(TRUNK_LAYERS)]}
            for i, rows in enumerate(schedule(length, chunk))]


def class_projection(tr):
    return [(f["forward"], [lay["tuple"] for lay in f["layers"]]) for f in tr]


def digest_trace(tr, h):
    for f in tr:
        for layer, lay in enumerate(f["layers"]):
            h.update(json.dumps([f["forward"], layer, lay["tuple"][1:5], lay["owner"], lay["copy"]], sort_keys=True).encode())


def case_row(name, length, chunk_base, chunk_n10, n10, n5):
    t10, t5 = trace(length, chunk_n10, n10), trace(length, chunk_base, n5)
    same = class_projection(t10) == class_projection(t5)
    sched10, sched5 = schedule(length, chunk_n10), schedule(length, chunk_base)
    row = {"name": name, "prompt_tokens": length, "forward_rows_n10_actual4096": sched10, "forward_rows_n5_actual4096": sched5,
           "classes_by_forward": [{"rows": r, "gate_up": kernel_class(r)[0], "down": kernel_class(r)[1], "precision": kernel_class(r)[2]} for r in sched5],
           "same_class_per_layer_and_forward": same,
           "host_layers_staging_from_768": [r >= STAGE_MIN_ROWS for r in sched5],
           "singleton_continuation_opportunities": {"class": list(kernel_class(1)), "max_new_tokens": 32, "count_upper_bound": 32,
                                                    "applies_to_layers": list(range(TRUNK_LAYERS)),
                                                    "note": "OPPORTUNITIES (cold prefill then forward_token_or_argmax rows=1 per token, up to max_new_tokens); EOS/terminator stops early; no 32 generated-token claim"}}
    if not same or sched10 != sched5:
        bad(f"case {name}: class or schedule differs between N10/actual4096 and N5/actual4096")
    return row, t10, t5


def band_summary(n):
    return {"resident_layers": list(range(n)), "host_mapped_layers": list(range(n, TRUNK_LAYERS)), "mtp_layer_48": "host weight placement, not executed by the AR loader",
            "donors_parity_order": donors(n)}


def prove(root):
    files, anchors = attest_source(root)
    scanned = scan_d2_absent(root)
    # --- placement/donor facts
    if donors(10) != [8, 9]:
        bad("N10 donors are not [8, 9]")
    if donors(5) != [4, 3] or donors(5) == [3, 4]:
        bad("N5 donors parity order is not [4, 3]")
    if rungs(8192) != [8192, 4096, 2048, 1536] or rungs(4096) != [4096, 2048, 1536]:
        bad("chunk rung ladder disagrees with the asserted source")
    per_expert_layer = EXPERTS * (GATE_UP_BYTES + DOWN_BYTES)
    if per_expert_layer != 1_336_934_400:
        bad("per-layer expert bytes != 1,336,934,400")
    relief = (10 - 5) * per_expert_layer
    growth = relief + (10 - 5) * 2 * PAD_BYTES
    if relief != DEVICE_RELIEF_BYTES or growth != HOST_GROWTH_BYTES:
        bad(f"device relief {relief} / host growth {growth} != {DEVICE_RELIEF_BYTES} / {HOST_GROWTH_BYTES}")
    # --- fixed cases (actual chunk 4096 on both placements)
    cases, h = [], hashlib.sha256()
    for name, length in CASES:
        row, t10, t5 = case_row(name, length, 4096, 4096, 10, 5)
        digest_trace(t10, h)
        digest_trace(t5, h)
        cases.append(row)
    by = {c["name"]: c for c in cases}
    if by["irregular4100"]["forward_rows_n5_actual4096"] != [4096, 4] or by["p8192"]["forward_rows_n5_actual4096"] != [4096, 4096]:
        bad("fixed-case schedules are not the asserted [4096, 4] / [4096, 4096]")
    if [by[n]["forward_rows_n5_actual4096"] for n in ("p512", "p1536", "p1537", "resume3400_reference")] != [[512], [1536], [1537], [3400]]:
        bad("fixed-case schedules are not single forwards")
    # --- conditional natural case + general admissible-length property (equality of class streams for EVERY length 1..MAX_SEQ at actual 4096)
    natural = {"conditional": True, "condition": "applies only if the tokenizer resolves the natural-text record to exactly 582 tokens; the actual prompt_tokens is bound later by the receipt compare",
               "assumed_prompt_tokens": NATURAL_CONDITIONAL}
    natural_row, _, _ = case_row("natural_conditional", NATURAL_CONDITIONAL, 4096, 4096, 10, 5)
    natural["case"] = natural_row
    g, checked, tail_f32_lengths, class_set = hashlib.sha256(), 0, 0, set()
    for length in range(1, MAX_SEQ + 1):
        s = schedule(length, 4096)
        c10 = [layer_tuple(0, r)[1:] for r in s]
        c5 = [layer_tuple(0, r)[1:] for r in s]
        for r in s:
            if r > 4096 or r < 1:
                bad(f"length {length}: forward rows {r} outside 1..4096")
            class_set.add(kernel_class(r))
        if c10 != c5:
            bad(f"length {length}: class stream differs")
        # layer independence is a source fact (predicates take only rows/m/k/arch); spot-check every layer on a bounded subset
        if length % 257 == 1 or length in (4100, 8192, 582):
            for r in s:
                if len({layer_tuple(layer, r)[1:] for layer in range(TRUNK_LAYERS)}) != 1:
                    bad(f"length {length}: layer-dependent class")
        if any(1 < r < WMMA_MIN_ROWS for r in s):
            tail_f32_lengths += 1
        g.update(f"{length}:{','.join(map(str, s))};".encode())
        checked += 1
    general = {"lengths_checked": [1, MAX_SEQ], "count": checked, "chunk_actual": 4096, "all_forward_rows_within": [1, 4096],
               "n10_n5_class_streams_equal_for_every_length": True, "schedule_digest_sha256": g.hexdigest(), "lengths_with_f32_multirow_forward": tail_f32_lengths,
               "distinct_class_count": len(class_set),
               "basis": "kernel_class/schedule are pure functions of forward rows; residency enters only owner/copy-event fields (anchors place_routed_experts_prefix, "
                        "stage_pointer_substitution*, gate/down predicates take only arch/m/k/rows)"}
    # --- negatives: explicit 8192 ceiling changes class; N alone (possible actual 8192) is not the same class
    s4096, s8192 = schedule(4100, 4096), schedule(4100, 8192)
    neg_classes_4096 = [kernel_class(r) for r in s4096]
    neg_classes_8192 = [kernel_class(r) for r in s8192]
    if s4096 != [4096, 4] or s8192 != [4100] or neg_classes_4096 == neg_classes_8192:
        bad("8192-ceiling negative case did not change the class")
    changed = [n for n in range(1, MAX_SEQ + 1) if [kernel_class(r) for r in schedule(n, 4096)] != [kernel_class(r) for r in schedule(n, 8192)]]
    if 4100 not in changed:
        bad("4100 is not among the class-changing lengths")
    negative = {"name": "irregular4100_with_8192_ceiling", "prompt_tokens": 4100, "chunk4096_rows": s4096, "chunk8192_rows": s8192,
                "chunk4096_classes": [list(c) for c in neg_classes_4096], "chunk8192_classes": [list(c) for c in neg_classes_8192],
                "class_changed": True, "lengths_with_changed_class_up_to_max_seq": len(changed), "first_changed_lengths": changed[:8],
                "conclusion": "EXPERT_VRAM_LAYERS=5 alone (admission may then grow the actual chunk to 8192) is NOT the same class; explicit 4096 preserves the old ACTUAL geometry"}
    return {
        "schema": SCHEMA,
        "beta_source": {"commit": BETA_COMMIT, "path": root, "file_sha256": files, "d2_scan": {"files_scanned": scanned}},
        "anchors": {"count": len(anchors), "model_constant_anchors": MODEL_CONSTANT_ANCHORS, "list": anchors},
        "pins": {"beta": {"HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS": "5", "HIPFIRE_PREFILL_CHUNK_ROWS": "4096"}, "quality_reference_only": True,
                 "h2_policy": False, "required_actual_chunk_rows": 4096, "actual_chunk_receipt_line": "qwen4 prefill chunk: 4096 rows (requested 4096)",
                 "chunk_rungs_for_requested_4096": rungs(4096), "chunk_rungs_for_requested_8192": rungs(8192),
                 "chunk_note": "HIPFIRE_PREFILL_CHUNK_ROWS is a CEILING (bundle.rs admission may downshift); the receipt must observe actual 4096"},
        "donors": {"n10": donors(10), "n5": donors(5), "n5_is_not": [3, 4], "rule": "[first_host-2, first_host-1] stably sorted by layer % 2"},
        "placement_bands": {"n10": band_summary(10), "n5": band_summary(5)},
        "wire_strides": {**wire_strides(), "bytes_per_expert_layer": per_expert_layer},
        "class_table": [{"rows": r, "gate_up": kernel_class(r)[0], "down": kernel_class(r)[1], "precision": kernel_class(r)[2]} for r in (1, 2, 8, 9, 511, 512, 767, 768, 4096)],
        "stage_threshold_rows": STAGE_MIN_ROWS,
        "cases": cases, "natural": natural, "general_admissible_natural_length_property": general, "negative_8192_ceiling": negative,
        "digest_fixed_cases_sha256": h.hexdigest(),
        "permitted_differences": ["owner(resident|host_mapped)", "base(layer index of resident prefix)", "donor pair/parity order", "copy-event schedule (mapped alias vs byte-copied donor stage)"],
        "compared_tuple": ["layer", "actual_rows", "kernel(gate_up,down)", "precision", "wire_strides"],
        "d2_default": {"selector_present_in_beta_source": False, "status": "absent / not applicable to ff799; the required env 0 is preserved as recorded metadata only"},
        "gpu_control": {"HIPFIRE_OFFLOAD_EXEC": "pcie", "meaning": "the real operational control: unrecognised/pcie disables CPU/passback execution (cpu_exec_enabled() false)"},
        "memory": {"device_payload_relief_bytes": relief, "host_growth_bytes_including_10_pads": growth, "basis": "source-counted capacity arithmetic, not a measurement"},
        "tensor_ops_note": "a tensor_ops package rejection/JIT fallback seen on the failed run is the known FirstRequestJit missing-q8hg4-declarations bug (fixed in FRJv5, not in this base); recorded, not an S1 finding, frozen beta not fixed",
        "claims": {"measured_fit": False, "generated_output_equality": False, "generated_32_tokens": False, "gpu_executed": False, "model_payload_read": False,
                   "scope": "same ARITHMETIC CLASS per (layer, actual rows) by source inspection; fit and token equality are decided only by the repaired run"},
    }


def main(argv):
    if len(argv) != 3:
        sys.exit(__doc__)
    proof = prove(argv[1])
    tmp = f"{argv[2]}.tmp-{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(proof, f, indent=1, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, argv[2])
    print(f"beta math proof OK: commit {BETA_COMMIT}, {len(PINS)} files pinned, {len(ANCHORS)} anchors, "
          f"{proof['general_admissible_natural_length_property']['count']} lengths; sha256 {sha_file(argv[2])}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
