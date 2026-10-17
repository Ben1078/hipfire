// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Developer hooks of the wide-verify oracle (`examples/qwen4_wide_verify`).
//!
//! The oracle compares the PRODUCTION verify transaction
//! (`Qwen4Bundle::verify_block_rows` through `SpecTarget::verify_block`:
//! snapshot, row-armed `spec_forward_rows` with `prefill_scope = false`,
//! `commit_prefix`) against sequential `forward_token` decode. These hooks
//! only expose that transaction, its resident logits and the live state's
//! digests to the example; no production route calls them, and the module is
//! compiled only with `reference-parity`.

use crate::bundle::Qwen4Bundle;
use crate::cache_quality::cq_state_families;
use crate::mtp_spec::native_mtp_row_capture;
use hipfire_runtime::session_cache::SessionRoute;
use hipfire_runtime::spec::SpecScratch;
use rdna_compute::Gpu;

/// Make the GDN row-capture ring hold at least `rows` rows, so
/// `verify_block_rows` arms capture for blocks of up to `rows` rows (its rule:
/// `block.len() >= 2 && row_capture_rows() >= block.len()`).
///
/// Mirrors the product drafter's `ensure_resources`: the ring is allocated
/// only where `native_mtp_row_capture` says the GDN route captures rows. On
/// any other GPU this is a no-op and [`dev_row_capture_rows`] stays 0 (the
/// verify then runs unarmed, exactly as the product does there).
pub fn dev_arm_row_capture(
    bundle: &mut Qwen4Bundle,
    gpu: &mut Gpu,
    rows: usize,
) -> Result<(), String> {
    if rows < 2 || !native_mtp_row_capture(gpu, &bundle.config) {
        return Ok(());
    }
    let conv_history = bundle
        .config
        .linear_conv_kernel_dim
        .checked_sub(1)
        .ok_or("linear_conv_kernel_dim is 0")?;
    bundle
        .state
        .ensure_row_capture(gpu, rows, conv_history)
        .map_err(|error| error.to_string())
}

/// Whether this GPU/config captures verify rows at all (the product drafter's
/// `native_mtp_row_capture` gate).
pub fn dev_row_capture_supported(bundle: &Qwen4Bundle, gpu: &Gpu) -> bool {
    native_mtp_row_capture(gpu, &bundle.config)
}

/// Rows the GDN row-capture ring currently holds (0 = none allocated).
pub fn dev_row_capture_rows(bundle: &Qwen4Bundle) -> usize {
    bundle.state.row_capture_rows()
}

/// Whether `verify_block_rows` arms row capture for an `n`-row block now: its
/// own arming rule, evaluated on the live state.
pub fn dev_would_arm(bundle: &Qwen4Bundle, n: usize) -> bool {
    n >= 2 && bundle.state.row_capture_rows() >= n
}

/// Live target position (tokens committed to the resident state).
pub fn dev_position(bundle: &Qwen4Bundle) -> usize {
    bundle.state.position
}

/// Run the production verify transaction over `block` at the live position
/// (picks downloaded, hidden captured) and download every row's logits:
/// returns `(picks, logits)` with `logits` row-major `block.len() * vocab`.
/// The state is left advanced by `block.len()` with the verify's snapshot
/// ticket active in `scratch`; finish with `SpecTarget::commit_prefix` or
/// [`dev_rollback_verify`].
/// A logits synchronization, view or download failure rolls the verify back
/// and consumes its ticket; a rollback failure is reported with the read error.
pub fn dev_verify_logits(
    bundle: &mut Qwen4Bundle,
    gpu: &mut Gpu,
    block: &[u32],
    scratch: &mut dyn SpecScratch,
) -> Result<(Vec<u32>, Vec<f32>), String> {
    let position = bundle.state.position;
    let picks = bundle.dev_verify_block_rows(gpu, block, position, scratch, true)?;
    let result = (|| -> Result<Vec<f32>, String> {
        gpu.hip
            .device_synchronize()
            .map_err(|error| error.to_string())?;
        let rows = bundle
            .spec_logits_rows(0, block.len())
            .map_err(|error| error.to_string())?;
        gpu.download_f32(&rows).map_err(|error| error.to_string())
    })();
    match result {
        Ok(logits) => Ok((picks, logits)),
        Err(error) => {
            bundle
                .dev_rollback_verify(gpu, scratch)
                .map_err(|rollback| format!("{error}; target rollback failed: {rollback}"))?;
            Err(error)
        }
    }
}

/// The same verify transaction with the logits left resident and no pick
/// readback (`verify_block_resident`'s route): the timed variant.
pub fn dev_verify_resident(
    bundle: &mut Qwen4Bundle,
    gpu: &mut Gpu,
    block: &[u32],
    scratch: &mut dyn SpecScratch,
) -> Result<(), String> {
    let position = bundle.state.position;
    bundle
        .dev_verify_block_rows(gpu, block, position, scratch, false)
        .map(|_| ())
}

/// Undo a finished verify without committing: restore the verify's snapshot
/// ticket (the state it started from) and consume it.
pub fn dev_rollback_verify(
    bundle: &mut Qwen4Bundle,
    gpu: &mut Gpu,
    scratch: &mut dyn SpecScratch,
) -> Result<(), String> {
    bundle.dev_rollback_verify(gpu, scratch)
}

/// SHA-256 hex of every part of the live target state at the live position,
/// as `cache_quality::cq_state_families` computes them
/// (`SessionState::snapshot_parts` on the AR route: part 0 is the metadata,
/// then every fixed part, then the valid rows of every row stream). Keys are
/// `part.<i>`, in part order.
pub fn dev_state_digest(
    bundle: &mut Qwen4Bundle,
    gpu: &mut Gpu,
) -> Result<Vec<(String, String)>, String> {
    let position = bundle.state.position;
    let families = cq_state_families(gpu, bundle, SessionRoute::Ar, position)?;
    let mut parts: Vec<(usize, String, String)> = families
        .into_iter()
        .map(|(key, family)| {
            let index = key
                .strip_prefix("part.")
                .and_then(|index| index.parse::<usize>().ok())
                .ok_or_else(|| format!("unexpected state part key '{key}'"))?;
            Ok((index, key, family.digest))
        })
        .collect::<Result<_, String>>()?;
    parts.sort_by_key(|(index, _, _)| *index);
    Ok(parts
        .into_iter()
        .map(|(_, key, digest)| (key, digest))
        .collect())
}
