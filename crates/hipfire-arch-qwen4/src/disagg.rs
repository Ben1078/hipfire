// SPDX-License-Identifier: Apache-2.0
//! Explicit, eager developer/lab Flash-Next sequence handoff (never routed).
//!
//! A producer bundle (gfx1151) exports the committed prompt state into one
//! portable pinned host buffer ([`HandoffState`]); a fresh consumer bundle
//! (gfx1100) imports it and decodes from `first_token`, the producer's greedy
//! sample, without re-running the last prompt token.
//!
//! Payload layout (no framing, no alignment): target fixed parts, target row
//! streams (rows below the marks), MTP fixed parts, MTP row streams, each in
//! the order of the existing session layout (`Qwen4State::session_parts`,
//! `Qwen4MtpGpu::session_parts`). The metadata is the existing session meta
//! (target marks/position/PLE history, MTP marks/position/policy). The
//! manifest digest is SHA-256 over a domain tag, `first_token`, `position`,
//! both metas, every part length and the payload.
//!
//! Identity is bound once at load (`Qwen4Bundle::bind_disagg`): the bundle's
//! role and the complete producer/consumer [`HandoffDomain`]. Export and
//! import take a domain that must equal the bound one, and the bound local
//! half must name this `Gpu`'s architecture. Derived device-only caches (F16
//! K/V arena, pooled BF16 mirror) are never wire state and are invalidated by
//! an import; the QSA partial buffers are uncommitted scratch outside the
//! session layout, so a mark carrying live partial rows is refused.
use std::fmt;

use hip_bridge::{DeviceBuffer, PinnedHostBuffer};
use hipfire_runtime::session_cache::StateLayout;
use rdna_compute::replay::ReplayState;
use rdna_compute::Gpu;
use sha2::{Digest, Sha256};

use crate::bundle::Qwen4Bundle;
use crate::config::{LayerType, Qwen4Config};
use crate::disagg_manifest::{
    flash_next_layer_layout, ExpectedHandoff, ExpectedMtp, HandoffDomain, HandoffManifest,
    StateFormats, CONSUMER_ARCH, FLASH_NEXT_LAYERS, HANDOFF_SCHEMA_VERSION, PRODUCER_ARCH,
};
use crate::state::Qwen4StateFormat;

const DIGEST_DOMAIN: &[u8] = b"hipfire-flash-next-handoff-v1\0";

/// Which side of the bound [`HandoffDomain`] a bundle is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DisaggRole {
    Producer,
    Consumer,
}

/// Immutable role and domain bound by the loader (`Qwen4Bundle::bind_disagg`).
#[derive(Clone, Debug)]
pub struct HandoffBinding {
    pub(crate) role: DisaggRole,
    pub(crate) domain: HandoffDomain,
    /// [`disagg_domain_id`] of `domain`.
    pub(crate) id: String,
}

/// An exported or received prompt state: the frozen manifest plus the packed
/// payload in portable pinned host memory (usable from any device).
pub struct HandoffState {
    pub manifest: HandoffManifest,
    payload: PinnedHostBuffer,
}

// SAFETY: the payload is initialized pinned host memory that is never mutated
// after construction (`as_bytes` is the only access); `hipHostMalloc`
// allocations are process-wide and freed from any thread (the owner keeps its
// library loaded). `manifest` is plain data.
unsafe impl Send for HandoffState {}
unsafe impl Sync for HandoffState {}

impl fmt::Debug for HandoffState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HandoffState")
            .field("position", &self.manifest.position)
            .field("first_token", &self.manifest.first_token)
            .field("payload_bytes", &self.payload.size())
            .finish()
    }
}

impl HandoffState {
    /// The packed payload bytes.
    pub fn as_bytes(&self) -> &[u8] {
        self.payload.as_bytes()
    }

    /// Own `bytes` (a saved payload) as pinned portable host memory after
    /// checking its length and SHA-256 against `manifest`. No device state is
    /// touched; the domain is checked by `import_handoff`.
    pub fn from_bytes(
        gpu: &mut Gpu,
        manifest: HandoffManifest,
        bytes: &[u8],
    ) -> Result<Self, String> {
        if manifest.version != HANDOFF_SCHEMA_VERSION {
            return Err(format!(
                "handoff schema version {} unsupported (expected {HANDOFF_SCHEMA_VERSION})",
                manifest.version
            ));
        }
        let expected = manifest.payload_bytes()?;
        if expected == 0 || expected != bytes.len() {
            return Err(format!(
                "handoff payload length {} does not match manifest {expected}",
                bytes.len()
            ));
        }
        let digest = handoff_digest(&manifest, bytes);
        if digest != manifest.payload_digest {
            return Err("handoff payload digest mismatch".to_string());
        }
        gpu.bind_thread().map_err(|e| format!("bind device: {e}"))?;
        let mut payload = gpu
            .hip
            .host_buffer(bytes.len())
            .map_err(|e| format!("pinned handoff buffer: {e}"))?;
        payload.as_bytes_mut().copy_from_slice(bytes);
        Ok(Self { manifest, payload })
    }
}

/// `disagg-v1-<sha256 hex of the domain's canonical JSON>`: the scope token (no
/// slash) a consumer's session cache carries after an import.
pub fn disagg_domain_id(domain: &HandoffDomain) -> String {
    let json = serde_json::to_vec(domain).expect("handoff domain serializes");
    format!("disagg-v1-{}", hex(&Sha256::digest(&json)))
}

/// The wire state formats of `format`: K/V, GDN recurrent and index-key
/// storage. The device-only F16 arena and pooled BF16 mirror are excluded.
pub fn handoff_formats(format: &Qwen4StateFormat) -> StateFormats {
    StateFormats {
        kv: format!("{:?}", format.qsa),
        gdn: format!("{:?}", format.gdn),
        qsa_index: format!("{:?}", format.qsa.index_dtype()),
    }
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn handoff_digest(manifest: &HandoffManifest, payload: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(DIGEST_DOMAIN);
    hasher.update(manifest.first_token.to_le_bytes());
    hasher.update((manifest.position as u64).to_le_bytes());
    for meta in [&manifest.target_meta, &manifest.mtp_meta] {
        hasher.update((meta.len() as u64).to_le_bytes());
        hasher.update(meta);
    }
    for fixed in [&manifest.fixed_bytes, &manifest.mtp_fixed_bytes] {
        hasher.update((fixed.len() as u64).to_le_bytes());
        for &bytes in fixed {
            hasher.update((bytes as u64).to_le_bytes());
        }
    }
    for rows in [&manifest.row_bytes, &manifest.mtp_row_bytes] {
        hasher.update((rows.len() as u64).to_le_bytes());
        for &(row_bytes, count) in rows {
            hasher.update((row_bytes as u64).to_le_bytes());
            hasher.update((count as u64).to_le_bytes());
        }
    }
    hasher.update((payload.len() as u64).to_le_bytes());
    hasher.update(payload);
    hex(&hasher.finalize())
}

/// One contiguous device byte range of the payload, in payload order.
struct Chunk<'a> {
    buf: &'a DeviceBuffer,
    offset: usize,
    bytes: usize,
}

fn push_fixed<'a>(out: &mut Vec<Chunk<'a>>, layout: &StateLayout<'a>) {
    out.extend(layout.fixed.iter().map(|part| Chunk {
        buf: part.buf,
        offset: part.offset,
        bytes: part.bytes,
    }));
}

fn push_rows<'a>(out: &mut Vec<Chunk<'a>>, layout: &StateLayout<'a>) -> Result<(), String> {
    for stream in &layout.rows {
        let bytes = stream
            .row_bytes
            .checked_mul(stream.rows)
            .ok_or("handoff row stream overflows")?;
        out.push(Chunk { buf: stream.buf, offset: 0, bytes });
    }
    Ok(())
}

/// Payload order: target fixed, target rows, MTP fixed, MTP rows. Returns the
/// chunks and the total byte count, with every range inside its buffer.
fn plan_chunks<'a>(
    target: &StateLayout<'a>,
    mtp: &StateLayout<'a>,
) -> Result<(Vec<Chunk<'a>>, usize), String> {
    let mut chunks = Vec::with_capacity(
        target.fixed.len() + target.rows.len() + mtp.fixed.len() + mtp.rows.len(),
    );
    push_fixed(&mut chunks, target);
    push_rows(&mut chunks, target)?;
    push_fixed(&mut chunks, mtp);
    push_rows(&mut chunks, mtp)?;
    let mut total = 0usize;
    for chunk in &chunks {
        let end = chunk
            .offset
            .checked_add(chunk.bytes)
            .ok_or("handoff range overflows")?;
        if end > chunk.buf.size() {
            return Err(format!(
                "handoff range [{}, {end}) exceeds its device buffer ({} bytes)",
                chunk.offset,
                chunk.buf.size()
            ));
        }
        total = total.checked_add(chunk.bytes).ok_or("handoff payload overflows")?;
    }
    Ok((chunks, total))
}

/// `(fixed part bytes, (row bytes, rows) per stream)` of a layout.
fn describe(layout: &StateLayout<'_>) -> (Vec<usize>, Vec<(usize, usize)>) {
    (
        layout.fixed.iter().map(|part| part.bytes).collect(),
        layout.rows.iter().map(|stream| (stream.row_bytes, stream.rows)).collect(),
    )
}

/// `Err` text when the HIP device cannot be proven drained; the Gpu is then
/// quarantined (it issues no further work).
fn drain_or_quarantine(gpu: &mut Gpu, what: &str) -> Option<String> {
    gpu.hip.device_synchronize().err().map(|error| {
        let message = format!(
            "{what}: device could not be drained ({error}); the Gpu is quarantined and its \
             allocations are retained"
        );
        gpu.quarantine(&message);
        message
    })
}

fn check_identity(side: &str, id: &crate::disagg_manifest::NumericalManifest) -> Result<(), String> {
    if id.route_identity.is_empty() || id.artifact_digest.is_empty() || id.policy_digest.is_empty() {
        return Err(format!("{side} numerical identity has an empty field"));
    }
    Ok(())
}

fn flash_next_config(config: &Qwen4Config) -> bool {
    config.layer_types.len() == FLASH_NEXT_LAYERS
        && config.layer_types.iter().enumerate().all(|(i, kind)| {
            matches!(
                (i % 4 == 3, kind),
                (true, LayerType::FullAttention) | (false, LayerType::LinearAttention)
            )
        })
}

impl Qwen4Bundle {
    /// Bind this bundle's immutable handoff role and domain (loader, after
    /// the forward and the native MTP head attach). A second call fails even
    /// with identical arguments. The bundle's own half of `domain` is checked
    /// against the `Gpu` architecture at every export/import.
    pub fn bind_disagg(
        &mut self,
        role: DisaggRole,
        domain: &HandoffDomain,
    ) -> Result<(), String> {
        if self.handoff_binding.is_some() {
            return Err("handoff domain is already bound".to_string());
        }
        check_identity("producer", &domain.producer)?;
        check_identity("consumer", &domain.consumer)?;
        if domain.producer.artifact_digest != domain.consumer.artifact_digest {
            return Err("producer and consumer artifact digests differ".to_string());
        }
        if domain.producer.arch != PRODUCER_ARCH || domain.consumer.arch != CONSUMER_ARCH {
            return Err(format!(
                "unsupported arch pair {}->{} (only {PRODUCER_ARCH}->{CONSUMER_ARCH})",
                domain.producer.arch, domain.consumer.arch
            ));
        }
        if self.ep.is_some() {
            return Err("handoff refuses an expert-parallel bundle".to_string());
        }
        if !self.state.is_whole_model() {
            return Err("handoff refuses a pipeline-band state".to_string());
        }
        if !flash_next_config(&self.config) {
            return Err("handoff needs the Flash-Next [gdn,gdn,gdn,qsa] x 12 layout".to_string());
        }
        if self.mtp.is_none() {
            return Err("handoff needs the native MTP head attached".to_string());
        }
        self.handoff_binding = Some(HandoffBinding {
            role,
            id: disagg_domain_id(domain),
            domain: domain.clone(),
        });
        Ok(())
    }

    /// `disagg-v1-<hex>` id of the domain whose imported state the live state
    /// descends from (`None` after a cold reset or without an import).
    pub fn disagg_domain(&self) -> Option<&str> {
        self.disagg_domain.as_deref()
    }

    /// Everything that must hold before an export or import touches state:
    /// bound role and an equal domain, this `Gpu`'s arch, whole-model single
    /// device state, no row capture / rollback ticket / pair stage, native MTP
    /// present, and no graph capture or retained replay body. Returns the
    /// role and bound domain id.
    fn handoff_gate(
        &self,
        gpu: &Gpu,
        domain: &HandoffDomain,
    ) -> Result<(DisaggRole, String), String> {
        let binding = self
            .handoff_binding
            .as_ref()
            .ok_or("no handoff domain is bound to this bundle")?;
        if binding.domain != *domain {
            return Err("handoff domain does not equal the bound domain".to_string());
        }
        let local = match binding.role {
            DisaggRole::Producer => &binding.domain.producer,
            DisaggRole::Consumer => &binding.domain.consumer,
        };
        if gpu.arch != local.arch {
            return Err(format!(
                "bound {:?} identity is {} but this Gpu is {}",
                binding.role, local.arch, gpu.arch
            ));
        }
        if self.ep.is_some() {
            return Err("handoff refuses an expert-parallel bundle".to_string());
        }
        if !self.state.is_whole_model() {
            return Err("handoff refuses a pipeline-band state".to_string());
        }
        if self.state.row_capture_armed || self.state.snapshot_active() {
            return Err("handoff refuses an armed verify row capture or rollback ticket".to_string());
        }
        if self.pair_session_stage.as_ref().is_some_and(|stage| stage.position.is_some()) {
            return Err("handoff refuses an armed pair session stage".to_string());
        }
        if self.mtp.is_none() {
            return Err("handoff needs the native MTP head".to_string());
        }
        if gpu.graphs.capture_mode
            || gpu.replay.is_recording()
            || gpu.replay.shadow_body_route().is_some()
            || matches!(
                gpu.replay.state(),
                ReplayState::Captured | ReplayState::ShadowValidated | ReplayState::Ready
            )
        {
            return Err(
                "handoff refuses graph capture or a retained replay body (use replay.backend=hip \
                 or a bundle that has not prepared one)"
                    .to_string(),
            );
        }
        if let Some(stream) = gpu.active_stream.as_ref() {
            if gpu
                .hip
                .stream_is_capturing(stream)
                .map_err(|e| format!("stream capture query: {e}"))?
            {
                return Err("handoff refuses a capturing stream".to_string());
            }
        }
        Ok((binding.role, binding.id.clone()))
    }

    /// Export the committed state at the current position, with the producer's
    /// externally sampled greedy `first_token`, into pinned host memory. Read
    /// only: the bundle's state is untouched. A consumer may export only right
    /// after importing this same bound `domain` (the round-trip compare); the
    /// manifest keeps the bound producer/consumer metadata.
    pub fn export_handoff(
        &mut self,
        gpu: &mut Gpu,
        domain: &HandoffDomain,
        first_token: u32,
    ) -> Result<HandoffState, String> {
        let (role, id) = self.handoff_gate(gpu, domain)?;
        match role {
            DisaggRole::Producer if self.disagg_domain.is_some() => {
                return Err("a producer bundle cannot hold imported state".to_string());
            }
            DisaggRole::Consumer if self.disagg_domain.as_deref() != Some(id.as_str()) => {
                return Err(
                    "a consumer exports only state imported from the bound domain".to_string()
                );
            }
            _ => {}
        }
        let vocab = self.config.vocab_size;
        if first_token as usize >= vocab {
            return Err(format!("first_token {first_token} is outside the vocabulary ({vocab})"));
        }
        let position = self.state.position;
        if position == 0 {
            return Err("handoff export needs a prefilled state (position 0)".to_string());
        }
        self.quiesce_ple().map_err(|e| format!("quiesce PLE: {e}"))?;
        gpu.bind_thread().map_err(|e| format!("bind device: {e}"))?;
        gpu.hip.device_synchronize().map_err(|e| format!("synchronize before export: {e}"))?;
        let mtp = self.mtp.as_ref().ok_or("handoff needs the native MTP head")?;
        if mtp.position() != position {
            return Err(format!(
                "MTP head is at {} but the target is at {position} (cold or lagging head)",
                mtp.position()
            ));
        }
        let (target_meta, target_layout) =
            self.state.session_parts().map_err(|e| format!("target session parts: {e}"))?;
        let (mtp_meta, mtp_layout) =
            mtp.session_parts().map_err(|e| format!("MTP session parts: {e}"))?;
        self.state
            .validate_handoff_meta(&target_meta, position, vocab)
            .map_err(|e| format!("target handoff state: {e}"))?;
        mtp.handoff_layout_for_meta(&mtp_meta, position)
            .map_err(|e| format!("MTP handoff state: {e}"))?;
        let (chunks, total) = plan_chunks(&target_layout, &mtp_layout)?;
        if total == 0 {
            return Err("handoff payload is empty".to_string());
        }
        let mut payload = gpu
            .hip
            .host_buffer(total)
            .map_err(|e| format!("pinned handoff buffer ({total} bytes): {e}"))?;
        let mut cursor = 0usize;
        for chunk in &chunks {
            if chunk.bytes == 0 {
                continue;
            }
            let dst = &mut payload.as_bytes_mut()[cursor..cursor + chunk.bytes];
            if let Err(error) = gpu.hip.memcpy_dtoh_at(dst, chunk.buf, chunk.offset) {
                let message = format!("handoff d2h copy failed: {error}");
                return Err(drain_or_quarantine(gpu, &message).unwrap_or(message));
            }
            cursor += chunk.bytes;
        }
        if let Some(message) = drain_or_quarantine(gpu, "handoff export") {
            return Err(message);
        }
        let (fixed_bytes, row_bytes) = describe(&target_layout);
        let (mtp_fixed_bytes, mtp_row_bytes) = describe(&mtp_layout);
        let mut manifest = HandoffManifest {
            version: HANDOFF_SCHEMA_VERSION,
            domain: domain.clone(),
            layer_layout: flash_next_layer_layout(),
            formats: handoff_formats(&self.state_format()),
            position,
            target_meta,
            mtp_meta,
            fixed_bytes,
            row_bytes,
            mtp_fixed_bytes,
            mtp_row_bytes,
            first_token,
            payload_digest: String::new(),
        };
        manifest.payload_digest = handoff_digest(&manifest, payload.as_bytes());
        debug_assert_eq!(manifest.payload_bytes().ok(), Some(total));
        Ok(HandoffState { manifest, payload })
    }

    /// Import `handoff` into this fresh (reset, position 0) consumer bundle
    /// and return its `first_token`. Everything is validated before any
    /// mutation: bound role/domain/arch, schema, formats, layer layout,
    /// capacity, exact target and MTP metadata (marks, positions, policy), part
    /// lengths against the live layout, and the payload digest. Then the
    /// target and MTP state are copied into this bundle's own arenas
    /// (synchronous h2d), the device is synchronized, and only then are the
    /// marks, position, PLE history and draft policy published and the derived
    /// F16/BF16 caches invalidated. A failure after the first mutation resets
    /// the request (and quarantines the `Gpu` when the device cannot be
    /// drained); it is never retried on partial state.
    pub fn import_handoff(
        &mut self,
        gpu: &mut Gpu,
        handoff: &HandoffState,
        domain: &HandoffDomain,
    ) -> Result<u32, String> {
        let (role, id) = self.handoff_gate(gpu, domain)?;
        if role != DisaggRole::Consumer {
            return Err("only a consumer bundle imports a handoff".to_string());
        }
        let manifest = &handoff.manifest;
        if manifest.domain != *domain {
            return Err("handoff manifest domain does not equal the bound domain".to_string());
        }
        if self.state.position != 0
            || self.mtp.as_ref().is_some_and(|mtp| mtp.position() != 0)
            || self.disagg_domain.is_some()
        {
            return Err("handoff import needs a reset bundle (position 0)".to_string());
        }
        let vocab = self.config.vocab_size;
        if manifest.first_token as usize >= vocab {
            return Err(format!(
                "first_token {} is outside the vocabulary ({vocab})",
                manifest.first_token
            ));
        }
        let bytes = handoff.as_bytes();
        if handoff_digest(manifest, bytes) != manifest.payload_digest {
            return Err("handoff payload digest mismatch".to_string());
        }
        let position = manifest.position;
        self.state
            .validate_handoff_meta(&manifest.target_meta, position, vocab)
            .map_err(|e| format!("target handoff metadata: {e}"))?;
        {
            let mtp = self.mtp.as_ref().ok_or("handoff needs the native MTP head")?;
            let target_layout = self
                .state
                .session_layout_for_meta(&manifest.target_meta)
                .map_err(|e| format!("target layout: {e}"))?;
            let mtp_layout = mtp
                .handoff_layout_for_meta(&manifest.mtp_meta, position)
                .map_err(|e| format!("MTP handoff metadata: {e}"))?;
            let (target_fixed, target_rows) = describe(&target_layout);
            let (mtp_fixed, mtp_rows) = describe(&mtp_layout);
            let expected = ExpectedHandoff {
                domain: domain.clone(),
                formats: handoff_formats(&self.state_format()),
                layer_layout: flash_next_layer_layout(),
                max_position: self.state.max_seq_len,
                fixed_bytes: target_fixed,
                row_sizes: target_rows.iter().map(|&(row, _)| row).collect(),
                mtp: Some(ExpectedMtp {
                    fixed_bytes: mtp_fixed,
                    row_sizes: mtp_rows.iter().map(|&(row, _)| row).collect(),
                }),
            };
            manifest.validate(&expected, bytes.len())?;
            if manifest.row_bytes != target_rows || manifest.mtp_row_bytes != mtp_rows {
                return Err("handoff row counts do not match the metadata marks".to_string());
            }
        }
        // From here on state mutates: quiesce readers of the discarded state.
        self.invalidate_ple_epoch().map_err(|e| format!("invalidate PLE epoch: {e}"))?;
        gpu.bind_thread().map_err(|e| format!("bind device: {e}"))?;
        if let Some(session) = self.session.as_mut() {
            session.forget_live();
        }
        if let Some(stage) = &mut self.pair_session_stage {
            stage.position = None;
        }
        match self.import_apply(gpu, manifest, bytes) {
            Ok(()) => {
                self.disagg_domain = Some(id);
                Ok(manifest.first_token)
            }
            Err(cause) => Err(self.handoff_abort(gpu, cause)),
        }
    }

    /// The mutating half of an import; the caller aborts on `Err`.
    fn import_apply(
        &mut self,
        gpu: &mut Gpu,
        manifest: &HandoffManifest,
        bytes: &[u8],
    ) -> Result<(), String> {
        let Self { state, mtp, .. } = self;
        let mtp = mtp.as_mut().ok_or("handoff needs the native MTP head")?;
        {
            let target_layout = state
                .prepare_session_restore(gpu, &manifest.target_meta)
                .map_err(|e| format!("prepare target restore: {e}"))?;
            let mtp_layout = mtp
                .prepare_session_restore(gpu, &manifest.mtp_meta)
                .map_err(|e| format!("prepare MTP restore: {e}"))?;
            let (fixed, rows) = describe(&target_layout);
            let (mtp_fixed, mtp_rows) = describe(&mtp_layout);
            if fixed != manifest.fixed_bytes
                || rows != manifest.row_bytes
                || mtp_fixed != manifest.mtp_fixed_bytes
                || mtp_rows != manifest.mtp_row_bytes
            {
                return Err("destination layout changed after validation".to_string());
            }
            let (chunks, total) = plan_chunks(&target_layout, &mtp_layout)?;
            if total != bytes.len() {
                return Err(format!(
                    "destination holds {total} bytes but the payload is {}",
                    bytes.len()
                ));
            }
            let mut cursor = 0usize;
            for chunk in &chunks {
                if chunk.bytes == 0 {
                    continue;
                }
                gpu.hip
                    .memcpy_htod_offset(chunk.buf, chunk.offset, &bytes[cursor..cursor + chunk.bytes])
                    .map_err(|e| format!("handoff h2d copy: {e}"))?;
                cursor += chunk.bytes;
            }
        }
        gpu.hip
            .device_synchronize()
            .map_err(|e| format!("synchronize after import copies: {e}"))?;
        state
            .finish_session_restore(&manifest.target_meta)
            .map_err(|e| format!("publish target state: {e}"))?;
        mtp.finish_session_restore(&manifest.mtp_meta)
            .map_err(|e| format!("publish MTP state: {e}"))?;
        Ok(())
    }

    /// Invalidate a failed import: quarantine the `Gpu` when the device
    /// cannot be drained, else cold-reset every owner.
    fn handoff_abort(&mut self, gpu: &mut Gpu, cause: String) -> String {
        if let Some(message) =
            drain_or_quarantine(gpu, &format!("handoff import failed ({cause})"))
        {
            return message;
        }
        self.disagg_domain = None;
        match self.reset(gpu) {
            Ok(()) => format!("handoff import failed ({cause}); the request state was reset"),
            Err(error) => format!(
                "handoff import failed ({cause}); reset after the failure also failed ({error}): \
                 the bundle state is invalid"
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::disagg_manifest::{NumericalManifest, StateFormats};

    fn identity(arch: &str) -> NumericalManifest {
        NumericalManifest {
            arch: arch.into(),
            route_identity: "route".into(),
            artifact_digest: "artifact".into(),
            policy_digest: "policy".into(),
        }
    }

    fn manifest() -> HandoffManifest {
        HandoffManifest {
            version: HANDOFF_SCHEMA_VERSION,
            domain: HandoffDomain { producer: identity("gfx1151"), consumer: identity("gfx1100") },
            layer_layout: flash_next_layer_layout(),
            formats: StateFormats { kv: "F32".into(), gdn: "F32".into(), qsa_index: "F32".into() },
            position: 4,
            target_meta: vec![1, 2, 3, 4, 5, 6, 7, 8],
            mtp_meta: vec![9; 16],
            fixed_bytes: vec![8],
            row_bytes: vec![(4, 2)],
            mtp_fixed_bytes: vec![4],
            mtp_row_bytes: vec![(2, 2)],
            first_token: 7,
            payload_digest: String::new(),
        }
    }

    #[test]
    fn digest_binds_meta_token_layout_and_payload() {
        let base = manifest();
        let payload = [3u8; 24];
        let digest = handoff_digest(&base, &payload);
        assert_eq!(digest.len(), 64);
        assert_eq!(digest, handoff_digest(&base, &payload));
        let mut other = base.clone();
        other.first_token = 8;
        assert_ne!(digest, handoff_digest(&other, &payload));
        let mut other = base.clone();
        other.mtp_meta[0] ^= 1;
        assert_ne!(digest, handoff_digest(&other, &payload));
        let mut other = base.clone();
        other.row_bytes = vec![(2, 4)];
        assert_ne!(digest, handoff_digest(&other, &payload));
        let mut flipped = payload;
        flipped[23] ^= 1;
        assert_ne!(digest, handoff_digest(&base, &flipped));
    }

    #[test]
    fn domain_id_is_ordered_and_stable() {
        let forward = HandoffDomain { producer: identity("gfx1151"), consumer: identity("gfx1100") };
        let swapped = HandoffDomain { producer: identity("gfx1100"), consumer: identity("gfx1151") };
        assert_eq!(disagg_domain_id(&forward), disagg_domain_id(&forward));
        assert_ne!(disagg_domain_id(&forward), disagg_domain_id(&swapped));
        assert!(disagg_domain_id(&forward).starts_with("disagg-v1-"));
    }
}
