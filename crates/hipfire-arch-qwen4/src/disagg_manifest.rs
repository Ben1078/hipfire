// SPDX-License-Identifier: Apache-2.0
//! Schema-v1 arch-neutral handoff identity for a Flash-Next prefill→decode
//! split (gfx1151 producer → gfx1100 consumer). Pure CPU: no device state.
//!
//! The manifest is immutable identity plus explicit lengths. Session
//! metadata (`target_meta` / `mtp_meta`) is opaque bytes produced by the
//! existing session parser (QSA marks/selection, PLE history, ...); this
//! module never interprets them, the caller does. `first_token` is the greedy
//! sample of the producer's logits and is never recomputed by the consumer.
use serde::{Deserialize, Serialize};

/// The only schema version this module produces or accepts.
pub const HANDOFF_SCHEMA_VERSION: u32 = 1;
/// Layer-layout entry for a gated-delta-net layer.
pub const LAYER_GDN: &str = "gdn";
/// Layer-layout entry for a QSA attention layer.
pub const LAYER_QSA: &str = "qsa";
/// Flash-Next layer count: `[gdn, gdn, gdn, qsa]` repeated 12 times.
pub const FLASH_NEXT_LAYERS: usize = 48;
/// Producer arch accepted by this schema version.
pub const PRODUCER_ARCH: &str = "gfx1151";
/// Consumer arch accepted by this schema version.
pub const CONSUMER_ARCH: &str = "gfx1100";

const GDN_LAYERS: u64 = 36;
const QSA_LAYERS: u64 = 12;
const GDN_RECURRENT_BYTES: u64 = 3_145_728;
const GDN_CONV_BYTES: u64 = 122_880;
const QSA_SELECTED_BYTES: u64 = 8_204;
/// Per QSA layer per token: K/V rows (F32).
const QSA_KV_ROW_BYTES: u64 = 4_096;
/// Per QSA layer per token: raw index key row.
const QSA_INDEX_ROW_BYTES: u64 = 512;
/// Per QSA layer per 4 tokens: pooled index key row.
const QSA_POOLED_ROW_BYTES: u64 = 512;
const POOL_STRIDE: u64 = 4;

/// Numerical identity of one side of the handoff.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NumericalManifest {
    pub arch: String,
    pub route_identity: String,
    pub artifact_digest: String,
    pub policy_digest: String,
}

/// State formats on the wire. Must match exactly on both sides; no
/// conversion happens in the handoff.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StateFormats {
    pub kv: String,
    pub gdn: String,
    pub qsa_index: String,
}

/// Ordered (producer, consumer) pair; the pair is the cache/handoff domain.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HandoffDomain {
    pub producer: NumericalManifest,
    pub consumer: NumericalManifest,
}

/// Frozen handoff description. `row_bytes` / `mtp_row_bytes` entries are
/// `(row_bytes, rows)`, matching the session-cache row-stream layout.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HandoffManifest {
    pub version: u32,
    pub domain: HandoffDomain,
    pub layer_layout: Vec<String>,
    pub formats: StateFormats,
    pub position: usize,
    pub target_meta: Vec<u8>,
    pub mtp_meta: Vec<u8>,
    pub fixed_bytes: Vec<usize>,
    pub row_bytes: Vec<(usize, usize)>,
    pub mtp_fixed_bytes: Vec<usize>,
    pub mtp_row_bytes: Vec<(usize, usize)>,
    /// Greedy sample of the producer's logits at `position`.
    pub first_token: u32,
    pub payload_digest: String,
}

/// Consumer-side expected MTP layout: fixed part sizes and per-row sizes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExpectedMtp {
    pub fixed_bytes: Vec<usize>,
    pub row_sizes: Vec<usize>,
}

/// What the consumer requires of an incoming manifest.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExpectedHandoff {
    pub domain: HandoffDomain,
    pub formats: StateFormats,
    pub layer_layout: Vec<String>,
    /// Consumer state capacity in tokens; `position` must be in `1..=max_position`.
    pub max_position: usize,
    pub fixed_bytes: Vec<usize>,
    /// Bytes per row of each target row stream (row counts come from the manifest).
    pub row_sizes: Vec<usize>,
    /// `None` = consumer imports no MTP state, and the manifest must carry none.
    pub mtp: Option<ExpectedMtp>,
}

/// The canonical Flash-Next layer layout: `[gdn, gdn, gdn, qsa]` × 12.
pub fn flash_next_layer_layout() -> Vec<String> {
    (0..FLASH_NEXT_LAYERS)
        .map(|i| (if i % 4 == 3 { LAYER_QSA } else { LAYER_GDN }).to_string())
        .collect()
}

fn checked_streams(fixed: &[usize], rows: &[(usize, usize)], what: &str) -> Result<usize, String> {
    let mut total = 0usize;
    for &b in fixed {
        total = total.checked_add(b).ok_or_else(|| format!("{what} fixed payload overflows"))?;
    }
    for &(row, n) in rows {
        let bytes = row.checked_mul(n).ok_or_else(|| format!("{what} row stream overflows"))?;
        total = total.checked_add(bytes).ok_or_else(|| format!("{what} payload overflows"))?;
    }
    Ok(total)
}

fn check_identity(side: &str, id: &NumericalManifest) -> Result<(), String> {
    if id.route_identity.is_empty() {
        return Err(format!("{side} route identity is empty"));
    }
    if id.policy_digest.is_empty() {
        return Err(format!("{side} policy digest is empty"));
    }
    if id.artifact_digest.is_empty() {
        return Err(format!("{side} artifact digest is empty"));
    }
    Ok(())
}

fn check_layout(layout: &[String], what: &str) -> Result<(), String> {
    if layout.len() != FLASH_NEXT_LAYERS {
        return Err(format!("{what} layer layout has {} entries, expected {FLASH_NEXT_LAYERS}", layout.len()));
    }
    if layout != flash_next_layer_layout().as_slice() {
        return Err(format!("{what} layer layout is not [gdn,gdn,gdn,qsa] x 12"));
    }
    Ok(())
}

fn check_mtp(
    meta: &[u8], fixed: &[usize], rows: &[(usize, usize)], expected: Option<&ExpectedMtp>, max_position: usize,
) -> Result<(), String> {
    let present = !meta.is_empty() || !fixed.is_empty() || !rows.is_empty();
    let exp = match (expected, present) {
        (None, false) => return Ok(()),
        (None, true) => return Err("manifest carries MTP state but the consumer expects none".into()),
        (Some(_), false) => return Err("consumer expects MTP state but the manifest carries none".into()),
        (Some(exp), true) => exp,
    };
    if meta.is_empty() {
        return Err("MTP metadata is empty while MTP streams are present".into());
    }
    if fixed != exp.fixed_bytes.as_slice() {
        return Err("MTP fixed byte lengths mismatch".into());
    }
    if rows.len() != exp.row_sizes.len() || rows.iter().zip(&exp.row_sizes).any(|(r, &s)| r.0 != s) {
        return Err("MTP row stream layout mismatch".into());
    }
    if rows.iter().any(|&(_, n)| n > max_position) {
        return Err("MTP row count exceeds consumer capacity".into());
    }
    Ok(())
}

impl HandoffManifest {
    /// Total payload bytes: Σ fixed + Σ `row_bytes × rows`, for the target
    /// state plus the MTP state. Errors on arithmetic overflow.
    pub fn payload_bytes(&self) -> Result<usize, String> {
        let target = checked_streams(&self.fixed_bytes, &self.row_bytes, "target")?;
        let mtp = checked_streams(&self.mtp_fixed_bytes, &self.mtp_row_bytes, "mtp")?;
        target.checked_add(mtp).ok_or_else(|| "handoff payload overflows".to_string())
    }

    /// Validate against the consumer's expectation and the received payload
    /// length. The caller additionally parses `target_meta` / `mtp_meta`.
    pub fn validate(&self, expected: &ExpectedHandoff, payload_len: usize) -> Result<(), String> {
        if self.version != HANDOFF_SCHEMA_VERSION {
            return Err(format!("handoff schema version {} unsupported (expected {HANDOFF_SCHEMA_VERSION})", self.version));
        }
        let (p, c) = (&self.domain.producer, &self.domain.consumer);
        check_identity("producer", p)?;
        check_identity("consumer", c)?;
        if p.artifact_digest != c.artifact_digest {
            return Err("producer and consumer artifact digests differ".into());
        }
        if p.arch != PRODUCER_ARCH || c.arch != CONSUMER_ARCH {
            return Err(format!(
                "unsupported arch pair {}->{} (only {PRODUCER_ARCH}->{CONSUMER_ARCH})", p.arch, c.arch
            ));
        }
        check_identity("expected producer", &expected.domain.producer)?;
        check_identity("expected consumer", &expected.domain.consumer)?;
        if self.domain.producer != expected.domain.producer {
            return Err("producer identity does not match the expected producer".into());
        }
        if self.domain.consumer != expected.domain.consumer {
            return Err("consumer identity does not match the expected consumer".into());
        }
        if self.formats != expected.formats {
            return Err("state formats do not match the consumer".into());
        }
        check_layout(&self.layer_layout, "manifest")?;
        check_layout(&expected.layer_layout, "expected")?;
        if self.position == 0 {
            return Err("handoff position is zero".into());
        }
        if self.position > expected.max_position {
            return Err(format!("handoff position {} exceeds consumer capacity {}", self.position, expected.max_position));
        }
        if self.target_meta.is_empty() {
            return Err("target metadata is empty".into());
        }
        if self.fixed_bytes != expected.fixed_bytes {
            return Err("target fixed byte lengths mismatch".into());
        }
        if self.row_bytes.len() != expected.row_sizes.len()
            || self.row_bytes.iter().zip(&expected.row_sizes).any(|(r, &s)| r.0 != s)
        {
            return Err("target row stream layout mismatch".into());
        }
        if self.row_bytes.iter().any(|&(_, n)| n > expected.max_position) {
            return Err("target row count exceeds consumer capacity".into());
        }
        check_mtp(
            &self.mtp_meta, &self.mtp_fixed_bytes, &self.mtp_row_bytes, expected.mtp.as_ref(), expected.max_position,
        )?;
        if self.payload_digest.is_empty() {
            return Err("payload digest is empty".into());
        }
        let bytes = self.payload_bytes()?;
        if bytes != payload_len {
            return Err(format!("payload length {payload_len} does not match manifest {bytes}"));
        }
        Ok(())
    }
}

/// Exact main-model state bytes for a Flash-Next handoff of `tokens` tokens.
///
/// Covers only the DISAGG table: fixed state, per-token QSA K/V + raw index
/// rows, and pooled index rows. It EXCLUDES PLE history, hyper-connection
/// feedback and every MTP byte; those are carried separately by the manifest
/// streams and are not part of this table.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FlashNextStateBytes {
    /// 36 GDN recurrent + 36 GDN conv + 12 QSA selected indices.
    pub fixed: u64,
    /// `12 * (4096 + 512) * tokens`.
    pub per_token: u64,
    /// `12 * 512 * ceil(tokens / 4)`.
    pub pooled: u64,
    pub total: u64,
}

/// `117768336 + 12*(4096+512)*tokens + 12*512*ceil(tokens/4)`, overflow-checked.
/// See [`FlashNextStateBytes`] for what is excluded.
pub fn flash_next_state_bytes(tokens: usize) -> Result<FlashNextStateBytes, String> {
    let overflow = || format!("Flash-Next state bytes overflow for {tokens} tokens");
    let t = tokens as u64;
    let fixed = GDN_LAYERS * (GDN_RECURRENT_BYTES + GDN_CONV_BYTES) + QSA_LAYERS * QSA_SELECTED_BYTES;
    let per_token = (QSA_LAYERS * (QSA_KV_ROW_BYTES + QSA_INDEX_ROW_BYTES)).checked_mul(t).ok_or_else(overflow)?;
    let pooled = (QSA_LAYERS * QSA_POOLED_ROW_BYTES).checked_mul(t.div_ceil(POOL_STRIDE)).ok_or_else(overflow)?;
    let total = fixed
        .checked_add(per_token)
        .and_then(|b| b.checked_add(pooled))
        .ok_or_else(overflow)?;
    Ok(FlashNextStateBytes { fixed, per_token, pooled, total })
}

/// Fixed bytes of the complete default-F32 packed handoff: the 117768336 main-model
/// fixed state + 368640 PLE conv history + 40960 hyper-connection feedback +
/// 49168 MTP fixed (ByteAccounting-verified figures).
const PACKED_FIXED_BYTES: u64 = 118_227_104;
/// Row streams in the packed payload: 12 QSA layers + the MTP head's layer.
const PACKED_ROW_LAYERS: u64 = 13;

/// Opaque session metadata observed with the packed payload: 608 target + 72 MTP.
/// Informational only; it is NOT part of [`flash_next_handoff_payload_bytes`].
pub const FLASH_NEXT_HANDOFF_META_BYTES: u64 = 608 + 72;

/// Exact packed payload bytes of a complete default-F32 Flash-Next handoff
/// (target + PLE + HC + MTP) at `p` committed tokens:
/// `118227104 + 13 * (4608 * p + 512 * floor(p / 4))`.
///
/// Excludes the envelope (manifest) and all opaque metadata
/// ([`FLASH_NEXT_HANDOFF_META_BYTES`]). The pooled index rows use the *committed*
/// count `floor(p / 4)`, unlike [`flash_next_state_bytes`], whose `ceil(tokens / 4)`
/// is the DISAGG design-inventory figure for the main model only (12 QSA layers, no
/// PLE/HC/MTP). This function covers 13 row layers plus PLE/HC/MTP fixed state.
pub fn flash_next_handoff_payload_bytes(tokens: usize) -> Result<u64, String> {
    let overflow = || format!("Flash-Next handoff payload bytes overflow for {tokens} tokens");
    let p = tokens as u64;
    let rows = 4_608u64
        .checked_mul(p)
        .and_then(|a| a.checked_add(QSA_POOLED_ROW_BYTES.checked_mul(p / POOL_STRIDE)?))
        .and_then(|a| a.checked_mul(PACKED_ROW_LAYERS))
        .ok_or_else(overflow)?;
    PACKED_FIXED_BYTES.checked_add(rows).ok_or_else(overflow)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn id(arch: &str, route: &str) -> NumericalManifest {
        NumericalManifest {
            arch: arch.into(), route_identity: route.into(),
            artifact_digest: "art".into(), policy_digest: format!("pol-{arch}"),
        }
    }

    fn domain() -> HandoffDomain {
        HandoffDomain { producer: id("gfx1151", "halo-prefill"), consumer: id("gfx1100", "xtx-d2") }
    }

    fn formats() -> StateFormats {
        StateFormats { kv: "f32".into(), gdn: "f32".into(), qsa_index: "bf16".into() }
    }

    fn manifest() -> HandoffManifest {
        HandoffManifest {
            version: HANDOFF_SCHEMA_VERSION, domain: domain(), layer_layout: flash_next_layer_layout(),
            formats: formats(), position: 512, target_meta: vec![1, 2, 3], mtp_meta: vec![9],
            fixed_bytes: vec![100, 28], row_bytes: vec![(8, 512), (4, 128)],
            mtp_fixed_bytes: vec![16], mtp_row_bytes: vec![(2, 512)],
            first_token: 42, payload_digest: "deadbeef".into(),
        }
    }

    fn expected() -> ExpectedHandoff {
        ExpectedHandoff {
            domain: domain(), formats: formats(), layer_layout: flash_next_layer_layout(),
            max_position: 4096, fixed_bytes: vec![100, 28], row_sizes: vec![8, 4],
            mtp: Some(ExpectedMtp { fixed_bytes: vec![16], row_sizes: vec![2] }),
        }
    }

    const PAYLOAD: usize = 100 + 28 + 8 * 512 + 4 * 128 + 16 + 2 * 512;

    #[test]
    fn state_bytes_table() {
        for (t, want) in [(512, 146_866_320u64), (2048, 234_160_272), (16384, 1_048_903_824), (32768, 1_980_039_312)] {
            let s = flash_next_state_bytes(t).unwrap();
            assert_eq!(s.total, want, "{t} tokens");
            assert_eq!(s.fixed, 117_768_336);
            assert_eq!(s.total, s.fixed + s.per_token + s.pooled);
        }
        assert_eq!(flash_next_state_bytes(0).unwrap().total, 117_768_336);
        assert_eq!(flash_next_state_bytes(5).unwrap().pooled, 12 * 512 * 2);
    }

    #[test]
    fn state_bytes_overflow_refused() {
        assert!(flash_next_state_bytes(usize::MAX).is_err());
    }

    #[test]
    fn handoff_payload_table() {
        for (p, want) in [
            (512, 149_749_920u64), (2048, 244_318_368), (16384, 1_126_957_216), (32768, 2_135_687_328),
        ] {
            assert_eq!(flash_next_handoff_payload_bytes(p).unwrap(), want, "{p} tokens");
        }
        assert_eq!(flash_next_handoff_payload_bytes(0).unwrap(), 118_227_104);
        assert_eq!(
            118_227_104,
            flash_next_state_bytes(0).unwrap().fixed + 368_640 + 40_960 + 49_168,
        );
        assert_eq!(FLASH_NEXT_HANDOFF_META_BYTES, 680);
    }

    #[test]
    fn handoff_payload_uses_floor_pooled_not_design_ceil() {
        // p = 5: committed pooled rows = floor(5/4) = 1; the design inventory uses ceil = 2.
        assert_eq!(flash_next_handoff_payload_bytes(5).unwrap(), 118_227_104 + 13 * (4_608 * 5 + 512));
        assert_eq!(flash_next_state_bytes(5).unwrap().pooled, 12 * 512 * 2);
        // Divisible p: the pooled term matches per-layer ceil == floor.
        assert_eq!(
            flash_next_handoff_payload_bytes(8).unwrap() - flash_next_handoff_payload_bytes(7).unwrap(),
            13 * (4_608 + 512),
        );
        assert_eq!(
            flash_next_handoff_payload_bytes(7).unwrap() - flash_next_handoff_payload_bytes(6).unwrap(),
            13 * 4_608,
        );
    }

    #[test]
    fn handoff_payload_overflow_refused() {
        assert!(flash_next_handoff_payload_bytes(usize::MAX).unwrap_err().contains("overflow"));
    }

    #[test]
    fn layout_is_48_with_3_gdn_per_qsa() {
        let l = flash_next_layer_layout();
        assert_eq!(l.len(), 48);
        assert_eq!(l.iter().filter(|s| *s == LAYER_QSA).count(), 12);
        assert!(l.chunks(4).all(|c| c[..3].iter().all(|s| s == LAYER_GDN) && c[3] == LAYER_QSA));
    }

    #[test]
    fn valid_manifest_accepts_and_sums_payload() {
        let m = manifest();
        assert_eq!(m.payload_bytes().unwrap(), PAYLOAD);
        m.validate(&expected(), PAYLOAD).unwrap();
    }

    #[test]
    fn json_roundtrip_and_unknown_field_refused() {
        let m = manifest();
        let json = serde_json::to_string(&m).unwrap();
        assert_eq!(serde_json::from_str::<HandoffManifest>(&json).unwrap(), m);
        let bad = json.replacen('{', "{\"extra\":1,", 1);
        assert!(serde_json::from_str::<HandoffManifest>(&bad).is_err());
    }

    #[test]
    fn no_mtp_manifest_accepts_without_expected_mtp() {
        let mut m = manifest();
        m.mtp_meta.clear();
        m.mtp_fixed_bytes.clear();
        m.mtp_row_bytes.clear();
        let mut e = expected();
        e.mtp = None;
        m.validate(&e, PAYLOAD - 16 - 2 * 512).unwrap();
    }

    fn refused(edit: impl FnOnce(&mut HandoffManifest, &mut ExpectedHandoff, &mut usize), needle: &str) {
        let (mut m, mut e, mut len) = (manifest(), expected(), PAYLOAD);
        edit(&mut m, &mut e, &mut len);
        let err = m.validate(&e, len).expect_err(needle);
        assert!(err.contains(needle), "{err:?} lacks {needle:?}");
    }

    #[test]
    fn refusal_matrix() {
        refused(|m, _, _| m.version = 2, "schema version");
        refused(|m, _, _| m.version = 0, "schema version");
        refused(|m, _, _| m.domain.producer.artifact_digest.clear(), "artifact digest is empty");
        refused(|m, _, _| m.domain.consumer.artifact_digest.clear(), "artifact digest is empty");
        refused(|m, _, _| m.domain.consumer.artifact_digest = "other".into(), "artifact digests differ");
        refused(|m, _, _| m.domain.producer.route_identity.clear(), "route identity is empty");
        refused(|m, _, _| m.domain.consumer.policy_digest.clear(), "policy digest is empty");
        refused(|m, _, _| std::mem::swap(&mut m.domain.producer, &mut m.domain.consumer), "unsupported arch pair");
        refused(|m, _, _| m.domain.consumer.arch = "gfx1151".into(), "unsupported arch pair");
        refused(|m, _, _| m.domain.producer.arch = "gfx1100".into(), "unsupported arch pair");
        refused(|m, _, _| m.domain.producer.route_identity = "other".into(), "producer identity");
        refused(|m, _, _| m.domain.consumer.policy_digest = "other".into(), "consumer identity");
        refused(|m, _, _| m.formats.kv = "bf16".into(), "state formats");
        refused(|m, _, _| m.formats.qsa_index = "f32".into(), "state formats");
        refused(|m, _, _| { m.layer_layout.pop(); }, "layer layout has 47");
        refused(|m, _, _| m.layer_layout.swap(2, 3), "not [gdn,gdn,gdn,qsa]");
        refused(|_, e, _| e.layer_layout[0] = LAYER_QSA.into(), "expected layer layout");
        refused(|m, _, _| m.position = 0, "position is zero");
        refused(|m, _, _| m.position = 4097, "exceeds consumer capacity");
        refused(|m, _, _| m.target_meta.clear(), "target metadata is empty");
        refused(|m, _, _| m.fixed_bytes[0] += 1, "target fixed byte");
        refused(|m, _, _| m.row_bytes[0].0 = 16, "target row stream");
        refused(|m, _, _| { m.row_bytes.pop(); }, "target row stream");
        refused(|m, _, _| m.row_bytes[0].1 = 4097, "target row count");
        refused(|m, _, _| m.mtp_meta.clear(), "MTP metadata is empty");
        refused(|m, _, _| m.mtp_fixed_bytes[0] = 17, "MTP fixed");
        refused(|m, _, _| m.mtp_row_bytes[0].0 = 3, "MTP row stream");
        refused(|m, _, _| m.mtp_row_bytes[0].1 = 4097, "MTP row count");
        refused(
            |m, _, _| { m.mtp_meta.clear(); m.mtp_fixed_bytes.clear(); m.mtp_row_bytes.clear(); },
            "expects MTP state but the manifest carries none",
        );
        refused(|_, e, _| e.mtp = None, "expects none");
        refused(|m, _, _| m.payload_digest.clear(), "payload digest");
        refused(|_, _, len| *len += 1, "payload length");
        refused(|_, _, len| *len -= 1, "payload length");
    }

    #[test]
    fn overflow_is_refused() {
        refused(|m, _, _| m.row_bytes[0] = (usize::MAX, 2), "target row stream layout");
        let mut m = manifest();
        m.row_bytes[0] = (usize::MAX, 2);
        assert!(m.payload_bytes().unwrap_err().contains("overflows"));
        m.row_bytes[0] = (8, 512);
        m.fixed_bytes = vec![usize::MAX, 1];
        assert!(m.payload_bytes().unwrap_err().contains("overflows"));
        m.fixed_bytes = vec![usize::MAX];
        m.row_bytes[0] = (1, 1);
        assert!(m.payload_bytes().unwrap_err().contains("overflows"));
    }

    #[test]
    fn domain_separation() {
        // Same artifact, same arches, different policy/route => different domain; refused.
        let mut other = expected();
        other.domain.consumer.policy_digest = "pol-other".into();
        assert!(manifest().validate(&other, PAYLOAD).unwrap_err().contains("consumer identity"));
        let mut other = expected();
        other.domain.producer.route_identity = "halo-prefill-b".into();
        assert!(manifest().validate(&other, PAYLOAD).unwrap_err().contains("producer identity"));
        // The reversed domain is not the same domain.
        let d = domain();
        let rev = HandoffDomain { producer: d.consumer.clone(), consumer: d.producer.clone() };
        assert_ne!(d, rev);
    }
}
