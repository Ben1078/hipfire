// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Pure Qwen4 capability and artifact admission.
//!
//! Admission is intentionally a value-only operation.  It validates source
//! shape, request modality, effective mesh, container format, and quantization
//! geometry before a loader is allowed to allocate weights or start a reader.

use crate::config::{Qwen4Config, ARCH_ID};
use crate::weights::Qwen4TierRecipe;
use std::fmt;

/// Request modality understood by the Qwen4 carrier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum InputModality {
    Text,
    Image,
    Video,
}

/// Short alias for callers that use request rather than input terminology.
pub type RequestKind = InputModality;

/// Capability bits exposed by this first Qwen4 implementation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Qwen4Capability {
    Text,
    NativeMtp,
    /// qt=55 `FP8E4M3B128` routed experts (the stock-FP8 artifact).
    Fp8Experts,
}

/// The only GPU architecture with FP8 expert kernels (MI300X, wave64 MFMA).
pub const FP8_EXPERTS_ARCH: &str = "gfx942";

/// Qwen4 is text-only at the modality boundary. Image and video requests are
/// represented as unsupported rather than silently routed to text. Native
/// greedy-MTP is a capability of the executable artifact when its validated
/// MTP manifest is requested by the caller.  `fp8_experts` is a device
/// capability: true only on [`FP8_EXPERTS_ARCH`], so every other card refuses
/// a stock-FP8 artifact (fail closed) until it has kernels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Qwen4Capabilities {
    pub text: bool,
    pub native_mtp: bool,
    pub image: bool,
    pub video: bool,
    pub fp8_experts: bool,
}

impl Qwen4Capabilities {
    pub const fn text_ar() -> Self {
        Self {
            text: true,
            native_mtp: false,
            image: false,
            video: false,
            fp8_experts: false,
        }
    }

    /// Native MTP is enabled only for an explicitly requested executable
    /// artifact. The artifact parser remains responsible for validating that
    /// the one-layer MTP manifest/config is present.
    pub const fn text_mtp() -> Self {
        Self {
            text: true,
            native_mtp: true,
            image: false,
            video: false,
            fp8_experts: false,
        }
    }

    /// These capabilities on the device `gpu_arch`: `fp8_experts` only on
    /// [`FP8_EXPERTS_ARCH`].
    pub fn on_device(self, gpu_arch: &str) -> Self {
        Self {
            fp8_experts: gpu_arch == FP8_EXPERTS_ARCH,
            ..self
        }
    }

    pub const fn supports_modality(self, modality: InputModality) -> bool {
        match modality {
            InputModality::Text => self.text,
            InputModality::Image => self.image,
            InputModality::Video => self.video,
        }
    }

    pub const fn supports(self, capability: Qwen4Capability) -> bool {
        match capability {
            Qwen4Capability::Text => self.text,
            Qwen4Capability::NativeMtp => self.native_mtp,
            Qwen4Capability::Fp8Experts => self.fp8_experts,
        }
    }
}

impl Default for Qwen4Capabilities {
    fn default() -> Self {
        Self::text_ar()
    }
}

/// Effective (post-policy) mesh sizes.  A size of zero is retained so a
/// malformed plan is refused rather than normalized into a single-device
/// request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EffectiveMesh {
    pub pp: usize,
    pub tp: usize,
    pub ep: usize,
}

/// Alias used by mesh planners and tests.
pub type MeshShape = EffectiveMesh;

impl EffectiveMesh {
    pub const fn single() -> Self {
        Self {
            pp: 1,
            tp: 1,
            ep: 1,
        }
    }

    pub const fn new(pp: usize, tp: usize, ep: usize) -> Self {
        Self { pp, tp, ep }
    }

    /// The degrees of an already-normalized `DeviceMesh`: the loader's single
    /// mesh normalization is the only source of an admitted `EffectiveMesh`,
    /// so a request's raw pp/tp/ep are never reinterpreted a second time.
    /// An absent axis is degree 1.
    pub fn from_device_mesh(mesh: &hipfire_runtime::device_mesh::DeviceMesh) -> Self {
        use hipfire_runtime::device_mesh::DimKind;
        Self::new(
            mesh.size_of(DimKind::Pp),
            mesh.size_of(DimKind::Tp),
            mesh.size_of(DimKind::Ep),
        )
    }

    pub const fn is_single(self) -> bool {
        self.pp == 1 && self.tp == 1 && self.ep == 1
    }

    /// Largest expert-parallel degree Qwen4 admits. Experts shard `e % N`
    /// across `N` ranks and the replicated program runs on each card, so
    /// uneven splits (512 over 6 ranks = 86/86/85/85/85/85) are admitted.
    pub const EP_MAX_DEGREE: usize = 8;

    /// Smallest and largest pipeline-parallel degree Qwen4 admits
    /// (`topology.pp`): contiguous layer bands over homogeneous cards with
    /// tp = ep = 1.
    pub const PP_MIN_DEGREE: usize = 2;
    pub const PP_MAX_DEGREE: usize = 8;

    /// A pure pipeline-parallel mesh in the admitted degree range.
    pub const fn is_pipeline(self) -> bool {
        self.pp >= Self::PP_MIN_DEGREE
            && self.pp <= Self::PP_MAX_DEGREE
            && self.tp == 1
            && self.ep == 1
    }

    /// The meshes Qwen4 admits: Single, expert parallelism of 1..=8 ranks
    /// with no pipeline or tensor parallelism, or a pure pipeline-parallel
    /// mesh of [`Self::PP_MIN_DEGREE`]..=[`Self::PP_MAX_DEGREE`] stages
    /// (tp = ep = 1; PP never composes with EP or TP).
    pub const fn is_qwen4_admitted(self) -> bool {
        (self.pp == 1 && self.tp == 1 && self.ep >= 1 && self.ep <= Self::EP_MAX_DEGREE)
            || self.is_pipeline()
    }
}

impl Default for EffectiveMesh {
    fn default() -> Self {
        Self::single()
    }
}

/// Wire/storage format admitted by the runtime Qwen4 artifact boundary.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Qwen4ArtifactFormat {
    /// Versioned HFQM package containing all resident and external-row records.
    Hfqm,
    /// Safetensors is a source/conversion input, not a serving artifact.
    Safetensors,
    /// Explicitly retain unknown formats so they fail closed with a useful
    /// diagnostic instead of being mistaken for HFQM.
    Other(String),
}

/// Short format alias for callers sharing an artifact admission interface.
pub type ArtifactFormat = Qwen4ArtifactFormat;

impl Qwen4ArtifactFormat {
    pub const fn is_hfqm(&self) -> bool {
        matches!(self, Self::Hfqm)
    }
}

/// Quantized tensor format relevant to the Qwen4 manifest recipe.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum QuantFormat {
    /// MQ4G256V2, HFQM quant_type 44, 136 B per 256 weights.
    Mq4G256V2,
    /// MQ4G128V2, HFQM quant_type 53, 68 B per 128 weights.
    Mq4G128V2,
    /// Native bfloat16 record, HFQM quant_type 16.
    BF16,
    /// Raw signed-I64 metadata record, HFQM quant_type 54.
    I64,
    /// OCP e4m3 codes + BF16 128x128 `s_inv`, expert-blocked, HFQM quant_type
    /// 55 (stock-FP8 routed experts).
    Fp8E4M3B128,
    /// OCP e4m3 codes behind a 16-byte per-tensor BF16 scale header, HFQM
    /// quant_type 56 (stock-FP8 PLE rows).
    Fp8E4M3PerTensor,
    Other(String),
}

/// Shape/format identity for one logical matrix family.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct QuantTensorGeometry {
    pub format: QuantFormat,
    /// HFQM quant_type byte (44 for MQ4G256V2, 53 for MQ4G128V2, 16 for BF16,
    /// 55 for FP8E4M3B128).
    pub quant_type: u8,
    /// Logical input width K of one projection row.
    pub k: usize,
    /// Quantization block width (1 for BF16).
    pub block_size: usize,
}

impl QuantTensorGeometry {
    pub const fn mq4g256v2(k: usize) -> Self {
        Self {
            format: QuantFormat::Mq4G256V2,
            quant_type: 44,
            k,
            block_size: 256,
        }
    }

    pub const fn mq4g128v2(k: usize) -> Self {
        Self {
            format: QuantFormat::Mq4G128V2,
            quant_type: 53,
            k,
            block_size: 128,
        }
    }

    /// Source BF16 matrix rows (qt=16), loaded raw.
    pub const fn bf16(k: usize) -> Self {
        Self {
            format: QuantFormat::BF16,
            quant_type: 16,
            k,
            block_size: 1,
        }
    }

    /// qt=55 FP8E4M3B128 expert blocks with reduction width `k`.
    pub const fn fp8e4m3b128(k: usize) -> Self {
        Self {
            format: QuantFormat::Fp8E4M3B128,
            quant_type: 55,
            k,
            block_size: 128,
        }
    }

    /// Bytes emitted for `rows` logical rows, including every row-local group.
    ///
    /// `None` means the geometry is not a valid exact extent (zero dimensions,
    /// overflow, or an aligned MQ4G256V2 tensor whose K is not divisible by
    /// 256).  The row product is intentionally supplied by the caller so
    /// stacked expert and ordinary matrix shapes share this check.  For the
    /// expert-blocked FP8E4M3B128 format `rows` is one block's rows (a single
    /// expert, `E = 1`); stacked extents come from
    /// [`Self::expert_payload_bytes`].
    pub fn payload_bytes(&self, rows: usize) -> Option<usize> {
        self.expert_payload_bytes(1, rows)
    }

    /// Bytes of `experts` stacked `[rows, k]` matrices.  Row-strided formats
    /// are `experts * rows` rows; FP8E4M3B128 is `experts` blocks of
    /// `S(rows, k) = round_up(rows*k + 2*ceil(rows/128)*ceil(k/128), 256)`.
    pub fn expert_payload_bytes(&self, experts: usize, rows: usize) -> Option<usize> {
        if experts == 0 || rows == 0 || self.k == 0 || self.block_size == 0 {
            return None;
        }
        let group_bytes = match self.format {
            QuantFormat::Mq4G256V2 if self.quant_type == 44 && self.block_size == 256 => 136,
            QuantFormat::Mq4G128V2 if self.quant_type == 53 && self.block_size == 128 => 68,
            QuantFormat::BF16 if self.quant_type == 16 && self.block_size == 1 => 2,
            QuantFormat::Fp8E4M3B128 if self.quant_type == 55 && self.block_size == 128 => {
                return rdna_compute::fp8_e4m3::b128_tensor_bytes(experts, rows, self.k);
            }
            _ => return None,
        };
        if matches!(self.format, QuantFormat::Mq4G256V2) && self.k % 256 != 0 {
            return None;
        }
        let groups = self
            .k
            .checked_add(self.block_size - 1)?
            .checked_div(self.block_size)?;
        experts
            .checked_mul(rows)?
            .checked_mul(groups)?
            .checked_mul(group_bytes)
    }
}

/// Exact Qwen4 manifest storage recipe.  The packed recipe uses the aligned
/// MQ4G256V2 form for every logical matrix whose K is divisible by 256 and the
/// row-local MQ4G128V2 form otherwise; the stock-FP8 recipe
/// ([`Self::stock_fp8`]) keeps every matrix source BF16 and ships the routed
/// experts as FP8E4M3B128 blocks and the PLE rows as FP8E4M3PT.  Non-matrix
/// payload classes retain their native wire representation.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Qwen4QuantGeometry {
    pub matrix_aligned: QuantTensorGeometry,
    pub matrix_unaligned: QuantTensorGeometry,
    pub routed_gate_up: QuantTensorGeometry,
    pub routed_down: QuantTensorGeometry,
    pub ple_row_dtype: QuantFormat,
    pub scalar_dtype: QuantFormat,
    pub vector_dtype: QuantFormat,
    pub conv_dtype: QuantFormat,
    pub metadata_dtype: QuantFormat,
}

/// Compatibility spelling for callers that call expert projections simply
/// gate/up and down.
pub type QuantGeometry = Qwen4QuantGeometry;

const ROUTED_EXPERTS: usize = 512;
const ROUTED_GATE_UP_EXPERT_ROWS: usize = 1_280;
const ROUTED_DOWN_EXPERT_ROWS: usize = 2_560;
const ROUTED_GATE_UP_ROWS: usize = ROUTED_EXPERTS * ROUTED_GATE_UP_EXPERT_ROWS;
const ROUTED_GATE_UP_BYTES: usize = 891_289_600;
const ROUTED_DOWN_ROWS: usize = ROUTED_EXPERTS * ROUTED_DOWN_EXPERT_ROWS;
const ROUTED_DOWN_BYTES: usize = 445_644_800;
/// Stock-FP8 gate/up: 512 blocks of `S(1280, 2560) = 3,277,312` bytes.
const FP8_ROUTED_GATE_UP_BYTES: usize = 512 * 3_277_312;
/// Stock-FP8 down: 512 blocks of `S(2560, 640) = 1,638,656` bytes.
const FP8_ROUTED_DOWN_BYTES: usize = 512 * 1_638_656;

impl Qwen4QuantGeometry {
    /// Manifest-derived recipe for the pinned Qwen4 text checkpoint.
    pub const fn manifest() -> Self {
        Self {
            matrix_aligned: QuantTensorGeometry::mq4g256v2(2_560),
            matrix_unaligned: QuantTensorGeometry::mq4g128v2(640),
            routed_gate_up: QuantTensorGeometry::mq4g256v2(2_560),
            routed_down: QuantTensorGeometry::mq4g128v2(640),
            ple_row_dtype: QuantFormat::BF16,
            scalar_dtype: QuantFormat::BF16,
            vector_dtype: QuantFormat::BF16,
            conv_dtype: QuantFormat::BF16,
            metadata_dtype: QuantFormat::I64,
        }
    }

    /// The stock-FP8 recipe (`Qwen4TierRecipe::StockFp8`): BF16 trunk
    /// matrices at both widths, FP8E4M3B128 routed experts, FP8E4M3PT PLE rows.
    pub const fn stock_fp8() -> Self {
        Self {
            matrix_aligned: QuantTensorGeometry::bf16(2_560),
            matrix_unaligned: QuantTensorGeometry::bf16(640),
            routed_gate_up: QuantTensorGeometry::fp8e4m3b128(2_560),
            routed_down: QuantTensorGeometry::fp8e4m3b128(640),
            ple_row_dtype: QuantFormat::Fp8E4M3PerTensor,
            scalar_dtype: QuantFormat::BF16,
            vector_dtype: QuantFormat::BF16,
            conv_dtype: QuantFormat::BF16,
            metadata_dtype: QuantFormat::I64,
        }
    }

    /// The geometry of `recipe`'s artifact (the packed and synthetic recipes
    /// are judged against the pinned manifest geometry).
    pub const fn for_recipe(recipe: &Qwen4TierRecipe) -> Self {
        if recipe.is_stock_fp8() {
            Self::stock_fp8()
        } else {
            Self::manifest()
        }
    }

    pub const fn expected() -> Self {
        Self::manifest()
    }

    /// Whether any component names a stock-FP8 format; such a geometry is
    /// judged against [`Self::stock_fp8`] as a whole, so a mix is refused.
    pub fn is_stock_fp8(&self) -> bool {
        self.routed_gate_up.format == QuantFormat::Fp8E4M3B128
            || self.routed_down.format == QuantFormat::Fp8E4M3B128
            || self.ple_row_dtype == QuantFormat::Fp8E4M3PerTensor
    }

    /// Validate the exact matrix policy, native payload classes, and routed
    /// extents selected by the Qwen4 manifest (or the stock-FP8 recipe).
    pub fn validate(&self) -> Result<(), QuantAdmissionError> {
        let stock_fp8 = self.is_stock_fp8();
        let expected = if stock_fp8 {
            Self::stock_fp8()
        } else {
            Self::expected()
        };
        for (component, got, expected_geometry) in [
            (
                "aligned matrix",
                &self.matrix_aligned,
                &expected.matrix_aligned,
            ),
            (
                "unaligned matrix",
                &self.matrix_unaligned,
                &expected.matrix_unaligned,
            ),
            (
                "routed gate/up",
                &self.routed_gate_up,
                &expected.routed_gate_up,
            ),
            ("routed down", &self.routed_down, &expected.routed_down),
        ] {
            if got != expected_geometry {
                return Err(QuantAdmissionError::WrongGeometry {
                    component,
                    expected: expected_geometry.clone(),
                    got: got.clone(),
                });
            }
        }
        for (component, got, expected_dtype) in [
            ("PLE rows", &self.ple_row_dtype, &expected.ple_row_dtype),
            ("scalars", &self.scalar_dtype, &expected.scalar_dtype),
            ("vectors", &self.vector_dtype, &expected.vector_dtype),
            ("convolution", &self.conv_dtype, &expected.conv_dtype),
            ("metadata", &self.metadata_dtype, &expected.metadata_dtype),
        ] {
            if got != expected_dtype {
                return Err(QuantAdmissionError::WrongDtype {
                    component,
                    expected: expected_dtype.clone(),
                    got: got.clone(),
                });
            }
        }
        let (gate_up_bytes, down_bytes) = if stock_fp8 {
            (
                self.routed_gate_up
                    .expert_payload_bytes(ROUTED_EXPERTS, ROUTED_GATE_UP_EXPERT_ROWS)
                    == Some(FP8_ROUTED_GATE_UP_BYTES),
                self.routed_down
                    .expert_payload_bytes(ROUTED_EXPERTS, ROUTED_DOWN_EXPERT_ROWS)
                    == Some(FP8_ROUTED_DOWN_BYTES),
            )
        } else {
            (
                self.routed_gate_up.payload_bytes(ROUTED_GATE_UP_ROWS) == Some(ROUTED_GATE_UP_BYTES),
                self.routed_down.payload_bytes(ROUTED_DOWN_ROWS) == Some(ROUTED_DOWN_BYTES),
            )
        };
        if !gate_up_bytes {
            return Err(QuantAdmissionError::WrongGeometry {
                component: "routed gate/up extent",
                expected: expected.routed_gate_up,
                got: self.routed_gate_up.clone(),
            });
        }
        if !down_bytes {
            return Err(QuantAdmissionError::WrongGeometry {
                component: "routed down extent",
                expected: expected.routed_down,
                got: self.routed_down.clone(),
            });
        }
        Ok(())
    }
}

/// One artifact identity presented to admission.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen4Artifact {
    pub format: Qwen4ArtifactFormat,
    pub quant_geometry: Qwen4QuantGeometry,
}

impl Qwen4Artifact {
    pub const fn new(format: Qwen4ArtifactFormat, quant_geometry: Qwen4QuantGeometry) -> Self {
        Self {
            format,
            quant_geometry,
        }
    }

    pub const fn manifest_hfqm() -> Self {
        Self::new(Qwen4ArtifactFormat::Hfqm, Qwen4QuantGeometry::manifest())
    }

    /// The stock-FP8 HFQM artifact (`--recipe stock-fp8`).
    pub const fn stock_fp8_hfqm() -> Self {
        Self::new(Qwen4ArtifactFormat::Hfqm, Qwen4QuantGeometry::stock_fp8())
    }
}

/// Pure request to the Qwen4 admission boundary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen4AdmissionRequest {
    pub modality: InputModality,
    pub native_mtp: bool,
    pub effective_mesh: EffectiveMesh,
    pub artifact: Qwen4Artifact,
    /// GPU architecture the load targets (`gfx942`, `gfx1201`, ...).  `None`
    /// is an unknown device, which has no FP8 expert kernels: a stock-FP8
    /// artifact is refused there (fail closed).
    pub gpu_arch: Option<String>,
}

impl Qwen4AdmissionRequest {
    pub const fn text_ar(effective_mesh: EffectiveMesh, artifact: Qwen4Artifact) -> Self {
        Self {
            modality: InputModality::Text,
            native_mtp: false,
            effective_mesh,
            artifact,
            gpu_arch: None,
        }
    }

    pub const fn text_mtp(effective_mesh: EffectiveMesh, artifact: Qwen4Artifact) -> Self {
        Self {
            modality: InputModality::Text,
            native_mtp: true,
            effective_mesh,
            artifact,
            gpu_arch: None,
        }
    }

    /// This request on the device `gpu_arch`.
    pub fn on_device(self, gpu_arch: impl Into<String>) -> Self {
        Self {
            gpu_arch: Some(gpu_arch.into()),
            ..self
        }
    }
}

/// Successful pure admission result.  This value carries only validated
/// architecture/capability/topology facts; GPU allocations and source readers
/// remain owned by the later carrier lifecycle.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen4Admission {
    pub architecture_id: u32,
    pub capabilities: Qwen4Capabilities,
    pub effective_mesh: EffectiveMesh,
    pub native_mtp: bool,
}

/// Why a request was refused.  Every variant is detected before allocation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AdmissionError {
    InvalidConfig(String),
    UnsupportedModality(InputModality),
    UnsupportedCapability(Qwen4Capability),
    NonSingleMesh(EffectiveMesh),
    UnsupportedFormat(Qwen4ArtifactFormat),
    WrongGeometry {
        component: &'static str,
        expected: QuantTensorGeometry,
        got: QuantTensorGeometry,
    },
    WrongDtype {
        component: &'static str,
        expected: QuantFormat,
        got: QuantFormat,
    },
    /// A stock-FP8 artifact (qt=55 experts) on a device without FP8 expert
    /// kernels (anything but [`FP8_EXPERTS_ARCH`]).
    Fp8ExpertsUnsupported { gpu_arch: Option<String> },
    /// Native MTP requested for the AR-only stock-FP8 artifact.
    ArOnlyArtifact,
    /// The stock-FP8 artifact is admitted fully resident on one device only
    /// (P1): no expert parallelism.
    Fp8SingleDeviceOnly(EffectiveMesh),
    /// The stock-FP8 artifact with routed experts placed in host RAM.
    Fp8HostMappedExperts,
    /// The stock-FP8 artifact under a config that is not the AR-only
    /// Flash-Next profile.
    Fp8ProfileMismatch(String),
}

/// Alias retaining the distinction between format/quant admission errors and
/// config parser errors for callers that expose one error type.
pub type Qwen4AdmissionError = AdmissionError;

impl fmt::Display for AdmissionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(error) => write!(f, "qwen4 admission: invalid config: {error}"),
            Self::UnsupportedModality(modality) => {
                write!(f, "qwen4 admission: unsupported modality {modality:?}")
            }
            Self::UnsupportedCapability(capability) => {
                write!(f, "qwen4 admission: unsupported capability {capability:?}")
            }
            Self::NonSingleMesh(mesh) => write!(
                f,
                "qwen4 admission: only pp=1, tp=1 with expert-parallel ep in 1..=8, or tp=1, ep=1 with pipeline-parallel pp in 2..=8 is admitted (pp={}, tp={}, ep={})",
                mesh.pp, mesh.tp, mesh.ep
            ),
            Self::UnsupportedFormat(format) => {
                write!(f, "qwen4 admission: unsupported artifact format {format:?}")
            }
            Self::WrongGeometry {
                component,
                expected,
                got,
            } => write!(
                f,
                "qwen4 admission: {component} geometry {:?} does not match {:?}",
                got, expected
            ),
            Self::WrongDtype {
                component,
                expected,
                got,
            } => write!(
                f,
                "qwen4 admission: {component} dtype {got:?} does not match {expected:?}"
            ),
            Self::Fp8ExpertsUnsupported { gpu_arch } => write!(
                f,
                "qwen4 admission: the stock-FP8 artifact (qt=55 FP8E4M3B128 routed experts) \
                 runs only on {FP8_EXPERTS_ARCH} (MI300X); this device is {}; refusing it \
                 (no FP8 expert kernels for this arch)",
                gpu_arch.as_deref().unwrap_or("unknown")
            ),
            Self::ArOnlyArtifact => write!(
                f,
                "qwen4 admission: the stock-FP8 artifact is AR-only (it carries no MTP head); \
                 native MTP (--spec mtp) cannot be admitted"
            ),
            Self::Fp8SingleDeviceOnly(mesh) => write!(
                f,
                "qwen4 admission: the stock-FP8 artifact loads fully resident on one device \
                 (pp=1, tp=1, ep=1); got pp={}, tp={}, ep={}",
                mesh.pp, mesh.tp, mesh.ep
            ),
            Self::Fp8HostMappedExperts => write!(
                f,
                "qwen4 admission: the stock-FP8 artifact keeps every routed expert resident in \
                 device memory; host-mapped experts ({}) are refused",
                crate::expert_residency::EXPERT_VRAM_LAYERS_ENV
            ),
            Self::Fp8ProfileMismatch(profile) => write!(
                f,
                "qwen4 admission: the stock-FP8 artifact is admitted under the AR-only \
                 Flash-Next profile, not {profile}"
            ),
        }
    }
}

impl std::error::Error for AdmissionError {}

/// Successful, immutable result passed to a later loader/carrier.  It carries
/// no GPU or reader handle and therefore cannot imply production admission.
impl Qwen4Admission {
    /// Validate all pure boundaries in order: config, capability, mesh,
    /// container, then exact quant recipe.
    pub fn admit(
        config: &Qwen4Config,
        request: &Qwen4AdmissionRequest,
    ) -> Result<Self, AdmissionError> {
        config.validate().map_err(AdmissionError::InvalidConfig)?;
        let stock_fp8 = request.artifact.quant_geometry.is_stock_fp8();
        if stock_fp8 && !config.profile.is_ar_only() {
            return Err(AdmissionError::Fp8ProfileMismatch(
                config.profile.id().to_string(),
            ));
        }
        let capabilities = if request.native_mtp {
            Qwen4Capabilities::text_mtp()
        } else {
            Qwen4Capabilities::text_ar()
        };
        let capabilities = match request.gpu_arch.as_deref() {
            Some(gpu_arch) => capabilities.on_device(gpu_arch),
            None => capabilities,
        };
        if !capabilities.supports_modality(request.modality) {
            return Err(AdmissionError::UnsupportedModality(request.modality));
        }
        if request.native_mtp && !capabilities.supports(Qwen4Capability::NativeMtp) {
            return Err(AdmissionError::UnsupportedCapability(
                Qwen4Capability::NativeMtp,
            ));
        }
        if !request.effective_mesh.is_qwen4_admitted() {
            return Err(AdmissionError::NonSingleMesh(request.effective_mesh));
        }
        if !request.artifact.format.is_hfqm() {
            return Err(AdmissionError::UnsupportedFormat(
                request.artifact.format.clone(),
            ));
        }
        if stock_fp8 {
            check_fp8_device(
                capabilities,
                request.gpu_arch.as_deref(),
                request.effective_mesh,
                request.native_mtp,
                false,
            )?;
        }
        request
            .artifact
            .quant_geometry
            .validate()
            .map_err(|error| match error {
                QuantAdmissionError::WrongGeometry {
                    component,
                    expected,
                    got,
                } => AdmissionError::WrongGeometry {
                    component,
                    expected,
                    got,
                },
                QuantAdmissionError::WrongDtype {
                    component,
                    expected,
                    got,
                } => AdmissionError::WrongDtype {
                    component,
                    expected,
                    got,
                },
            })?;
        Ok(Self {
            architecture_id: ARCH_ID,
            capabilities,
            effective_mesh: request.effective_mesh,
            native_mtp: request.native_mtp,
        })
    }

    /// Free-function style entry point for callers that do not retain an
    /// admission object as a type.
    pub fn admit_text_mtp(
        config: &Qwen4Config,
        request: &Qwen4AdmissionRequest,
    ) -> Result<Self, AdmissionError> {
        Self::admit(config, request)
    }
}

/// The stock-FP8 artifact's device admission: FP8 expert kernels (gfx942
/// only), AR-only, single device, every routed expert resident.
fn check_fp8_device(
    capabilities: Qwen4Capabilities,
    gpu_arch: Option<&str>,
    mesh: EffectiveMesh,
    native_mtp: bool,
    host_mapped_experts: bool,
) -> Result<(), AdmissionError> {
    if !capabilities.supports(Qwen4Capability::Fp8Experts) {
        return Err(AdmissionError::Fp8ExpertsUnsupported {
            gpu_arch: gpu_arch.map(str::to_string),
        });
    }
    if native_mtp {
        return Err(AdmissionError::ArOnlyArtifact);
    }
    if !mesh.is_single() {
        return Err(AdmissionError::Fp8SingleDeviceOnly(mesh));
    }
    if host_mapped_experts {
        return Err(AdmissionError::Fp8HostMappedExperts);
    }
    Ok(())
}

/// Loader-side device admission of an admitted artifact's `recipe`: the
/// capabilities `gpu_arch` offers it.  A recipe without FP8 experts is
/// device-neutral here; the stock-FP8 recipe is admitted only on
/// [`FP8_EXPERTS_ARCH`], AR-only (`native_mtp` refused), on a single device,
/// and with no host-mapped expert placement requested
/// (`host_mapped_experts`: an explicit `HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS`).
pub fn admit_recipe_on_device(
    recipe: &Qwen4TierRecipe,
    gpu_arch: &str,
    mesh: EffectiveMesh,
    native_mtp: bool,
    host_mapped_experts: bool,
) -> Result<Qwen4Capabilities, AdmissionError> {
    let capabilities = if native_mtp {
        Qwen4Capabilities::text_mtp()
    } else {
        Qwen4Capabilities::text_ar()
    }
    .on_device(gpu_arch);
    if recipe.fp8_experts() {
        check_fp8_device(capabilities, Some(gpu_arch), mesh, native_mtp, host_mapped_experts)?;
    }
    Ok(capabilities)
}

/// Errors local to quant geometry validation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QuantAdmissionError {
    WrongGeometry {
        component: &'static str,
        expected: QuantTensorGeometry,
        got: QuantTensorGeometry,
    },
    WrongDtype {
        component: &'static str,
        expected: QuantFormat,
        got: QuantFormat,
    },
}

/// Free device bytes a pipeline-parallel rank keeps beyond its band census:
/// the first long request's other lazy allocations, kernel-cache warmup and
/// VMM page rounding (the loader's single-device headroom, 2 GiB).
pub const QWEN4_PP_RESERVE_BYTES: u64 = 2 << 30;

/// Inputs of [`qwen4_pp_band_budget`]. Everything is a value the loader has
/// before its first allocation: the manifest's entries, payload sizes from the
/// open artifact, the resolved state format, context and scratch mode, and
/// each rank's measured free VRAM.
pub struct Qwen4PpBudgetInputs<'a> {
    pub config: &'a Qwen4Config,
    /// The admitted placement (`BandPlacement::with_first_band_min(2)`).
    pub placement: &'a hipfire_runtime::mesh_placement::BandPlacement,
    /// `manifest.weights`, the whole manifest. The output band's `mtp.*` head
    /// entries (experts, projections, the tied embedding's materialization)
    /// are counted: the band loads them whether or not the head attaches.
    pub weights: &'a [hipfire_runtime::weight_manifest::WeightEntry],
    pub bytes_of: &'a dyn Fn(&hipfire_runtime::weight_manifest::WeightEntry) -> Option<u64>,
    pub format: crate::Qwen4StateFormat,
    /// The actual backend with the whole admitted context mapped.
    pub context: &'a crate::Qwen4ContextCommit,
    pub chunk_rows: usize,
    pub mode: crate::gpu_forward::Qwen4ForwardScratchMode,
    /// Free VRAM by LOGICAL rank (`AdmittedMesh::ranks` order).
    pub free_per_rank: &'a [u64],
    pub reserve_bytes: u64,
    /// The native MTP request, `None` for an AR load (no MTP reserve and no
    /// row capture on any band): the output band charges the head and its
    /// request buffers, every band its own row capture.
    pub native_mtp: Option<crate::expert_residency::NativeMtpCensusInputs>,
}

/// One rank's admitted budget line.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Qwen4PpRankFit {
    pub stage: usize,
    pub band: std::ops::Range<usize>,
    /// Weights + F16 shadows + QSA context + state + forward scratch + native
    /// MTP + reserve.
    pub required: u64,
    /// The native MTP term of `required` (0 for an AR load).
    pub native_mtp: u64,
    pub free: u64,
}

/// Per-rank band budget from the S5 band census: every rank's required bytes
/// must fit its free VRAM (no host-mapped band layers). Pure; refuses before
/// any allocation, naming `topology.pp`, every failing rank and its shortfall.
pub fn qwen4_pp_band_budget(i: &Qwen4PpBudgetInputs<'_>) -> Result<Vec<Qwen4PpRankFit>, String> {
    use crate::weights::Qwen4BandScope;
    let stages = i.placement.stages();
    let prefix = format!("qwen4: pipeline parallelism (topology.pp = {stages})");
    if i.free_per_rank.len() != stages {
        return Err(format!(
            "{prefix}: {} free-VRAM readings for {stages} ranks",
            i.free_per_rank.len()
        ));
    }
    let overflow = || format!("{prefix}: band budget overflows");
    let mut fits = Vec::with_capacity(stages);
    let mut failures = Vec::new();
    for stage in 0..stages {
        let scope = Qwen4BandScope::from_placement(i.config, i.placement, stage)
            .map_err(|error| format!("{prefix}: {error}"))?;
        let census = crate::expert_residency::band_census_native_mtp(
            i.config,
            i.weights,
            |entry| (i.bytes_of)(entry),
            &scope,
            i.format,
            i.context,
            i.chunk_rows,
            i.mode,
            i.native_mtp.as_ref(),
        )
        .map_err(|error| format!("{prefix}: rank {stage}: {error}"))?;
        let weight_bytes = census
            .weight_bytes()
            .map_err(|e| format!("{prefix}: {e}"))?;
        let native_mtp = census
            .mtp
            .reserve()
            .map_err(|e| format!("{prefix}: rank {stage}: {e}"))?;
        let required = [
            census.f16_shadow_bytes,
            census.state.qsa_context,
            census.state.state,
            census.state.forward,
            native_mtp,
            i.reserve_bytes,
        ]
        .into_iter()
        .try_fold(weight_bytes, |total, bytes| total.checked_add(bytes))
        .ok_or_else(overflow)?;
        let free = i.free_per_rank[stage];
        if required > free {
            failures.push(format!(
                "rank {stage} (layers {}..{}) needs {required} bytes ({weight_bytes} weights + {} F16 shadows + {} QSA context + {} state + {} forward scratch + {native_mtp} native MTP + {} reserve) but {free} bytes are free (short by {})",
                scope.band.start,
                scope.band.end,
                census.f16_shadow_bytes,
                census.state.qsa_context,
                census.state.state,
                census.state.forward,
                i.reserve_bytes,
                required - free
            ));
        }
        fits.push(Qwen4PpRankFit {
            stage,
            band: scope.band,
            required,
            native_mtp,
            free,
        });
    }
    if failures.is_empty() {
        Ok(fits)
    } else {
        Err(format!(
            "{prefix}: band does not fit device memory: {}; rebalance HIPFIRE_PP_LAYERS or lower memory.max_seq (host-mapped band layers are not admitted)",
            failures.join("; ")
        ))
    }
}

/// Largest streams payload of one band edge at `chunk_rows` canonical rows:
/// `rows * hc_count * hidden * 4` (the F32 worst case; BF16 halves it), the
/// `max_bytes` for `Gpus::provision_mesh_edges`.
pub fn qwen4_pp_edge_max_bytes(config: &Qwen4Config, chunk_rows: usize) -> Result<usize, String> {
    if chunk_rows == 0 {
        return Err("qwen4: pipeline-parallel edge payload for zero rows".to_string());
    }
    chunk_rows
        .checked_mul(config.hc_count)
        .and_then(|bytes| bytes.checked_mul(config.hidden_size))
        .and_then(|bytes| bytes.checked_mul(std::mem::size_of::<f32>()))
        .filter(|bytes| bytes % 16 == 0)
        .ok_or_else(|| {
            format!("qwen4: pipeline-parallel edge payload for {chunk_rows} rows is not a 16-byte multiple or overflows")
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Qwen4Config;
    use serde_json::json;

    fn config() -> Qwen4Config {
        let layers: Vec<_> = (0..48)
            .map(|idx| {
                if idx % 4 == 3 {
                    "full_attention"
                } else {
                    "linear_attention"
                }
            })
            .collect();
        Qwen4Config::from_value(&json!({
            "model_type": "qwen4_exp",
            "text_config": {
                "model_type": "qwen4_exp_text", "dtype": "bfloat16",
                "hidden_size": 2560, "vocab_size": 248320, "num_hidden_layers": 48,
                "max_position_embeddings": 262144, "num_attention_heads": 24,
                "num_key_value_heads": 2, "head_dim": 256, "partial_rotary_factor": 0.25,
                "rope_parameters": {"rope_theta": 10000000.0}, "attention_bias": false,
                "layer_types": layers, "full_attention_interval": 4,
                "linear_num_key_heads": 16, "linear_num_value_heads": 48,
                "linear_key_head_dim": 128, "linear_value_head_dim": 128,
                "linear_conv_kernel_dim": 4, "mamba_ssm_dtype": "float32",
                "indexer_n_heads": 4, "indexer_kv_heads": 1, "indexer_head_dim": 128,
                "indexer_budget": 2048, "indexer_compress_ratio": 4,
                "num_experts": 512, "num_experts_per_tok": 10,
                "moe_intermediate_size": 640, "shared_expert_intermediate_size": 640,
                "hc_count": 4, "hc_lowrank": 320, "ple_layer_ids": [2],
                "ple_conv_kernel_size": 4, "ple_embed_dim": 2560,
                "split_ngram_parts": 128, "heads_per_ngram": 8, "ngram_size": 3,
                "ngram_vocab_size_base": 20000000, "make_ngram_vocab_size_divisible_by": 128,
                "eos_token_id": 248044, "tie_word_embeddings": false,
                "mtp_num_hidden_layers": 1, "mtp_use_dedicated_embeddings": false,
                "output_gate_type": "sigmoid",
                "mtp": {"hybrid": true, "layer_types": ["full_attention"],
                        "num_hidden_layers": 1, "rope_theta": 10000000.0}
            }
        }))
        .unwrap()
    }

    #[test]
    fn native_mtp_is_admitted_when_requested() {
        let request = Qwen4AdmissionRequest::text_mtp(
            EffectiveMesh::single(),
            Qwen4Artifact::manifest_hfqm(),
        );
        let mtp = Qwen4Admission::admit(&config(), &request).expect("native MTP is admitted");
        assert!(mtp.capabilities.text);
        assert!(mtp.capabilities.native_mtp);
        assert!(mtp.native_mtp);

        let ar = Qwen4Admission::admit(
            &config(),
            &Qwen4AdmissionRequest::text_ar(
                EffectiveMesh::single(),
                Qwen4Artifact::manifest_hfqm(),
            ),
        )
        .expect("AR remains admitted");
        assert!(ar.capabilities.text);
        assert!(!ar.capabilities.native_mtp);
        assert!(!ar.native_mtp);
    }

    #[test]
    fn images_videos_meshes_and_formats_fail_closed() {
        let mut request =
            Qwen4AdmissionRequest::text_ar(EffectiveMesh::single(), Qwen4Artifact::manifest_hfqm());
        request.modality = InputModality::Image;
        assert!(matches!(
            Qwen4Admission::admit(&config(), &request),
            Err(AdmissionError::UnsupportedModality(InputModality::Image))
        ));
        request.modality = InputModality::Video;
        assert!(matches!(
            Qwen4Admission::admit(&config(), &request),
            Err(AdmissionError::UnsupportedModality(InputModality::Video))
        ));
        request.modality = InputModality::Text;
        request.effective_mesh = EffectiveMesh::new(1, 2, 1);
        assert!(matches!(
            Qwen4Admission::admit(&config(), &request),
            Err(AdmissionError::NonSingleMesh(_))
        ));
        request.effective_mesh = EffectiveMesh::single();
        request.artifact.format = Qwen4ArtifactFormat::Safetensors;
        assert!(matches!(
            Qwen4Admission::admit(&config(), &request),
            Err(AdmissionError::UnsupportedFormat(
                Qwen4ArtifactFormat::Safetensors
            ))
        ));
    }

    #[test]
    fn wrong_quant_geometry_is_refused() {
        let mut artifact = Qwen4Artifact::manifest_hfqm();
        artifact.quant_geometry.routed_down.k = 2560;
        let request = Qwen4AdmissionRequest::text_ar(EffectiveMesh::single(), artifact);
        assert!(matches!(
            Qwen4Admission::admit(&config(), &request),
            Err(AdmissionError::WrongGeometry {
                component: "routed down",
                ..
            })
        ));
    }
    #[test]
    fn manifest_recipe_has_exact_extents_and_native_payload_classes() {
        let geometry = Qwen4QuantGeometry::manifest();
        assert_eq!(
            geometry
                .matrix_aligned
                .payload_bytes(2)
                .expect("aligned matrix extent"),
            2 * 10 * 136
        );
        assert_eq!(
            geometry
                .matrix_unaligned
                .payload_bytes(2)
                .expect("row-local matrix extent"),
            2 * 5 * 68
        );
        assert_eq!(
            geometry.routed_gate_up.payload_bytes(ROUTED_GATE_UP_ROWS),
            Some(ROUTED_GATE_UP_BYTES)
        );
        assert_eq!(
            geometry.routed_down.payload_bytes(ROUTED_DOWN_ROWS),
            Some(ROUTED_DOWN_BYTES)
        );
        assert_eq!(geometry.ple_row_dtype, QuantFormat::BF16);
        assert_eq!(geometry.scalar_dtype, QuantFormat::BF16);
        assert_eq!(geometry.vector_dtype, QuantFormat::BF16);
        assert_eq!(geometry.conv_dtype, QuantFormat::BF16);
        assert_eq!(geometry.metadata_dtype, QuantFormat::I64);
        geometry.validate().expect("manifest recipe");
    }

    #[test]
    fn effective_mesh_is_derived_from_the_normalized_device_mesh() {
        use hipfire_runtime::device_mesh::{DeviceMesh, DimKind};
        let single = DeviceMesh::single().expect("single mesh");
        assert_eq!(
            EffectiveMesh::from_device_mesh(&single),
            EffectiveMesh::single()
        );
        let ep = DeviceMesh::rect(&[(DimKind::Ep, 6)]).expect("ep mesh");
        let derived = EffectiveMesh::from_device_mesh(&ep);
        assert_eq!(derived, EffectiveMesh::new(1, 1, 6));
        assert!(derived.is_qwen4_admitted());
        let pp = DeviceMesh::rect(&[(DimKind::Pp, 2), (DimKind::Ep, 2)]).expect("pp mesh");
        let derived = EffectiveMesh::from_device_mesh(&pp);
        assert_eq!(derived, EffectiveMesh::new(2, 1, 2));
        assert!(!derived.is_qwen4_admitted());
    }

    #[test]
    fn ep_mesh_matrix_admits_one_through_eight() {
        let admit = |mesh: EffectiveMesh, mtp: bool| {
            let artifact = Qwen4Artifact::manifest_hfqm();
            let request = if mtp {
                Qwen4AdmissionRequest::text_mtp(mesh, artifact)
            } else {
                Qwen4AdmissionRequest::text_ar(mesh, artifact)
            };
            Qwen4Admission::admit(&config(), &request)
        };
        for ep in 1..=8 {
            let mesh = EffectiveMesh::new(1, 1, ep);
            assert!(mesh.is_qwen4_admitted());
            assert_eq!(
                admit(mesh, false)
                    .expect("ep AR admitted")
                    .effective_mesh
                    .ep,
                ep
            );
        }
        assert!(EffectiveMesh::single().is_single());
        assert!(!EffectiveMesh::new(1, 1, 2).is_single());
        // 0 is malformed; above the 8-rank maximum is refused.
        for ep in [0, 9, 16] {
            assert!(matches!(
                admit(EffectiveMesh::new(1, 1, ep), false),
                Err(AdmissionError::NonSingleMesh(_))
            ));
        }
        // tp stays refused at every pp/ep, and pp never composes with ep.
        for (pp, tp, ep) in [
            (1, 2, 1),
            (2, 1, 2),
            (1, 2, 4),
            (4, 1, 4),
            (1, 2, 6),
            (2, 2, 1),
            (9, 1, 1),
            (0, 1, 1),
        ] {
            assert!(matches!(
                admit(EffectiveMesh::new(pp, tp, ep), false),
                Err(AdmissionError::NonSingleMesh(_))
            ));
        }
        // Pure pipeline parallelism is admitted at pp 2..=8 only, AR and
        // native MTP alike (the output band attaches the one head).
        for pp in 2..=8 {
            let mesh = EffectiveMesh::new(pp, 1, 1);
            assert!(mesh.is_qwen4_admitted() && mesh.is_pipeline() && !mesh.is_single());
            let admitted = admit(mesh, false).expect("PP AR admitted");
            assert_eq!(admitted.effective_mesh.pp, pp);
            assert!(!admitted.native_mtp);
            let admitted = admit(mesh, true).expect("PP native MTP admitted");
            assert_eq!(admitted.effective_mesh.pp, pp);
            assert!(admitted.native_mtp && admitted.capabilities.native_mtp);
        }
        assert!(!EffectiveMesh::single().is_pipeline());
        assert!(!EffectiveMesh::new(1, 1, 4).is_pipeline());
        // Native MTP is admitted on every admitted mesh: under EP the drafter
        // is replicated on every rank.
        assert!(admit(EffectiveMesh::single(), true).is_ok());
        for ep in [2, 3, 4, 6, 8] {
            let admitted = admit(EffectiveMesh::new(1, 1, ep), true).expect("EP native MTP");
            assert!(admitted.native_mtp);
            assert_eq!(admitted.effective_mesh.ep, ep);
        }
    }

    /// The stock-FP8 geometry: BF16 trunk, qt=55 experts at the contract
    /// stride, qt=56 PLE rows; a mix with the packed geometry is refused.
    #[test]
    fn stock_fp8_geometry_has_contract_extents() {
        let geometry = Qwen4QuantGeometry::stock_fp8();
        geometry.validate().expect("stock-FP8 recipe");
        assert!(geometry.is_stock_fp8() && !Qwen4QuantGeometry::manifest().is_stock_fp8());
        assert_eq!(
            Qwen4QuantGeometry::for_recipe(&Qwen4TierRecipe::StockFp8),
            geometry
        );
        // One block: S(R,K) = round_up(R*K + 2*ceil(R/128)*ceil(K/128), 256).
        assert_eq!(geometry.routed_gate_up.payload_bytes(1_280), Some(3_277_312));
        assert_eq!(geometry.routed_down.payload_bytes(2_560), Some(1_638_656));
        assert_eq!(
            geometry.routed_gate_up.expert_payload_bytes(512, 1_280),
            Some(1_677_983_744)
        );
        assert_eq!(
            geometry.routed_down.expert_payload_bytes(512, 2_560),
            Some(838_991_872)
        );
        // Non-multiple-of-128 shapes pad their scale grid and the block.
        assert_eq!(
            QuantTensorGeometry::fp8e4m3b128(129).payload_bytes(3),
            Some((3usize * 129 + 2 * 2).div_ceil(256) * 256)
        );
        assert_eq!(geometry.matrix_aligned.payload_bytes(2), Some(2 * 2_560 * 2));
        assert_eq!(geometry.matrix_unaligned.payload_bytes(2), Some(2 * 640 * 2));
        assert_eq!(geometry.ple_row_dtype, QuantFormat::Fp8E4M3PerTensor);

        let mut mixed = Qwen4QuantGeometry::manifest();
        mixed.routed_down = QuantTensorGeometry::fp8e4m3b128(640);
        assert!(matches!(
            mixed.validate(),
            Err(QuantAdmissionError::WrongGeometry {
                component: "aligned matrix",
                ..
            })
        ));
        let mut short = Qwen4QuantGeometry::stock_fp8();
        short.routed_gate_up.block_size = 256;
        assert!(short.validate().is_err());
    }

    /// A stock-FP8 artifact is admitted on gfx942 only (fail closed on every
    /// other device and on an unknown one), AR-only, single device, and only
    /// under the AR-only profile.
    #[test]
    fn stock_fp8_artifact_admits_on_gfx942_only() {
        let ar_only = config().into_ar_only().expect("AR-only Flash-Next");
        let request = |arch: Option<&str>| {
            let request = Qwen4AdmissionRequest::text_ar(
                EffectiveMesh::single(),
                Qwen4Artifact::stock_fp8_hfqm(),
            );
            match arch {
                Some(arch) => request.on_device(arch),
                None => request,
            }
        };
        let admitted =
            Qwen4Admission::admit(&ar_only, &request(Some("gfx942"))).expect("gfx942 admits");
        assert!(admitted.capabilities.fp8_experts);
        assert!(!admitted.native_mtp && !admitted.capabilities.native_mtp);
        for arch in [Some("gfx1201"), Some("gfx1151"), Some("gfx1100"), Some("gfx940"), None] {
            let error = Qwen4Admission::admit(&ar_only, &request(arch)).unwrap_err();
            assert_eq!(
                error,
                AdmissionError::Fp8ExpertsUnsupported {
                    gpu_arch: arch.map(str::to_string)
                }
            );
            assert!(error.to_string().contains("gfx942"), "{error}");
        }
        // Native MTP: the artifact carries no head.
        let mut mtp = request(Some("gfx942"));
        mtp.native_mtp = true;
        assert_eq!(
            Qwen4Admission::admit(&ar_only, &mtp),
            Err(AdmissionError::ArOnlyArtifact)
        );
        // Expert parallelism is refused for P1.
        let mut ep = request(Some("gfx942"));
        ep.effective_mesh = EffectiveMesh::new(1, 1, 2);
        assert!(matches!(
            Qwen4Admission::admit(&ar_only, &ep),
            Err(AdmissionError::Fp8SingleDeviceOnly(_))
        ));
        // The checkpoint profile (which names an MTP head) is not the
        // stock-FP8 artifact's profile.
        assert!(matches!(
            Qwen4Admission::admit(&config(), &request(Some("gfx942"))),
            Err(AdmissionError::Fp8ProfileMismatch(_))
        ));
        // The packed artifact stays device-neutral at this boundary.
        let packed = Qwen4AdmissionRequest::text_ar(
            EffectiveMesh::single(),
            Qwen4Artifact::manifest_hfqm(),
        )
        .on_device("gfx1201");
        let packed = Qwen4Admission::admit(&config(), &packed).expect("packed on gfx1201");
        assert!(!packed.capabilities.fp8_experts);
    }

    /// The loader-side recipe check mirrors the pure admission.
    #[test]
    fn recipe_device_admission_gates_stock_fp8() {
        let single = EffectiveMesh::single();
        let stock = Qwen4TierRecipe::StockFp8;
        let caps = admit_recipe_on_device(&stock, "gfx942", single, false, false).unwrap();
        assert!(caps.fp8_experts && caps.supports(Qwen4Capability::Fp8Experts));
        for arch in ["gfx1201", "gfx1151"] {
            assert!(matches!(
                admit_recipe_on_device(&stock, arch, single, false, false),
                Err(AdmissionError::Fp8ExpertsUnsupported { .. })
            ));
        }
        assert_eq!(
            admit_recipe_on_device(&stock, "gfx942", single, true, false),
            Err(AdmissionError::ArOnlyArtifact)
        );
        assert_eq!(
            admit_recipe_on_device(&stock, "gfx942", single, false, true),
            Err(AdmissionError::Fp8HostMappedExperts)
        );
        assert!(matches!(
            admit_recipe_on_device(&stock, "gfx942", EffectiveMesh::new(1, 1, 4), false, false),
            Err(AdmissionError::Fp8SingleDeviceOnly(_))
        ));
        let packed = Qwen4TierRecipe::FlashNextPacked {
            trunk_tier: rdna_compute::DType::Q8_0,
            mtp_source: false,
        };
        let caps = admit_recipe_on_device(&packed, "gfx1201", single, true, true).unwrap();
        assert!(!caps.fp8_experts && caps.native_mtp);
    }

    mod pp_budget {
        use super::*;
        use crate::expert_residency::NativeMtpCensusInputs;
        use crate::gpu_forward::Qwen4ForwardScratchMode::Full;
        use hipfire_runtime::device_mesh::{DeviceMesh, DimKind};
        use hipfire_runtime::mesh_placement::BandPlacement;
        use hipfire_runtime::weight_manifest::WeightEntry;
        use rdna_compute::tensor_ops::QsaKvFormat;

        fn bytes_of(entry: &WeightEntry) -> Option<u64> {
            Some(entry.logical_shape.iter().product::<usize>() as u64 * 2)
        }

        fn placement(counts: &[usize]) -> BandPlacement {
            let mesh = DeviceMesh::rect(&[(DimKind::Pp, counts.len())]).unwrap();
            BandPlacement::from_counts(&mesh, 48, counts).unwrap()
        }

        fn budget(
            config: &Qwen4Config,
            weights: &[WeightEntry],
            placement: &BandPlacement,
            free: &[u64],
        ) -> Result<Vec<Qwen4PpRankFit>, String> {
            budget_mtp(config, weights, placement, free, None)
        }

        fn budget_mtp(
            config: &Qwen4Config,
            weights: &[WeightEntry],
            placement: &BandPlacement,
            free: &[u64],
            native_mtp: Option<NativeMtpCensusInputs>,
        ) -> Result<Vec<Qwen4PpRankFit>, String> {
            let context = crate::Qwen4ContextCommit::legacy(1024);
            qwen4_pp_band_budget(&Qwen4PpBudgetInputs {
                config,
                placement,
                weights,
                bytes_of: &bytes_of,
                format: crate::Qwen4StateFormat {
                    qsa: QsaKvFormat::Fp8,
                    gdn: crate::GdnStateFormat::Q8,
                    qsa_f16_arena: false,
                    qsa_pooled_bf16: false,
                },
                context: &context,
                chunk_rows: 256,
                mode: Full,
                free_per_rank: free,
                reserve_bytes: QWEN4_PP_RESERVE_BYTES,
                native_mtp,
            })
        }

        #[test]
        fn every_rank_must_fit_its_own_census() {
            let config = crate::config::compact_test_config();
            let manifest = crate::weights::Qwen4Manifest::build(&config).unwrap();
            let placement = placement(&[9, 8, 8, 8, 8, 7]);
            let fits = budget(&config, &manifest.weights, &placement, &[u64::MAX; 6]).unwrap();
            assert_eq!(fits.len(), 6);
            assert_eq!(fits[0].band, 0..9);
            assert_eq!(fits[5].band, 41..48);
            assert!(fits.iter().all(|fit| fit.required > QWEN4_PP_RESERVE_BYTES));
            // Exactly enough fits; one byte short on one rank refuses, naming
            // topology.pp and that rank only.
            let exact: Vec<u64> = fits.iter().map(|fit| fit.required).collect();
            assert!(budget(&config, &manifest.weights, &placement, &exact).is_ok());
            let mut short = exact.clone();
            short[3] -= 1;
            let error = budget(&config, &manifest.weights, &placement, &short).unwrap_err();
            assert!(
                error.contains("topology.pp = 6")
                    && error.contains("rank 3 (layers 25..33)")
                    && error.contains("short by 1")
                    && !error.contains("rank 2")
                    && !error.contains("rank 4"),
                "{error}"
            );
            // The reading count must match the rank count.
            let error = budget(&config, &manifest.weights, &placement, &exact[..5]).unwrap_err();
            assert!(error.contains("topology.pp = 6"), "{error}");
        }

        #[test]
        fn the_output_rank_counts_the_mtp_heads_weights_even_for_ar() {
            // The output band loads its `mtp.*` entries whether or not the
            // head attaches, so an AR budget counts them on that rank only.
            let config = crate::config::compact_test_config();
            let manifest = crate::weights::Qwen4Manifest::build(&config).unwrap();
            assert!(manifest.weights.iter().any(|e| e.name.starts_with("mtp.")));
            let without: Vec<WeightEntry> = manifest
                .weights
                .iter()
                .filter(|e| !e.name.starts_with("mtp."))
                .cloned()
                .collect();
            let placement = placement(&[24, 24]);
            let with = budget(&config, &manifest.weights, &placement, &[u64::MAX; 2]).unwrap();
            let without = budget(&config, &without, &placement, &[u64::MAX; 2]).unwrap();
            assert_eq!(with[0], without[0]);
            assert!(with[1].required > without[1].required);
            assert!(with.iter().all(|fit| fit.native_mtp == 0));
        }

        #[test]
        fn native_mtp_charges_the_head_on_the_output_rank_and_capture_on_every_rank() {
            let config = crate::config::compact_test_config();
            let manifest = crate::weights::Qwen4Manifest::build(&config).unwrap();
            let placement = placement(&[9, 8, 8, 8, 8, 7]);
            let gdn = crate::GdnStateFormat::Q8;
            let mtp = NativeMtpCensusInputs {
                max_k: 3,
                row_capture: Some(gdn),
                batched_fill: false,
            };
            let free = [u64::MAX; 6];
            let ar = budget(&config, &manifest.weights, &placement, &free).unwrap();
            let on = budget_mtp(&config, &manifest.weights, &placement, &free, Some(mtp)).unwrap();
            let capture = |band: &std::ops::Range<usize>| {
                let gdn_layers = band
                    .clone()
                    .filter(|&layer| {
                        config.layer_types[layer] == crate::config::LayerType::LinearAttention
                    })
                    .count();
                crate::state::Qwen4State::row_capture_bytes_scoped(&config, gdn, 4, gdn_layers)
                    .unwrap() as u64
            };
            for (stage, (ar, on)) in ar.iter().zip(&on).enumerate() {
                // The whole MTP term, and nothing else, is the difference.
                assert_eq!(on.required - ar.required, on.native_mtp, "rank {stage}");
                assert_eq!(ar.native_mtp, 0);
                if stage + 1 < 6 {
                    // A rank without the head pays its own capture alone.
                    assert_eq!(on.native_mtp, capture(&on.band), "rank {stage}");
                } else {
                    assert!(on.native_mtp > capture(&on.band));
                }
            }
            // A byte short on the output rank alone refuses, naming the term.
            let mut exact: Vec<u64> = on.iter().map(|fit| fit.required).collect();
            assert!(budget_mtp(&config, &manifest.weights, &placement, &exact, Some(mtp)).is_ok());
            exact[5] -= 1;
            let error =
                budget_mtp(&config, &manifest.weights, &placement, &exact, Some(mtp)).unwrap_err();
            assert!(
                error.contains("rank 5 (layers 41..48)")
                    && error.contains("native MTP")
                    && error.contains("short by 1")
                    && !error.contains("rank 4"),
                "{error}"
            );
            // The same free memory does not fit once MTP is on where the AR
            // budget fit it exactly.
            let ar_exact: Vec<u64> = ar.iter().map(|fit| fit.required).collect();
            assert!(budget(&config, &manifest.weights, &placement, &ar_exact).is_ok());
            assert!(
                budget_mtp(&config, &manifest.weights, &placement, &ar_exact, Some(mtp)).is_err()
            );
        }

        #[test]
        fn a_band_without_the_ple_layer_is_refused_before_any_census() {
            let config = crate::config::compact_test_config();
            let manifest = crate::weights::Qwen4Manifest::build(&config).unwrap();
            // `with_first_band_min(2)` already refuses this split; the budget
            // refuses it independently.
            assert!(placement(&[1, 47]).with_first_band_min(2).is_err());
            let error = budget(
                &config,
                &manifest.weights,
                &placement(&[1, 47]),
                &[u64::MAX; 2],
            )
            .unwrap_err();
            assert!(
                error.contains("topology.pp = 2") && error.contains("PLE"),
                "{error}"
            );
        }

        #[test]
        fn edge_payload_matches_the_frozen_chunk_sizes() {
            let config = crate::config::compact_test_config();
            assert_eq!(qwen4_pp_edge_max_bytes(&config, 1), Ok(40_960));
            assert_eq!(qwen4_pp_edge_max_bytes(&config, 1536), Ok(62_914_560));
            assert_eq!(qwen4_pp_edge_max_bytes(&config, 4096), Ok(167_772_160));
            assert!(qwen4_pp_edge_max_bytes(&config, 0).is_err());
            assert!(qwen4_pp_edge_max_bytes(&config, usize::MAX).is_err());
        }
    }
}
