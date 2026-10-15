// SPDX-License-Identifier: Apache-2.0
//! Weight preparation, program selection and thin GPU glue for the opt-in
//! Flash-Next NPU trunk. References live in peacemaker-npu; no device opens here.

pub(crate) mod admission;
pub(crate) use calibration::{host_facts, npu_firmware, runtime_key_with_firmware};
mod calibration;
mod config;
pub(crate) mod containment;
mod gpu;
mod packing;
mod plan;
mod programs;
mod rates;
mod shapes;

pub(crate) use packing::pack_matrix_direct;
pub(crate) use programs::build_rowmajor_programs_for_families;

pub use calibration::{
    CalKey, Calibration, ContentionCalibration, DeviceKey, FamilyCalibration, GlueCalibration,
    GpuCalibration, KernelKey, NpuCalibration, OutFamily, QsaOrder, CALIBRATION_SCHEMA_VERSION,
    calibration_build_identity, calibration_key, runtime_calibration_key, model_index_identity,
};
pub use plan::{
    bank_segments, bank_segments_moved, contention_extra, crossover, gpu_chunk, load_log_line, npu_ms,
    pair_time, pair_time_moved, plan, plan_executable, plan_moved,
    request_log_line, select_families, BankSegments, GpuChunk, LoadLogKey, Moved, PairTime, Plan,
    SelectionReport, SelectionRow, Step, CHUNK, SELECTION_ROWS,
    auto_admission, executable_plan, executable_plan_for, planned_pair, planned_pair_rows,
};
pub use rates::{
    choose_families, pair_model, scaled_calibration, LayerKind, PairModel, RateEstimator, RateSample, RateSnapshot,
    APPLY_EPS, EWMA_ALPHA, FREEZE_AFTER, FREEZE_EPS, FREEZE_WINDOW, MIN_MODEL_MS, REOPEN_CONFIRM, REOPEN_EPS,
    SCALE_MAX, SCALE_MIN,
};
pub use config::{NpuTrunkMode, NPU_TRUNK_DAEMON_KEY, NPU_TRUNK_ENV};
pub use packing::{
    pack_matrix, pack_out, pack_out_layers, pack_shape, pack_shape_with_contract, pack_trunk, pack_trunk_with_contract,
    LayerTrunkRefs, MatrixRef, PackedB, PackedOut, PackedTrunk,
};
pub use programs::{
    build_out_programs, build_program_for_rows, build_programs, build_programs_for_families, wave_padded_rows,
    NpuOutPrograms, NpuPrograms, ShapeProgram,
};
pub use shapes::{
    ColumnMap, EpilogueContract, Family, FamilyMask, Shape, OUT_COLS, OUT_K, OUT_PANEL_COLS, PANEL_COLS, TRUNK_K,
};

#[cfg(test)]
mod plan_tests;
