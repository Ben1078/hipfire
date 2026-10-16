// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
use rdna_compute::{DType, Gpu};

fn bf16_round_trip(v: f32) -> f32 {
    if !v.is_finite() { return v; }
    let u = v.to_bits();
    f32::from_bits(u.wrapping_add(0x7fff + ((u >> 16) & 1)) & 0xffff0000)
}

/// Full index-order float32 softmax, then top-10 selected renormalization.
/// Input rows retain original ids in increasing order, even when sliced.
fn route(mut logits: Vec<(usize, f32)>, normalize: bool, round: bool) -> (Vec<usize>, Vec<f32>) {
    if round {
        for (_, value) in &mut logits { *value = bf16_round_trip(*value); }
    }
    let mut selected = logits.clone();
    selected.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap().then(a.0.cmp(&b.0)));
    selected.truncate(10);
    let max = selected[0].1;
    let denominator = logits.iter().fold(0.0f32, |sum, (_, v)| sum + (*v - max).exp());
    let inverse = 1.0f32 / denominator;
    let mut probabilities: Vec<f32> = selected.iter().map(|(_, v)| (*v - max).exp() * inverse).collect();
    if normalize {
        let selected_sum = probabilities.iter().fold(0.0f32, |sum, &p| sum + p);
        let inverse_selected = 1.0f32 / selected_sum;
        for p in &mut probabilities { *p *= inverse_selected; }
    }
    if round {
        for p in &mut probabilities { *p = bf16_round_trip(*p); }
    }
    (selected.iter().map(|(id, _)| *id).collect(), probabilities)
}

fn bits(values: &[f32]) -> Vec<u32> { values.iter().map(|v| v.to_bits()).collect() }

#[test]
fn masked_router_equals_sliced_router_original_id_ties() {
    for count in [16, 31, 128, 511, 512] {
        let keep: Vec<usize> = (0..512).filter(|e| (e * 37) % 512 < count).collect();
        for ties in [false, true] {
            let logits: Vec<f32> = (0..512).map(|e| if ties { 0.25 } else { ((e * 13) % 47) as f32 * 0.1273 }).collect();
            for normalize in [false, true] {
                for round in [false, true] {
                    let masked = route(logits.iter().enumerate().map(|(e, &v)| (e, if keep.contains(&e) { v } else { f32::NEG_INFINITY })).collect(), normalize, round);
                    let sliced = route(keep.iter().map(|&e| (e, logits[e])).collect(), normalize, round);
                    assert_eq!(masked.0, sliced.0);
                    assert_eq!(bits(&masked.1), bits(&sliced.1));
                    if ties { assert_eq!(masked.0, keep[..10]); }
                }
            }
        }
    }
}

#[test]
#[ignore = "requires gfx1201 HIP GPU; decode and grouped-prefill masked route"]
fn masked_router_decode_and_prefill_gfx1201() {
    let mut gpu = Gpu::init().expect("HIP GPU");
    assert_eq!(gpu.arch, "gfx1201", "run on the R9700");
    for rows in [1, 2, 4, 17, 128] {
        let keep: Vec<u8> = (0..512).map(|e| u8::from((e * 37) % 512 < 31)).collect();
        let logits: Vec<f32> = (0..rows * 512).map(|i| if i / 512 % 2 == 0 { ((i * 13) % 47) as f32 * 0.1273 } else { 0.25 }).collect();
        let masked_logits: Vec<f32> = logits.iter().enumerate().map(|(i, &v)| if keep[i % 512] != 0 { v } else { f32::NEG_INFINITY }).collect();
        let mask = gpu.upload_raw(&keep, &[512]).unwrap();
        let input = gpu.upload_f32(&logits, &[rows, 512]).unwrap();
        let reference_input = gpu.upload_f32(&masked_logits, &[rows, 512]).unwrap();
        let ids = gpu.alloc_tensor(&[rows, 10], DType::F32).unwrap();
        let weights = gpu.alloc_tensor(&[rows, 10], DType::F32).unwrap();
        let reference_ids = gpu.alloc_tensor(&[rows, 10], DType::F32).unwrap();
        let reference_weights = gpu.alloc_tensor(&[rows, 10], DType::F32).unwrap();
        for normalize in [false, true] {
            for round in [false, true] {
                gpu.moe_router_softmax_top10_masked_f32(&input, &ids, &weights, rows, normalize, mask.buf.as_ptr() as usize, round).unwrap();
                gpu.moe_router_softmax_top10_f32(&reference_input, &reference_ids, &reference_weights, rows, normalize, round).unwrap();
                let got_ids = gpu.download_f32(&ids).unwrap();
                let got_weights = gpu.download_f32(&weights).unwrap();
                assert_eq!(bits(&got_ids), bits(&gpu.download_f32(&reference_ids).unwrap()), "ids rows={rows} normalize={normalize} round={round}");
                assert_eq!(bits(&got_weights), bits(&gpu.download_f32(&reference_weights).unwrap()), "weights rows={rows} normalize={normalize} round={round}");
                for row in 0..rows {
                    let expected = route((0..512).filter(|&e| keep[e] != 0).map(|e| (e, logits[row * 512 + e])).collect(), normalize, round);
                    assert_eq!(got_ids[row * 10..(row + 1) * 10].iter().map(|v| v.to_bits() as usize).collect::<Vec<_>>(), expected.0);
                }
            }
        }
        for tensor in [mask, input, reference_input, ids, weights, reference_ids, reference_weights] { gpu.free_tensor(tensor).unwrap(); }
    }
}
