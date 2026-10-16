// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
use rdna_compute::{DType, Gpu};

fn route(mut logits: Vec<(usize, f32)>) -> (Vec<usize>, Vec<f32>) {
    logits.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    logits.truncate(10);
    let max = logits[0].1;
    let p: Vec<f32> = logits.iter().map(|(_, v)| (*v - max).exp()).collect();
    let inverse = 1.0 / p.iter().sum::<f32>();
    (logits.iter().map(|(id, _)| *id).collect(), p.iter().map(|p| p * inverse).collect())
}

#[test]
fn masked_router_equals_sliced_router_original_id_ties() {
    for count in [16, 31, 128, 511, 512] {
        let keep: Vec<usize> = (0..512).filter(|e| (e * 37) % 512 < count).collect();
        for ties in [false, true] {
            let logits: Vec<f32> = (0..512).map(|e| if ties { 0.25 } else { ((e * 13) % 47) as f32 * 0.125 }).collect();
            let masked = route(logits.iter().enumerate().map(|(e, &v)| (e, if keep.contains(&e) { v } else { f32::NEG_INFINITY })).collect());
            let sliced = route(keep.iter().map(|&e| (e, logits[e])).collect());
            assert_eq!(masked.0, sliced.0);
            assert_eq!(masked.1.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), sliced.1.iter().map(|v| v.to_bits()).collect::<Vec<_>>());
            if ties { assert_eq!(masked.0, keep[..10]); }
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
        let logits: Vec<f32> = (0..rows * 512).map(|i| if i / 512 % 2 == 0 { ((i * 13) % 47) as f32 * 0.125 } else { 0.25 }).collect();
        let mask = gpu.upload_raw(&keep, &[512]).unwrap();
        let input = gpu.upload_f32(&logits, &[rows, 512]).unwrap();
        let ids = gpu.alloc_tensor(&[rows, 10], DType::F32).unwrap();
        let weights = gpu.alloc_tensor(&[rows, 10], DType::F32).unwrap();
        gpu.moe_router_softmax_top10_masked_f32(&input, &ids, &weights, rows, mask.buf.as_ptr() as usize, false).unwrap();
        let got_ids = gpu.download_f32(&ids).unwrap();
        let got_weights = gpu.download_f32(&weights).unwrap();
        for row in 0..rows {
            let expected = route((0..512).filter(|&e| keep[e] != 0).map(|e| (e, logits[row * 512 + e])).collect());
            assert_eq!(got_ids[row * 10..(row + 1) * 10].iter().map(|v| v.to_bits() as usize).collect::<Vec<_>>(), expected.0);
            for j in 0..10 {
                assert!((got_weights[row * 10 + j] - expected.1[j]).abs() < 2e-6, "row={row} slot={j}");
            }
        }
        for tensor in [mask, input, ids, weights] { gpu.free_tensor(tensor).unwrap(); }
    }
}
