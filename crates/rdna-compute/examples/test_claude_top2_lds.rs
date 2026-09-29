// claude C6 (CLAUDE_TOP2_LDS): byte-identity + timing of claude_top2lse_s2_lds
// (partials staged in LDS by 128 threads, identical serial merge on thread 0)
// against claude_top2lse_s2 (one thread, global loads).
//
// Cases: (a) partials produced by claude_top2lse_s1 from full-vocab (248320)
// logits, rows 1..4, including duplicated maxima (index tie-break) and a block of
// underflowing logits; (b) 2000 crafted partial rows (random maxima incl. -inf blocks,
// equal block maxima, random sums and indices) to stress the merge branches.
// top_idx and top_logp are compared bit for bit.
// Timing: GPU time per stage-2 launch and per s1+s2, queued, events.
//
// Usage: test_claude_top2_lds [reps]   (exit 1 on any FAIL)
#[path = "claude_fusion/common.rs"]
mod common;
use common::*;
use rdna_compute::{DType, Gpu};

const NB: usize = 128;
const VOCAB: usize = 248_320;

fn main() {
    let reps: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(200);
    let mut gpu = Gpu::init().expect("gpu init");
    println!("test_claude_top2_lds: arch={} reps={reps}", gpu.arch);
    let mut fails = 0usize;

    // (a) real stage-1 partials
    for rows in 1..=4usize {
        let mut r = Lcg(0xC6 ^ ((rows as u64) << 12));
        let mut lg: Vec<f32> = (0..rows * VOCAB).map(|_| r.sym(12.0)).collect();
        for row in 0..rows {
            let base = row * VOCAB;
            // a clear winner duplicated at two indices (tie -> lower index)
            let hi = 30.0 + row as f32;
            lg[base + 1000 + 17 * row] = hi;
            lg[base + 200_000 - row] = hi;
            // runner-up duplicated too
            lg[base + 5000] = hi - 0.75;
            lg[base + 150_000] = hi - 0.75;
            // a stretch of hugely negative logits (a whole s1 block whose
            // exponentials underflow). Not -inf: claude_top2lse_s1 itself
            // (unchanged) turns an all -inf lane into a NaN partial sum.
            for x in &mut lg[base + 240_000..base + 243_000] {
                *x = -1.0e30;
            }
        }
        let logits = up(&mut gpu, &lg);
        let part = gpu.zeros(&[rows * NB * 6], DType::F32).unwrap();
        let ia = gpu.zeros(&[rows * 2], DType::F32).unwrap();
        let la = gpu.zeros(&[rows * 2], DType::F32).unwrap();
        let ib = poison(&mut gpu, rows * 2);
        let lb = poison(&mut gpu, rows * 2);
        gpu.claude_top2lse_s1_into(&logits, part.buf.as_ptr(), VOCAB, rows).unwrap();
        gpu.claude_top2lse_s2_from(part.buf.as_ptr(), &ia, &la, rows, false).unwrap();
        gpu.claude_top2lse_s2_from(part.buf.as_ptr(), &ib, &lb, rows, true).unwrap();
        gpu.hip.device_synchronize().unwrap();
        let (ia_h, la_h) = (bits(&gpu, &ia), bits(&gpu, &la));
        let idx_mm = mism(&ia_h, &bits(&gpu, &ib));
        let lp_mm = mism(&la_h, &bits(&gpu, &lb));
        // sanity: the duplicated winner resolves to the lower index
        let win0 = ia_h[0] as usize;
        let ok = idx_mm == 0 && lp_mm == 0 && win0 == 1000;
        if !ok {
            fails += 1;
        }
        let t_old = time_us(&mut gpu, reps, |g| {
            g.claude_top2lse_s2_from(part.buf.as_ptr(), &ia, &la, rows, false).unwrap();
        });
        let t_new = time_us(&mut gpu, reps, |g| {
            g.claude_top2lse_s2_from(part.buf.as_ptr(), &ib, &lb, rows, true).unwrap();
        });
        let t_old2 = time_us(&mut gpu, reps / 2, |g| {
            g.claude_top2lse_s1_into(&logits, part.buf.as_ptr(), VOCAB, rows).unwrap();
            g.claude_top2lse_s2_from(part.buf.as_ptr(), &ia, &la, rows, false).unwrap();
        });
        let t_new2 = time_us(&mut gpu, reps / 2, |g| {
            g.claude_top2lse_s1_into(&logits, part.buf.as_ptr(), VOCAB, rows).unwrap();
            g.claude_top2lse_s2_from(part.buf.as_ptr(), &ib, &lb, rows, true).unwrap();
        });
        println!(
            "{} s1-partials rows={rows} idx_mm={idx_mm} logp_mm={lp_mm} top=({}, {}) logp0={:.6} | s2 {t_old:6.2} -> {t_new:6.2} us ; s1+s2 {t_old2:6.2} -> {t_new2:6.2} us",
            if ok { "PASS" } else { "FAIL" },
            ia_h[0],
            ia_h[1],
            f32::from_bits(la_h[0])
        );
        for t in [logits, part, ia, la, ib, lb] {
            gpu.free_tensor(t).unwrap();
        }
    }

    // (b) crafted partials
    let rows = 16usize;
    let mut total_mm = 0usize;
    for it in 0..125usize {
        let mut r = Lcg(0x5EED_C6 ^ ((it as u64) << 8));
        let mut p = vec![0f32; rows * NB * 6];
        for row in 0..rows {
            for j in 0..NB {
                let o = (row * NB + j) * 6;
                let v0 = if r.next() % 7 == 0 {
                    5.0 // frequent equal maxima
                } else if r.next() % 301 == 0 {
                    f32::INFINITY // a +inf block maximum (inf - inf paths)
                } else {
                    r.sym(20.0)
                };
                let v1 = v0 - r.unit() * 3.0;
                let i0 = (r.next() % VOCAB as u32) as i32;
                let i1 = (r.next() % VOCAB as u32) as i32;
                let masked = r.next() % 11 == 0;
                p[o] = if masked { f32::NEG_INFINITY } else { v0 };
                p[o + 1] = f32::from_bits(if masked { 0x7fff_ffff } else { i0 as u32 });
                p[o + 2] = if masked { f32::NEG_INFINITY } else { v1 };
                p[o + 3] = f32::from_bits(if masked { 0x7fff_ffff } else { i1 as u32 });
                p[o + 4] = if masked { f32::NEG_INFINITY } else if r.next() % 5 == 0 { 5.0 } else { v0 };
                p[o + 5] = if masked { 0.0 } else { r.range(1.0, 2000.0) };
            }
        }
        let part = up(&mut gpu, &p);
        let ia = gpu.zeros(&[rows * 2], DType::F32).unwrap();
        let la = gpu.zeros(&[rows * 2], DType::F32).unwrap();
        let ib = poison(&mut gpu, rows * 2);
        let lb = poison(&mut gpu, rows * 2);
        gpu.claude_top2lse_s2_from(part.buf.as_ptr(), &ia, &la, rows, false).unwrap();
        gpu.claude_top2lse_s2_from(part.buf.as_ptr(), &ib, &lb, rows, true).unwrap();
        gpu.hip.device_synchronize().unwrap();
        total_mm += mism(&bits(&gpu, &ia), &bits(&gpu, &ib)) + mism(&bits(&gpu, &la), &bits(&gpu, &lb));
        for t in [part, ia, la, ib, lb] {
            gpu.free_tensor(t).unwrap();
        }
    }
    let ok = total_mm == 0;
    if !ok {
        fails += 1;
    }
    println!(
        "{} crafted partials: {} rows, idx+logp mismatches={total_mm}",
        if ok { "PASS" } else { "FAIL" },
        125 * rows
    );

    if fails == 0 {
        println!("ALL PASS");
    } else {
        println!("FAILURES: {fails}");
        std::process::exit(1);
    }
}
