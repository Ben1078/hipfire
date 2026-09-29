// claude C1 (CLAUDE_KV_PAIR): byte-identity + timing of the paired K/V Q8_0
// cache writes against the two-launch sequences they replace.
//
//   verify (batched):  kv_cache_write_q8_0_batched(K) + (V)  vs  kv_cache_write_q8_0_pair_batched
//   draft (1 row):     kv_cache_write_q8_0(K) + (V)          vs  kv_cache_write_q8_0_pair
//
// Caches are pre-filled with the same random bytes; sources are random with
// outliers and all-zero 32-blocks (amax == 0 path). The whole K and V cache
// buffers are compared byte for byte (so stray writes would show up too).
// Timing: GPU time per sequence, reps back to back behind a spin kernel,
// event-bracketed (dispatch gaps included).
//
// Usage: test_claude_kv_pair [reps]   (exit 1 on any FAIL)
#[path = "claude_fusion/common.rs"]
mod common;
use common::*;
use rdna_compute::Gpu;

fn main() {
    let reps: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(200);
    let mut gpu = Gpu::init().expect("gpu init");
    println!("test_claude_kv_pair: arch={} reps={reps}", gpu.arch);
    let mut fails = 0usize;
    let max_seq = 96usize;
    // (n_kv_heads, head_dim): Qwen3.8-27B trunk + MTP head is (4, 256).
    for &(nkv, hd) in &[(4usize, 256usize), (2, 128), (8, 128), (1, 64)] {
        let kv_dim = nkv * hd;
        let per_pos = nkv * (hd / 32) * 34;
        let cache_bytes = max_seq * per_pos;
        for b in 1..=4usize {
            for (pi, pos_kind) in ["consecutive", "scattered"].iter().enumerate() {
                let mut r = Lcg(0xC1 ^ ((nkv as u64) << 8) ^ ((hd as u64) << 16) ^ ((b as u64) << 24) ^ pi as u64);
                let start = (r.next() as usize) % (max_seq - 4);
                let positions: Vec<i32> = (0..b)
                    .map(|i| {
                        if pi == 0 {
                            (start + i) as i32
                        } else {
                            ((start + 17 * i + (r.next() as usize % 3)) % max_seq) as i32
                        }
                    })
                    .collect();
                // distinct positions (a real verify never writes one slot twice)
                let mut sorted = positions.clone();
                sorted.sort();
                sorted.dedup();
                if sorted.len() != positions.len() {
                    continue;
                }
                let init_k = r.bytes(cache_bytes);
                let init_v = r.bytes(cache_bytes);
                let ksrc = r.acts(b * kv_dim, 3.0, 7);
                let vsrc = r.acts(b * kv_dim, 1.5, 5);
                let k_src = up(&mut gpu, &ksrc);
                let v_src = up(&mut gpu, &vsrc);
                let pos_t = up_i32(&mut gpu, &positions);
                let k_ref = up_raw(&mut gpu, &init_k);
                let v_ref = up_raw(&mut gpu, &init_v);
                let k_new = up_raw(&mut gpu, &init_k);
                let v_new = up_raw(&mut gpu, &init_v);

                // ---- batched (verify) ----
                gpu.kv_cache_write_q8_0_batched(&k_ref, &k_src, &pos_t, nkv, hd, b).unwrap();
                gpu.kv_cache_write_q8_0_batched(&v_ref, &v_src, &pos_t, nkv, hd, b).unwrap();
                gpu.kv_cache_write_q8_0_pair_batched(&k_new, &v_new, &k_src, &v_src, &pos_t, nkv, hd, b)
                    .unwrap();
                gpu.hip.device_synchronize().unwrap();
                let kr = raw(&gpu, &k_ref, cache_bytes);
                let vr = raw(&gpu, &v_ref, cache_bytes);
                let k_mm = mism(&kr, &raw(&gpu, &k_new, cache_bytes));
                let v_mm = mism(&vr, &raw(&gpu, &v_new, cache_bytes));
                let k_changed = mism(&kr, &init_k);
                let v_changed = mism(&vr, &init_v);
                let ok = k_mm == 0 && v_mm == 0 && k_changed > 0 && v_changed > 0;
                if !ok {
                    fails += 1;
                }
                let (t_old, t_new) = if pi == 0 {
                    let t_old = time_us(&mut gpu, reps, |g| {
                        g.kv_cache_write_q8_0_batched(&k_ref, &k_src, &pos_t, nkv, hd, b).unwrap();
                        g.kv_cache_write_q8_0_batched(&v_ref, &v_src, &pos_t, nkv, hd, b).unwrap();
                    });
                    let t_new = time_us(&mut gpu, reps, |g| {
                        g.kv_cache_write_q8_0_pair_batched(&k_new, &v_new, &k_src, &v_src, &pos_t, nkv, hd, b)
                            .unwrap();
                    });
                    (format!("{t_old:6.2}"), format!("{t_new:6.2}"))
                } else {
                    ("   -  ".into(), "   -  ".into())
                };
                println!(
                    "{} batched nkv={nkv} hd={hd} b={b} {:<11} K_mm={k_mm} V_mm={v_mm} (bytes changed K={k_changed} V={v_changed}) | 2 launches {t_old} us  pair {t_new} us",
                    if ok { "PASS" } else { "FAIL" },
                    pos_kind
                );

                // ---- single row (draft head), row 0 of the sources ----
                if b == 1 {
                    let k1 = k_src.sub_offset(0, kv_dim);
                    let v1 = v_src.sub_offset(0, kv_dim);
                    let k_ref1 = up_raw(&mut gpu, &init_k);
                    let v_ref1 = up_raw(&mut gpu, &init_v);
                    let k_new1 = up_raw(&mut gpu, &init_k);
                    let v_new1 = up_raw(&mut gpu, &init_v);
                    gpu.kv_cache_write_q8_0(&k_ref1, &k1, &pos_t.buf, nkv, hd).unwrap();
                    gpu.kv_cache_write_q8_0(&v_ref1, &v1, &pos_t.buf, nkv, hd).unwrap();
                    gpu.kv_cache_write_q8_0_pair(&k_new1, &v_new1, &k1, &v1, &pos_t.buf, nkv, hd).unwrap();
                    gpu.hip.device_synchronize().unwrap();
                    let kr1 = raw(&gpu, &k_ref1, cache_bytes);
                    let vr1 = raw(&gpu, &v_ref1, cache_bytes);
                    let k_mm1 = mism(&kr1, &raw(&gpu, &k_new1, cache_bytes));
                    let v_mm1 = mism(&vr1, &raw(&gpu, &v_new1, cache_bytes));
                    let ok1 = k_mm1 == 0 && v_mm1 == 0 && mism(&kr1, &init_k) > 0 && mism(&vr1, &init_v) > 0;
                    if !ok1 {
                        fails += 1;
                    }
                    let (t_old, t_new) = if pi == 0 {
                        let t_old = time_us(&mut gpu, reps, |g| {
                            g.kv_cache_write_q8_0(&k_ref1, &k1, &pos_t.buf, nkv, hd).unwrap();
                            g.kv_cache_write_q8_0(&v_ref1, &v1, &pos_t.buf, nkv, hd).unwrap();
                        });
                        let t_new = time_us(&mut gpu, reps, |g| {
                            g.kv_cache_write_q8_0_pair(&k_new1, &v_new1, &k1, &v1, &pos_t.buf, nkv, hd).unwrap();
                        });
                        (format!("{t_old:6.2}"), format!("{t_new:6.2}"))
                    } else {
                        ("   -  ".into(), "   -  ".into())
                    };
                    println!(
                        "{} single  nkv={nkv} hd={hd} pos={} K_mm={k_mm1} V_mm={v_mm1} | 2 launches {t_old} us  pair {t_new} us",
                        if ok1 { "PASS" } else { "FAIL" },
                        positions[0]
                    );
                    for t in [k_ref1, v_ref1, k_new1, v_new1] {
                        gpu.free_tensor(t).unwrap();
                    }
                }
                for t in [k_src, v_src, pos_t, k_ref, v_ref, k_new, v_new] {
                    gpu.free_tensor(t).unwrap();
                }
            }
        }
    }
    if fails == 0 {
        println!("ALL PASS");
    } else {
        println!("FAILURES: {fails}");
        std::process::exit(1);
    }
}
