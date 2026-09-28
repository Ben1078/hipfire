// claude: byte-exactness + timing of the fused int8-quant verify producers.
//
// For each producer (RMSNorm+FWHT AWQ, SiLU(gate)*up+FWHT AWQ, FWHT AWQ), k and
// batch, with synthetic LCG inputs (plus an all-zero group for the amax == 0 path):
//   ref   = original producer (CLAUDE_Q8_FUSE forced off) + standalone claude_q8x_quant
//   fused = claude_{rmsnorm,silu,}rotate_q8 writing x_rot + xq + dx in one launch
// and asserts x_rot, xq and dx are byte-equal (fused outputs are pre-poisoned).
// For batch >= 2 it also drives the dispatch path: the host producer takes the
// fused kernel and records the prequant entry, `claude_q8x_project` consumes
// it (skipping its quant launch) and must produce the same GEMV bits as the
// standalone-quant path, and a later non-fused rewrite of the same buffer
// must drop the entry. Timings: us per call (back-to-back launches, wall clock
// with a sync) for the original pair (2 launches) vs the fused producer.
//
// Usage: test_claude_q8_fuse [reps]   (exit code 1 on any FAIL)
use rdna_compute::{DType, Gpu, GpuTensor};
use std::ffi::c_void;
use std::time::Instant;

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 33) as u32
    }
    /// [0, 1)
    fn unit(&mut self) -> f32 {
        self.next() as f32 / 2_147_483_648.0
    }
    fn sym(&mut self, a: f32) -> f32 {
        (self.unit() * 2.0 - 1.0) * a
    }
    fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + self.unit() * (hi - lo)
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Prod {
    RmsNorm,
    Silu,
    Rotate,
}

impl Prod {
    fn name(self) -> &'static str {
        match self {
            Prod::RmsNorm => "rmsnorm_rotate_awq",
            Prod::Silu => "silu_mul_rotate_awq",
            Prod::Rotate => "rotate_x_awq",
        }
    }
}

/// Producer inputs: `a` = x (rmsnorm / rotate) or gate (silu), `up` (silu),
/// `w` = RMSNorm weight (rmsnorm), `awq` = per-channel AWQ scale.
struct Inputs {
    a: GpuTensor,
    up: Option<GpuTensor>,
    w: Option<GpuTensor>,
    awq: GpuTensor,
}

const EPS: f32 = 1e-6;

fn make_inputs(gpu: &mut Gpu, p: Prod, k: usize, b: usize, seed: u64) -> Inputs {
    let mut r = Lcg(seed);
    let act = |amp: f32, r: &mut Lcg| -> Vec<f32> {
        let mut v: Vec<f32> = (0..b * k)
            .map(|i| {
                let x = r.sym(amp);
                // sparse outliers so some 32-blocks have a dominant element
                if i % 97 == 13 { x * 9.0 } else { x }
            })
            .collect();
        // one all-zero 256-group (row 0, group 1): FWHT keeps it zero, so its
        // 32-element blocks take the amax == 0 (d = 0, id = 0) path.
        for x in &mut v[256..512] {
            *x = 0.0;
        }
        v
    };
    let (a, up) = match p {
        Prod::RmsNorm => (act(3.0, &mut r), None),
        Prod::Silu => {
            let g = act(6.0, &mut r);
            let u: Vec<f32> = (0..b * k).map(|_| r.sym(3.0)).collect();
            (g, Some(u))
        }
        Prod::Rotate => (act(2.0, &mut r), None),
    };
    let w: Option<Vec<f32>> =
        (p == Prod::RmsNorm).then(|| (0..k).map(|_| r.range(0.3, 1.7)).collect());
    let awq: Vec<f32> = (0..k).map(|_| r.range(0.25, 4.0)).collect();
    Inputs {
        a: gpu.upload_f32(&a, &[b * k]).expect("upload a"),
        up: up.map(|u| gpu.upload_f32(&u, &[b * k]).expect("upload up")),
        w: w.map(|w| gpu.upload_f32(&w, &[k]).expect("upload w")),
        awq: gpu.upload_f32(&awq, &[k]).expect("upload awq"),
    }
}

fn free_inputs(gpu: &mut Gpu, inp: Inputs) {
    gpu.free_tensor(inp.a).unwrap();
    if let Some(t) = inp.up {
        gpu.free_tensor(t).unwrap();
    }
    if let Some(t) = inp.w {
        gpu.free_tensor(t).unwrap();
    }
    gpu.free_tensor(inp.awq).unwrap();
}

/// The public host producer (`*_batched`): original kernel when fusion is
/// forced off, fused kernel + prequant entry when on and eligible.
fn run_host(gpu: &mut Gpu, p: Prod, inp: &Inputs, x_rot: &GpuTensor, k: usize, b: usize) {
    match p {
        Prod::RmsNorm => gpu
            .fused_rmsnorm_rotate_mq_awq_batched(&inp.a, inp.w.as_ref().unwrap(), &inp.awq, x_rot, k, EPS, b)
            .expect("rmsnorm host"),
        Prod::Silu => gpu
            .fused_silu_mul_rotate_mq_awq_batched(&inp.a, inp.up.as_ref().unwrap(), &inp.awq, x_rot, k, b)
            .expect("silu host"),
        Prod::Rotate => gpu
            .rotate_x_mq_awq_batched(&inp.a, &inp.awq, x_rot, k, b)
            .expect("rotate host"),
    }
}

/// The fused kernel launched explicitly into caller buffers.
#[allow(clippy::too_many_arguments)]
fn run_fused(
    gpu: &mut Gpu,
    p: Prod,
    inp: &Inputs,
    x_rot: &GpuTensor,
    xq: *mut c_void,
    dx: *mut c_void,
    k: usize,
    b: usize,
) {
    match p {
        Prod::RmsNorm => gpu
            .claude_rmsnorm_rotate_q8_batched(&inp.a, inp.w.as_ref().unwrap(), &inp.awq, x_rot, xq, dx, k, EPS, b)
            .expect("rmsnorm fused"),
        Prod::Silu => gpu
            .claude_silu_rotate_q8_batched(&inp.a, inp.up.as_ref().unwrap(), &inp.awq, x_rot, xq, dx, k, b)
            .expect("silu fused"),
        Prod::Rotate => gpu
            .claude_rotate_q8_batched(&inp.a, &inp.awq, x_rot, xq, dx, k, b)
            .expect("rotate fused"),
    }
}

fn bits(gpu: &Gpu, t: &GpuTensor) -> Vec<u32> {
    gpu.download_f32(t).expect("download").iter().map(|v| v.to_bits()).collect()
}

fn mismatches(a: &[u32], b: &[u32]) -> usize {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).filter(|(x, y)| x != y).count()
}

fn poisoned(gpu: &mut Gpu, n: usize) -> GpuTensor {
    gpu.upload_f32(&vec![f32::from_bits(0x7FC0_1234); n], &[n]).expect("poison")
}

/// Random MQ4G256V2 rows: per 256-group 8 bytes of (scale, zero) f16 pairs
/// followed by 128 bytes of nibbles (layout read by claude_q8x_gemv).
fn mq4v2_weights(gpu: &mut Gpu, m: usize, k: usize, seed: u64) -> GpuTensor {
    const SC: [u16; 4] = [0x2148, 0x2A66, 0x1E66, 0x2466]; // ~0.0103, 0.05, 0.0062, 0.0187
    const ZP: [u16; 4] = [0xB266, 0x3266, 0xAE66, 0x2E66]; // -0.2, 0.2, -0.1, 0.1
    let mut r = Lcg(seed);
    let gpr = k / 256;
    let mut bytes = Vec::with_capacity(m * gpr * 136);
    for _ in 0..m * gpr {
        for _ in 0..2 {
            bytes.extend_from_slice(&SC[(r.next() & 3) as usize].to_le_bytes());
            bytes.extend_from_slice(&ZP[(r.next() & 3) as usize].to_le_bytes());
        }
        for _ in 0..128 {
            bytes.push(r.next() as u8);
        }
    }
    let mut t = gpu.upload_raw(&bytes, &[bytes.len()]).expect("upload weights");
    t.dtype = DType::MQ4G256V2;
    t
}

fn main() {
    let reps: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(200);
    let mut gpu = Gpu::init().expect("gpu init");
    eprintln!("test_claude_q8_fuse: arch={} reps={reps}", gpu.arch);
    let mut failures = 0usize;
    let mut e2e_skipped = 0usize;

    for p in [Prod::RmsNorm, Prod::Silu, Prod::Rotate] {
        for k in [5120usize, 6144, 17408] {
            for b in [1usize, 2, 4, 8] {
                let n = b * k;
                let inp = make_inputs(&mut gpu, p, k, b, 0x5EED ^ ((k as u64) << 8) ^ b as u64);

                // --- byte exactness: original + standalone quant vs fused ---
                let ref_rot = gpu.zeros(&[n], DType::F32).unwrap();
                let ref_xq = gpu.zeros(&[n / 4], DType::F32).unwrap();
                let ref_dx = gpu.zeros(&[n / 32], DType::F32).unwrap();
                let fus_rot = poisoned(&mut gpu, n);
                let fus_xq = poisoned(&mut gpu, n / 4);
                let fus_dx = poisoned(&mut gpu, n / 32);

                gpu.claude_q8_fuse_override(Some(false));
                run_host(&mut gpu, p, &inp, &ref_rot, k, b);
                gpu.claude_q8x_quant_into(&ref_rot, ref_xq.buf.as_ptr(), ref_dx.buf.as_ptr(), k, b)
                    .unwrap();
                run_fused(&mut gpu, p, &inp, &fus_rot, fus_xq.buf.as_ptr(), fus_dx.buf.as_ptr(), k, b);
                gpu.hip.device_synchronize().unwrap();
                let rot_mm = mismatches(&bits(&gpu, &ref_rot), &bits(&gpu, &fus_rot));
                let xq_mm = mismatches(&bits(&gpu, &ref_xq), &bits(&gpu, &fus_xq));
                let dx_mm = mismatches(&bits(&gpu, &ref_dx), &bits(&gpu, &fus_dx));
                // sanity: the reference quant is not degenerate
                let ref_q = bits(&gpu, &ref_xq);
                let nonzero_q = ref_q.iter().filter(|&&w| w != 0).count();
                let exact = rot_mm == 0 && xq_mm == 0 && dx_mm == 0 && nonzero_q > n / 8;

                // --- dispatch path: fused host producer -> claude_q8x_project ---
                let mut e2e = String::from("e2e=n/a(b=1)");
                let mut e2e_ok = true;
                if b >= 2 {
                    gpu.claude_q8_fuse_override(Some(true));
                    if !gpu.claude_q8_fuse_eligible(k, b) {
                        e2e = "e2e=SKIP(not eligible on this arch)".into();
                        e2e_skipped += 1;
                    } else {
                        let m = 256usize;
                        let wts = mq4v2_weights(&mut gpu, m, k, 0xC0FFEE ^ k as u64);
                        let other = gpu.upload_f32(&vec![0.37f32; n], &[n]).unwrap();
                        let d_rot = poisoned(&mut gpu, n);
                        let y_hit = gpu.zeros(&[b * m], DType::F32).unwrap();
                        let y_miss = gpu.zeros(&[b * m], DType::F32).unwrap();
                        let y_ref = gpu.zeros(&[b * m], DType::F32).unwrap();
                        // Fill the shared scratch with another input's quant first,
                        // so a hit that ignored the fused output would show up.
                        gpu.claude_q8x_project(&[(&wts, &y_miss, m)], &other, k, b, false).unwrap();
                        run_host(&mut gpu, p, &inp, &d_rot, k, b); // fused + prequant entry
                        let pending = gpu.claude_q8_prequant_pending(&d_rot, k, b);
                        gpu.claude_q8x_project(&[(&wts, &y_hit, m)], &d_rot, k, b, false).unwrap();
                        let consumed = !gpu.claude_q8_prequant_pending(&d_rot, k, b);
                        gpu.claude_q8x_project(&[(&wts, &y_miss, m)], &d_rot, k, b, false).unwrap(); // miss
                        gpu.hip.device_synchronize().unwrap();
                        let d_rot_mm = mismatches(&bits(&gpu, &ref_rot), &bits(&gpu, &d_rot));
                        let y_mm = mismatches(&bits(&gpu, &y_hit), &bits(&gpu, &y_miss));

                        // Invalidation: fused write of `d_rot` from other inputs, then
                        // an original (non-fused) rewrite from `inp` must drop the entry.
                        let inp2 = make_inputs(&mut gpu, p, k, b, 0xBAD ^ k as u64 ^ ((b as u64) << 20));
                        run_host(&mut gpu, p, &inp2, &d_rot, k, b); // fused, entry set
                        let pending2 = gpu.claude_q8_prequant_pending(&d_rot, k, b);
                        gpu.claude_q8_fuse_override(Some(false));
                        run_host(&mut gpu, p, &inp, &d_rot, k, b); // original kernel
                        let dropped = !gpu.claude_q8_prequant_pending(&d_rot, k, b);
                        gpu.claude_q8x_project(&[(&wts, &y_ref, m)], &d_rot, k, b, false).unwrap();
                        gpu.hip.device_synchronize().unwrap();
                        let inv_mm = mismatches(&bits(&gpu, &y_ref), &bits(&gpu, &y_miss));
                        e2e_ok = pending && consumed && pending2 && dropped
                            && d_rot_mm == 0 && y_mm == 0 && inv_mm == 0;
                        e2e = format!(
                            "e2e: pending={pending} consumed={consumed} x_rot_mm={d_rot_mm} \
                             y_hit_vs_miss_mm={y_mm} | inval: pending={pending2} dropped={dropped} y_mm={inv_mm}"
                        );
                        free_inputs(&mut gpu, inp2);
                        for t in [wts, other, d_rot, y_hit, y_miss, y_ref] {
                            gpu.free_tensor(t).unwrap();
                        }
                    }
                }

                // --- timing: original pair vs fused ---
                gpu.claude_q8_fuse_override(Some(false));
                gpu.hip.device_synchronize().unwrap();
                let t0 = Instant::now();
                for _ in 0..reps {
                    run_host(&mut gpu, p, &inp, &ref_rot, k, b);
                    gpu.claude_q8x_quant_into(&ref_rot, ref_xq.buf.as_ptr(), ref_dx.buf.as_ptr(), k, b)
                        .unwrap();
                }
                gpu.hip.device_synchronize().unwrap();
                let pair_us = t0.elapsed().as_secs_f64() * 1e6 / reps as f64;
                let t1 = Instant::now();
                for _ in 0..reps {
                    run_fused(&mut gpu, p, &inp, &fus_rot, fus_xq.buf.as_ptr(), fus_dx.buf.as_ptr(), k, b);
                }
                gpu.hip.device_synchronize().unwrap();
                let fused_us = t1.elapsed().as_secs_f64() * 1e6 / reps as f64;

                let ok = exact && e2e_ok;
                if !ok {
                    failures += 1;
                }
                println!(
                    "{} {:<20} k={k:<5} b={b}  x_rot_mm={rot_mm} xq_mm={xq_mm} dx_mm={dx_mm} \
                     (nonzero q words {nonzero_q}/{})  {e2e}  | pair {pair_us:7.2} us  fused {fused_us:7.2} us",
                    if ok { "PASS" } else { "FAIL" },
                    p.name(),
                    n / 4
                );
                free_inputs(&mut gpu, inp);
                for t in [ref_rot, ref_xq, ref_dx, fus_rot, fus_xq, fus_dx] {
                    gpu.free_tensor(t).unwrap();
                }
            }
        }
    }
    gpu.claude_q8_fuse_override(None);
    if failures == 0 {
        println!("ALL PASS{}", if e2e_skipped > 0 { " (e2e skipped on this arch)" } else { "" });
    } else {
        println!("FAILURES: {failures}");
        std::process::exit(1);
    }
}
