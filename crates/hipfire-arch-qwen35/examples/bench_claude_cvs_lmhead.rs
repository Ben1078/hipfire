//! claude r2: standalone check + timing of the compressed MTP draft lm_head.
//!
//!   bench_claude_cvs_lmhead --trunk <trunk.mq4-xts> --cvs <head-cvsK.mtp>
//!       [--rows 65536] [--iters 30] [--trials 8]
//!
//! 1. Bit-exactness: the slice (`lm_head_draft.weight`, byte-sliced rows of the
//!    trunk lm_head) and the first `--rows` rows of the real trunk lm_head go
//!    through the production draft call (`llama::weight_gemv` = FWHT rotate +
//!    `gemv_mq4g256v2`) on the same x; every slice row whose token id is
//!    < rows must produce the identical f32 bits.
//! 2. GPU time (hip events): slice GEMV warm (back to back; the slice fits the
//!    96 MB Infinity Cache) and cold (a >96 MB GEMV over the trunk rows before
//!    each timed call, as in the real draft step where ~225 MB of MTP-head
//!    weights stream between lm_head passes); the trunk-row GEMV (streaming,
//!    cold) and its per-row extrapolation to the full 248320 rows (the full
//!    675 MB matrix is not allocated: the run stays under ~400 MB of VRAM).
//! 3. The top-2 + logsumexp readout at n = K and n = rows (stock vs claude
//!    two-stage kernel, selected by CLAUDE_FAST_REDUCE_MIN_N).
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::llama::{self, WeightTensor};
use rdna_compute::{DType, Gpu, GpuTensor};

struct Rng(u64);
impl Rng {
    fn u32(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
    fn unit(&mut self) -> f32 {
        self.u32() as f32 / 2147483648.0
    }
    fn gauss(&mut self) -> f32 {
        let s: f32 = (0..4).map(|_| self.unit()).sum();
        (s - 2.0) * 1.732
    }
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn wt(buf: GpuTensor, m: usize, k: usize) -> WeightTensor {
    WeightTensor {
        buf,
        gpu_dtype: DType::MQ4G256V2,
        m,
        k,
        row_stride: 0,
        paro: None,
        awq_scale: None,
        lloyd_lut_e4m3: None,
        lloyd_lut_f16: None,
        lloyd_lut_c16: None,
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let arg = |k: &str| args.iter().position(|a| a == k).map(|i| args[i + 1].clone());
    let trunk = arg("--trunk").expect("--trunk");
    let cvs_path = arg("--cvs").expect("--cvs");
    let rows: usize = arg("--rows").map(|v| v.parse().unwrap()).unwrap_or(65536);
    let iters: usize = arg("--iters").map(|v| v.parse().unwrap()).unwrap_or(30);
    let trials: usize = arg("--trials").map(|v| v.parse().unwrap()).unwrap_or(8);
    let k = 5120usize;
    let row_bytes = (k / 256) * 136;

    let head = HfqFile::open(std::path::Path::new(&cvs_path)).expect("open cvs .mtp");
    let (si, sbytes) = head.tensor_data("lm_head_draft.weight").expect("lm_head_draft.weight");
    assert_eq!(si.quant_type, 44, "expect MQ4G256V2 slice");
    let cvs = si.shape[0] as usize;
    let (_, vbytes) = head.tensor_data("lm_head_draft.vocab_map").expect("vocab_map");
    let vmap: Vec<u32> = vbytes.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    let (_, nbytes) = head.tensor_data("shared_head_norm").expect("shared_head_norm");
    let norm_w: Vec<f32> = nbytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    let tr = HfqFile::open(std::path::Path::new(&trunk)).expect("open trunk");
    let (li, lbytes) = tr.tensor_data("lm_head.weight").expect("trunk lm_head");
    assert_eq!(li.quant_type, 44);
    let full_rows = li.shape[0] as usize;
    assert!(rows <= full_rows);

    let mut gpu = Gpu::init().expect("GPU init");
    eprintln!("GPU {}: slice K={cvs} ({:.1} MB), trunk rows {rows} of {full_rows} ({:.1} MB)",
        gpu.arch, (cvs * row_bytes) as f64 / 1e6, (rows * row_bytes) as f64 / 1e6);
    let slice = wt(gpu.upload_raw(sbytes, &[sbytes.len()]).unwrap(), cvs, k);
    let big = wt(gpu.upload_raw(&lbytes[..rows * row_bytes], &[rows * row_bytes]).unwrap(), rows, k);
    let x = gpu.alloc_tensor(&[k], DType::F32).unwrap();
    let y_s = gpu.zeros(&[cvs], DType::F32).unwrap();
    let y_b = gpu.zeros(&[rows], DType::F32).unwrap();
    let top_i = gpu.zeros(&[2], DType::F32).unwrap();
    let top_l = gpu.zeros(&[2], DType::F32).unwrap();
    let mut rng = Rng(0x5eed_c0ffee);
    let upload_x = |gpu: &mut Gpu, rng: &mut Rng| {
        // post-shared_head_norm scale: unit-RMS gaussian times (1 + w)
        let v: Vec<f32> = (0..k).map(|i| rng.gauss() * (1.0 + norm_w[i])).collect();
        let b = unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) };
        gpu.hip.memcpy_htod(&x.buf, b).unwrap();
    };

    // 1. bit-exactness
    let (mut checked, mut equal, mut top_match, mut top_cases) = (0usize, 0usize, 0usize, 0usize);
    for _ in 0..trials {
        upload_x(&mut gpu, &mut rng);
        llama::weight_gemv(&mut gpu, &slice, &x, &y_s).unwrap();
        llama::weight_gemv(&mut gpu, &big, &x, &y_b).unwrap();
        gpu.hip.device_synchronize().unwrap();
        let ys = gpu.download_f32(&y_s).unwrap();
        let yb = gpu.download_f32(&y_b).unwrap();
        for (j, &id) in vmap.iter().enumerate() {
            if (id as usize) < rows {
                checked += 1;
                equal += (ys[j].to_bits() == yb[id as usize].to_bits()) as usize;
            }
        }
        // argmax over the slice restricted to ids < rows == argmax of big over the same id set
        let mut best_s = (f32::NEG_INFINITY, u32::MAX);
        for (j, &id) in vmap.iter().enumerate() {
            if (id as usize) < rows && (ys[j] > best_s.0 || (ys[j] == best_s.0 && id < best_s.1)) {
                best_s = (ys[j], id);
            }
        }
        let mut best_b = (f32::NEG_INFINITY, u32::MAX);
        for &id in vmap.iter() {
            if (id as usize) < rows {
                let v = yb[id as usize];
                if v > best_b.0 || (v == best_b.0 && id < best_b.1) {
                    best_b = (v, id);
                }
            }
        }
        top_cases += 1;
        top_match += (best_s.1 == best_b.1) as usize;
    }
    println!("bit-exact: {equal}/{checked} slice logits identical to the trunk-row logits ({trials} x); restricted argmax match {top_match}/{top_cases}");

    // 2. timing
    let ev0 = gpu.hip.event_create().unwrap();
    let ev1 = gpu.hip.event_create().unwrap();
    upload_x(&mut gpu, &mut rng);
    let t = std::time::Instant::now();
    while t.elapsed().as_secs_f64() < 1.0 {
        llama::weight_gemv(&mut gpu, &big, &x, &y_b).unwrap();
        gpu.hip.device_synchronize().unwrap();
    }
    let time_one = |gpu: &mut Gpu, pre: &dyn Fn(&mut Gpu), f: &dyn Fn(&mut Gpu)| -> f64 {
        let mut ts = Vec::new();
        for _ in 0..iters {
            pre(gpu);
            gpu.hip.event_record(&ev0, gpu.active_stream.as_ref()).unwrap();
            f(gpu);
            gpu.hip.event_record(&ev1, gpu.active_stream.as_ref()).unwrap();
            gpu.hip.event_synchronize(&ev1).unwrap();
            ts.push(gpu.hip.event_elapsed_ms(&ev0, &ev1).unwrap() as f64 * 1e3);
        }
        median(&mut ts)
    };
    let flush = |gpu: &mut Gpu| {
        llama::weight_gemv(gpu, &big, &x, &y_b).unwrap();
    };
    let none = |_: &mut Gpu| {};
    let gemv_s = |gpu: &mut Gpu| llama::weight_gemv(gpu, &slice, &x, &y_s).unwrap();
    let gemv_b = |gpu: &mut Gpu| llama::weight_gemv(gpu, &big, &x, &y_b).unwrap();
    let us_s_cold = time_one(&mut gpu, &flush, &gemv_s);
    let us_s_warm = {
        // 10 back-to-back calls per sample
        let mut ts = Vec::new();
        for _ in 0..iters {
            gemv_s(&mut gpu);
            gpu.hip.event_record(&ev0, gpu.active_stream.as_ref()).unwrap();
            for _ in 0..10 {
                gemv_s(&mut gpu);
            }
            gpu.hip.event_record(&ev1, gpu.active_stream.as_ref()).unwrap();
            gpu.hip.event_synchronize(&ev1).unwrap();
            ts.push(gpu.hip.event_elapsed_ms(&ev0, &ev1).unwrap() as f64 * 1e3 / 10.0);
        }
        median(&mut ts)
    };
    let us_b = time_one(&mut gpu, &none, &gemv_b);
    let us_full_est = us_b * full_rows as f64 / rows as f64;
    let gbs = |bytes: usize, us: f64| bytes as f64 / (us * 1e3);
    println!(
        "draft lm_head GEMV (rotate + gemv_mq4g256v2, events): slice K={cvs}: cold {us_s_cold:.1} us ({:.0} GB/s), warm {us_s_warm:.1} us | trunk rows {rows}: {us_b:.1} us ({:.0} GB/s) -> full {full_rows} rows est {us_full_est:.1} us | saving/step est {:.1} us",
        gbs(cvs * row_bytes, us_s_cold),
        gbs(rows * row_bytes, us_b),
        us_full_est - us_s_cold
    );

    // 3. top-2 + logsumexp readout
    let top2 = |gpu: &mut Gpu, y: &GpuTensor, n: usize| {
        gpu.topk_logsumexp_batched_f32(y, &top_i, &top_l, n, 2, 1).unwrap();
    };
    // 10 back-to-back readouts per sample (a lone ~10 us pair is dominated by
    // clock ramp / event overhead), preceded by the producing GEMV.
    let us_t_s = time_one(&mut gpu, &gemv_s, &|g: &mut Gpu| {
        for _ in 0..10 {
            top2(g, &y_s, cvs)
        }
    }) / 10.0;
    let us_t_b = time_one(&mut gpu, &gemv_b, &|g: &mut Gpu| {
        for _ in 0..10 {
            top2(g, &y_b, rows)
        }
    }) / 10.0;
    // The whole draft tail as the step runs it: GEMV + top-2 readout, cold.
    let us_tail_s = time_one(&mut gpu, &flush, &|g: &mut Gpu| {
        gemv_s(g);
        top2(g, &y_s, cvs)
    });
    println!(
        "top-2+lse readout (CLAUDE_FAST_REDUCE_MIN_N={}): n={cvs} {us_t_s:.1} us, n={rows} {us_t_b:.1} us | slice GEMV + readout (cold) {us_tail_s:.1} us",
        std::env::var("CLAUDE_FAST_REDUCE_MIN_N").unwrap_or_else(|_| "default".into())
    );
    // Top-2 agreement between the slice readout and a host reference over the slice.
    upload_x(&mut gpu, &mut rng);
    gemv_s(&mut gpu);
    top2(&mut gpu, &y_s, cvs);
    gpu.hip.device_synchronize().unwrap();
    let ys = gpu.download_f32(&y_s).unwrap();
    let ti = gpu.download_f32(&top_i).unwrap();
    let tl = gpu.download_f32(&top_l).unwrap();
    let i0 = ti[0].to_bits() as usize;
    let m = ys.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let lse = m as f64 + ys.iter().map(|&v| ((v - m) as f64).exp()).sum::<f64>().ln();
    let mut host_best = 0usize;
    for j in 1..cvs {
        if ys[j] > ys[host_best] {
            host_best = j;
        }
    }
    println!(
        "slice top-1: gpu idx {i0} logp {:.6} | host idx {host_best} logp {:.6}",
        tl[0],
        ys[host_best] as f64 - lse
    );

    // 4. The real loader on the compressed file (`--load-head`): frees the
    // trunk rows first (the head is ~300 MB), loads through
    // mtp_head::load_mtp_head, and checks the draft lm_head it built produces
    // the same logits as the raw slice upload above.
    if args.iter().any(|a| a == "--load-head") {
        let _ = gpu.free_tensor(big.buf);
        let head = hipfire_arch_qwen35::mtp_head::load_mtp_head(
            std::path::Path::new(&cvs_path),
            &mut gpu,
            256,
        )
        .expect("load_mtp_head");
        let w = &head.weights;
        let lm = w.lm_head_draft.as_ref().expect("lm_head_draft loaded");
        assert_eq!(w.compressed_vocab_size, Some(cvs));
        assert_eq!(lm.gpu_dtype, DType::MQ4G256V2);
        assert_eq!((lm.m, lm.k), (cvs, k));
        assert_eq!(w.lm_head_draft_vocab_map.as_ref().unwrap(), &vmap);
        let vg = w.lm_head_draft_vocab_map_gpu.as_ref().unwrap();
        let mut vb = vec![0u8; cvs * 4];
        gpu.hip.memcpy_dtoh(&mut vb, &vg.buf).unwrap();
        assert_eq!(vb.as_slice(), vbytes, "GPU vocab_map bytes");
        let y2 = gpu.zeros(&[cvs], DType::F32).unwrap();
        upload_x(&mut gpu, &mut rng);
        llama::weight_gemv(&mut gpu, &slice, &x, &y_s).unwrap();
        llama::weight_gemv(&mut gpu, lm, &x, &y2).unwrap();
        gpu.hip.device_synchronize().unwrap();
        let a = gpu.download_f32(&y_s).unwrap();
        let b = gpu.download_f32(&y2).unwrap();
        let same = a.iter().zip(&b).filter(|(p, q)| p.to_bits() == q.to_bits()).count();
        println!(
            "load_mtp_head: compressed_vocab_size={:?} lm_head_draft {:?} [{} x {}], vocab_map host+GPU ok, logits {same}/{cvs} bit-identical to the raw slice",
            w.compressed_vocab_size, lm.gpu_dtype, lm.m, lm.k
        );
    }
}
