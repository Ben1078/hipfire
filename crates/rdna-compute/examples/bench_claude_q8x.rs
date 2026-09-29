// claude: standalone bandwidth bench + f64 check of the `claude_q8x`
// small-batch MQ4G256V2 GEMVs (int8 activations, speculative-verify batch).
//
// Per shape (one weight `--m`, or several sharing one activation `--ms`,
// launched fused through the `_multi` twin like `claude_q8x_project` does):
//   * random MQ4G256V2 weights in a ring of copies of ~--ring-mb total, so
//     back-to-back launches stream from DRAM rather than from the 96 MB
//     Infinity Cache (production reads each weight once per step);
//   * random f32 activations quantized by `claude_q8x_quant` (the production
//     int8 path: xq + one f32 scale per 32 elements);
//   * every variant is launched through `Gpu::claude_q8x_gemv_launch`, i.e.
//     the launcher's own JIT names/sources/geometry;
//   * check: y vs an f64 CPU reference built from the same int8 activations
//     and the dequantized weights; err_i = |y_i - r_i| / max(|r_i|, rms(r)),
//     reported as the max over all outputs (and rel_l2);
//   * timing: hipEvents around N back-to-back launches that cycle the ring,
//     repeated --reps times after >= --warmup launches (and >= 0.3 s of
//     clock warm-up); prints the median (and min) us per launch and GB/s with
//     bytes = sum(M)*(K/256)*136 + B*K + B*(K/32)*4.
// `stream` is a bench-only roofline: a pure b128 streaming read of the same
// weight bytes (no math), for the achievable-bandwidth ceiling (`stream_t`:
// temporal loads; `rowstream`: one wave per weight row, whole row in flight;
// `lprow`: lane-per-row, each lane streams its own row group by group;
// `empty`: a do-nothing launch per weight, the back-to-back launch floor).
// `--launcher` times `claude_q8x_project` itself (quant + the picked variant).
// `--iso` times isolated launches (GPU idle before each) instead of back-to-back.
//
// Usage: bench_claude_q8x [--shapes prod|task|all] [--m M | --ms M0,M1,.. | --shape-list M0+M1/K[r],..]
//          [--k K] [--res] [--bs 4,2] [--variants v1:u4:o8,v2:s1:r1:p2:x1,stream,pick]
//          [--reps 50] [--warmup 30] [--ring-mb 256] [--no-check] [--multi 0|1]
//          [--launcher] [--iso] [--csv]
use rayon::prelude::*;
use rdna_compute::gemv::{claude_q8x_pick, ClaudeQ8xVariant};
use rdna_compute::{DType, Gpu, GpuTensor};

const GROUP_BYTES: usize = 136;
const PEAK_GBS: f64 = 960.0;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        // splitmix64
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32
    }
    fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.unit()
    }
}

fn f32_to_f16(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u32;
    let exp = ((b >> 23) & 0xff) as i32 - 127 + 15;
    let man = b & 0x7f_ffff;
    assert!(exp > 0 && exp < 31, "f32_to_f16: {x} outside the normal f16 range");
    let mut h = sign | ((exp as u32) << 10) | (man >> 13);
    let rem = man & 0x1fff;
    if rem > 0x1000 || (rem == 0x1000 && (h & 1) == 1) {
        h += 1;
    }
    h as u16
}

fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h as u32) & 0x8000) << 16;
    let exp = ((h >> 10) & 0x1f) as u32;
    let man = (h & 0x3ff) as u32;
    let bits = if exp == 0 {
        if man == 0 {
            sign
        } else {
            // subnormal
            let mut e = 127 - 15 + 1;
            let mut m = man;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            sign | ((e as u32) << 23) | ((m & 0x3ff) << 13)
        }
    } else if exp == 31 {
        sign | 0x7f80_0000 | (man << 13)
    } else {
        sign | ((exp + 127 - 15) << 23) | (man << 13)
    };
    f32::from_bits(bits)
}

/// Random MQ4G256V2 rows: per 256-group two (f16 scale, f16 zero) pairs, then
/// 128 bytes of 4-bit codes.
fn gen_weights(m: usize, k: usize, seed: u64) -> Vec<u8> {
    let gpr = k / 256;
    let mut v = vec![0u8; m * gpr * GROUP_BYTES];
    v.par_chunks_mut(gpr * GROUP_BYTES).enumerate().for_each(|(row, rb)| {
        let mut r = Rng(seed ^ (row as u64).wrapping_mul(0x2545_F491_4F6C_DD1D));
        for g in 0..gpr {
            let gb = &mut rb[g * GROUP_BYTES..(g + 1) * GROUP_BYTES];
            for h in 0..2 {
                let sc = r.range(0.002, 0.02);
                let zp = -sc * r.range(6.0, 9.0);
                gb[4 * h..4 * h + 2].copy_from_slice(&f32_to_f16(sc).to_le_bytes());
                gb[4 * h + 2..4 * h + 4].copy_from_slice(&f32_to_f16(zp).to_le_bytes());
            }
            for c in gb[8..].chunks_mut(8) {
                c.copy_from_slice(&r.next().to_le_bytes());
            }
        }
    });
    v
}

/// f64 reference `y[b][row] = sum_k W[row][k] * xhat[b][k]` with W dequantized
/// from MQ4G256V2 and xhat = d * xq (the int8 activations the GPU consumes).
fn reference(w: &[u8], m: usize, k: usize, xhat: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let gpr = k / 256;
    let nb = xhat.len();
    // per b, per half-group: sum of xhat (the zero-point term)
    let hsum: Vec<Vec<f64>> = xhat
        .iter()
        .map(|x| (0..gpr * 2).map(|i| x[i * 128..(i + 1) * 128].iter().sum()).collect())
        .collect();
    let rows: Vec<Vec<f64>> = (0..m)
        .into_par_iter()
        .map(|row| {
            let rb = &w[row * gpr * GROUP_BYTES..(row + 1) * gpr * GROUP_BYTES];
            let mut out = vec![0.0f64; nb];
            for g in 0..gpr {
                let gb = &rb[g * GROUP_BYTES..(g + 1) * GROUP_BYTES];
                for h in 0..2 {
                    let sc = f16_to_f32(u16::from_le_bytes([gb[4 * h], gb[4 * h + 1]])) as f64;
                    let zp = f16_to_f32(u16::from_le_bytes([gb[4 * h + 2], gb[4 * h + 3]])) as f64;
                    let mut dot = vec![0.0f64; nb];
                    for t in 16 * h..16 * h + 16 {
                        for jj in 0..4 {
                            let byte = gb[8 + 4 * t + jj];
                            let e = g * 256 + 8 * t + 2 * jj;
                            let (q0, q1) = ((byte & 15) as f64, (byte >> 4) as f64);
                            for b in 0..nb {
                                dot[b] += q0 * xhat[b][e] + q1 * xhat[b][e + 1];
                            }
                        }
                    }
                    for b in 0..nb {
                        out[b] += sc * dot[b] + zp * hsum[b][g * 2 + h];
                    }
                }
            }
            out
        })
        .collect();
    (0..nb).map(|b| rows.iter().map(|r| r[b]).collect()).collect()
}

/// xhat[b][k] from the device int8 activations: xq [b][k/256][32] uint2
/// (even elements in .x, odd in .y), dx [b][k/32].
fn dequant_acts(xq: &[u8], dx: &[f32], k: usize, nb: usize) -> Vec<Vec<f64>> {
    let gpr = k / 256;
    (0..nb)
        .map(|b| {
            let mut x = vec![0.0f64; k];
            for g in 0..gpr {
                for t in 0..32 {
                    let base = ((b * gpr + g) * 32 + t) * 8;
                    let e0 = g * 256 + 8 * t;
                    let d = dx[b * (k / 32) + e0 / 32] as f64;
                    for jj in 0..4 {
                        x[e0 + 2 * jj] = d * (xq[base + jj] as i8) as f64;
                        x[e0 + 2 * jj + 1] = d * (xq[base + 4 + jj] as i8) as f64;
                    }
                }
            }
            x
        })
        .collect()
}

#[derive(Clone)]
struct Shape {
    name: String,
    ms: Vec<usize>,
    k: usize,
    res: bool,
}

fn shape(name: &str, ms: &[usize], k: usize, res: bool) -> Shape {
    Shape { name: name.to_string(), ms: ms.to_vec(), k, res }
}

fn preset(name: &str) -> Vec<Shape> {
    // Qwen 3.8 27B verify projections (hidden 5120, 48 GDN + 16 attention layers).
    let prod = vec![
        shape("gdn_qkvza", &[10240, 6144, 48, 48], 5120, false),
        shape("attn_qkv", &[12288, 1024, 1024], 5120, false),
        shape("attn_qkv_nogate", &[6144, 1024, 1024], 5120, false),
        shape("mlp_gate_up", &[17408, 17408], 5120, false),
        shape("out_proj_res", &[5120], 6144, true),
        shape("mlp_down_res", &[5120], 17408, true),
        shape("lm_head", &[248320], 5120, false),
    ];
    let task = vec![
        shape("m16384", &[16384], 5120, false),
        shape("m96", &[96], 5120, false),
        shape("m34816", &[34816], 5120, false),
        shape("m5120", &[5120], 5120, false),
        shape("m14336", &[14336], 5120, false),
        shape("m8192", &[8192], 5120, false),
        shape("out_proj_res", &[5120], 6144, true),
        shape("mlp_down_res", &[5120], 17408, true),
    ];
    match name {
        "prod" => prod,
        "task" => task,
        "all" => {
            let mut v = prod;
            v.extend(task.into_iter().filter(|s| !s.name.ends_with("_res")));
            v
        }
        other => panic!("unknown --shapes {other} (prod|task|all)"),
    }
}

const STREAM_SRC: &str = r#"
#include <hip/hip_runtime.h>
// claude bench roofline: pure streaming read, 8 x b128 per thread in flight.
typedef unsigned int su4 __attribute__((ext_vector_type(4)));
extern "C" __global__ __launch_bounds__(256) void claude_q8x_bench_stream(
    const su4* __restrict__ p, long long n16, unsigned int* __restrict__ out)
{
    const long long base = (long long)blockIdx.x * 256 * 8 + threadIdx.x;
    su4 v[8];
#pragma unroll
    for (int i = 0; i < 8; i++) {
        const long long idx = base + i * 256;
        v[i] = idx < n16 ? __builtin_nontemporal_load(p + idx) : (su4){0, 0, 0, 0};
    }
    unsigned int x = 0;
#pragma unroll
    for (int i = 0; i < 8; i++) x ^= v[i].x ^ v[i].y ^ v[i].z ^ v[i].w;
    if (x == 0x9e3779b9u) out[0] = x;
}
// same, plain (temporal) loads
extern "C" __global__ __launch_bounds__(256) void claude_q8x_bench_stream_t(
    const su4* __restrict__ p, long long n16, unsigned int* __restrict__ out)
{
    const long long base = (long long)blockIdx.x * 256 * 8 + threadIdx.x;
    su4 v[8];
#pragma unroll
    for (int i = 0; i < 8; i++) {
        const long long idx = base + i * 256;
        v[i] = idx < n16 ? p[idx] : (su4){0, 0, 0, 0};
    }
    unsigned int x = 0;
#pragma unroll
    for (int i = 0; i < 8; i++) x ^= v[i].x ^ v[i].y ^ v[i].z ^ v[i].w;
    if (x == 0x9e3779b9u) out[0] = x;
}
// one wave per weight row (row_bytes, multiple of 16): all of the row's
// b128 loads issued at once, nontemporal; 8 rows per 256-thread workgroup.
extern "C" __global__ __launch_bounds__(256) void claude_q8x_bench_rowstream(
    const su4* __restrict__ p, long long n16, unsigned int* __restrict__ out, int row16)
{
    const long long row = (long long)blockIdx.x * 8 + (threadIdx.x >> 5);
    const int t = threadIdx.x & 31;
    unsigned int x = 0;
    su4 v[16];
#pragma unroll
    for (int i = 0; i < 16; i++) {
        const int c = t + 32 * i;
        const long long idx = row * row16 + c;
        v[i] = (c < row16 && idx < n16) ? __builtin_nontemporal_load(p + idx) : (su4){0, 0, 0, 0};
    }
#pragma unroll
    for (int i = 0; i < 16; i++) x ^= v[i].x ^ v[i].y ^ v[i].z ^ v[i].w;
    if (x == 0x9e3779b9u) out[0] = x;
}
// launch floor: one workgroup that does nothing
extern "C" __global__ __launch_bounds__(256) void claude_q8x_bench_empty(
    const su4* __restrict__ p, long long n16, unsigned int* __restrict__ out)
{
    if (n16 < 0) out[0] = p[0].x;
}
// lane-per-row: lane = weight row, each lane streams its own row one 136-byte
// group at a time (8 x b128 codes at +8, one b64 header), two groups per
// iteration; 256 rows per workgroup.
extern "C" __global__ __launch_bounds__(256) void claude_q8x_bench_lprow(
    const su4* __restrict__ p, long long n16, unsigned int* __restrict__ out, int row16)
{
    const long long row = (long long)blockIdx.x * 256 + threadIdx.x;
    const long long nrow = n16 / row16;
    if (row >= nrow) return;
    const char* rp = (const char*)p + row * row16 * 16;
    const int gpr = row16 * 16 / 136;
    unsigned int x = 0;
    for (int g = 0; g + 1 < gpr; g += 2) {
        su4 v[16];
        unsigned int h[4];
#pragma unroll
        for (int u = 0; u < 2; u++) {
            const char* gp = rp + (g + u) * 136;
#pragma unroll
            for (int i = 0; i < 8; i++) v[u * 8 + i] = __builtin_nontemporal_load((const su4*)(gp + 8 + 16 * i));
            h[2 * u] = __builtin_nontemporal_load((const unsigned int*)gp);
            h[2 * u + 1] = __builtin_nontemporal_load((const unsigned int*)(gp + 4));
        }
#pragma unroll
        for (int i = 0; i < 16; i++) x ^= v[i].x ^ v[i].y ^ v[i].z ^ v[i].w;
        x ^= h[0] ^ h[1] ^ h[2] ^ h[3];
    }
    if (x == 0x9e3779b9u) out[0] = x;
}
"#;

enum Kind {
    Variant(ClaudeQ8xVariant),
    Pick,
    Stream(&'static str),
}

fn parse_kind(s: &str) -> Kind {
    match s {
        "stream" => Kind::Stream("claude_q8x_bench_stream"),
        "stream_t" => Kind::Stream("claude_q8x_bench_stream_t"),
        "rowstream" => Kind::Stream("claude_q8x_bench_rowstream"),
        "lprow" => Kind::Stream("claude_q8x_bench_lprow"),
        "empty" => Kind::Stream("claude_q8x_bench_empty"),
        "pick" => Kind::Pick,
        _ => Kind::Variant(ClaudeQ8xVariant::parse(s).unwrap_or_else(|| panic!("bad variant spec {s:?}"))),
    }
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n % 2 == 1 { v[n / 2] } else { 0.5 * (v[n / 2 - 1] + v[n / 2]) }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let val = |k: &str| args.iter().position(|a| a == k).map(|i| args[i + 1].clone());
    let flag = |k: &str| args.iter().any(|a| a == k);
    let list = |s: &str| -> Vec<usize> { s.split(',').filter(|x| !x.is_empty()).map(|x| x.parse().unwrap()).collect() };
    let reps: usize = val("--reps").map(|v| v.parse().unwrap()).unwrap_or(50);
    let warmup: usize = val("--warmup").map(|v| v.parse().unwrap()).unwrap_or(30);
    let ring_mb: usize = val("--ring-mb").map(|v| v.parse().unwrap()).unwrap_or(256);
    let check = !flag("--no-check");
    let multi = val("--multi").map(|v| v != "0").unwrap_or(true);
    let launcher = flag("--launcher");
    // --iso: time single launches (GPU idle before each) instead of N back-to-back
    let iso = flag("--iso");
    let csv = flag("--csv");
    let bs: Vec<usize> = val("--bs").or(val("--b")).map(|v| list(&v)).unwrap_or_else(|| vec![4]);
    let variants: Vec<String> = val("--variants")
        .unwrap_or_else(|| "v1:u4:o8,pick".to_string())
        .split(',')
        .map(|s| s.to_string())
        .collect();
    let kinds: Vec<(String, Kind)> = variants.iter().map(|s| (s.clone(), parse_kind(s))).collect();
    // --shape-list "M0+M1/K[r],...": several shapes in one run (r = residual)
    let shapes: Vec<Shape> = if let Some(sl) = val("--shape-list") {
        sl.split(',')
            .map(|it| {
                let (ms, kk) = it.split_once('/').expect("--shape-list item M0+M1/K[r]");
                let res = kk.ends_with('r');
                let k: usize = kk.trim_end_matches('r').parse().unwrap();
                let ms: Vec<usize> = ms.split('+').map(|m| m.parse().unwrap()).collect();
                let name = if ms.len() > 1 { format!("ms{}_k{k}", ms.iter().map(|m| m.to_string()).collect::<Vec<_>>().join("+")) } else { format!("m{}_k{k}", ms[0]) };
                shape(&name, &ms, k, res)
            })
            .collect()
    } else if let Some(m) = val("--m") {
        let k = val("--k").map(|v| v.parse().unwrap()).unwrap_or(5120);
        list(&m).into_iter().map(|m| shape(&format!("m{m}_k{k}"), &[m], k, flag("--res"))).collect()
    } else if let Some(ms) = val("--ms") {
        let k = val("--k").map(|v| v.parse().unwrap()).unwrap_or(5120);
        vec![shape(&format!("ms{}_k{k}", ms.replace(',', "+")), &list(&ms), k, flag("--res"))]
    } else {
        preset(&val("--shapes").unwrap_or_else(|| "prod".to_string()))
    };

    let mut gpu = Gpu::init().expect("GPU init");
    eprintln!(
        "GPU {}  reps {reps}  warmup {warmup}  ring {ring_mb} MB  multi {multi}  check {check}{}",
        gpu.arch,
        if launcher { "  (launcher path: quant + claude_q8x_project)" } else { "" }
    );
    for f in ["claude_q8x_bench_stream", "claude_q8x_bench_stream_t", "claude_q8x_bench_rowstream", "claude_q8x_bench_lprow", "claude_q8x_bench_empty"] {
        gpu.ensure_kernel_public("claude_q8x_bench_stream", STREAM_SRC, f).expect("stream kernel");
    }
    let dummy = gpu.zeros(&[64], DType::F32).unwrap();
    let ev0 = gpu.hip.event_create().unwrap();
    let ev1 = gpu.hip.event_create().unwrap();
    if csv {
        println!("shape,k,ms,b,res,variant,us_median,us_min,gbs,pct_peak,max_rel_err,rel_l2,ring");
    }

    for sh in &shapes {
        let k = sh.k;
        let gpr = k / 256;
        let row_bytes = gpr * GROUP_BYTES;
        let budget = ring_mb << 20;
        let mut ms = sh.ms.clone();
        let total: usize = ms.iter().sum();
        if total * row_bytes > budget {
            let cap = budget / row_bytes;
            eprintln!("note: {} rows ({} MB) exceed --ring-mb; capped to {cap} rows (one copy)", total, total * row_bytes >> 20);
            ms = vec![cap];
        }
        let total: usize = ms.iter().sum();
        let copy_bytes = total * row_bytes;
        let ring = (budget / copy_bytes).clamp(1, 16);
        // host weights of copy 0 (for the reference), device ring
        let host0: Vec<Vec<u8>> = ms.iter().enumerate().map(|(i, &m)| gen_weights(m, k, 0x5eed_0000 + i as u64)).collect();
        let mut ring_w: Vec<Vec<GpuTensor>> = Vec::new();
        for c in 0..ring {
            let mut set = Vec::new();
            for (i, &m) in ms.iter().enumerate() {
                let bytes = if c == 0 { host0[i].clone() } else { gen_weights(m, k, 0x5eed_0000 + (c * 16 + i) as u64) };
                set.push(gpu.upload_raw(&bytes, &[bytes.len()]).expect("upload weights"));
            }
            ring_w.push(set);
        }
        let ys: Vec<GpuTensor> = ms.iter().map(|&m| gpu.zeros(&[8 * m], DType::F32).unwrap()).collect();
        eprintln!(
            "\n== {}  K={k}  M={:?} (sum {total}, {:.1} MB/copy, ring {ring})  {}",
            sh.name,
            ms,
            copy_bytes as f64 / 1e6,
            if sh.res { "residual" } else { "" }
        );

        for &b in &bs {
            // activations: random f32 with sparse outliers, then the production int8 quant
            let mut r = Rng(0xacc0_0000 + b as u64 * 7 + k as u64);
            let xf: Vec<f32> = (0..b * k)
                .map(|i| {
                    let v = r.range(-1.0, 1.0);
                    if i % 97 == 13 { v * 6.0 } else { v }
                })
                .collect();
            let x = gpu.upload_f32(&xf, &[b * k]).unwrap();
            let xq = gpu.zeros(&[b * k / 4], DType::F32).unwrap();
            let dx = gpu.zeros(&[b * k / 32], DType::F32).unwrap();
            let (xq_p, dx_p) = (xq.buf.as_ptr(), dx.buf.as_ptr());
            gpu.claude_q8x_quant_into(&x, xq_p, dx_p, k, b).unwrap();
            gpu.hip.device_synchronize().unwrap();
            let refs: Option<Vec<Vec<Vec<f64>>>> = check.then(|| {
                let mut xq_h = vec![0u8; b * k];
                gpu.hip.memcpy_dtoh(&mut xq_h, &xq.buf).unwrap();
                let dx_h = gpu.download_f32(&dx).unwrap();
                let xhat = dequant_acts(&xq_h, &dx_h, k, b);
                ms.iter().enumerate().map(|(i, &m)| reference(&host0[i], m, k, &xhat)).collect()
            });
            let act_bytes = b * k + b * (k / 32) * 4;
            let bytes = (copy_bytes + act_bytes) as f64;

            for (_spec, kind) in &kinds {
                let (vname, v) = match kind {
                    Kind::Variant(v) => (v.spec(), Some(*v)),
                    Kind::Pick => {
                        let v = claude_q8x_pick(k, total, b, sh.res);
                        (format!("pick={}", v.spec()), Some(v))
                    }
                    Kind::Stream(f) => (f.trim_start_matches("claude_q8x_bench_").to_string(), None),
                };
                if let Some(v) = v {
                    if !v.supports(k, b) {
                        eprintln!("  B={b} {vname:<34} (not supported at K={k} batch {b})");
                        continue;
                    }
                }
                // one launch of ring copy c
                let sfunc: &str = match kind {
                    Kind::Stream(f) => f,
                    _ => "",
                };
                let launch = |gpu: &mut Gpu, c: usize| {
                    let outs: Vec<(&GpuTensor, &GpuTensor, usize)> =
                        ring_w[c].iter().zip(&ys).zip(&ms).map(|((a, y), &m)| (a, y, m)).collect();
                    match v {
                        Some(v) if !launcher => {
                            gpu.claude_q8x_gemv_launch(v, &outs, xq_p, dx_p, k, b, sh.res, multi).expect("launch")
                        }
                        Some(_) => gpu.claude_q8x_project(&outs, &x, k, b, sh.res).expect("project"),
                        None => {
                            for (a, &m) in ring_w[c].iter().zip(&ms) {
                                let n16 = (a.buf.size() / 16) as i64;
                                let mut blob = hip_bridge::KernargBlob::new();
                                blob.push_ptr(a.buf.as_ptr());
                                blob.push_u64(n16 as u64);
                                blob.push_ptr(dummy.buf.as_ptr());
                                let grid = if sfunc == "claude_q8x_bench_rowstream" {
                                    blob.push_i32((row_bytes / 16) as i32);
                                    [m.div_ceil(8) as u32, 1, 1]
                                } else if sfunc == "claude_q8x_bench_empty" {
                                    [1, 1, 1]
                                } else if sfunc == "claude_q8x_bench_lprow" {
                                    blob.push_i32((row_bytes / 16) as i32);
                                    [m.div_ceil(256) as u32, 1, 1]
                                } else {
                                    [((n16 as usize).div_ceil(256 * 8)) as u32, 1, 1]
                                };
                                gpu.launch_kernel_blob(sfunc, grid, [256, 1, 1], 0, blob.as_mut_slice())
                                    .expect("stream launch");
                            }
                        }
                    }
                };
                // correctness (copy 0)
                let (mut max_err, mut rel_l2) = (f64::NAN, f64::NAN);
                if let (Some(refs), Some(_)) = (&refs, v) {
                    let mut inits: Vec<Vec<f32>> = Vec::new();
                    for (i, &m) in ms.iter().enumerate() {
                        let init: Vec<f32> = if sh.res {
                            let mut rr = Rng(0x1234 + i as u64);
                            (0..8 * m).map(|_| rr.range(-4.0, 4.0)).collect()
                        } else {
                            vec![f32::NAN; 8 * m]
                        };
                        let bytes: Vec<u8> = init.iter().flat_map(|f| f.to_le_bytes()).collect();
                        gpu.hip.memcpy_htod(&ys[i].buf, &bytes).unwrap();
                        inits.push(init);
                    }
                    launch(&mut gpu, 0);
                    gpu.hip.device_synchronize().unwrap();
                    max_err = 0.0;
                    let (mut sq_d, mut sq_r) = (0.0f64, 0.0f64);
                    for (i, &m) in ms.iter().enumerate() {
                        let got = gpu.download_f32(&ys[i]).unwrap();
                        let rr = &refs[i];
                        let n = (b * m) as f64;
                        let rms = (rr.iter().flat_map(|v| v.iter()).map(|v| v * v).sum::<f64>() / n).sqrt().max(1e-30);
                        for bb in 0..b {
                            for row in 0..m {
                                let want = rr[bb][row] + if sh.res { inits[i][bb * m + row] as f64 } else { 0.0 };
                                let g = got[bb * m + row] as f64;
                                let d = (g - want).abs();
                                let e = if d.is_finite() { d / rr[bb][row].abs().max(rms) } else { f64::INFINITY };
                                max_err = max_err.max(e);
                                sq_d += d * d;
                                sq_r += rr[bb][row] * rr[bb][row];
                            }
                        }
                    }
                    rel_l2 = (sq_d / sq_r.max(1e-300)).sqrt();
                }
                // timing
                let n_launch = ring.max(8);
                let t0 = std::time::Instant::now();
                let mut i = 0usize;
                while i < warmup || t0.elapsed().as_secs_f64() < 0.3 {
                    launch(&mut gpu, i % ring);
                    i += 1;
                    if i % 64 == 0 {
                        gpu.hip.device_synchronize().unwrap();
                    }
                }
                gpu.hip.device_synchronize().unwrap();
                let mut ts: Vec<f64> = Vec::with_capacity(reps);
                for rep in 0..reps {
                    if iso {
                        // one launch between its own events, GPU idle before it
                        gpu.hip.device_synchronize().unwrap();
                        gpu.hip.event_record(&ev0, gpu.active_stream.as_ref()).unwrap();
                        launch(&mut gpu, rep % ring);
                        gpu.hip.event_record(&ev1, gpu.active_stream.as_ref()).unwrap();
                        gpu.hip.event_synchronize(&ev1).unwrap();
                        ts.push(gpu.hip.event_elapsed_ms(&ev0, &ev1).unwrap() as f64 * 1e3);
                        continue;
                    }
                    gpu.hip.event_record(&ev0, gpu.active_stream.as_ref()).unwrap();
                    for j in 0..n_launch {
                        launch(&mut gpu, (rep * n_launch + j) % ring);
                    }
                    gpu.hip.event_record(&ev1, gpu.active_stream.as_ref()).unwrap();
                    gpu.hip.event_synchronize(&ev1).unwrap();
                    ts.push(gpu.hip.event_elapsed_ms(&ev0, &ev1).unwrap() as f64 * 1e3 / n_launch as f64);
                }
                let us_min = ts.iter().cloned().fold(f64::INFINITY, f64::min);
                let us = median(&mut ts);
                let gbs = if v.is_none() { copy_bytes as f64 / (us * 1e3) } else { bytes / (us * 1e3) };
                eprintln!(
                    "  B={b} {vname:<34} {us:8.2} us (min {us_min:7.2})  {gbs:6.1} GB/s  {:5.1}%  err {max_err:.2e}  l2 {rel_l2:.2e}",
                    100.0 * gbs / PEAK_GBS
                );
                if csv {
                    println!(
                        "{},{k},{},{b},{},{vname},{us:.3},{us_min:.3},{gbs:.1},{:.2},{max_err:.3e},{rel_l2:.3e},{ring}",
                        sh.name,
                        ms.iter().map(|m| m.to_string()).collect::<Vec<_>>().join("+"),
                        sh.res as u8,
                        100.0 * gbs / PEAK_GBS
                    );
                }
            }
            gpu.free_tensor(x).unwrap();
            gpu.free_tensor(xq).unwrap();
            gpu.free_tensor(dx).unwrap();
        }
        for y in ys {
            gpu.free_tensor(y).unwrap();
        }
        for set in ring_w {
            for a in set {
                gpu.free_tensor(a).unwrap();
            }
        }
    }
}
