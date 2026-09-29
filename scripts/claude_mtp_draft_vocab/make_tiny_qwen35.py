#!/usr/bin/env python3
"""claude r2: tiny synthetic Qwen3.5 (qwen3_5, arch 5) trunk + matching MTP head
(.mtp, arch 21) for end-to-end tests of the MTP decode loop on the real
runtime code paths (MQ4G256V2 trunk weights with AWQ=1 sidecars, Q8_0
embedding, F16 norms, production head_dim/rope/mrope config), under 200 MB.

Construction (so the draft agrees with the verifier often, but not always):
  * every residual-branch output projection (GDN out_proj, FA o_proj, MLP
    down_proj) is ZERO -> the trunk's hidden at every position is its input
    embedding, and trunk logits(token y) = lm_head . rmsnorm(e_y);
  * all other trunk weights are random (the recurrent / KV state is exercised);
  * MTP head: eh_proj = [I | 0] (F32, exact), wo = 0, ffn_down = 0, norms 1
    -> the head's hidden = enorm(e_y) and its draft logits are the trunk's
    logits for the NEXT token (the verify lm_head uses int8 activations, so
    near-ties flip -> a mix of accepts and rejects). `--noisy` perturbs the
    head's shared_head_norm weights to lower the acceptance.
Usage: make_tiny_qwen35.py OUTDIR [--seed S] [--vocab V] [--noisy]
"""
import json, struct, sys, os
import numpy as np

args = sys.argv[1:]
out = args[0]
seed = int(args[args.index('--seed') + 1]) if '--seed' in args else 1234
V = int(args[args.index('--vocab') + 1]) if '--vocab' in args else 69632
noisy = '--noisy' in args
lm_scale = float(args[args.index('--lm-scale') + 1]) if '--lm-scale' in args else 0.05
rng = np.random.default_rng(seed)

D = 1024; H = 8; KV = 2; HD = 256; F = 1024
LKH = 4; LKD = 128; LVH = 12; LVD = 128; CONV = 4
NL = 4
layer_types = ['linear_attention', 'linear_attention', 'linear_attention', 'full_attention']
qkv_dim = LKH * LKD * 2 + LVH * LVD
d_inner = LVH * LVD
EOS = V - 632  # an ordinary id inside the vocab

def f16(a):
    return np.asarray(a, dtype=np.float16).tobytes()

def f32(a):
    return np.asarray(a, dtype=np.float32).tobytes()

def mq4v2(m, k, scale=0.01, zero=False):
    """MQ4G256V2: per 256-group 136 B: [s0 f16][z0 f16][s1 f16][z1 f16] + 128 B nibbles."""
    g = k // 256
    n = m * g
    if zero:
        return bytes(n * 136)
    buf = np.zeros((n, 136), dtype=np.uint8)
    for half in range(2):
        sc = (scale * (0.5 + rng.random(n))).astype(np.float16)
        zp = (-7.5 * sc.astype(np.float32) + (rng.random(n) - 0.5) * sc.astype(np.float32)).astype(np.float16)
        buf[:, half * 4:half * 4 + 2] = sc.view(np.uint8).reshape(n, 2)
        buf[:, half * 4 + 2:half * 4 + 4] = zp.view(np.uint8).reshape(n, 2)
    buf[:, 8:] = rng.integers(0, 256, size=(n, 128), dtype=np.uint8)
    return buf.tobytes()

def mq4v1(m, k, scale=0.01, zero=False):
    """MQ4G256 (v1): per 256-group 136 B: [f32 scale][f32 zero][128 B nibbles]."""
    g = k // 256
    n = m * g
    if zero:
        return bytes(n * 136)
    buf = np.zeros((n, 136), dtype=np.uint8)
    sc = (scale * (0.5 + rng.random(n))).astype(np.float32)
    zp = (-7.5 * sc + (rng.random(n) - 0.5) * sc * 0.02).astype(np.float32)
    buf[:, 0:4] = sc.view(np.uint8).reshape(n, 4)
    buf[:, 4:8] = zp.view(np.uint8).reshape(n, 4)
    buf[:, 8:] = rng.integers(0, 256, size=(n, 128), dtype=np.uint8)
    return buf.tobytes()

def q8f16(m, k, scale=0.02):
    """Q8_F16 (GGML Q8_0): per 32 elements [f16 scale][32 x i8]."""
    n = m * k // 32
    buf = np.zeros((n, 34), dtype=np.uint8)
    sc = (scale * (0.5 + rng.random(n))).astype(np.float16)
    buf[:, 0:2] = sc.view(np.uint8).reshape(n, 2)
    buf[:, 2:] = rng.integers(0, 256, size=(n, 32), dtype=np.uint8)
    return buf.tobytes()

def write_hfq(path, arch, meta, tensors):
    meta_b = json.dumps(meta).encode()
    idx = bytearray()
    idx += struct.pack('<I', len(tensors))
    for (name, qt, shape, gs, data) in tensors:
        nb = name.encode()
        idx += struct.pack('<H', len(nb)) + nb + struct.pack('<BB', qt, len(shape))
        for d in shape:
            idx += struct.pack('<I', d)
        idx += struct.pack('<IQ', gs, len(data))
    data_start = 32 + len(meta_b) + len(idx)
    data_off = (data_start + 4095) & ~4095
    with open(path, 'wb') as f:
        f.write(b'HFQM' + struct.pack('<IIIQQ', 1, arch, len(tensors), 32, data_off))
        f.write(meta_b)
        f.write(idx)
        f.write(bytes(data_off - data_start))
        for t in tensors:
            f.write(t[4])

# ---------------- trunk ----------------
T = []
P = 'model.language_model.'
T.append((P + 'embed_tokens.weight', 3, [V, D], 32, q8f16(V, D, 0.02)))
T.append(('lm_head.weight', 44, [V, D], 256, mq4v2(V, D, lm_scale)))
T.append((P + 'norm.weight', 1, [D], 0, f16(np.zeros(D))))

def proj(name, m, k, zero=False, scale=0.01):
    T.append((name + '.weight', 44, [m, k], 256, mq4v2(m, k, scale, zero)))
    T.append((name + '.awq_scale.weight', 1, [k], 0, f16(np.ones(k))))

for i, lt in enumerate(layer_types):
    L = P + f'layers.{i}.'
    T.append((L + 'input_layernorm.weight', 1, [D], 0, f16(rng.normal(0, 0.05, D))))
    T.append((L + 'post_attention_layernorm.weight', 1, [D], 0, f16(rng.normal(0, 0.05, D))))
    if lt == 'linear_attention':
        proj(L + 'linear_attn.in_proj_qkv', qkv_dim, D)
        proj(L + 'linear_attn.in_proj_z', d_inner, D)
        proj(L + 'linear_attn.in_proj_a', LVH, D)
        proj(L + 'linear_attn.in_proj_b', LVH, D)
        T.append((L + 'linear_attn.A_log', 2, [LVH], 0, f32(np.log(rng.uniform(1, 16, LVH)))))
        T.append((L + 'linear_attn.dt_bias', 2, [LVH], 0, f32(rng.normal(0, 0.5, LVH))))
        T.append((L + 'linear_attn.conv1d.weight', 2, [qkv_dim, 1, CONV], 0, f32(rng.normal(0, 0.3, qkv_dim * CONV))))
        T.append((L + 'linear_attn.norm.weight', 2, [LVD], 0, f32(np.ones(LVD))))
        proj(L + 'linear_attn.out_proj', D, d_inner, zero=True)
    else:
        proj(L + 'self_attn.q_proj', 2 * H * HD, D)
        proj(L + 'self_attn.k_proj', KV * HD, D)
        proj(L + 'self_attn.v_proj', KV * HD, D)
        proj(L + 'self_attn.o_proj', D, H * HD, zero=True)
        T.append((L + 'self_attn.q_norm.weight', 1, [HD], 0, f16(rng.normal(0, 0.05, HD))))
        T.append((L + 'self_attn.k_norm.weight', 1, [HD], 0, f16(rng.normal(0, 0.05, HD))))
    proj(L + 'mlp.gate_proj', F, D)
    proj(L + 'mlp.up_proj', F, D)
    proj(L + 'mlp.down_proj', D, F, zero=True)

text_config = {
    "attention_bias": False, "attn_output_gate": True, "bos_token_id": EOS, "eos_token_id": EOS,
    "full_attention_interval": 4, "head_dim": HD, "hidden_act": "silu", "hidden_size": D,
    "intermediate_size": F, "linear_conv_kernel_dim": CONV, "linear_key_head_dim": LKD,
    "linear_num_key_heads": LKH, "linear_num_value_heads": LVH, "linear_value_head_dim": LVD,
    "max_position_embeddings": 262144, "model_type": "qwen3_5_text", "mtp_num_hidden_layers": 1,
    "num_attention_heads": H, "num_hidden_layers": NL, "num_key_value_heads": KV,
    "partial_rotary_factor": 0.25, "rms_norm_eps": 1e-6,
    "rope_parameters": {"mrope_interleaved": True, "mrope_section": [11, 11, 10],
                        "partial_rotary_factor": 0.25, "rope_theta": 10000000, "rope_type": "default"},
    "tie_word_embeddings": False, "vocab_size": V, "layer_types": layer_types,
}
config = {"architectures": ["Qwen3_5ForConditionalGeneration"], "model_type": "qwen3_5",
          "text_config": text_config, "tie_word_embeddings": False,
          "vision_config": {"depth": 1, "hidden_size": 64}}
meta = {"architecture": "qwen3_5", "config": config,
        "generation_config": {"bos_token_id": EOS, "eos_token_id": EOS}}
os.makedirs(out, exist_ok=True)
write_hfq(os.path.join(out, 'tiny-qwen35.mq4'), 5, meta, T)

# ---------------- MTP head ----------------
M = []
def nrm(name, n, noise=0.0):
    M.append((name, 2, [n], 0, f32(rng.normal(0, noise, n) if noise > 0 else np.zeros(n))))
nrm('shared_head_norm', D, 0.3 if noisy else 0.0)
nrm('enorm', D); nrm('hnorm', D); nrm('attn_norm', D); nrm('attn_post_norm', D)
nrm('attn_q_norm', HD); nrm('attn_k_norm', HD)
eh = np.zeros((D, 2 * D), dtype=np.float32)
eh[np.arange(D), np.arange(D)] = 1.0
M.append(('eh_proj', 2, [D, 2 * D], 0, eh.tobytes()))
M.append(('wq', 13, [2 * H * HD, D], 256, mq4v1(2 * H * HD, D)))
M.append(('wk', 13, [KV * HD, D], 256, mq4v1(KV * HD, D)))
M.append(('wv', 13, [KV * HD, D], 256, mq4v1(KV * HD, D)))
M.append(('wo', 13, [D, H * HD], 256, mq4v1(D, H * HD, zero=True)))
M.append(('ffn_gate', 13, [F, D], 256, mq4v1(F, D)))
M.append(('ffn_up', 13, [F, D], 256, mq4v1(F, D)))
M.append(('ffn_down', 13, [D, F], 256, mq4v1(D, F, zero=True)))
mmeta = {"arch": "qwen35_mtp_head", "arch_id": 21, "source_model": "tiny-qwen35", "n_embd": D,
         "n_layer": NL, "nextn_predict_layers": 1, "n_head": H, "n_head_kv": KV, "n_embd_head": HD,
         "n_ff": F, "ffn_kind": "dense", "num_experts": 0, "num_experts_per_tok": 0,
         "moe_intermediate_size": 0, "shared_expert_intermediate_size": 0, "norm_topk_prob": True,
         "vocab_size": V, "rope_theta": 10000000.0, "rope_sections": [11, 11, 10], "rms_norm_eps": 1e-6,
         "shared_embed_with_trunk": True, "shared_lm_head_with_trunk": True,
         "shared_output_norm_with_trunk": False, "tie_word_embeddings": False,
         "weight_quant": "MQ4G256", "has_compressed_lm_head_draft": False, "compressed_vocab_size": 0,
         "config_text_config": text_config}
write_hfq(os.path.join(out, 'tiny-qwen35' + ('-noisy' if noisy else '') + '.mtp'), 21, mmeta, M)
print('wrote', out, 'vocab', V, 'eos', EOS, 'noisy', noisy)
