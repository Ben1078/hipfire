#!/usr/bin/env python3
"""A4 fused-ladder study harness: quality (KLD), timing (ABBA prefill), summaries.

Subcommands
  plan             print the ABBA timing schedule (no GPU, no files written)
  quality          one arm, both corpora, fresh HOME + kernel cache per corpus (run inside a
                   fabric `--class correct` hold, <=600 s)
  timing           all selected arms in palindromic ABBA blocks, fresh process / HOME / kernel
                   cache per run, inside ONE fabric `--class timing` hold (<=1800 s)
  summarize        recompute summary.json from an existing timing raw.jsonl
  quality-compare  absolute KLD, delta vs control, recovered WMM4 fraction, +0.010 cap check

The harness never starts a fabric hold itself. quality/timing refuse to run unless FABRIC_JOB is
set (i.e. they were launched by fabric-run.sh). See README.md next to this file.
"""
import argparse
import glob
import hashlib
import json
import math
import os
import pathlib
import re
import signal
import statistics
import struct
import subprocess
import sys
import threading
import time

T0 = time.monotonic()
HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]  # <repo>/experiments/a4-ladder/runner.py

# ---------------------------------------------------------------- pins and tables (AGENTS.md section 5)
CANON_MODEL_SHA256 = '3e38ccbae3776470eb5a89344d300e9279d6b9ab6c31fd40ca1758c4f7c6f8ae'
CANON_MODEL_BYTES = 14987185152
REFS = {  # corpus -> (default path, reference sha256)
    'wt2': ('/home/kaden/kldrefs/qwen3.8-27b.ref_wt2.bin',
            '8c545178fb43647499c4c7bdd33178c24fd402615c641add9ed51a0c9a43234a'),
    'code24': ('/home/kaden/qcal/qat/v3/heldout-code24.gptqsym.kldref.bin',
               '43ab7591f2acf6b5b4058b77253109230d96591be02469ddef55235102ec7582'),
}
PINS = {  # arch -> corpus -> default-route .kldseq sha256 (gfx1201, kv fp8/q8, cold cache)
    'r9700': {
        'wt2': 'ff88c49910be618a98a917afc40d4b01ceeb7c6299c62f1c553444c11b402841',
        'code24': '9b0616316f50e9a654690d06d594d57466430d0a4a67041ec4ff0128d830910e',
    },
}
WMM4_GAIN = {'wt2': 0.0023473457, 'code24': 0.0019178636}  # standalone WMM4 KLD reductions (PLAN.md)
KLD_CAP = 0.010  # one-sided: arm KLD may exceed control by at most this
N_CHUNKS = 24
ARCHES = {
    'r9700': {'gfx': 'gfx1201', 'env_key': 'HIPFIRE_G12_A4_CANDIDATES', 'quality_kv': ('fp8', 'q8')},
    'halo': {'gfx': 'gfx1151', 'env_key': 'HIPFIRE_GFX11_A4_CANDIDATES', 'quality_kv': ('q8', 'q8')},
}
TIMING_KV = ('q8', 'q8')  # bench_qwen35_mq4 supports q8 only; identical on both architectures
KEEP_ENV = {'HIPFIRE_DEVICES', 'HIPFIRE_LOCK_DIR', 'HIPFIRE_FABRIC_ADMISSION'}
UNSET = '@unset'
DEFAULT_SOURCES = [
    'crates/hipfire-runtime/examples/bench_qwen35_mq4.rs',
    'crates/hipfire-runtime/examples/eval_hipfire.rs',
    'crates/rdna-compute/src/feature_flags.rs',
    'crates/rdna-compute/src/compiler.rs',
    'experiments/a4-ladder/runner.py',
]
EVIDENCE_RE = re.compile(r'A4_CANDIDATES|per-kernel flags|unknown HIPFIRE_|falling back|WARN|fallback', re.I)

# ---------------------------------------------------------------- process control
STOP = threading.Event()
CHILD = [None]


def _on_term(signum, _frame):
    STOP.set()
    p = CHILD[0]
    if p is not None:
        try:
            os.kill(p.pid, signal.SIGKILL)  # never Popen.kill/poll: they could reap before os.wait4
        except OSError:
            pass


signal.signal(signal.SIGTERM, _on_term)
signal.signal(signal.SIGINT, _on_term)


def die(msg):
    raise SystemExit('a4-ladder: ' + msg)


def require_fabric():
    if not os.environ.get('FABRIC_JOB'):
        die('run only inside fabric-run.sh (FABRIC_JOB is not set)')


# ---------------------------------------------------------------- hashing / provenance
def file_digests(path, md5=False):
    h = hashlib.sha256()
    m = hashlib.md5() if md5 else None
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(16 << 20), b''):
            h.update(b)
            if m:
                m.update(b)
    return h.hexdigest(), (m.hexdigest() if m else None)


def git(repo, *args):
    try:
        r = subprocess.run(['git', '-C', str(repo), *args], capture_output=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return r.stdout if r.returncode == 0 else None


def provenance(repo, extra_sources):
    repo = pathlib.Path(repo).resolve()
    head = git(repo, 'rev-parse', 'HEAD')
    status = git(repo, 'status', '--porcelain')
    diff = git(repo, 'diff', 'HEAD')
    untracked = git(repo, 'ls-files', '--others', '--exclude-standard')
    out = {
        'repo': str(repo),
        'git_head': head.decode().strip() if head else None,
        'git_status_porcelain': status.decode().splitlines() if status is not None else None,
        'git_diff_head_sha256': hashlib.sha256(diff).hexdigest() if diff is not None else None,
        'untracked_sha256': {},
        'sources': {},
        'harness_sha256': file_digests(__file__)[0],
    }
    if untracked:
        for rel in untracked.decode().splitlines()[:200]:
            p = repo / rel
            if p.is_file() and p.stat().st_size <= (16 << 20):
                out['untracked_sha256'][rel] = file_digests(p)[0]
    for rel in dict.fromkeys([*DEFAULT_SOURCES, *extra_sources]):
        p = repo / rel
        out['sources'][rel] = file_digests(p)[0] if p.is_file() else None
    return out


def binary_receipt(path):
    p = pathlib.Path(path)
    if not p.is_file():
        die('binary not found: %s' % path)
    sha, md5 = file_digests(p, md5=True)
    st = p.stat()
    return {'path': str(p.resolve()), 'sha256': sha, 'md5': md5, 'bytes': st.st_size, 'mtime': st.st_mtime}


def recorded_env(env):
    return {k: v for k, v in sorted(env.items())
            if k.startswith(('HIPFIRE_', 'ROCR_', 'HIP_', 'FABRIC_', 'HSA_')) or k == 'HOME'}


# ---------------------------------------------------------------- arms and environments
def parse_arms(items):
    arms = {}
    for item in items:
        if '=' not in item:
            die('--arm expects LABEL=VALUE (VALUE %s leaves the selector unset): %r' % (UNSET, item))
        label, value = item.split('=', 1)
        if not re.fullmatch(r'[A-Za-z0-9_.-]+', label):
            die('bad arm label %r' % label)
        if label in arms:
            die('duplicate arm label %r' % label)
        arms[label] = None if value == UNSET else value
    return arms


def selector_key(args):
    return args.env_key or ARCHES[args.arch]['env_key']


def child_env(home, cache, arm_key, arm_value, extra):
    env = os.environ.copy()
    for k in list(env):
        if k.startswith('HIPFIRE_') and k not in KEEP_ENV:
            env.pop(k)
    env.update(HOME=str(home), HIPFIRE_KERNEL_CACHE=str(cache), HIPFIRE_GRAPH='0',
               HIPFIRE_NORMALIZE_PROMPT='0')
    env.update(extra)
    env.pop(arm_key, None)
    if arm_value is not None:
        env[arm_key] = arm_value
    return env


def parse_kv_pairs(items, what):
    out = {}
    for item in items:
        if '=' not in item:
            die('%s expects KEY=VALUE: %r' % (what, item))
        k, v = item.split('=', 1)
        out[k] = v
    return out


def fresh_run_dir(run, mkhome):
    run.mkdir(parents=True, exist_ok=False)
    r = subprocess.run([mkhome, str(run / 'home')], stdout=subprocess.DEVNULL)
    if r.returncode != 0:
        die('mkhome failed (%s)' % mkhome)
    cache = run / 'kernel-cache'
    cache.mkdir()
    return run / 'home', cache


def cache_stats(cache):
    n = size = 0
    for root, _dirs, files in os.walk(cache):
        for f in files:
            n += 1
            try:
                size += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return {'files': n, 'bytes': size}


# ---------------------------------------------------------------- memory sampling and process runner
class MemSampler:
    """Peak of per-card VRAM/GTT use from sysfs (all cards; baseline taken before spawn)."""
    KEYS = ('mem_info_vram_used', 'mem_info_gtt_used')

    def __init__(self):
        self.cards = [d for d in sorted(glob.glob('/sys/class/drm/card*/device'))
                      if re.fullmatch(r'card\d+', d.split('/')[-2])]
        self.baseline = self.read()
        self.peak = dict(self.baseline)
        self.total = {}
        for d in self.cards:
            for key in ('mem_info_vram_total', 'mem_info_gtt_total'):
                v = self._get(d, key)
                if v is not None:
                    self.total['%s/%s' % (d.split('/')[-2], key)] = v

    @staticmethod
    def _get(d, key):
        try:
            with open(os.path.join(d, key)) as f:
                return int(f.read().strip())
        except (OSError, ValueError):
            return None

    def read(self):
        out = {}
        for d in self.cards:
            for key in self.KEYS:
                v = self._get(d, key)
                if v is not None:
                    out['%s/%s' % (d.split('/')[-2], key)] = v
        return out

    def sample(self):
        for k, v in self.read().items():
            if v > self.peak.get(k, -1):
                self.peak[k] = v

    def receipt(self):
        return {'baseline_bytes': self.baseline, 'peak_bytes': self.peak,
                'peak_minus_baseline_bytes': {k: self.peak[k] - self.baseline.get(k, 0) for k in self.peak},
                'total_bytes': self.total}


def run_monitored(cmd, env, cwd, log_path, timeout):
    mem = MemSampler()
    t0 = time.monotonic()
    timed_out = False
    with open(log_path, 'wb') as log:
        p = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT)
        CHILD[0] = p
        status = ru = None
        while True:
            pid, status, ru = os.wait4(p.pid, os.WNOHANG)
            mem.sample()
            if pid:
                break
            if STOP.is_set() or time.monotonic() - t0 > timeout:
                timed_out = not STOP.is_set()
                try:
                    os.kill(p.pid, signal.SIGKILL)
                except OSError:
                    pass
            time.sleep(0.25)
        code = os.waitstatus_to_exitcode(status)
        p.returncode = code
        CHILD[0] = None
    return {'exit_code': code, 'timed_out': timed_out, 'terminated': STOP.is_set(),
            'wall_seconds': time.monotonic() - t0, 'host_max_rss_kib': ru.ru_maxrss,
            'cpu_user_s': ru.ru_utime, 'cpu_sys_s': ru.ru_stime, 'device_memory': mem.receipt()}


def route_evidence(text, limit=60):
    lines = [ln.strip() for ln in text.splitlines() if EVIDENCE_RE.search(ln)]
    seen = list(dict.fromkeys(lines))
    return {'lines': seen[:limit], 'truncated': len(seen) > limit,
            'selector_rejected': any('unknown HIPFIRE_' in ln and 'A4_CANDIDATES' in ln for ln in seen)}


# ---------------------------------------------------------------- statistics
def geomean(xs):
    xs = list(xs)
    return math.exp(math.fsum(math.log(x) for x in xs) / len(xs))


def spread_pct(xs, center):
    return (max(xs) - min(xs)) / center * 100.0


def cv_pct(xs):
    return statistics.stdev(xs) / statistics.fmean(xs) * 100.0 if len(xs) > 1 else 0.0


# ---------------------------------------------------------------- quality
def parse_kldseq(path):
    data = pathlib.Path(path).read_bytes()
    magic, version, n, reserved = struct.unpack_from('<8sIII', data)
    if magic != b'HFKSEQ\0\0' or version != 2 or n != N_CHUNKS or len(data) != 20 + n * 24:
        raise ValueError('unexpected kldseq layout: magic=%r version=%d n=%d bytes=%d' % (magic, version, n, len(data)))
    rows = [struct.unpack_from('<ddd', data, 20 + 24 * i) for i in range(n)]
    return {
        'kld': math.fsum(r[0] for r in rows) / n,
        'p99_kld_mean': math.fsum(r[1] for r in rows) / n,
        'nll_mean': math.fsum(r[2] for r in rows) / n,
        'chunk_kld': [r[0] for r in rows],
        'chunk_p99': [r[1] for r in rows],
        'chunk_nll': [r[2] for r in rows],
        'kldseq_sha256': hashlib.sha256(data).hexdigest(),
        'kldseq_bytes': len(data),
    }


def cmd_quality(a):
    require_fabric()
    arms = parse_arms(a.arm)
    if len(arms) != 1:
        die('quality takes exactly one --arm LABEL=VALUE')
    (label, value), = arms.items()
    key = selector_key(a)
    arch = ARCHES[a.arch]
    kv_mode, kv_v = arch['quality_kv']
    refs = {c: list(v) for c, v in REFS.items()}
    for c, p in parse_kv_pairs(a.ref, '--ref').items():
        if c not in refs:
            die('unknown corpus %r' % c)
        refs[c][0] = p
    corpora = [c.strip() for c in a.corpora.split(',') if c.strip()]
    for c in corpora:
        if c not in refs:
            die('unknown corpus %r' % c)
    pins = dict(PINS.get(a.arch, {}))
    pins.update(parse_kv_pairs(a.pin, '--pin'))
    if a.validate_pin:
        for c in corpora:
            if c not in pins:
                die('--validate-pin: no pin known for %s/%s (use --pin %s=SHA256)' % (a.arch, c, c))
    model = pathlib.Path(a.model).resolve()
    if not model.is_file():
        die('model not found: %s' % model)
    out = pathlib.Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    model_sha, _ = file_digests(model)
    model_bytes = model.stat().st_size
    if not a.allow_noncanonical_model and (model_sha != CANON_MODEL_SHA256 or model_bytes != CANON_MODEL_BYTES):
        die('model is not the canonical fixture (sha256 %s, %d bytes)' % (model_sha, model_bytes))
    binrec = binary_receipt(a.bin)
    prov = provenance(a.repo, a.source)
    result = {
        'kind': 'a4-quality', 'arm': label, 'arch': a.arch, 'env_key': key,
        'env_value': value, 'kv_mode': kv_mode, 'kv_v': kv_v,
        'model': {'path': str(model), 'sha256': model_sha, 'bytes': model_bytes},
        'binary': binrec, 'provenance': prov,
        'fabric': {k: os.environ.get(k) for k in ('FABRIC_JOB', 'FABRIC_CARDS', 'HIPFIRE_FABRIC_ADMISSION',
                                                   'HIPFIRE_DEVICES', 'ROCR_VISIBLE_DEVICES')},
        'corpora': {}, 'failures': [],
    }
    for corpus in corpora:
        ref, refsha = refs[corpus]
        if file_digests(ref)[0] != refsha:
            die('reference hash mismatch: %s' % ref)
        run = out / ('%s.%s' % (label, corpus))
        home, cache = fresh_run_dir(run, a.mkhome)
        env = child_env(home, cache, key, value, parse_kv_pairs(a.env, '--env'))
        cmd = [binrec['path'], '--model', str(model), '--ref', ref, '--output', str(run / 'output.kldseq'),
               '--scoring-mode', 'prefill', '--max-chunks', str(N_CHUNKS), '--kv-mode', kv_mode, '--kv-v', kv_v]
        left = a.budget_secs - (time.monotonic() - T0)
        rec = {'corpus': corpus, 'reference': ref, 'reference_sha256': refsha, 'command': cmd,
               'environment': recorded_env(env), 'run_dir': str(run)}
        (run / 'command.json').write_text(json.dumps(rec, indent=2) + '\n')
        if left <= 30:
            rec['error'] = 'budget exhausted before start'
            result['failures'].append(corpus + ' budget exhausted')
        else:
            mon = run_monitored(cmd, env, a.cwd, run / 'eval.log', left)
            rec.update(mon)
            text = (run / 'eval.log').read_text(errors='replace')
            rec['route_evidence'] = route_evidence(text)
            rec['kernel_cache'] = cache_stats(cache)
            if rec['route_evidence']['selector_rejected']:
                result['failures'].append(corpus + ' selector value rejected by runtime')
            if mon['exit_code'] == 0:
                try:
                    rec.update(parse_kldseq(run / 'output.kldseq'))
                except ValueError as e:
                    result['failures'].append('%s %s' % (corpus, e))
                else:
                    pin = pins.get(corpus)
                    rec['pin_sha256'] = pin
                    rec['pin_match'] = (rec['kldseq_sha256'] == pin) if pin else None
                    if a.validate_pin and not rec['pin_match']:
                        result['failures'].append(corpus + ' pin mismatch')
            else:
                result['failures'].append('%s evaluator failed (exit %s)' % (corpus, mon['exit_code']))
        result['corpora'][corpus] = rec
        (run / 'result.json').write_text(json.dumps(rec, indent=2) + '\n')
        (out / ('quality-%s.json' % label)).write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps({k: rec.get(k) for k in ('corpus', 'kld', 'kldseq_sha256', 'pin_match', 'exit_code')}), flush=True)
        if STOP.is_set():
            result['failures'].append('terminated')
            break
    result['elapsed_seconds'] = time.monotonic() - T0
    (out / ('quality-%s.json' % label)).write_text(json.dumps(result, indent=2) + '\n')
    if result['failures']:
        die('; '.join(result['failures']))


def cmd_quality_compare(a):
    results = {}
    for p in a.results:
        r = json.loads(pathlib.Path(p).read_text())
        results[r['arm']] = r
    if a.control not in results:
        die('control %r not among result files' % a.control)
    ctrl = results[a.control]
    table = {}
    for label, r in results.items():
        row = {}
        for corpus, rec in r['corpora'].items():
            if 'kld' not in rec:
                row[corpus] = {'error': rec.get('error', 'no kld')}
                continue
            entry = {'kld': rec['kld'], 'kldseq_sha256': rec['kldseq_sha256'], 'pin_match': rec.get('pin_match')}
            c = ctrl['corpora'].get(corpus, {})
            if 'kld' in c:
                delta = rec['kld'] - c['kld']
                gain = WMM4_GAIN.get(corpus)
                entry.update(delta_vs_control=delta, bytes_equal_control=rec['kldseq_sha256'] == c['kldseq_sha256'],
                             within_cap=delta <= KLD_CAP)
                if gain:
                    entry['recovered_fraction_of_wmm4_gain'] = -delta / gain
            row[corpus] = entry
        table[label] = row
    out = {'control': a.control, 'wmm4_gain': WMM4_GAIN, 'one_sided_cap': KLD_CAP, 'arms': table}
    text = json.dumps(out, indent=2) + '\n'
    if a.out:
        pathlib.Path(a.out).write_text(text)
    sys.stdout.write(text)


# ---------------------------------------------------------------- timing
def build_schedule(order, lengths, blocks):
    """Palindromic ABBA blocks (A B B A for two arms; A B C C B A for three...). Every pair of arms
    is an ABBA subsequence inside each block. Block-major, lengths inside a block."""
    sched = []
    for b in range(1, blocks + 1):
        for length in lengths:
            seq = list(order) + list(reversed(order))
            for pos, arm in enumerate(seq):
                sched.append({'seq': len(sched) + 1, 'block': b, 'length': length, 'pos': pos, 'arm': arm})
    return sched


RUN_RE = re.compile(r'^\s*run\s+(\d+):\s+([0-9.]+)ms\s+([0-9.]+) tok/s\s*$', re.M)


def find_key(obj, key):
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for v in obj.values():
            r = find_key(v, key)
            if r is not None:
                return r
    elif isinstance(obj, list):
        for v in obj:
            r = find_key(v, key)
            if r is not None:
                return r
    return None


def parse_bench(text, atlas_path):
    """Per-run prefill samples come from the bench's stderr `run N: X ms` lines (the atlas row and
    PREFILL_SUMMARY only carry the LAST sample)."""
    runs = [(int(n), float(ms), float(tps)) for n, ms, tps in RUN_RE.findall(text)]

    def one(pat, cast=str):
        m = re.search(pat, text, re.M)
        return cast(m.group(1)) if m else None
    ev = {
        'gpu_arch': one(r'^GPU:\s*(\S+)'),
        'kv_mode': one(r'^KV mode:\s*(\S+)'),
        'phases': one(r'^Phases:\s*(.+)$'),
        'model_size_line': one(r'^Model size:\s*(.+)$'),
        'weights_loaded_s': one(r'Weights loaded in ([0-9.]+)s', float),
        'dpm_warmup_seconds': [float(x) for x in re.findall(r'warming caches\.\.\. took ([0-9.]+)s', text)],
        'dpm_warmup_headers': len(re.findall(r'=== DPM warmup', text)),
        'prefill_next_token': one(r'PREFILL_NEXT_TOKEN\s+token_id=(\d+)', int),
        'summary_tok_s_last': one(r'PREFILL_SUMMARY\s+prefill_tok_s=([0-9.]+)', float),
        'summary_wall_ms_last': one(r'PREFILL_SUMMARY\s+prefill_tok_s=[0-9.]+\s+prefill_wall_ms=([0-9.]+)', float),
    }
    atlas = None
    if atlas_path and pathlib.Path(atlas_path).is_file():
        rows = []
        for ln in pathlib.Path(atlas_path).read_text().splitlines():
            if ln.strip():
                try:
                    rows.append(json.loads(ln))
                except ValueError:
                    pass
        atlas = {'rows': len(rows), 'prefill_wall_ms': None, 'prefill_tok_s': None}
        for r in rows:
            w = find_key(r, 'prefill_wall_ms')
            if w is not None:
                atlas['prefill_wall_ms'] = w
                atlas['prefill_tok_s'] = find_key(r, 'prefill_tok_s')
                atlas['prefill_runs'] = find_key(r, 'prefill_runs')
                atlas['kv_mode'] = find_key(r, 'kv_mode')
                break
    return runs, ev, atlas


def cmd_plan(a):
    arms = parse_arms(a.arm)
    order = resolve_order(a, arms)
    lengths = parse_lengths(a.lengths)
    sched = build_schedule(order, lengths, a.blocks)
    print(json.dumps({'env_key': selector_key(a), 'arms': {k: arms[k] for k in order},
                      'processes': len(sched), 'processes_per_arm_per_length': 2 * a.blocks,
                      'schedule': sched}, indent=2))


def parse_lengths(s):
    try:
        lengths = [int(x) for x in s.split(',') if x.strip()]
    except ValueError:
        die('--lengths expects comma-separated integers')
    if not lengths or any(n <= 0 for n in lengths):
        die('--lengths must be positive integers')
    return lengths


def resolve_order(a, arms):
    if not arms:
        die('at least one --arm LABEL=VALUE is required')
    control = a.control or next(iter(arms))
    if control not in arms:
        die('control %r is not a mapped arm' % control)
    cands = [c.strip() for c in a.candidates.split(',') if c.strip()] if a.candidates else \
        [k for k in arms if k != control]
    for c in cands:
        if c not in arms:
            die('candidate %r is not a mapped arm' % c)
        if c == control:
            die('control %r listed among candidates' % c)
    if len(set(cands)) != len(cands) or not cands:
        die('need at least one distinct candidate arm')
    return [control] + cands


def summarize(rows, order, lengths):
    good = [r for r in rows if r.get('valid')]
    by = {}
    for r in good:
        by.setdefault((r['length'], r['arm']), []).append(r)
    arms_out = {}
    for (length, arm), rs in sorted(by.items()):
        rs.sort(key=lambda r: r['seq'])
        meds = [r['proc_median_tok_s'] for r in rs]
        g = geomean(meds)
        pooled = [t for r in rs for t in r['retained_tok_s']]
        arms_out.setdefault('pp%d' % length, {})[arm] = {
            'processes': len(rs), 'retained_samples': len(pooled),
            'proc_median_tok_s': meds, 'geomean_of_proc_medians_tok_s': g,
            'min_tok_s': min(meds), 'max_tok_s': max(meds),
            'spread_pct': spread_pct(meds, g), 'cv_pct': cv_pct(meds),
            'pooled_sample_median_tok_s': statistics.median(pooled),
            'discarded_first_ms': [r['discarded_first_ms'] for r in rs],
            'device_peak_minus_baseline_bytes': [r['device_memory']['peak_minus_baseline_bytes'] for r in rs],
            'host_max_rss_kib': [r['host_max_rss_kib'] for r in rs],
            'route_evidence_lines': sorted({ln for r in rs for ln in r['route_evidence']['lines']}),
        }
    pairs = []
    control = order[0]
    for length in lengths:
        crs = sorted(by.get((length, control), []), key=lambda r: r['seq'])
        for cand in order[1:]:
            xrs = sorted(by.get((length, cand), []), key=lambda r: r['seq'])
            blocks = sorted({r['block'] for r in crs} & {r['block'] for r in xrs})
            ratios, self_ratios, detail = [], [], []
            for b in blocks:
                cb = [r for r in crs if r['block'] == b]
                xb = [r for r in xrs if r['block'] == b]
                c = geomean(r['proc_median_tok_s'] for r in cb)
                x = geomean(r['proc_median_tok_s'] for r in xb)
                ratios.append(x / c)
                if len(cb) >= 2:
                    self_ratios.append(cb[-1]['proc_median_tok_s'] / cb[0]['proc_median_tok_s'])
                detail.append({'block': b, 'control_procs': len(cb), 'candidate_procs': len(xb),
                               'control_geomean_tok_s': c, 'candidate_geomean_tok_s': x, 'ratio': x / c})
            entry = {'length': length, 'control': control, 'candidate': cand, 'blocks': detail,
                     'control_procs': len(crs), 'candidate_procs': len(xrs)}
            if ratios:
                med = statistics.median(ratios)
                gm = geomean(ratios)
                noise = max([abs(s - 1) * 100 for s in self_ratios] + [(max(ratios) - min(ratios)) / 2 * 100])
                entry.update({
                    'block_ratios': ratios, 'geomean_ratio': gm, 'median_ratio': med,
                    'geomean_delta_pct': (gm - 1) * 100, 'median_delta_pct': (med - 1) * 100,
                    'median_slowdown_pct': (1 - med) * 100,
                    'block_ratio_spread_pct': spread_pct(ratios, gm),
                    'control_first_over_last_ratios': self_ratios,
                    'noise_pct': noise,
                    'delta_resolved_above_noise': abs(med - 1) * 100 > noise,
                    'within_1pct_median_slowdown': (1 - med) * 100 <= 1.0,
                    'abba_complete': all(d['control_procs'] >= 2 and d['candidate_procs'] >= 2 for d in detail),
                })
            pairs.append(entry)
    invalid = [{k: r.get(k) for k in ('seq', 'arm', 'length', 'problems', 'exit_code')} for r in rows if not r.get('valid')]
    return {'order': order, 'control': control, 'lengths': lengths, 'arms': arms_out,
            'pairs': pairs, 'invalid_processes': invalid, 'notes': [
                'per-process value = median of retained (non-first) prefill samples; arm value = geomean over process medians',
                'delta_resolved_above_noise=false means the delta is not distinguishable from drift/spread: do not call it a win or loss',
                'noise_pct = max(|control last/first - 1| over blocks, half block-ratio range)']}


def render(summary):
    lines = []
    for pp, arms in summary['arms'].items():
        lines.append(pp)
        for arm, s in arms.items():
            lines.append('  %-16s n=%d geomean=%.2f tok/s spread=%.2f%% cv=%.2f%%' % (
                arm, s['processes'], s['geomean_of_proc_medians_tok_s'], s['spread_pct'], s['cv_pct']))
    for p in summary['pairs']:
        if 'median_ratio' in p:
            lines.append('pp%d %s vs %s: median delta %+.3f%% geomean %+.3f%% noise %.3f%% resolved=%s' % (
                p['length'], p['candidate'], p['control'], p['median_delta_pct'], p['geomean_delta_pct'],
                p['noise_pct'], p['delta_resolved_above_noise']))
    return '\n'.join(lines)


def cmd_summarize(a):
    raw = pathlib.Path(a.raw)
    rows = [json.loads(ln) for ln in raw.read_text().splitlines() if ln.strip()]
    manifest = json.loads((raw.parent / 'manifest.json').read_text())
    s = summarize(rows, manifest['order'], manifest['lengths'])
    (raw.parent / 'summary.json').write_text(json.dumps(s, indent=2) + '\n')
    print(render(s))


def cmd_timing(a):
    require_fabric()
    arch = ARCHES[a.arch]
    key = selector_key(a)
    arms = parse_arms(a.arm)
    order = resolve_order(a, arms)
    lengths = parse_lengths(a.lengths)
    if a.prefill_runs < 3:
        die('--prefill-runs must be >= 3 (first sample is discarded, >=2 warm samples remain)')
    if a.blocks < 1:
        die('--blocks must be >= 1')
    sched = build_schedule(order, lengths, a.blocks)
    model = pathlib.Path(a.model).resolve()
    if not model.is_file():
        die('model not found: %s' % model)
    model_bytes = model.stat().st_size
    if not a.allow_noncanonical_model and model_bytes != CANON_MODEL_BYTES:
        die('model size %d != canonical %d' % (model_bytes, CANON_MODEL_BYTES))
    model_rec = {'path': str(model), 'bytes': model_bytes, 'expected_sha256': a.model_sha256,
                 'sha256_verified': None}
    if a.hash_model:
        model_rec['sha256'] = file_digests(model)[0]
        model_rec['sha256_verified'] = model_rec['sha256'] == a.model_sha256
        if not model_rec['sha256_verified']:
            die('model sha256 mismatch: %s' % model_rec['sha256'])
    if not a.allow_noncanonical_model and a.model_sha256 != CANON_MODEL_SHA256:
        die('--model-sha256 is not the canonical fixture digest')
    binrec = binary_receipt(a.bin)
    out = pathlib.Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    extra_env = parse_kv_pairs(a.env, '--env')
    if a.dpm_warmup_secs > 0:
        extra_env['HIPFIRE_DPM_WARMUP_SECS'] = str(a.dpm_warmup_secs)
    extra_env['HIPFIRE_KV_MODE'] = TIMING_KV[0]
    extra_env['HIPFIRE_KV_V'] = TIMING_KV[1]
    manifest = {
        'kind': 'a4-timing', 'arch': a.arch, 'expected_gpu_arch': arch['gfx'], 'env_key': key,
        'arms': {k: arms[k] for k in order}, 'order': order, 'control': order[0], 'lengths': lengths,
        'blocks': a.blocks, 'prefill_runs': a.prefill_runs, 'discard_first_samples': 1,
        'kv': {'mode': TIMING_KV[0], 'v': TIMING_KV[1]}, 'gen': 0, 'warmup_tokens': 0,
        'prompt': 'literal token ids 0..N-1 (bench_qwen35_mq4 fake prompt)',
        'budget_secs': a.budget_secs, 'extra_env': extra_env, 'model': model_rec, 'binary': binrec,
        'provenance': provenance(a.repo, a.source), 'schedule': sched,
        'fabric': {k: os.environ.get(k) for k in ('FABRIC_JOB', 'FABRIC_CARDS', 'HIPFIRE_FABRIC_ADMISSION',
                                                   'HIPFIRE_DEVICES', 'ROCR_VISIBLE_DEVICES')},
        'started_unix': time.time(),
    }
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    rows, status, durations = [], 'complete', {}
    raw = open(out / 'raw.jsonl', 'a')
    try:
        for step in sched:
            arm, length = step['arm'], step['length']
            left = a.budget_secs - (time.monotonic() - T0)
            known = durations.get(length) or (max(durations[k] for k in durations) if durations else 0)
            if STOP.is_set():
                status = 'terminated'
                break
            if left < known * 1.15 or left < 30:
                status = 'budget: stopped before seq %d (%.0fs left, need ~%.0fs)' % (step['seq'], left, known * 1.15)
                break
            tag = '%03d-b%d-pp%d-%s-p%d' % (step['seq'], step['block'], length, arm, step['pos'])
            run = out / 'proc' / tag
            run.parent.mkdir(exist_ok=True)
            home, cache = fresh_run_dir(run, a.mkhome)
            env = child_env(home, cache, key, arms[arm], extra_env)
            atlas = run / 'atlas.jsonl'
            cmd = [binrec['path'], str(model), '--prefill', str(length), '--prefill-runs', str(a.prefill_runs),
                   '--gen', '0', '--warmup', '0', '--emit-atlas', str(atlas)]
            (run / 'command.json').write_text(json.dumps(
                {'command': cmd, 'environment': recorded_env(env), 'step': step}, indent=2) + '\n')
            mon = run_monitored(cmd, env, a.cwd, run / 'bench.log', min(a.proc_timeout_secs, max(left, 1)))
            text = (run / 'bench.log').read_text(errors='replace')
            runs, ev, atl = parse_bench(text, atlas)
            problems = []
            if mon['exit_code'] != 0:
                problems.append('exit %s%s' % (mon['exit_code'], ' (timeout)' if mon['timed_out'] else ''))
            if [n for n, _, _ in runs] != list(range(1, a.prefill_runs + 1)):
                problems.append('expected %d run lines, got %d' % (a.prefill_runs, len(runs)))
            if ev['gpu_arch'] != arch['gfx']:
                problems.append('gpu arch %r != %s' % (ev['gpu_arch'], arch['gfx']))
            if ev['kv_mode'] != TIMING_KV[0]:
                problems.append('kv mode %r' % ev['kv_mode'])
            if ev['phases'] is None or 'prefill=%d ' % length not in ev['phases']:
                problems.append('phases line mismatch: %r' % ev['phases'])
            rev = route_evidence(text)
            if rev['selector_rejected']:
                problems.append('selector value rejected by runtime')
            if atl and atl.get('prefill_wall_ms') is not None and runs and abs(atl['prefill_wall_ms'] - runs[-1][1]) > 0.06:
                problems.append('atlas last sample %.3f ms != last run line %.1f ms' % (atl['prefill_wall_ms'], runs[-1][1]))
            samples = [ms for _, ms, _ in runs]
            retained = samples[1:]
            tok_s = [length / (ms / 1000.0) for ms in retained]
            row = {
                'seq': step['seq'], 'block': step['block'], 'pos': step['pos'], 'length': length, 'arm': arm,
                'env_key': key, 'env_value': arms[arm], 'run_dir': str(run), 'valid': not problems,
                'problems': problems, 'prefill_runs': a.prefill_runs,
                'samples_ms': samples, 'discarded_first_ms': samples[0] if samples else None,
                'retained_ms': retained, 'retained_tok_s': tok_s,
                'proc_median_tok_s': statistics.median(tok_s) if tok_s else None,
                'bench': ev, 'atlas': atl, 'route_evidence': rev, 'kernel_cache': cache_stats(cache),
                'start_offset_s': time.monotonic() - T0 - mon['wall_seconds'], **mon,
            }
            raw.write(json.dumps(row) + '\n')
            raw.flush()
            rows.append(row)
            durations[length] = max(durations.get(length, 0), mon['wall_seconds'])
            print(json.dumps({'seq': step['seq'], 'arm': arm, 'pp': length, 'valid': row['valid'],
                              'tok_s': row['proc_median_tok_s'], 'wall_s': round(mon['wall_seconds'], 1),
                              'problems': problems}), flush=True)
            if problems and not a.keep_going:
                status = 'invalid process at seq %d: %s' % (step['seq'], '; '.join(problems))
                break
    finally:
        raw.close()
        summary = summarize(rows, order, lengths)
        summary['status'] = status
        summary['complete'] = status == 'complete'
        summary['processes_run'] = len(rows)
        summary['processes_planned'] = len(sched)
        summary['elapsed_seconds'] = time.monotonic() - T0
        (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        print(render(summary))
        print('status: ' + status)
    if status != 'complete':
        raise SystemExit(3)


# ---------------------------------------------------------------- CLI
def common(p):
    p.add_argument('--arch', choices=sorted(ARCHES), default='r9700',
                   help='selects the selector env key (r9700: HIPFIRE_G12_A4_CANDIDATES, halo: HIPFIRE_GFX11_A4_CANDIDATES) and expected gfx arch')
    p.add_argument('--env-key', default=None, help='override the selector env key chosen by --arch')
    p.add_argument('--arm', action='append', default=[], metavar='LABEL=VALUE',
                   help='arm label to selector value; VALUE %s leaves the selector unset (repeatable)' % UNSET)


def paths(p, binary_name):
    p.add_argument('--bin', required=True, help='%s binary' % binary_name)
    p.add_argument('--out', required=True, help='new output directory')
    p.add_argument('--model', required=True, help='canonical Qwen3.8-27B MQ4-XTS .hfq path')
    p.add_argument('--repo', default=str(REPO), help='source tree for provenance hashes (default: this worktree)')
    p.add_argument('--source', action='append', default=[], help='extra repo-relative source file to hash (repeatable)')
    p.add_argument('--mkhome', default=os.environ.get('A4_MKHOME', '/home/kaden/qcal/tools/mkhome.sh'),
                   help='fresh-HOME creator (HOME/.hipfire/models -> canonical store)')
    p.add_argument('--cwd', default=str(REPO), help='working directory of the measured process')
    p.add_argument('--env', action='append', default=[], metavar='KEY=VALUE',
                   help='extra child environment applied to every arm (recorded)')
    p.add_argument('--allow-noncanonical-model', action='store_true')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)

    q = sub.add_parser('quality', help='one arm, both corpora, fresh HOME/cache per corpus')
    common(q)
    paths(q, 'eval_hipfire')
    q.add_argument('--corpora', default='wt2,code24')
    q.add_argument('--ref', action='append', default=[], metavar='CORPUS=PATH')
    q.add_argument('--pin', action='append', default=[], metavar='CORPUS=SHA256', help='override/add a .kldseq pin')
    q.add_argument('--validate-pin', action='store_true', help='fail unless each .kldseq matches its pin (use for the control arms)')
    q.add_argument('--budget-secs', type=float, default=570.0, help='hard budget inside the <=600 s correct hold')
    q.set_defaults(fn=cmd_quality)

    t = sub.add_parser('timing', help='ABBA prefill timing of all selected arms in one hold')
    common(t)
    paths(t, 'bench_qwen35_mq4 (built --features deltanet)')
    t.add_argument('--model-sha256', default=CANON_MODEL_SHA256,
                   help='digest verified outside the hold (scripts/check_fixture.sh); recorded as expected, not recomputed unless --hash-model')
    t.add_argument('--hash-model', action='store_true', help='recompute the model sha256 inside the hold (~1 min)')
    t.add_argument('--control', default=None, help='control arm label (default: first --arm)')
    t.add_argument('--candidates', default=None, help='comma-separated candidate arm labels (default: all other --arm)')
    t.add_argument('--lengths', default='512,8192')
    t.add_argument('--blocks', type=int, default=2, help='palindromic ABBA blocks (2 => 4 processes per arm per length)')
    t.add_argument('--prefill-runs', type=int, default=3, help='prefill samples per process; the first is discarded')
    t.add_argument('--dpm-warmup-secs', type=float, default=3.0, help='HIPFIRE_DPM_WARMUP_SECS before prefill (0 disables)')
    t.add_argument('--budget-secs', type=float, default=1740.0, help='stop launching processes that cannot finish inside the 1800 s hold')
    t.add_argument('--proc-timeout-secs', type=float, default=600.0)
    t.add_argument('--keep-going', action='store_true', help='continue after an invalid process (default: stop)')
    t.set_defaults(fn=cmd_timing)

    pl = sub.add_parser('plan', help='print the ABBA schedule')
    common(pl)
    pl.add_argument('--control', default=None)
    pl.add_argument('--candidates', default=None)
    pl.add_argument('--lengths', default='512,8192')
    pl.add_argument('--blocks', type=int, default=2)
    pl.set_defaults(fn=cmd_plan)

    s = sub.add_parser('summarize', help='recompute summary.json from timing raw.jsonl')
    s.add_argument('raw')
    s.set_defaults(fn=cmd_summarize)

    qc = sub.add_parser('quality-compare', help='KLD deltas, recovered WMM4 fraction, +0.010 cap')
    qc.add_argument('results', nargs='+', help='quality-<arm>.json files')
    qc.add_argument('--control', required=True, help='arm label of the control result')
    qc.add_argument('--out', default=None)
    qc.set_defaults(fn=cmd_quality_compare)

    a = ap.parse_args()
    a.fn(a)


if __name__ == '__main__':
    main()
