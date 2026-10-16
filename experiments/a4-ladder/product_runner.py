#!/usr/bin/env python3
"""A4 fused-ladder PRODUCT timing: `hipfire bench ... --json` per fresh process.

Measures what the shipped CLI measures, not `bench_qwen35_mq4` (whose timing in runner.py is stale):

  hipfire bench MODEL --backend noslots --workload stateless --prompt-file FILE \
         --max-tokens 1 --runs 6 --json

Each invocation is a fresh CLI that spawns a fresh stdio daemon (HIPFIRE_DAEMON_BIN pinned), with a
fresh HOME (mkhome.sh: models symlink to the canonical store), an empty HIPFIRE_KERNEL_CACHE, and a
scrubbed environment. Product defaults are untouched: no HIPFIRE_KV*, GRAPH*, NORMALIZE*, no --spec,
no DPM override (add one only via an explicit, recorded `--env`). The only varied knob is
HIPFIRE_G12_A4_CANDIDATES (unset == default C2, or 4 / 8).

Modes
  baseline  unset-C2 only, one process per length (512 and 8192); parent approves before ABBA
  abba      per candidate (4, 8) and length: --blocks palindromic ABBA blocks (A = unset C2,
            B = candidate) = 4 fresh processes per arm per comparison length at --blocks 2
  summary   recompute summary.json from an existing <out>/raw.jsonl + manifest.json

baseline/abba refuse to run unless FABRIC_JOB is set (launched by fabric-run.sh inside the
already-admitted hold). Metrics (formulas in summary.json `notes`):
  per-process value  = median of the LAST 5 `samples.prefill` (sample 0 discarded: the Hello/16
                       warmup does not JIT the long-prompt shapes); prefill_tok_s is the daemon's
                       prefill_tokens / prefill_s per run
  arm value          = geomean of its process medians
  block ratio        = geomean(candidate proc medians) / geomean(control proc medians) in a block
  reported           = block ratios, geomean + median of block ratios, spread; NOT significance
"""
import argparse
import hashlib
import json
import math
import os
import pathlib
import re
import signal
import statistics
import subprocess
import sys
import time

T0 = time.monotonic()
HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]

CANON_MODEL_SHA256 = '3e38ccbae3776470eb5a89344d300e9279d6b9ab6c31fd40ca1758c4f7c6f8ae'
CANON_MODEL_BYTES = 14987185152
CANON_STORE = '/home/kaden/.hipfire/models'
# parent-verified exact fixtures (product/prompt-fixtures.json): 500 / 8180 hello words; the closed-think
# template adds 12 tokens, so the daemon must report exactly 512 / 8192 prompt tokens.
PROMPT_PINS = {
    512: '735624a52c6282551665e9d91e4eac4ff73860d99939faa20dffa2cfcb4cb018',
    8192: '7996a13c9a0416ad6948ae2aa9133a1c17faed867fda35390eb5949d587daa9a',
}
SELECTOR = 'HIPFIRE_G12_A4_CANDIDATES'
ARMS = {'c2': None, 'c4': '4', 'c8': '8'}  # label -> selector value (None = unset = shipped default C2)
KEEP_HIPFIRE = {'HIPFIRE_DEVICES', 'HIPFIRE_LOCK_DIR', 'HIPFIRE_FABRIC_ADMISSION'}
KEEP_OTHER = {'HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES', 'HIP_PATH'}
SCRUB_PREFIXES = ('HIPFIRE_', 'HIP_', 'HSA_', 'ROCR_', 'GPU_', 'AMD_', 'ROCBLAS_', 'MIOPEN_')
FORBIDDEN_ENV = ('HIPFIRE_KV', 'HIPFIRE_GRAPH', 'HIPFIRE_NORMALIZE', SELECTOR)
BASELINE_PP8192 = (5100.0, 4590.0, 5610.0)  # center, lo, hi (+-10%) operational guard, tok/s
EVIDENCE_RE = re.compile(r'A4_CANDIDATES|unknown HIPFIRE_|falling back|fallback|WARN|warning:|cache hit|cached', re.I)
SAMPLES = 6

STOP = False
CHILD_PGID = [None]


def _on_term(_signum, _frame):
    global STOP
    STOP = True
    if CHILD_PGID[0] is not None:
        try:
            os.killpg(CHILD_PGID[0], signal.SIGKILL)  # only the group we created
        except OSError:
            pass


signal.signal(signal.SIGTERM, _on_term)
signal.signal(signal.SIGINT, _on_term)


def die(msg):
    raise SystemExit('product_runner: ' + msg)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(16 << 20), b''):
            h.update(b)
    return h.hexdigest()


def git(repo, *args):
    try:
        r = subprocess.run(['git', '-C', str(repo), *args], capture_output=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return r.stdout if r.returncode == 0 else None


def geomean(xs):
    xs = list(xs)
    return math.exp(math.fsum(math.log(x) for x in xs) / len(xs))


def spread_pct(xs, center):
    return (max(xs) - min(xs)) / center * 100.0


def cv_pct(xs):
    return statistics.stdev(xs) / statistics.fmean(xs) * 100.0 if len(xs) > 1 else 0.0


# ---------------------------------------------------------------- environment / process
def scrubbed_env(home, cache, daemon, arm_value, extra):
    env, scrubbed = {}, []
    for k, v in os.environ.items():
        if k.startswith(SCRUB_PREFIXES) and k not in KEEP_HIPFIRE and k not in KEEP_OTHER:
            scrubbed.append(k)
        else:
            env[k] = v
    env.update(HOME=str(home), HIPFIRE_KERNEL_CACHE=str(cache), HIPFIRE_DAEMON_BIN=str(daemon))
    env.update(extra)
    if arm_value is not None:
        env[SELECTOR] = arm_value
    return env, sorted(scrubbed)


def recorded_env(env):
    return {k: v for k, v in sorted(env.items()) if k.startswith(SCRUB_PREFIXES + ('FABRIC_', 'ROCM_')) or k == 'HOME'}


def fresh_run_dir(run, mkhome):
    run.mkdir(parents=True, exist_ok=False)
    r = subprocess.run([mkhome, str(run / 'home')], stdout=subprocess.DEVNULL)
    if r.returncode != 0:
        die('mkhome failed (%s)' % mkhome)
    cache = run / 'kernel-cache'
    cache.mkdir()
    return run / 'home', cache


def live_group_members(pgid):
    """pids in process group pgid that are not zombies (the unreaped CLI leader is a zombie)."""
    out = []
    for d in os.listdir('/proc'):
        if d.isdigit():
            try:
                with open('/proc/%s/stat' % d) as f:
                    rest = f.read().rsplit(')', 1)[1].split()
            except (OSError, IndexError):
                continue
            if rest[2] == str(pgid) and rest[0] != 'Z':
                out.append(int(d))
    return out


def run_group(cmd, env, cwd, out_path, err_path, timeout):
    """Run cmd (CLI) in its own session so CLI + its daemon form one process group we can kill.
    stdout/stderr go straight to files (raw). Never pkill; only killpg of the group we created."""
    t0 = time.monotonic()
    timed_out = False
    with open(out_path, 'wb') as so, open(err_path, 'wb') as se:
        p = subprocess.Popen(cmd, cwd=cwd, env=env, stdin=subprocess.DEVNULL, stdout=so, stderr=se,
                             start_new_session=True)
        pgid = p.pid
        CHILD_PGID[0] = pgid
        while True:
            # detect exit WITHOUT reaping so the pgid cannot be recycled before the sweep below
            if os.waitid(os.P_PID, pgid, os.WEXITED | os.WNOHANG | os.WNOWAIT) is not None:
                break
            if STOP or time.monotonic() - t0 > timeout:
                timed_out = not STOP
                try:
                    os.killpg(pgid, signal.SIGKILL)
                except OSError:
                    pass
                os.waitid(os.P_PID, pgid, os.WEXITED | os.WNOWAIT)
                break
            time.sleep(0.2)
        # CLI's EngineDrop kills its daemon; any live (non-zombie) group member now is a leak
        stray = bool(live_group_members(pgid))
        try:
            os.killpg(pgid, signal.SIGKILL)
        except OSError:
            pass
        _, status, ru = os.wait4(pgid, 0)
        CHILD_PGID[0] = None
    return {'exit_code': os.waitstatus_to_exitcode(status), 'timed_out': timed_out, 'terminated': STOP,
            'stray_group_members_killed': stray, 'wall_seconds': time.monotonic() - t0,
            'host_max_rss_kib': ru.ru_maxrss, 'cpu_user_s': ru.ru_utime, 'cpu_sys_s': ru.ru_stime}


# ---------------------------------------------------------------- validation of one process
def evaluate(a, length, rep, stderr, mon, arm):
    """Return (problems, derived). rep may be None."""
    problems, d = [], {}
    if mon['exit_code'] != 0:
        problems.append('exit %s%s' % (mon['exit_code'], ' (timeout)' if mon['timed_out'] else ''))
    if mon['stray_group_members_killed']:
        problems.append('processes still alive in the CLI process group after exit (killed)')
    if rep is None:
        problems.append('stdout is not a single JSON document')
        return problems, d
    if rep.get('protocol') != 'native-generate-v1':
        problems.append('protocol %r != native-generate-v1' % rep.get('protocol'))
    if rep.get('runs') != SAMPLES or rep.get('max_tokens') != 1:
        problems.append('runs/max_tokens %r/%r != %d/1' % (rep.get('runs'), rep.get('max_tokens'), SAMPLES))
    if rep.get('prompt_tokens') != a.expect_tokens[length]:
        problems.append('prompt_tokens %r != expected %d' % (rep.get('prompt_tokens'), a.expect_tokens[length]))
    gfx = (rep.get('gpu') or {}).get('arch')
    if gfx != a.expect_gfx:
        problems.append('gpu.arch %r != %s' % (gfx, a.expect_gfx))
    kv = rep.get('kv_mode')
    if kv != a.expect_kv_mode:
        problems.append('effective kv_mode %r != %s' % (kv, a.expect_kv_mode))
    if rep.get('kv_backend_warning'):
        problems.append('kv_backend_warning: %r' % (rep.get('kv_backend_warning'),))
    samples = (rep.get('samples') or {})
    pre, ttft = samples.get('prefill') or [], samples.get('ttft_ms') or []
    if len(pre) != SAMPLES:
        problems.append('samples.prefill has %d entries, expected %d' % (len(pre), SAMPLES))
        return problems, d
    # Fresh-prefill evidence. The JSON has no per-sample cached_tokens, but prefill_tok_s * ttft_ms is
    # the rows actually prefilled in that sample (prefill_tokens/prefill_s * prefill_s); a prompt-cache
    # hit would make it smaller than the full prompt.
    need = a.expect_tokens[length]
    implied = [t * m / 1000.0 for t, m in zip(pre, ttft)] if len(ttft) == len(pre) else []
    d['implied_prefilled_tokens'] = implied
    if not implied:
        problems.append('samples.ttft_ms missing: cannot prove fresh prefill per sample')
    elif any(x < 0.99 * need for x in implied):
        problems.append('cached prefix suspected: implied prefilled tokens %r < prompt_tokens %d' % (implied, need))
    elif any(x > 1.01 * need for x in implied):
        d['ttft_exceeds_prefill_note'] = 'ttft_ms > prefill time on some samples (implied tokens above prompt_tokens)'
    retained = pre[1:]
    d.update(discarded_first_tok_s=pre[0], retained_tok_s=retained, proc_median_tok_s=statistics.median(retained),
             retained_min=min(retained), retained_max=max(retained))
    if any(not (x > 0) for x in pre):
        problems.append('non-positive prefill sample')
    if re.search(r'unknown HIPFIRE_', stderr):
        problems.append('runtime rejected an environment selector')
    return problems, d


# ---------------------------------------------------------------- statistics
def summarize(rows, manifest):
    good = sorted((r for r in rows if r['valid']), key=lambda r: r['seq'])
    groups = {}
    for r in good:
        groups.setdefault((r['length'], r['comparison'], r['arm']), []).append(r)
    arms = []
    for (length, comp, arm), rs in sorted(groups.items()):
        meds = [r['proc_median_tok_s'] for r in rs]
        g = geomean(meds)
        arms.append({'length': length, 'comparison': comp, 'arm': arm, 'processes': len(rs),
                     'proc_median_tok_s': meds, 'geomean_of_proc_medians_tok_s': g,
                     'min_tok_s': min(meds), 'max_tok_s': max(meds), 'spread_pct': spread_pct(meds, g),
                     'cv_pct': cv_pct(meds), 'discarded_first_tok_s': [r['discarded_first_tok_s'] for r in rs],
                     'effective': sorted({json.dumps(r['effective'], sort_keys=True) for r in rs})})
    pairs = []
    for length in manifest['lengths']:
        for comp in manifest.get('candidates', []):
            cand = 'c' + comp
            crs = [r for r in good if (r['length'], r['comparison'], r['arm']) == (length, comp, 'c2')]
            xrs = [r for r in good if (r['length'], r['comparison'], r['arm']) == (length, comp, cand)]
            blocks = sorted({r['block'] for r in crs} & {r['block'] for r in xrs})
            detail, ratios, selfr = [], [], []
            for b in blocks:
                cb = [r for r in crs if r['block'] == b]
                xb = [r for r in xrs if r['block'] == b]
                c = geomean(r['proc_median_tok_s'] for r in cb)
                x = geomean(r['proc_median_tok_s'] for r in xb)
                ratios.append(x / c)
                if len(cb) >= 2:
                    selfr.append(cb[-1]['proc_median_tok_s'] / cb[0]['proc_median_tok_s'])
                detail.append({'block': b, 'control_procs': len(cb), 'candidate_procs': len(xb),
                               'control_geomean_tok_s': c, 'candidate_geomean_tok_s': x, 'ratio': x / c})
            e = {'length': length, 'control': 'c2', 'candidate': cand, 'blocks': detail,
                 'control_procs': len(crs), 'candidate_procs': len(xrs)}
            if ratios:
                gm = geomean(ratios)
                e.update({'block_ratios': ratios, 'geomean_ratio': gm, 'median_ratio': statistics.median(ratios),
                          'geomean_delta_pct': (gm - 1) * 100, 'block_ratio_spread_pct': spread_pct(ratios, gm),
                          'control_last_over_first_ratios': selfr,
                          'abba_complete': len(blocks) >= manifest['blocks'] and all(
                              d['control_procs'] >= 2 and d['candidate_procs'] >= 2 for d in detail)})
            pairs.append(e)
    out = {'mode': manifest['mode'], 'lengths': manifest['lengths'], 'arms': arms, 'pairs': pairs,
           'invalid_processes': [{k: r.get(k) for k in ('seq', 'arm', 'length', 'comparison', 'problems', 'exit_code')}
                                 for r in rows if not r['valid']],
           'notes': [
               'per-process value = median of the last 5 samples.prefill (sample 0 discarded); tok/s as reported by the daemon (prefill_tokens / prefill_s)',
               'arm value = geomean of process medians; block ratio = geomean(candidate proc medians) / geomean(control proc medians)',
               'paired ratios, geomean/median and spread are descriptive; no statistical-significance claim']}
    if manifest['mode'] == 'baseline':
        g = [r for r in good if r['length'] == 8192 and r['arm'] == 'c2']
        if g:
            v = statistics.median(r['proc_median_tok_s'] for r in g)
            c, lo, hi = BASELINE_PP8192
            out['baseline_guard_pp8192'] = {'median_tok_s': v, 'expected_center': c, 'lo': lo, 'hi': hi,
                                            'within': lo <= v <= hi,
                                            'action': None if lo <= v <= hi else 'WARNING: outside +-10% of ~5100; parent must explain before ABBA (no automatic substitution)'}
    return out


def render(s):
    lines = []
    for x in s['arms']:
        lines.append('pp%-5d %-8s %-3s n=%d geomean=%.1f tok/s spread=%.2f%% cv=%.2f%% procs=%s' % (
            x['length'], x['comparison'], x['arm'], x['processes'], x['geomean_of_proc_medians_tok_s'],
            x['spread_pct'], x['cv_pct'], ['%.1f' % m for m in x['proc_median_tok_s']]))
    for p in s['pairs']:
        if 'geomean_ratio' in p:
            lines.append('pp%d %s vs c2: block ratios %s geomean %+.3f%% median %+.3f%% spread %.3f%% abba_complete=%s' % (
                p['length'], p['candidate'], ['%.4f' % r for r in p['block_ratios']], p['geomean_delta_pct'],
                (p['median_ratio'] - 1) * 100, p['block_ratio_spread_pct'], p['abba_complete']))
    if 'baseline_guard_pp8192' in s:
        lines.append('baseline guard pp8192: %s' % json.dumps(s['baseline_guard_pp8192']))
    return '\n'.join(lines)


# ---------------------------------------------------------------- run
def build_schedule(mode, candidates, lengths, blocks):
    if mode == 'baseline':
        return [{'seq': i, 'comparison': 'baseline', 'block': 1, 'pos': 0, 'length': L, 'arm': 'c2'}
                for i, L in enumerate(lengths)]
    sched = []
    for block in range(1, blocks + 1):  # block-major: every comparison has block 1 before any has block 2
        for L in lengths:
            for c in candidates:
                for pos, arm in enumerate(('c2', 'c' + c, 'c' + c, 'c2')):
                    sched.append({'seq': len(sched), 'comparison': c, 'block': block, 'pos': pos,
                                  'length': L, 'arm': arm})
    return sched


def resolve_model_file(model):
    p = pathlib.Path(model)
    if p.is_file():
        return p.resolve()
    q = pathlib.Path(CANON_STORE) / model
    if q.is_file():
        return q.resolve()
    die('model %r is neither a file nor in %s' % (model, CANON_STORE))


def cmd_run(a):
    if not os.environ.get('FABRIC_JOB'):
        die('run only inside fabric-run.sh (FABRIC_JOB is not set)')
    mode = a.cmd
    lengths = [512, 8192]
    candidates = [c.strip() for c in a.candidates.split(',') if c.strip()] if mode == 'abba' else []
    if any(c not in ('4', '8') for c in candidates) or (mode == 'abba' and not candidates):
        die('--candidates must be a subset of 4,8')
    if mode == 'abba' and a.blocks < 1:
        die('--blocks must be >= 1')
    a.expect_tokens = {512: a.tokens512, 8192: a.tokens8192}
    extra_env = {}
    for item in a.env:
        if '=' not in item:
            die('--env expects KEY=VALUE: %r' % item)
        k, v = item.split('=', 1)
        if k.startswith(FORBIDDEN_ENV):
            die('--env %s would change a product default or the studied selector' % k)
        extra_env[k] = v
    prompts = {512: pathlib.Path(a.prompt512).resolve(), 8192: pathlib.Path(a.prompt8192).resolve()}
    prompt_rec = {}
    for L, p in prompts.items():
        if not p.is_file():
            die('prompt not found: %s' % p)
        raw = p.read_bytes()
        sha = hashlib.sha256(raw).hexdigest()
        want = {512: a.prompt512_sha256, 8192: a.prompt8192_sha256}[L] or PROMPT_PINS[L]
        if sha != want:
            die('prompt %d sha256 %s != pinned %s' % (L, sha, want))
        prompt_rec[str(L)] = {'path': str(p), 'bytes': len(raw), 'sha256': sha, 'md5': hashlib.md5(raw).hexdigest(),
                              'chars': len(raw.decode('utf-8')), 'expected_prompt_tokens': a.expect_tokens[L]}
    mfile = resolve_model_file(a.model)
    msize = mfile.stat().st_size
    if not a.allow_noncanonical_model and (msize != CANON_MODEL_BYTES or a.model_sha256 != CANON_MODEL_SHA256):
        die('model size %d / sha pin not canonical (%d / %s)' % (msize, CANON_MODEL_BYTES, CANON_MODEL_SHA256))
    model_rec = {'arg': a.model, 'file': str(mfile), 'bytes': msize, 'expected_sha256': a.model_sha256,
                 'sha256_verified': None}
    if a.hash_model:
        model_rec['sha256'] = sha256(mfile)
        model_rec['sha256_verified'] = model_rec['sha256'] == a.model_sha256
        if not model_rec['sha256_verified']:
            die('model sha256 mismatch: %s' % model_rec['sha256'])
    bins = {}
    for name, path, want in (('cli', a.cli, a.expect_cli_sha256), ('daemon', a.daemon, a.expect_daemon_sha256)):
        p = pathlib.Path(path)
        if not p.is_file():
            die('%s binary not found: %s' % (name, path))
        h = sha256(p)
        if want and h != want:
            die('%s sha256 %s != expected %s' % (name, h, want))
        bins[name] = {'path': str(p.resolve()), 'sha256': h, 'bytes': p.stat().st_size, 'mtime': p.stat().st_mtime}
    repo = pathlib.Path(a.repo).resolve()
    head, status, diff = git(repo, 'rev-parse', 'HEAD'), git(repo, 'status', '--porcelain'), git(repo, 'diff', 'HEAD')
    head = head.decode().strip() if head else None
    if a.expect_source and (not head or not head.startswith(a.expect_source)):
        die('source HEAD %s != expected %s' % (head, a.expect_source))
    prov = {'repo': str(repo), 'git_head': head,
            'git_status_porcelain': status.decode().splitlines() if status is not None else None,
            'git_diff_head_sha256': hashlib.sha256(diff).hexdigest() if diff is not None else None,
            'runner_sha256': sha256(__file__)}
    sched = build_schedule(mode, candidates, lengths, a.blocks if mode == 'abba' else 1)
    out = pathlib.Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    manifest = {
        'kind': 'a4-product', 'mode': mode, 'lengths': lengths, 'candidates': candidates,
        'blocks': a.blocks if mode == 'abba' else 1, 'selector': SELECTOR, 'arms': ARMS, 'samples_per_process': SAMPLES,
        'discard_first_samples': 1,
        'command_template': [bins['cli']['path'], 'bench', a.model, '--backend', 'noslots', '--workload', 'stateless',
                             '--prompt-file', '<PROMPT>', '--max-tokens', '1', '--runs', str(SAMPLES), '--json'],
        'note_backend_workload': '--backend/--workload are recorded flags only; the standard (non---concurrency) bench is inherently a sequential fresh-request loop, enforcement is not claimed',
        'expect': {'gfx': a.expect_gfx, 'kv_mode': a.expect_kv_mode, 'tokens': a.expect_tokens, 'protocol': 'native-generate-v1'},
        'extra_env': extra_env, 'model': model_rec, 'prompts': prompt_rec, 'binaries': bins, 'provenance': prov,
        'schedule': sched, 'budget_secs': a.budget_secs, 'proc_timeout_secs': a.proc_timeout_secs,
        'fabric': {k: os.environ.get(k) for k in ('FABRIC_JOB', 'FABRIC_CARDS', 'HIPFIRE_FABRIC_ADMISSION',
                                                   'HIPFIRE_DEVICES', 'ROCR_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES')},
        'started_unix': time.time()}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    rows, status, durations = [], 'complete', {}
    raw = open(out / 'raw.jsonl', 'a')
    try:
        for step in sched:
            arm, length = step['arm'], step['length']
            left = a.budget_secs - (time.monotonic() - T0)
            known = durations.get(length) or (max(durations.values()) if durations else 0)
            if STOP:
                status = 'terminated'
                break
            if left < known * 1.15 or left < 30:
                status = 'budget: stopped before seq %d (%.0fs left, need ~%.0fs)' % (step['seq'], left, known * 1.15)
                break
            tag = '%03d-%s-b%d-pp%d-%s-p%d' % (step['seq'], step['comparison'], step['block'], length, arm, step['pos'])
            run = out / 'proc' / tag
            run.parent.mkdir(exist_ok=True)
            home, cache = fresh_run_dir(run, a.mkhome)
            env, scrubbed = scrubbed_env(home, cache, bins['daemon']['path'], ARMS[arm], extra_env)
            cmd = [bins['cli']['path'], 'bench', a.model, '--backend', 'noslots', '--workload', 'stateless',
                   '--prompt-file', str(prompts[length]), '--max-tokens', '1', '--runs', str(SAMPLES), '--json']
            (run / 'command.json').write_text(json.dumps(
                {'command': cmd, 'cwd': a.cwd, 'environment': recorded_env(env), 'scrubbed_env_names': scrubbed,
                 'step': step}, indent=2) + '\n')
            mon = run_group(cmd, env, a.cwd, run / 'stdout.json', run / 'stderr.txt',
                            min(a.proc_timeout_secs, max(left, 1)))
            stdout = (run / 'stdout.json').read_bytes()
            stderr = (run / 'stderr.txt').read_text(errors='replace')
            try:
                rep = json.loads(stdout)
            except ValueError:
                rep = None
            problems, d = evaluate(a, length, rep, stderr, mon, arm)
            if rep is not None and rep.get('prompt_md5') != prompt_rec[str(length)]['md5']:
                problems.append('prompt_md5 %r != file md5 %s' % (rep.get('prompt_md5'), prompt_rec[str(length)]['md5']))
            effective = {k: (rep or {}).get(k) for k in ('kv_mode', 'kv_backend', 'kv_backend_request', 'kv_backend_reason',
                                                         'kv_backend_legacy', 'kv_backend_warning')}
            effective['gpu_arch'] = ((rep or {}).get('gpu') or {}).get('arch')
            effective['gpu_kv_mode'] = ((rep or {}).get('gpu') or {}).get('kv_mode')
            effective['spec_requested'] = (rep or {}).get('spec_requested')
            effective['load_speculation'] = (rep or {}).get('load_speculation')
            effective['spec_route'] = (rep or {}).get('spec_route')
            ev = sorted({ln.strip() for ln in stderr.splitlines() if EVIDENCE_RE.search(ln)})[:60]
            row = {'seq': step['seq'], 'comparison': step['comparison'], 'block': step['block'], 'pos': step['pos'],
                   'length': length, 'arm': arm, 'env_value': ARMS[arm], 'run_dir': str(run), 'valid': not problems,
                   'problems': problems, 'effective': effective, 'prompt_tokens': (rep or {}).get('prompt_tokens'),
                   'warnings': (rep or {}).get('warnings'), 'stderr_evidence_lines': ev,
                   'start_offset_s': time.monotonic() - T0 - mon['wall_seconds'], 'bench_json': rep, **d, **mon}
            raw.write(json.dumps(row) + '\n')
            raw.flush()
            rows.append(row)
            durations[length] = max(durations.get(length, 0), mon['wall_seconds'])
            print(json.dumps({'seq': step['seq'], 'comparison': step['comparison'], 'arm': arm, 'pp': length,
                              'valid': row['valid'], 'tok_s': d.get('proc_median_tok_s'),
                              'wall_s': round(mon['wall_seconds'], 1), 'problems': problems}), flush=True)
            if problems and not a.keep_going:
                status = 'invalid process at seq %d: %s' % (step['seq'], '; '.join(problems))
                break
    finally:
        raw.close()
        s = summarize(rows, manifest)
        s.update(status=status, complete=status == 'complete' and len(rows) == len(sched),
                 processes_run=len(rows), processes_planned=len(sched), elapsed_seconds=time.monotonic() - T0)
        (out / 'summary.json').write_text(json.dumps(s, indent=2) + '\n')
        print(render(s))
        print('status: ' + status)
    if status != 'complete':
        raise SystemExit(3)
    guard = s.get('baseline_guard_pp8192')
    if guard and not guard['within']:
        raise SystemExit(4)  # complete, but outside the +-10% operational guard: parent decides


def cmd_summary(a):
    out = pathlib.Path(a.out).resolve()
    rows = [json.loads(ln) for ln in (out / 'raw.jsonl').read_text().splitlines() if ln.strip()]
    manifest = json.loads((out / 'manifest.json').read_text())
    s = summarize(rows, manifest)
    (out / 'summary.json').write_text(json.dumps(s, indent=2) + '\n')
    print(render(s))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    for name in ('baseline', 'abba'):
        p = sub.add_parser(name)
        p.add_argument('--cli', required=True, help='release hipfire binary')
        p.add_argument('--daemon', required=True, help='release daemon binary (pinned via HIPFIRE_DAEMON_BIN)')
        p.add_argument('--model', required=True, help='MODEL argument passed verbatim to `hipfire bench` (file or store name)')
        p.add_argument('--prompt512', required=True)
        p.add_argument('--prompt8192', required=True)
        p.add_argument('--out', required=True, help='new output directory')
        p.add_argument('--prompt512-sha256', default=None, help='override the pinned fixture digest')
        p.add_argument('--prompt8192-sha256', default=None)
        p.add_argument('--tokens512', type=int, default=512, help='exact expected daemon prompt_tokens')
        p.add_argument('--tokens8192', type=int, default=8192)
        p.add_argument('--model-sha256', default=CANON_MODEL_SHA256)
        p.add_argument('--hash-model', action='store_true', help='recompute model sha256 inside the hold (~1 min)')
        p.add_argument('--allow-noncanonical-model', action='store_true')
        p.add_argument('--expect-gfx', default='gfx1201')
        p.add_argument('--expect-kv-mode', default='fp8', help='effective kv_mode from the loaded event (product default auto)')
        p.add_argument('--expect-cli-sha256', default=None)
        p.add_argument('--expect-daemon-sha256', default=None)
        p.add_argument('--expect-source', default=None, help='required git HEAD prefix of --repo')
        p.add_argument('--repo', default=str(REPO))
        p.add_argument('--cwd', default=str(REPO))
        p.add_argument('--mkhome', default=os.environ.get('A4_MKHOME', '/home/kaden/qcal/tools/mkhome.sh'))
        p.add_argument('--env', action='append', default=[], metavar='KEY=VALUE',
                       help='extra recorded child env (e.g. HIPFIRE_DPM_WARMUP_SECS=10); KV/GRAPH/NORMALIZE/selector refused')
        p.add_argument('--budget-secs', type=float, default=1740.0)
        p.add_argument('--proc-timeout-secs', type=float, default=420.0)
        p.add_argument('--keep-going', action='store_true')
        if name == 'abba':
            p.add_argument('--candidates', default='4,8')
            p.add_argument('--blocks', type=int, default=2, help='ABBA blocks per candidate and length')
        else:
            p.set_defaults(blocks=1, candidates='')
        p.set_defaults(fn=cmd_run)
    s = sub.add_parser('summary', help='recompute summary.json from <out>/raw.jsonl + manifest.json')
    s.add_argument('out')
    s.set_defaults(fn=cmd_summary)
    a = ap.parse_args()
    a.fn(a)


if __name__ == '__main__':
    main()
