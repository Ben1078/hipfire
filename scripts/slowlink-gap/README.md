# slowlink-gap: S1 pp16384 >= 2000 tok/s ABBAAB qualification harness (performance only)

Adapted from the untracked `gap2000/scripts/gap-{cell,job,launch,wait}.sh`; deployed via git (push, then a worktree on
hipx), never rsynced as source. CPU scripts: `gap_lib.py gap_stage.py gap_plan.py gap_score.py gap_selftest.py`.
GPU scripts (run only inside a fabric hold on hipx/xtx): `gap-launch.sh -> gap-job.sh -> gap-cell.sh`, `gap-wait.sh`.

Contract encoded here: six fresh `hipfire bench` processes ABBAAB = A1 B1 B2 A2 A3 B3, each `--matrix --pp 16384 --tg 1
--ctx 1 --runs 3 --warmups 1 --max-seq 16416 --json` (3 daemon-measured prefill rates/cell, flat 9/arm). Arm A = all S1
levers off; arm B = placement + IU4 + QSA gather. Both arms: same stage, same `candidate.toml` (S1 enabled by config), same
env `HIPFIRE_STATE_QUANT=q8 HIPFIRE_KV_MODE=q8 HIPFIRE_QWEN4_D2_DEFAULT=0`, no f32 carry/global f32/q8w/trace.
Only `HIPFIRE_QWEN4_SLOWLINK_RESIDENT_LAYERS`, `HIPFIRE_QWEN4_MOE_SYM_IU4`, `HIPFIRE_FN_QSA_GATHER_GFX1100` vary.
Fabric: class timing, card xtx, 22.4 GiB, host-mem 67 GiB lock, max 1800 s, P1. Own HOME/config/cache/locks per cell,
UUID `GPU-43390a851e296ee5`, BDF `0000:67:00.0`, MemAvailable >= 80 GiB per cell.

## Flow (hipx; the harness checkout is the pushed worktree)

```
H=<hipx worktree>/scripts/slowlink-gap
A=/home/kaden/qcal/release-0.4.3/slowlink-prefill/gap2000/artifacts/<NAME>   # rsync -a ARTIFACTS hipx:$A  (no model)
# 0. CPU self-test of the harness python (fabricated artifacts, no GPU, no model)
python3 $H/gap_selftest.py
# 1. symlink-only stage + read-only identity manifest (same script local or remote; no payload copied)
python3 $H/gap_stage.py --out $S/stage-<tag> --cli $A/hipfire --daemon $A/daemon \
    --kernels $A/kernels/compiled/gfx1100 --pack-manifest $A/<pack>.manifest.json \
    [--pack-tar-sha256 <hex>] [--probe qwen4_slowlink_quality=$A/qwen4_slowlink_quality]
# 2. plan + pins (stage identity, config sha, model cached stat, env, synthetic digest) in a FRESH read-only dir
python3 $H/gap_plan.py make --stage-root $S/stage-<tag> --out-dir $P/plan-<tag> [--place 0,3,7,15,19,23,27,39,43]
# 3. detached launch (nohup setsid, fds>2 closed) + detached waiter; run dir must not exist
bash $H/gap-launch.sh <runname> $P/plan-<tag>           # -> $GAP_RUNS/<runname> (default gap2000/runs)
bash $H/gap-wait.sh --detach <runname>                  # -> <run>/waiter.log, waiter.exit
# 4. CPU scorer on the (rsynced) run dir
python3 $H/gap_score.py $GAP_RUNS/<runname> --out $GAP_RUNS/<runname>.score.json
```

`gap_score.py` exit: 0 PASS (candidate flat median over all nine >= 2000), 1 FAIL_GATE, 2 INVALID (contract assertion),
3 INCOMPLETE (XTX drop or partial run: stop, tell the parent, no retry). The score json keeps every rate sample,
per-process medians, applied env per cell, route lines, stage identity, and the synthetic stream digest
(sha256 of u32le ids `10 + (i % 1000)`, i < 16384, the daemon's `bench_prefill` stream).

Receipts: each run lives in a fresh directory (`runs/<name>`), the job writes `cells/MANIFEST.sha256` and makes `cells/`
read-only. Cell failure: no automatic rerun. Original gap2000 artifacts are untouched.
