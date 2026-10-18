# hold1-common.sh — sourced ONLY by hold1a-correct.sh and hold1b-correct.sh, after hold-lib.sh (never by hold2/hold3, never by hold-lib.sh).
# It carries the Hold 1-only spawn env pin, so the shared cell_env (which strips every HIPFIRE_*/HIP_*/HSA_*/ROCR_* variable and sets its fixed pins)
# is unchanged and H2/H3 keep their default (D2 pairing) behaviour.
#
# H1_OFFLOAD_PIN: the Hold 1 PCIe-only ruling. HIPFIRE_OFFLOAD_EXEC is a registered variable of the current base
# (OFFLOAD_EXECS = [pcie, cpu, passback], default pcie; crates/rdna-compute lib.rs). H1 pins the EXACT value `pcie` EXPLICITLY on EVERY H1 GPU cell
# (prim, state, greedy-beta, greedy-s1), as a per-cell extra KEY=VAL of `cell`, so the explicit pin also survives a future beta whose default
# is the D2 opt-out. `off` is NOT a registered value: an invalid fallback could mask the origin of a path, so it is never used.
# The pin actually applied to a cell is persisted in CELL/spawn-env.json and CELL/evidence.json, and h1_handoff.py requires it on every
# bound 1a cell and on both 1b greedy arms.
H1_OFFLOAD_PIN="HIPFIRE_OFFLOAD_EXEC=pcie"
