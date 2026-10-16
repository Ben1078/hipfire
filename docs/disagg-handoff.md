# Flash-Next eager disaggregated handoff (developer/lab)

This is an explicitly invoked experiment, not daemon routing or a serving default.
A full-model gfx1151 bundle produces a prompt state; a separate gfx1100 bundle
imports it and decodes. No streaming per tile/layer, daemon routing, adaptive
migration, or cross-domain prefix reuse is provided.

## Frozen packet

Schema version 1 (`disagg_manifest::HandoffManifest`) binds:

- The producing and consuming numerical manifests: architecture, model artifact
  digest, route identity, and policy digest. The ordered pair is the cache domain.
- The canonical 48-layer layout (`gdn,gdn,gdn,qsa`, repeated twelve times),
  resolved KV/GDN/index state formats, and committed absolute position.
- Target metadata containing QSA full/raw/pooled/partial lengths, selected lengths
  and positions, and PLE token history; MTP metadata containing head position,
  QSA marks, and request policy state.
- Exact fixed-part lengths and `(row bytes, live rows)` for target and MTP, plus
  the state payload digest. The payload carries the GDN recurrent/conv state,
  QSA keys/values/index rows/selection, PLE conv state, hyper-connection feedback,
  and the MTP head state and last wide hidden row.
- `first_token`: the greedy token sampled from the producer's final logits.
  This lab gate uses greedy decoding. The consumer **does not run the final
  prompt token again** with different arithmetic. Non-greedy sampling requires
  its own explicit sampler-state contract and is not implied by this packet.

State bytes are exported eagerly into portable pinned host storage. Device-local
derived F16 QSA arenas and pooled BF16 mirrors are not wire state; restore
invalidates them so their normal consumers rebuild them. A receiving bundle
restores into its own arenas and publishes committed marks/position only after
all state copies complete. Failure after mutation invalidates the request rather
than retrying on partially advanced state or switching arithmetic.

The packet is not an ordinary session snapshot. A restore must match both halves
of its producer/consumer domain; a different artifact, format, route, numerical
policy, or producer history is not interchangeable. MTP-versus-AR parity means
both decode on the XTX from the same imported state under the same decode policy,
not equality with an XTX cold prefill.

## Ownership and admission

The lab two-bundle path uses one immutable `MAPPED|PORTABLE` host expert backing,
including a host shadow of XTX-resident expert layers. Each device obtains its own
alias while bound through `Gpu::bind_thread`; CPU and device addresses are never
assumed equal. Alias views keep the original owner leased and never free it.
Admission accounts the shared allocation once and both bundles' independent
weights, state, scratch, tables and safety headroom before loading model bytes.
Unsupported admission returns an error; ordinary single-device loading is unchanged.

`DisaggPair::load` admits canonical Flash-Next aggregate QT44 gate/up + QT53 down
weights, including MTP, with fixed whole-layer XTX placement. An explicit
`HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS=N` fixes that placement; unset chooses the
largest whole-layer count fitting the admitted memory census, never timings.
`DisaggLoaded::run` binds the side's device and scoped execution policy for every
operation: native GPU arithmetic on the producer, CPU cold experts on the
consumer when `memory.offload_exec=cpu|passback` was requested, otherwise PCIe.
PLE rows are resolved once on the producer and shared in the artifact identity,
rather than letting the arches choose different `auto` PLE formats.

The lab loader defaults an **unset** `HIPFIRE_STATE_QUANT` to `fp32` for the F32
inventory below; an explicit state-format request still resolves normally and
must agree on both bundles. Ordinary single-device defaults are not changed.
Call `DisaggPair::unload` to release consumer aliases before producer backing;
the example does so on success and error.

Admission refuses wrong/equal device IDs or architectures, non-UMA producer or
UMA consumer, invalid context capacity, capture or retained/non-HIP replay,
EP/NPU/trunk-overlay composition, iGPU expert offload, hot-set/requant overrides,
slab/stock-FP8/noncanonical or mixed expert formats, absent MTP, differing
artifacts/PLE/state formats, explicit `auto` whole-layer placement, and host,
GTT or either device memory shortfalls. Handoff validation separately refuses
unknown schema, unbound or mismatched numerical domain, incompatible part
geometry/marks/position, live partial QSA scratch, cold MTP, armed row capture,
and payload digest/length failures. There is no arithmetic fallback after a
failed handoff.

## Byte accounting

The design's main-model inventory (F32 recurrent and KV, excluding PLE,
hyper-feedback, MTP and manifest metadata) is:

```
117,768,336 + 12 * (4096 + 512) * tokens + 12 * 512 * ceil(tokens / 4)
```

| Tokens | Design main-model bytes | Complete packed payload bytes |
|---:|---:|---:|
| 512 | 146,866,320 | 149,749,920 |
| 2,048 | 234,160,272 | 244,318,368 |
| 16,384 | 1,048,903,824 | 1,126,957,216 |
| 32,768 | 1,980,039,312 | 2,135,687,328 |

`flash_next_state_bytes` computes the design inventory with checked arithmetic.
The complete default F32 target plus MTP payload is
`118,227,104 + 13 * (4608 * tokens + 512 * floor(tokens / 4))`, computed by
`flash_next_handoff_payload_bytes`. PLE convolution is **368,640 bytes**
(`9 * 2560 * 4 * 4`), not the approximately 92 KB in the initial design inventory;
hyper feedback adds 40,960 bytes and MTP fixed state adds 49,168 bytes.
Committed pooled rows use floor; allocation capacity uses ceil. At the four
divisible-by-four lengths the design and committed pooled counts agree.

Existing target/MTP session metadata adds 680 bytes; the serialized versioned
manifest is additional and variable length. No disk framing or alignment is
included in the table. The manifest records every part's exact length, so
`payload_bytes()` is authoritative for the loaded model and live QSA marks.

## Exactness gate

Build the `qwen4_disagg_handoff` example with feature `lab`. It loads the producer
and consumer, prefills only on gfx1151, exports and saves the bytes, imports on
gfx1100, compares the re-exported state, and decodes greedy IDs. It then replaces
the decode bundle with a fresh XTX load sharing the same backing and repeats the
saved-state import. Finally it compares native MTP with AR from that imported
state. Its JSON result must report exact state bytes and identical greedy IDs;
this is a correctness gate, not a performance or KLD claim.

MTP policy of the gate (`--tokens` at least 4): the first window is the production
calibration AR step on the carried first token; the timed AR floor is then pinned
off with the lab-only `Qwen4MtpDrafter::lab_set_floor_enabled(false)` as a lab
correctness policy, so no timing-dependent retirement to AR changes the route.
The gate fails unless native windows drafted at least one token, and reports
drafted/accepted counts.

No GPU validation is implied by a successful CPU build or unit test. Hardware
runs must be scheduled with the `xtx,halo` card set and enough host/GTT memory.

CPU-only build:

```sh
ROCR_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 \
CARGO_TARGET_DIR=/home/kaden/qcal/release-0.4.3/target-offload-prefill \
nice -n 10 cargo build --release -p hipfire-arch-qwen4 \
  --features lab --example qwen4_disagg_handoff
```

Inside an admitted hipx `xtx,halo` hardware cell (replace model/prompt paths and
use the device indices exposed by that cell; the command assumes XTX=0, Halo=1):

```sh
HIPFIRE_OFFLOAD_EXEC=cpu HIPFIRE_REPLAY_BACKEND=hip \
/home/kaden/qcal/release-0.4.3/target-offload-prefill/release/examples/qwen4_disagg_handoff \
  --model /path/to/flash-next.mq4 --prefill-device 1 --decode-device 0 \
  --prompt-file /path/to/prompt.txt --tokens 64 --max-seq 4096 \
  --saved-state /tmp/flash-next-handoff.bin
```

The loader validates the architectures; an incorrect device ordering refuses
rather than running a different producer/consumer route.
