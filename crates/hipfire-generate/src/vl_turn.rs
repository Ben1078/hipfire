// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.

//! claude: Qwen3.5-VL image turns through the shared text/MTP path ("VL v2").
//!
//! The historical VL body (`vision::generate_vl`) built its own prompt from the
//! last user text + one image: no system prompt, no tools, no conversation, the
//! image always in the last turn. Agent harnesses (system prompt + tools +
//! history, images in earlier turns, several images) need the text path's
//! rendering, prompt cache and tool parsing. An image turn here IS a text turn
//! (`qwen::generate_dflash`, MTP speculative decode) with three additions:
//!
//! 1. **Placement.** The serve layer leaves [`IMAGE_MARKER`] (what the Qwen
//!    chat template emits for an image content part) where each image sat in
//!    its message. We expand marker `i` to `<|vision_start|>` + `rows_i` ×
//!    `<|image_pad|>` + `<|vision_end|>` in the message text, so every render
//!    of the conversation — cold, cached-history, prefix realign — tokenizes to
//!    the same expanded sequence. No markers (raw JSONL clients) → images go in
//!    front of the last user message.
//! 2. **Inputs.** After render + prompt-cache planning ([`turn_plan`], H1) the
//!    image-pad runs are located in the final token sequence and HF
//!    `get_rope_index` phases are built for all spans. After `gen_start`
//!    ([`turn_install`], H2) the images are encoded (content-keyed cache: a
//!    later turn re-sending the same image skips the tower) and
//!    `qwen35::vl_prefill` is installed for the whole spec turn, with a phase
//!    table covering every KV slot so any re-prefill from slot 0 is exact.
//! 3. **Reuse gate.** Phases are shifted by the request's `-rope_delta` (so
//!    decode and everything after the last image rotate at their slot). The KV
//!    of a turn is therefore only reusable by a later turn with the same shift
//!    and the same images at the same slots. [`Resident`] records the layout of
//!    the resident KV; H1 clamps the cache plan to the prefix both layouts
//!    agree on, and text turns ([`text_turn_guard`], H0) cold-reset when the
//!    resident KV carries shifted phases.

use crate::vision::ImageSource;
use hipfire_arch_qwen35::qwen35;
use hipfire_arch_qwen35_vl::image;
use hipfire_arch_qwen35_vl::qwen35_vl;
use hipfire_loader::LoadedModel;
use hipfire_runtime::prompt_frame::{AssistantPrefix, Message};
use std::cell::RefCell;
use std::sync::Arc;

/// What the Qwen chat template emits for one image content part; the serve
/// layer leaves it in the message text where the image was.
pub const IMAGE_MARKER: &str = "<|vision_start|><|image_pad|><|vision_end|>";
const VISION_START: &str = "<|vision_start|>";
const IMAGE_PAD: &str = "<|image_pad|>";
const VISION_END: &str = "<|vision_end|>";

/// `CLAUDE_VL_V2` (default on): image turns run through the text/MTP path
/// (full chat template with system prompt, tools, history; several images;
/// prompt cache). `=0` restores the historical single-image VL body.
pub fn v2_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("CLAUDE_VL_V2").as_deref() != Ok("0"))
}

/// Maximum images per request (bounds encoder time and host cache memory).
const MAX_IMAGES: usize = 8;
/// Encoded images kept for later turns of the same conversation.
const EMBED_CACHE_CAP: usize = 6;

/// Request parameters (the text path's, plus the images).
pub struct VlTurnParams<'a> {
    pub id: &'a str,
    pub prompt: &'a str,
    pub system_prompt: Option<&'a str>,
    pub images: Vec<ImageSource<'a>>,
    pub max_tokens: usize,
    pub max_think_tokens: usize,
    pub assistant_prefix: AssistantPrefix,
    pub tools: Option<&'a [serde_json::Value]>,
    pub messages_history: Option<&'a [Message]>,
    pub stop: &'a [String],
    pub temp: f32,
    pub top_p: f32,
    pub top_k: Option<u32>,
    pub min_p: Option<f32>,
    pub cactus_delta: f32,
    pub request_seed: u32,
    pub reasoning_effort: Option<&'a str>,
    pub enable_thinking: bool,
}

/// Whether this model/request can take the v2 path. `Err` names the reason
/// (the caller logs it and keeps the historical VL body).
pub fn v2_eligible(m: &LoadedModel, gpu: &rdna_compute::Gpu, temp: f32) -> Result<(), String> {
    if !v2_enabled() {
        return Err("CLAUDE_VL_V2=0".into());
    }
    if hipfire_config::developer_var("HIPFIRE_DFLASH_CHAT")
        .ok()
        .as_deref()
        == Some("0")
    {
        return Err("HIPFIRE_DFLASH_CHAT=0".into());
    }
    if m.kv_adaptive.is_some() || m.eviction.is_some() || m.ep.is_some() || m.pp > 1 {
        return Err("adaptive KV / eviction / EP / PP load".into());
    }
    let Some(spec) = m.speculator.as_ref() else {
        return Err("no speculator loaded".into());
    };
    if spec.name() != "mtp" {
        return Err(format!("speculator {} (v2 is validated with MTP)", spec.name()));
    }
    if temp > 1e-6 && !spec.supports_temp_verify() {
        return Err("sampled request on a greedy-only speculator".into());
    }
    let Some(b) = m.qwen35() else {
        return Err("not a qwen35 bundle".into());
    };
    if b.vision_config.is_none() || b.vision_weights.is_none() {
        return Err("no vision tower".into());
    }
    qwen35::vl_prefill::batched_supported(gpu, &b.weights, &b.config, &b.dn_state, &b.kv_cache, 1024)
}

// ── Resident KV layout ──────────────────────────────────────────────────────

#[derive(Clone, Debug, PartialEq, Eq)]
struct SpanSig {
    start: usize,
    rows: usize,
    grid_h: usize,
    grid_w: usize,
    key: u64,
}

/// Phase layout of the KV the model currently holds.
#[derive(Clone, Debug)]
enum Resident {
    /// Slot phases everywhere (text turns).
    Text,
    /// A v2 image turn: phases shifted by `shift`, images at `spans`.
    Vl { shift: i32, spans: Vec<SpanSig> },
    /// Written by a path that does not record its layout (historical VL body).
    Unknown,
}

/// Longest prefix on which the resident KV and a new layout agree.
fn reusable_prefix(resident: &Resident, shift: i32, spans: &[SpanSig]) -> usize {
    match resident {
        Resident::Unknown => 0,
        Resident::Text => {
            if shift == 0 {
                spans.first().map_or(usize::MAX, |s| s.start)
            } else {
                0
            }
        }
        Resident::Vl {
            shift: old_shift,
            spans: old,
        } => {
            if *old_shift != shift {
                return 0;
            }
            for (a, b) in old.iter().zip(spans) {
                if a != b {
                    return a.start.min(b.start);
                }
            }
            match old.len().cmp(&spans.len()) {
                std::cmp::Ordering::Less => spans[old.len()].start,
                std::cmp::Ordering::Greater => old[spans.len()].start,
                std::cmp::Ordering::Equal => usize::MAX,
            }
        }
    }
}

// ── Per-request session ─────────────────────────────────────────────────────

struct TurnImage {
    key: u64,
    grid_h: usize,
    grid_w: usize,
    rows: usize,
    /// CHW pixels + (h, w) when the embedding is not cached yet.
    pixels: Option<(Vec<f32>, usize, usize)>,
    embeds: Option<Arc<Vec<f32>>>,
}

struct Session {
    images: Vec<TurnImage>,
    image_pad_id: u32,
    spatial_merge: usize,
    /// Set by H1: the final prompt, its image spans and shifted phases.
    prompt_tokens: Vec<u32>,
    spans: Vec<SpanSig>,
    shift: i32,
    positions: Vec<[i32; 3]>,
    planned: bool,
    installed: bool,
}

thread_local! {
    static SESSION: RefCell<Option<Session>> = const { RefCell::new(None) };
    static RESIDENT: RefCell<Resident> = const { RefCell::new(Resident::Text) };
    static EMBED_CACHE: RefCell<Vec<(u64, usize, usize, Arc<Vec<f32>>)>> =
        const { RefCell::new(Vec::new()) };
}

fn session_armed() -> bool {
    SESSION.with(|s| s.borrow().is_some())
}

/// Mark the resident KV as written by a path that does not record its phase
/// layout (the historical VL body): the next text turn cold-resets.
pub fn note_unrecorded_vl_turn() {
    RESIDENT.with(|r| *r.borrow_mut() = Resident::Unknown);
}

/// H0 — text turns (`ar::generate` entry): if the resident KV carries shifted
/// VL phases, drop the conversation so this turn prefills cold instead of
/// reusing a prefix rotated at different phases.
pub fn text_turn_guard(m: &mut LoadedModel, gpu: &mut rdna_compute::Gpu) {
    if session_armed() {
        return;
    }
    let stale = RESIDENT.with(|r| !matches!(*r.borrow(), Resident::Text));
    if !stale {
        return;
    }
    if m.seq_pos > 0 || !m.conversation_tokens.is_empty() {
        eprintln!(
            "[vl-v2] text turn after an image turn: cold prefill (resident KV has VL phases, seq_pos={})",
            m.seq_pos
        );
        reset_conversation(m, gpu);
    }
    RESIDENT.with(|r| *r.borrow_mut() = Resident::Text);
}

fn reset_conversation(m: &mut LoadedModel, gpu: &mut rdna_compute::Gpu) {
    m.seq_pos = 0;
    m.conversation_tokens.clear();
    crate::common::free_checkpoints(&mut m.prefill_checkpoints, gpu);
    crate::common::free_checkpoints(&mut m.dflash_checkpoints, gpu);
    if let Some(s) = m.speculator.as_mut() {
        if let Err(e) = s.reset(gpu) {
            eprintln!("[vl-v2] speculator reset: {e}");
        }
    }
    if let Err(e) = crate::common::reset_qwen35_recurrent(m, gpu) {
        eprintln!("[vl-v2] recurrent reset: {e}");
    }
    if let Some(b) = m.qwen35_mut() {
        b.kv_cache.compact_offset = 0;
    }
}

/// Prompt-cache plan as `generate_dflash` computes it:
/// `(prefill_tokens, prefill_start, cache_hit, resume_from, cached_tokens)`.
pub(crate) type DflashPlan = (Vec<u32>, usize, bool, Option<usize>, usize);

/// H1 — after `generate_dflash` rendered the prompt and planned the cache:
/// locate the image spans, build their phases, and clamp the plan to the
/// prefix the resident KV agrees on. No-op for text turns.
pub(crate) fn turn_plan(prompt_tokens: &[u32], plan: DflashPlan) -> Result<DflashPlan, String> {
    if !session_armed() {
        return Ok(plan);
    }
    SESSION.with(|s| {
        let mut s = s.borrow_mut();
        let s = s.as_mut().expect("armed");
        // Image-pad runs, in order.
        let mut runs: Vec<(usize, usize)> = Vec::new();
        let mut i = 0usize;
        while i < prompt_tokens.len() {
            if prompt_tokens[i] == s.image_pad_id {
                let start = i;
                while i < prompt_tokens.len() && prompt_tokens[i] == s.image_pad_id {
                    i += 1;
                }
                runs.push((start, i - start));
            } else {
                i += 1;
            }
        }
        if runs.len() != s.images.len() {
            return Err(format!(
                "image placement: the rendered prompt has {} image-pad runs for {} images",
                runs.len(),
                s.images.len()
            ));
        }
        let mut spans = Vec::with_capacity(runs.len());
        let mut image_spans = Vec::with_capacity(runs.len());
        for (&(start, rows), img) in runs.iter().zip(&s.images) {
            if rows != img.rows {
                return Err(format!(
                    "image placement: image-pad run at {start} has {rows} rows, image needs {}",
                    img.rows
                ));
            }
            spans.push(SpanSig {
                start,
                rows,
                grid_h: img.grid_h,
                grid_w: img.grid_w,
                key: img.key,
            });
            image_spans.push(hipfire_arch_qwen35_vl::mrope::ImageSpan {
                start,
                len: rows,
                grid_h: img.grid_h,
                grid_w: img.grid_w,
            });
        }
        let built = hipfire_arch_qwen35_vl::mrope::build_mrope_positions(
            prompt_tokens.len(),
            &image_spans,
            s.spatial_merge,
        );
        if built.positions.len() != prompt_tokens.len() {
            return Err(format!(
                "mrope positions: {} rows for {} tokens",
                built.positions.len(),
                prompt_tokens.len()
            ));
        }
        let shift = -built.rope_delta;
        let reusable = RESIDENT.with(|r| reusable_prefix(&r.borrow(), shift, &spans));
        let (prefill_tokens, prefill_start, cache_hit, resume_from, cached) = plan;
        let reused = if cache_hit {
            prefill_start
        } else {
            resume_from.unwrap_or(0)
        };
        let plan = if reused > reusable {
            eprintln!(
                "[vl-v2] prompt cache: reuse {reused} > {} rows valid for this image layout — cold prefill",
                reusable.min(prompt_tokens.len())
            );
            (prompt_tokens.to_vec(), 0, false, None, 0)
        } else {
            (prefill_tokens, prefill_start, cache_hit, resume_from, cached)
        };
        eprintln!(
            "[vl-v2] prompt={} images={} spans={:?} rope_delta={} prefill_from={} cache_hit={}",
            prompt_tokens.len(),
            spans.len(),
            spans.iter().map(|x| (x.start, x.rows)).collect::<Vec<_>>(),
            built.rope_delta,
            if plan.2 { plan.1 } else { plan.3.unwrap_or(0) },
            plan.2,
        );
        s.positions = qwen35::vl_prefill::shift_positions(&built.positions, built.rope_delta);
        s.prompt_tokens = prompt_tokens.to_vec();
        s.spans = spans;
        s.shift = shift;
        s.planned = true;
        Ok(plan)
    })
}

/// H2 — after `gen_start`, before the spec prefill: encode the images (cache
/// first) and install the VL prefill context for the whole spec turn.
/// No-op for text turns.
pub(crate) fn turn_install(m: &mut LoadedModel, gpu: &mut rdna_compute::Gpu) -> Result<(), String> {
    if !session_armed() {
        return Ok(());
    }
    let planned = SESSION.with(|s| s.borrow().as_ref().is_some_and(|s| s.planned));
    if !planned {
        return Err("image turn reached the prefill without a plan".into());
    }
    // Encode what is not cached yet (pixels move out of the session so the
    // tower can borrow the model).
    let todo: Vec<(usize, Vec<f32>, usize, usize, usize, usize)> = SESSION.with(|s| {
        let mut s = s.borrow_mut();
        let s = s.as_mut().expect("armed");
        s.images
            .iter_mut()
            .enumerate()
            .filter(|(_, img)| img.embeds.is_none())
            .filter_map(|(i, img)| {
                img.pixels
                    .take()
                    .map(|(px, h, w)| (i, px, h, w, img.grid_h, img.grid_w))
            })
            .collect()
    });
    for (i, pixels, h, w, grid_h, grid_w) in todo {
        let (vc, vw) = match m.qwen35() {
            Some(b) => match (b.vision_config.as_ref(), b.vision_weights.as_ref()) {
                (Some(c), Some(w)) => (c, w),
                _ => return Err("vision tower missing".into()),
            },
            None => return Err("not a qwen35 bundle".into()),
        };
        let patches = image::extract_patches(
            &pixels,
            3,
            h,
            w,
            vc.patch_size,
            vc.temporal_patch_size,
            vc.spatial_merge_size,
        );
        let t = std::time::Instant::now();
        let embeds = qwen35_vl::vision_forward(gpu, vw, vc, &patches, grid_h, grid_w)
            .map_err(|e| format!("vision encoder: {e}"))?;
        let embeds = Arc::new(embeds);
        eprintln!(
            "[vl-v2] image {i}: encoded {grid_h}x{grid_w} patches in {:.2}s",
            t.elapsed().as_secs_f64()
        );
        let key = SESSION.with(|s| {
            let mut s = s.borrow_mut();
            let img = &mut s.as_mut().expect("armed").images[i];
            img.embeds = Some(embeds.clone());
            img.key
        });
        EMBED_CACHE.with(|c| {
            let mut c = c.borrow_mut();
            c.retain(|e| e.0 != key);
            c.push((key, grid_h, grid_w, embeds));
            if c.len() > EMBED_CACHE_CAP {
                c.remove(0);
            }
        });
    }
    let (dim, section, table_rows) = match m.qwen35() {
        Some(b) => (b.config.dim, b.config.mrope_section, b.kv_cache.physical_cap),
        None => return Err("not a qwen35 bundle".into()),
    };
    let inputs = SESSION.with(|s| {
        let s = s.borrow();
        let s = s.as_ref().expect("armed");
        let mut embeds = Vec::with_capacity(s.images.iter().map(|i| i.rows * dim).sum());
        for img in &s.images {
            let e = img.embeds.as_ref().expect("encoded above");
            if e.len() != img.rows * dim {
                return Err(format!(
                    "vision encoder returned {} values for {} rows x {dim}",
                    e.len(),
                    img.rows
                ));
            }
            embeds.extend_from_slice(e);
        }
        Ok(qwen35::vl_prefill::VlPrefillInputs {
            spans: s.spans.iter().map(|x| (x.start, x.rows)).collect(),
            embeds,
            dim,
            positions: s.positions.clone(),
            section,
        })
    })?;
    qwen35::vl_prefill::install(gpu, inputs, table_rows)
        .map_err(|e| format!("vl_prefill install: {e}"))?;
    SESSION.with(|s| s.borrow_mut().as_mut().expect("armed").installed = true);
    // Until the turn proves which KV it left behind, nothing is reusable.
    RESIDENT.with(|r| *r.borrow_mut() = Resident::Unknown);
    Ok(())
}

/// End of an image turn: remove the prefill context and record the layout of
/// the KV the turn left behind.
fn turn_finish(m: &LoadedModel, gpu: &mut rdna_compute::Gpu) {
    if let Err(e) = qwen35::vl_prefill::uninstall(gpu) {
        eprintln!("[vl-v2] vl_prefill uninstall: {e}");
    }
    let Some(s) = SESSION.with(|s| s.borrow_mut().take()) else {
        return;
    };
    if !s.installed {
        return; // KV untouched by this turn.
    }
    let resident = if !s.prompt_tokens.is_empty()
        && m.conversation_tokens.starts_with(&s.prompt_tokens)
    {
        Resident::Vl {
            shift: s.shift,
            spans: s.spans,
        }
    } else if m.seq_pos == 0 && m.conversation_tokens.is_empty() {
        Resident::Text
    } else {
        Resident::Unknown
    };
    RESIDENT.with(|r| *r.borrow_mut() = resident);
}

fn image_key(bytes: &[u8]) -> u64 {
    // FNV-1a over the encoded bytes (deterministic across requests).
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h ^ (bytes.len() as u64).rotate_left(32)
}

fn expanded_marker(rows: usize) -> String {
    let mut s = String::with_capacity(VISION_START.len() + VISION_END.len() + rows * IMAGE_PAD.len());
    s.push_str(VISION_START);
    for _ in 0..rows {
        s.push_str(IMAGE_PAD);
    }
    s.push_str(VISION_END);
    s
}

/// Replace the `i`-th [`IMAGE_MARKER`] (in message order) by its expansion.
/// Returns the number of markers found; unchanged text when it differs from
/// `rows.len()`.
fn expand_markers(texts: &mut [&mut String], rows: &[usize]) -> usize {
    let found: usize = texts.iter().map(|t| t.matches(IMAGE_MARKER).count()).sum();
    if found != rows.len() {
        return found;
    }
    let mut next = 0usize;
    for t in texts.iter_mut() {
        if !t.contains(IMAGE_MARKER) {
            continue;
        }
        let mut out = String::with_capacity(t.len());
        let mut rest: &str = t.as_str();
        while let Some(pos) = rest.find(IMAGE_MARKER) {
            out.push_str(&rest[..pos]);
            out.push_str(&expanded_marker(rows[next]));
            next += 1;
            rest = &rest[pos + IMAGE_MARKER.len()..];
        }
        out.push_str(rest);
        **t = out;
    }
    found
}

/// The image turn: decode + place the images, then run the text/MTP path.
pub fn generate_vl_v2(
    m: &mut LoadedModel,
    gpu: &mut rdna_compute::Gpu,
    stdout: &mut std::io::Stdout,
    p: &VlTurnParams,
) {
    let id = p.id;
    let _route_scope =
        crate::ar::GenerationRouteScope::enter(crate::ar::GenerationRoute::QwenDflash, id);
    hipfire_runtime::llama::reset_cpu_sampler_rng(p.request_seed);
    let fail = |stdout: &mut std::io::Stdout, msg: &str| {
        crate::dense::emit_active_attempt_error(stdout, Some(id), msg, "validation", false, false);
        let _ = std::io::Write::flush(stdout);
    };
    if p.images.is_empty() || p.images.len() > MAX_IMAGES {
        fail(stdout, &format!("image turns take 1..={MAX_IMAGES} images"));
        return;
    }
    let (patch, merge) = match m.vision_config() {
        Some(vc) => (vc.patch_size, vc.spatial_merge_size),
        None => {
            fail(stdout, "model has no vision encoder");
            return;
        }
    };
    let image_pad_id = match m
        .tokenizer
        .as_ref()
        .and_then(|t| t.special_token_id(IMAGE_PAD))
    {
        Some(id) => id,
        None => {
            fail(stdout, "tokenizer has no <|image_pad|> token");
            return;
        }
    };

    // 1. Images: content-keyed cache first, else decode + resize (the tower
    //    runs after gen_start, in H2).
    let mut images: Vec<TurnImage> = Vec::with_capacity(p.images.len());
    for (i, src) in p.images.iter().enumerate() {
        let bytes = match src {
            ImageSource::Base64(b64) => crate::vision::decode_image_bytes(b64),
            ImageSource::Path(path) => {
                std::fs::read(path).map_err(|e| format!("image {path}: {e}"))
            }
        };
        let bytes = match bytes {
            Ok(b) => b,
            Err(e) => {
                fail(stdout, &format!("image {i}: {e}"));
                return;
            }
        };
        let key = image_key(&bytes);
        let cached = EMBED_CACHE.with(|c| {
            c.borrow()
                .iter()
                .find(|e| e.0 == key)
                .map(|e| (e.1, e.2, e.3.clone()))
        });
        let img = match cached {
            Some((grid_h, grid_w, embeds)) => TurnImage {
                key,
                grid_h,
                grid_w,
                rows: (grid_h / merge) * (grid_w / merge),
                pixels: None,
                embeds: Some(embeds),
            },
            None => match image::load_and_preprocess_from_bytes(&bytes, patch, merge) {
                Ok((pixels, h, w)) => {
                    let (grid_h, grid_w) = (h / patch, w / patch);
                    TurnImage {
                        key,
                        grid_h,
                        grid_w,
                        rows: (grid_h / merge) * (grid_w / merge),
                        pixels: Some((pixels, h, w)),
                        embeds: None,
                    }
                }
                Err(e) => {
                    fail(stdout, &format!("image {i}: {e}"));
                    return;
                }
            },
        };
        eprintln!(
            "[vl-v2] image {i}: {}x{} patches -> {} tokens{}",
            img.grid_h,
            img.grid_w,
            img.rows,
            if img.embeds.is_some() { " (cached embedding)" } else { "" }
        );
        images.push(img);
    }
    let rows: Vec<usize> = images.iter().map(|i| i.rows).collect();

    // 2. Placement: expand the serve layer's markers in message order; raw
    //    clients without markers get the images in front of the last user turn.
    let mut history: Option<Vec<Message>> = p.messages_history.map(|h| h.to_vec());
    let mut prompt = p.prompt.to_string();
    let found = match history.as_mut() {
        Some(h) => {
            let mut texts: Vec<&mut String> = h.iter_mut().map(|m| &mut m.content).collect();
            expand_markers(&mut texts, &rows)
        }
        None => expand_markers(&mut [&mut prompt], &rows),
    };
    if found != rows.len() {
        if found != 0 {
            fail(
                stdout,
                &format!("request has {} images but {found} image markers", rows.len()),
            );
            return;
        }
        let lead: String = rows.iter().map(|&r| expanded_marker(r)).collect();
        match history.as_mut() {
            Some(h) => {
                let Some(last_user) = h
                    .iter_mut()
                    .rev()
                    .find(|m| m.role == hipfire_runtime::prompt_frame::Role::User)
                else {
                    fail(stdout, "image request without a user message");
                    return;
                };
                last_user.content = format!("{lead}{}", last_user.content);
            }
            None => prompt = format!("{lead}{prompt}"),
        }
    }

    // 3. Arm the session; H1/H2 inside `generate_dflash` pick it up.
    SESSION.with(|s| {
        *s.borrow_mut() = Some(Session {
            images,
            image_pad_id,
            spatial_merge: merge,
            prompt_tokens: Vec::new(),
            spans: Vec::new(),
            shift: 0,
            positions: Vec::new(),
            planned: false,
            installed: false,
        })
    });

    // 4. The text/MTP path.
    let handled = crate::qwen::generate_dflash(
        m,
        gpu,
        stdout,
        id,
        &prompt,
        p.system_prompt,
        p.max_tokens,
        p.max_think_tokens,
        p.assistant_prefix,
        None,
        None,
        p.tools,
        history.as_deref(),
        p.stop,
        p.temp,
        p.top_p,
        p.top_k.map(|k| k as usize).unwrap_or(0),
        p.min_p.unwrap_or(0.0),
        p.cactus_delta,
        p.request_seed as u64,
        p.reasoning_effort,
        p.enable_thinking,
    );
    turn_finish(m, gpu);
    if !handled {
        // Draft-context miss (no gen_start was emitted). Image turns have no
        // AR fallback on this path.
        fail(
            stdout,
            "image request does not fit the speculative decoder's context (prompt + max_tokens); lower max_tokens",
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sig(start: usize, key: u64) -> SpanSig {
        SpanSig {
            start,
            rows: 4,
            grid_h: 4,
            grid_w: 4,
            key,
        }
    }

    #[test]
    fn reuse_requires_same_shift_and_images() {
        let old = Resident::Vl {
            shift: 2,
            spans: vec![sig(5, 1)],
        };
        assert_eq!(reusable_prefix(&old, 2, &[sig(5, 1)]), usize::MAX);
        assert_eq!(reusable_prefix(&old, 4, &[sig(5, 1), sig(20, 2)]), 0);
        assert_eq!(reusable_prefix(&old, 2, &[sig(5, 9)]), 5);
        assert_eq!(reusable_prefix(&old, 2, &[sig(6, 1)]), 5);
        assert_eq!(reusable_prefix(&Resident::Text, 2, &[sig(5, 1)]), 0);
        assert_eq!(reusable_prefix(&Resident::Text, 0, &[sig(5, 1)]), 5);
        assert_eq!(reusable_prefix(&Resident::Unknown, 2, &[sig(5, 1)]), 0);
    }

    #[test]
    fn markers_expand_in_order() {
        let mut a = format!("look {IMAGE_MARKER} and");
        let mut b = format!("{IMAGE_MARKER}then");
        let n = expand_markers(&mut [&mut a, &mut b], &[2, 3]);
        assert_eq!(n, 2);
        assert_eq!(a, format!("look {} and", expanded_marker(2)));
        assert_eq!(b, format!("{}then", expanded_marker(3)));
        assert_eq!(a.matches(IMAGE_PAD).count(), 2);
        assert_eq!(b.matches(IMAGE_PAD).count(), 3);
        // Count mismatch leaves the text alone.
        let mut c = format!("{IMAGE_MARKER}");
        assert_eq!(expand_markers(&mut [&mut c], &[1, 1]), 1);
        assert_eq!(c, IMAGE_MARKER);
    }
}
