# Efficient LLM Inference, From the Metal Up
### A hands-on course built on a from-scratch OCaml Qwen engine

**Audience.** Graduate ML students (MVA-style) who know transformers mathematically
and use PyTorch, but have never seen what actually happens when a model runs.

**The question the course answers.** *You can write the attention equation. So why
does a 0.5B model run at 0.2 tokens/s in the obvious implementation and 23 tokens/s
in a good one — and which of those gaps is about math, and which about memory?*

**Prerequisites.** Transformers at the level of "I can derive attention"; basic C
literacy; willingness to learn just-enough OCaml (no prior FP required — the typed
core is small and the labs give skeletons).

**Hardware.** Any Apple Silicon Mac (the repo targets arm64 + Accelerate). x86 +
OpenBLAS works for the fp32 path with minor changes; note this if your lab is mixed.

**Format.** 10 sessions. Each = one lecture + one lab. Every lab has an **objective
auto-grade**: a `validate_*.exe` that diffs the student's engine against a trusted
reference (HuggingFace fp32 logits, or llama.cpp greedy tokens). "It runs" and "it's
correct" are graded separately — that distinction *is* a learning outcome.

**The spine.** The repo's five milestones are five labs that take one model from
*correct-but-slow* to *quantized-and-fast*, measuring the speedup at each step. The
numbers below are the real measurements from the reference build (Apple M3 Max).

| Milestone | What changes | tok/s | The lesson |
|---|---|---|---|
| M1 | correct fp32 forward | ~0.2 | correctness ≠ done; validate against a reference |
| M2 | BLAS / AMX matmul | ~18 | decode is **bandwidth-bound**; BLAS is the floor |
| M3 | Q4_K GGUF quantization | ~7→exact match | quantization = the real lever (bytes/weight) |
| M4 | Qwen3 architecture | — | the details (QK-norm, bias, head_dim) that break loaders |
| M5 | SIMD + parallel kernel | ~23 | fidelity vs speed; profile before you optimize |

---

## Assessment

- **5 labs (60%)** — auto-graded: each must reach `MATCH: true` against its reference,
  plus a tok/s target for the performance labs (M2, M5).
- **Code-review exercise (15%)** — Week 9: students audit a seeded-bug branch.
- **Final project (25%)** — one extension (menu at the end), with a written roofline
  analysis: *where did the bytes go, and did your change move the predicted ceiling?*

The grading philosophy mirrors the repo: a reference oracle, a fixed prompt, and a
diff. Students cannot "eyeball" their way to a pass.

---

## Week 1 — The inference problem & the roofline

**Lecture.** Why inference ≠ training. Prefill vs decode. Arithmetic intensity and
the roofline model (Williams et al. 2009). The central result:
`decode tok/s ≈ memory_bandwidth / model_bytes`. Derive the ceiling for the lab
hardware; predict the fp16 vs Q4 ceilings *before writing any code*.

**Lab 0 (warm-up, no engine yet).** Run the three reference paths that ship with the
repo: HF `transformers` (fp32), `llama-simple` (Q4_K GGUF). Measure tok/s. Compute
your machine's bandwidth ceiling and compare. Stand up the OCaml toolchain and run
`dune test`.

**Objective.** Internalize that the headline number is set by bytes moved, not FLOPs.

**Reading.** `original_plan.md` §4 (roofline); Williams/Waterman/Patterson CACM 2009.

---

## Week 2 — The forward pass, implementation-detail view

**Lecture.** The decoder block as *code*: RMSNorm (eps placement, fp32 accumulate),
RoPE and the **HF rotate_half convention** (the single most common from-scratch bug),
GQA (broadcast vs materialize), SwiGLU, the KV cache as the only growing buffer,
tied embeddings. Less "what is attention," more "the 5 things that silently corrupt
your logits."

**Lab 1 = Milestone 1.** Given the safetensors loader and op skeletons, implement the
fp32 forward in `lib/model.ml`. Validate against dumped HF logits.

**Auto-grade.** `validate.exe` → `MATCH: true`, step-0 logit diff < 1e-3.

**Teaching move.** Hand them a version with RoPE in the *interleaved* (paper) convention
instead of rotate_half. Their logits won't match. Debugging it is the lesson.

---

## Week 3 — From correct to fast (I): BLAS, AMX, and the FFI boundary

**Lecture.** Why a naive triple-loop matvec wastes the machine. BLAS GEMV; Apple AMX
via Accelerate; the OCaml↔C boundary (Bigarray as shared memory, `[@@noalloc]`,
`caml_release_runtime_system`). Why even "pure" tensor libraries call C here.

**Lab 2 = Milestone 2.** Replace the naive matvec with a `cblas_sgemv` C stub; add
bulk bf16→f32. Measure the jump (≈0.2 → ≈18 tok/s) and explain it with the roofline.

**Auto-grade.** Still `MATCH: true`; tok/s ≥ a threshold (e.g. 10× the naive baseline).

**Teaching move.** The repo's real `[@@noalloc]` + runtime-release **segfault** is a
ready-made "read the FFI contract" exercise.

---

## Week 4 — Quantization, the theory

**Lecture.** Why quantization is *the* lever (it divides the denominator of the
roofline). Bits-per-weight; symmetric vs asymmetric; per-block scales; K-quants and
double quantization; calibration (max-abs, MSE). The GGUF container format and why
it's mmap-designed. Activation quantization (Q8_0/Q8_K) and int8 dot products.

**Lab (paper + small code).** Implement and unit-test one dequant kernel (Q8_0, the
simplest: `w = d·q`) against `gguf.quants`. Compute the bpw and predicted speedup for
Q8_0 vs Q4_K. No full integration yet.

**Reading.** `original_plan.md` §3; `scripts/ggml_block_layouts.md`; llama.cpp
`ggml-quants.c`; the K-quants PR (#1684).

---

## Week 5 — Quantization, the practice = Milestone 3

**Lecture.** Reading a binary format from a spec: GGUF header, metadata, tensor table,
alignment, mmap. The dims convention (`ne0=cols, ne1=rows`). Tied `token_embd` as the
lm_head.

**Lab 3 = Milestone 3.** Write the GGUF parser and the dequant kernels (Q4_K/Q5_0/
Q6_K/Q8_0); load the quantized model; validate greedy output against llama.cpp.

**Auto-grade.** `validate_gguf.exe` → char-for-char match to llama.cpp.

**Teaching move.** Have them *discover* that the "Q4_K_M" file is actually a **mix** of
formats (it is). Lesson: the name on the box ≠ the bytes inside.

---

## Week 6 — Architectural variation = Milestone 4

**Lecture.** What actually differs across model families and why loaders break:
Qwen3's **QK-Norm** (per-head RMSNorm pre-RoPE), removed QKV bias, and **decoupled
head_dim** (q_proj output ≠ hidden_size). Tying decisions. How to make one forward
pass serve multiple architectures cleanly (the `option`/config-flag pattern).

**Lab 4 = Milestone 4.** Extend the forward to Qwen3-0.6B; validate against HF.

**Auto-grade.** `validate_qwen3.exe` → `MATCH: true`.

**Reading.** Qwen2.5 (2412.15115) and Qwen3 (2505.09388) papers; QK-Norm (Dehghani 2023).

---

## Week 7 — SIMD, parallelism, and the fidelity/speed tradeoff = Milestone 5

**Lecture.** NEON SIMD; threading the GEMV (GCD here; Domains/Domainslib as the OCaml
alternative); int8 dot products (`vdotq_s32`). The crucial systems lesson: **profile
before you optimize** — the bottleneck here was parallel scaling, not the dot.

**Lab 5 = Milestone 5.** Make the quant path beat the fp32 baseline (≈18 tok/s).

**Auto-grade.** tok/s > fp32 baseline **AND** `validate_gguf.exe` still matches.

**Teaching move (the centerpiece).** The int8-activation kernel is *faster* (≈27 tok/s)
but its rounding **diverges** from llama.cpp after ~30 tokens. Students must confront
the tradeoff: do you ship the fast-but-divergent kernel, or the exact one? This is the
real engineering judgment the course is about.

---

## Week 8 — Attention at scale: flash-attention & long context

**Lecture.** The two scaling regimes of attention and why they have different
bottlenecks: **prefill** (the `O(N²)` score matrix → flash-attention, online softmax,
tiling, IO-awareness) vs **decode** (KV-cache *bandwidth* grows with context → the
roofline's second-order term). The reframing hook: this repo's per-token decode never
materializes a `T×T` matrix, so flash-attention's headline win doesn't apply there —
understanding *why* is the lesson. Long-context levers: GQA (already in the model), KV
quantization, sliding-window/attention-sinks, and RoPE context extension (PI/NTK/YaRN).

Key correction students take away: **prefill speed comes from GEMM batching (AMX), not
from flash-attention** — flash buys *memory*. The lab isolates the two axes.

**Lab.** (A, in-session) Convert decode attention to the **streaming online-softmax**
recurrence — flash-attention's core — turning the per-head score buffer from O(T) into
O(d). (B, mini-project) Batched prefill in three versions — token-by-token GEMV →
**batched `cblas_sgemm`/AMX** with a naive `T×T` attention → batched GEMM with
flash-tiled attention — then two plots: prefill *time* vs `T` (the GEMM/AMX speedup) and
peak attention *memory* vs `T` (the flash O(T²)→O(T) win). (C, homework) Sweep context
length, measure tok/s vs context (KV bandwidth), implement one mitigation (fp16 KV /
sliding window / RoPE scaling).

**Auto-grade.** Attention is refactored but output is bit-stable: `validate.exe` and
`validate_gguf.exe` still `MATCH: true`; plus the memory-scaling and bandwidth plots.

**Full session plan:** `course/session-attention-at-scale.md`.

**Reading.** FlashAttention (2205.14135) & FA-2 (2307.08691); "Self-attention does not
need O(n²) memory" (2112.05682); StreamingLLM (2309.17453); YaRN (2309.00071).

---

## Week 9 — The rest of the pipeline: tokenization & sampling

**Lecture.** Byte-level BPE (the cl100k regex, byte↔unicode map, merge ranks, special
tokens), ChatML, and the sampling pipeline (repetition penalty → temperature → top-k →
top-p → multinomial) and why order matters.

**Lab (open).** Implement top-k/top-p/temperature sampling (the repo ships only greedy
— the sampling module is a deliberate stub) and a determinism test. Optionally harden
the tokenizer's ASCII-only pre-tokenizer toward full Unicode.

---

## Week 10 — Correctness at scale: code review & memory safety

**Lecture.** What review finds in real systems code: the `pos < max_seq` out-of-bounds
(silent corruption past the KV cache), unchecked file inputs (mmap bounds), integer
truncation at the FFI, dead code and dependency hygiene. The validation philosophy as
the through-line of the whole course.

**Exercise.** Students audit a branch with seeded bugs (drawn from the repo's real
audit findings) and submit a findings report + fixes. Re-run all validators.

**Capstone presentations.**

---

## Final project menu (pick one)

Each requires a **roofline write-up**: predict the effect on the bandwidth ceiling,
then measure, then explain the gap.

1. **Beat 23 tok/s without breaking the match** — a SIMD-vectorized Q5_0/Q4_K dequant,
   or a tiling + AMX scheme on dequantized tiles.
2. **Add a quant format** (e.g. Q4_0 or Q6_K-only) end-to-end with its own validator.
3. **Port Qwen3 to the GGUF/quant path** (it currently runs Qwen3 in fp32 only).
4. **Speculative decoding** with a tiny draft model; measure the acceptance/throughput
   tradeoff.
5. **Long-context correctness** — raise `max_seq`, measure KV-cache bandwidth becoming
   a non-trivial fraction of the budget (the roofline's second-order term).
6. **An x86 + OpenBLAS port** of the fp32 path; compare the roofline across machines.

---

## What this course deliberately does *not* cover

Training/backprop, GPU (Metal/CUDA), GPU flash-attention kernels (SRAM tiling, FA-3),
paged-KV serving (vLLM), and distributed serving. Note: the *algorithm* behind
flash-attention (online softmax, tiling) and the long-context KV-bandwidth problem **are**
covered, on CPU, in Week 8 — it's the GPU kernel engineering and serving stack that are
out of scope. It is CPU inference of small models, studied to the metal. Those topics
are natural *follow-on* courses that assume this mental model.

## Why OCaml doesn't get in the way

Students write ML-systems code, not OCaml exercises. The typed core is small and given
as skeletons; the interesting work is in the forward pass, the kernels, and the C
boundary. The language choice buys a clean separation between the *structure* (typed,
OCaml) and the *hot kernels* (C/SIMD) — which is itself the architectural lesson of
modern inference engines.
