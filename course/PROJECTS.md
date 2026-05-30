# Speed Projects — Closing the Gap to llama.cpp

Final-project menu for "Efficient LLM Inference, From the Metal Up." Every project is a
*measured* speed improvement to the quantized decode path, defended with a roofline
argument and guarded by the validation oracle.

## The goalposts (same M3 Max, same Q4_K_M file)

| | decode tok/s | reads/token | effective bandwidth |
|---|---|---|---|
| our engine (current) | **23** | ~379 MB | ~8.7 GB/s |
| **llama.cpp** (target) | **~226** | ~379 MB | ~86 GB/s |
| pure-bandwidth ceiling | ~790 | 379 MB | ~300 GB/s |

**The headline fact that drives every project:** at 23 tok/s we move ~8.7 GB/s — nowhere
near the chip's ~300 GB/s. We are **compute-bound in the dequant**, not bandwidth-bound.
llama.cpp, on the identical bytes, is ~10× faster. So the gold is in *compute* (kernels,
SIMD, int8) first; *bytes* (smaller quant, KV) only pays once you become bandwidth-bound.

## Current bottleneck (measure it yourself first)

The decode kernel (`lib/stubs.c: caml_qwen_qmatvec`) does, per weight row: fp32 dequant
into a scratch buffer + a NEON fp32 dot, GCD-parallelized over row chunks. From the M5
profile, the per-token cost splits roughly: **Q5_0 ≈ 50%** (q/k/o/gate/up projections),
Q8_0 ≈ 18% (the tied `token_embd` lm_head + attn_v), Q6_K ≈ 15% (ffn_down), Q4_K ≈ 11%.

## Shared methodology (required for every project)

- **Profile before optimizing.** Re-add a per-kernel timer (the M5 one was removed in the
  audit) and a tok/s harness. Sub-task `P0` for everyone: a `QWEN_PROF=1` build that prints
  per-ggml-type time and call counts.
- **Validation is the guardrail.** Unless a project states otherwise, `validate.exe` **and**
  `validate_gguf.exe` must still print `MATCH: true` after your change. "Fast but wrong" is
  a fail, and the oracle makes it impossible to hide.
- **Deliverable = number + roofline write-up.** Report before/after tok/s, and explain the
  result against the 23 → 226 → 790 ladder: what was the bottleneck, what did you predict,
  what did you measure, and why is there still a gap?

Tags below: **Difficulty** {Starter, Core, Ambitious} · **Headroom** = realistic share of
the 10× gap to llama.cpp · **Correctness bar**.

---

## P1 — Vectorize the fp32 dequant kernels (NEON)
**Difficulty:** Core · **Headroom:** large · **Correctness bar:** exact (`MATCH: true`)

**Problem.** The dequant functions are scalar bit-twiddling and dominate decode time.
Vectorize them with NEON while keeping fp32 output, so correctness cannot drift.

**Roofline hypothesis.** We are compute-bound; cutting dequant cycles moves tok/s up
roughly linearly until KV/weight bandwidth (~hundreds of tok/s) becomes the new wall.
Q5_0 is ~50% of the time, so it's the highest-value target.

**Where in the repo.** `lib/stubs.c`: `dequant_row_q8_0` (trivial — load int8, widen,
multiply by `d`), `dequant_row_q5_0` (5th-bit reconstruct — the int8 branch in M5 git
history has NEON bit-extraction to crib), `dequant_row_q4_k`, `dequant_row_q6_k`. The dot
is already NEON (`qwen_dot_f32`); a fused dequant-and-accumulate is the stretch.

**Steps.** (1) Vectorize Q8_0, measure. (2) Vectorize Q5_0 (biggest win). (3) Q6_K/Q4_K.
(4) Optionally fuse dequant directly into the dot to skip the scratch round-trip.

**Validation gate.** `validate_gguf.exe` MATCH true (fp32 dequant is bit-identical);
report per-type time before/after.

**Stretch.** Compare auto-vectorization (`-O3 -ffast-math` on a clean scalar loop) vs hand
intrinsics — does the compiler already get Q8_0?

---

## P2 — Int8 dot product, matching llama.cpp exactly
**Difficulty:** Ambitious · **Headroom:** largest · **Correctness bar:** match llama.cpp

**Problem.** This is *why* llama.cpp is 10× faster: it quantizes the activation vector to
int8 (Q8_0/Q8_1) once per matvec and uses `vdotq_s32` (16 int8 MACs/instruction),
skipping fp32 dequant entirely. The M5 attempt did this and hit ~27 tok/s but **diverged**
from llama.cpp after ~30 tokens because its activation rounding differed. Your job: make it
*both* fast *and* bit-faithful.

**Roofline hypothesis.** int8 dot has ~4–8× the MAC throughput of fp32 fma and removes the
dequant-to-fp32 store; this is the single change that should recover most of the 10×.

**Where in the repo.** `lib/stubs.c: caml_qwen_qmatvec` (the M5 git history has
`qwen_quantize_x`, `qdot_q8_0`, `qdot_q5_0` as a starting point); `lib/quant.ml`.

**Steps.** (1) Replicate llama.cpp's exact activation quant: per-block scale, the **Q8_1**
sum-correction term, and its rounding (`nearbyint` / round-half-to-even — verify which).
(2) Reconstruct int8 weights per format and `vdotq_s32`. (3) Fold the Q5_0 `-16` / Q4_K
min terms via the Q8_1 sum, as llama does, to keep arithmetic identical.

**Validation gate.** `validate_gguf.exe` MATCH true vs `scripts/llama_ref_gguf.txt`. (If you
can't reach exact, document the divergence point and fall back to top-1 agreement + report
the tok/s — partial credit.)

**Stretch.** Use `vmmlaq_s32` (int8 matrix-multiply, M2+) for a 2×-wide inner kernel.

---

## P3 — Threading & per-call overhead
**Difficulty:** Core · **Headroom:** medium · **Correctness bar:** exact

**Problem.** ~170 `qmatvec` calls/token, each releases the OCaml runtime and launches a
fresh `dispatch_apply`. Reduce overhead and improve parallel scaling.

**Roofline hypothesis.** If dequant compute is the limiter, near-linear scaling to the
P-core count should multiply throughput; today scaling is sub-linear (M5 saw ~2× before the
chunk-count fix). Quantify efficiency = speedup / cores.

**Where in the repo.** `lib/stubs.c: caml_qwen_qmatvec` (the `dispatch_apply` block, the
`rows <= 256` serial path, `n_chunks = 32`).

**Steps.** (1) Sweep chunk count; find the granularity knee. (2) Replace per-call GCD with a
persistent worker pool / `dispatch_apply` over a fused work-list. (3) P-core pinning via
`pthread_set_qos_class_self_np`. (4) Decide which small matvecs should stay serial.

**Validation gate.** MATCH true; report tok/s vs thread count (a scaling curve).

**Caution (from the audit).** Do all OCaml-raising validation *before* releasing the
runtime; never call `caml_*` from inside a GCD block.

---

## P4 — Fewer bytes: requantize to a lower-bpw format
**Difficulty:** Core · **Headroom:** medium (only after P1/P2) · **Correctness bar:** top-1 + perplexity

**Problem.** The shipped file is ~5.5 bpw (lots of Q5_0). Build a lower-bpw GGUF and measure
the bandwidth payoff and the quality cost.

**Roofline hypothesis.** Once compute-bound-ness is fixed (P1/P2), tok/s ∝ 1/bytes. Going
~5.5 → 4.5 bpw should give ~1.2× *if* you're bandwidth-bound — and ~nothing if you aren't.
That "nothing until bandwidth-bound" result is itself the lesson; sequence P4 after P1/P2.

**Where in the repo.** `llama-quantize` to produce e.g. Q4_0 or a Q4_K-heavy build; extend
the kernels/`Quant.row_bytes` if a new type appears; `llama-perplexity` for quality.

**Steps.** (1) Quantize to Q4_0 (single simple format) and/or Q4_K_S. (2) Ensure the kernel
covers it. (3) Measure tok/s and perplexity vs the Q4_K_M baseline.

**Validation gate.** Greedy top-1 agreement with llama.cpp *on the same new file* + a
perplexity number (quality is now an explicit tradeoff, not a pass/fail).

**Stretch.** An IQ-quant (codebook) format — note the gather-style access cost.

---

## P5 — Speculative / prompt-lookup decoding
**Difficulty:** Ambitious · **Headroom:** large (effective) · **Correctness bar:** exact (greedy identical)

**Problem.** A draft proposes K tokens; the model verifies them in one batched forward;
accepted tokens cost nothing extra. **For greedy decoding this is bit-exact** — the output
is unchanged — so the existing oracle still grades it.

**Roofline hypothesis.** Decode is bandwidth-bound per *step*; verifying K tokens in one
pass amortizes the weight read across K, so effective tok/s ≈ (accepted/step) × baseline.
"Prompt-lookup" (draft = longest match in the prompt/history) needs no model and wins big on
repetitive/structured text.

**Where in the repo.** `lib/generate.ml: run` (the decode loop); needs a small batched
forward (`forward_verify : tokens -> logits[]`), shared with P6. Draft: a length-N
prompt-lookup table, or Qwen2.5-0.5B drafting for a bigger target.

**Steps.** (1) Build a K-token batched forward. (2) Implement prompt-lookup drafting. (3)
Accept/reject by comparing draft tokens to argmax; resume from the first mismatch. (4)
Measure accepted-tokens/step and effective tok/s.

**Validation gate.** Output **identical** to plain greedy (assert token-for-token); report
the speedup and acceptance rate.

**Stretch.** A trained tiny draft model; measure the draft-cost vs acceptance tradeoff.

---

## P6 — Batched prefill = GEMM/AMX (time-to-first-token)
**Difficulty:** Core · **Headroom:** ~10× on *prefill* (different metric) · **Correctness bar:** exact

**Problem.** Prefill is currently the decode loop run per prompt token — GEMV, re-streaming
every weight T times. Batch the T prompt tokens into one GEMM so prefill becomes
compute-bound and rides AMX.

**Roofline hypothesis.** GEMV intensity ≈ 1 (bandwidth-bound); GEMM over a `[T×hidden]`
tile reuses each weight across T tokens → intensity ≈ T → compute-bound for T ≳ 100, where
`cblas_sgemm` (AMX) runs near peak. ~10× faster prompt processing, ~0 effect on decode
tok/s — be clear which metric you're moving.

**Where in the repo.** `lib/model.ml` (add `forward_prefill : tokens -> logits`), a new
`caml_qwen_sgemm` stub in `lib/stubs.c` (`cblas_sgemm`, `CblasNoTrans`/`CblasTrans`),
`Generate.run` prefill loop. (This is Lab B of the attention-at-scale session;
see `course/session-attention-at-scale.md`.)

**Validation gate.** Last-position prefill logits match the token-by-token reference; report
TTFT (time-to-first-token) vs prompt length.

**Stretch.** GEMM the attention scores too (`Q·Kᵀ`, `·V`) and compare naive `T×T` vs a
flash-tiled prefill (memory axis).

---

## P7 — KV-cache quantization (long-context throughput)
**Difficulty:** Core · **Headroom:** grows with context · **Correctness bar:** tolerance + quality note

**Problem.** Per-token decode reads weights **plus the KV cache**; the KV term grows with
context and eventually dominates. Store K/V in fp16 or int8 to cut it.

**Roofline hypothesis.** bytes/token = weights + `2·layers·n_kv·context·head_dim·bytes`.
fp16 KV halves, int8 quarters, the context-dependent term — a measurable tok/s gain that
*increases* with context length.

**Where in the repo.** `lib/attention.ml: Kv_cache` (currently fp32 `Tensor.t`); the write
and the score/`·V` reads in `forward`.

**Steps.** (1) Store K/V as fp16 (or int8 + per-token scale). (2) Dequant on read in the
attention dot. (3) Sweep context length; plot tok/s-vs-context with/without.

**Validation gate.** Output within tolerance of the fp32-KV run on a fixed prompt; report
the bandwidth reduction and any quality drift at long context.

**Stretch.** Sliding-window cache (bounded memory) and attention sinks (StreamingLLM).

---

## Suggested pairings & sequencing

- **Solo, high-confidence:** P1 or P3 (exact, measurable, self-contained).
- **Standout:** P2 (the real 10×) or P5 (exact *and* clever).
- **Sequencing lesson:** do P1/P2 *before* P4/P7 — bytes don't help until you're
  bandwidth-bound. A strong report can show exactly that crossover.
- **Shared infra:** P5 and P6 both need a batched forward — a natural two-person split.

## Reference points to cite
`original_plan.md` §4 (roofline, GEMM vs GEMV) and §3 (quantization); llama.cpp
`ggml-quants.c` and its `vec_dot_*` int8 kernels; `scripts/ggml_block_layouts.md`;
Salykova's CPU matmul tutorial; the FlashAttention papers (for P6 stretch).
