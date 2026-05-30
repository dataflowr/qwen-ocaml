# Session: Attention at Scale — Flash-Attention & Long Context

*Advanced session for "Efficient LLM Inference, From the Metal Up." Slots in as
Week 8, after the kernel work (M5) and before tokenization/sampling.*

## The hook (start here, it reframes everything)

Students arrive thinking "flash-attention = the trick that makes attention fast."
Open by showing them this repo's `lib/attention.ml`: it decodes **one token at a
time**, so per step it computes one query against `T` cached keys and forms a
length-`T` score vector — never a `T×T` matrix. **Flash-attention's signature win,
not materializing the `N×N` score matrix, has nothing to buy here.**

That contradiction is the lesson: attention has *two* scaling regimes with *different*
bottlenecks, and "flash-attention" and "long context" live in different ones.

| Regime | When | What you compute | Bottleneck | The lever |
|---|---|---|---|---|
| **Prefill** | the prompt, all at once | `T_q × T_k` scores | compute + the `O(N²)` score matrix | **flash-attention** (online softmax, tiling) |
| **Decode** | one token per step | `1 × T_k` scores | **KV-cache bandwidth** (grows with context) | GQA, KV quantization, sliding window |

The roofline from Week 1 returns with a second-order term: in decode, total bytes/token
= `weights + KV_cache(context)`. At short context, weights dominate (what M2–M5 optimized).
At long context, the KV cache takes over — and *that's* the long-context problem.

---

## Lecture

### Part 1 — Online softmax (the actual core of flash-attention)

Derive it. Standard softmax over scores `s_1..s_n` needs all scores in memory (two
passes: max, then exp-sum). **Online softmax** computes the same result in one
streaming pass with O(1) state by carrying a running max `m` and running denominator
`l`, rescaling when a new max appears:

```
m = -inf ; l = 0 ; acc = 0      (acc is the weighted value accumulator, length d)
for each key/value (k_j, v_j):
    s     = (q · k_j) * scale
    m_new = max(m, s)
    c     = exp(m - m_new)        # correction for the old running totals
    p     = exp(s - m_new)
    l     = l*c + p
    acc   = acc*c + p * v_j
    m     = m_new
out = acc / l
```

This is *exactly* flash-attention's inner recurrence. Key points to land:
- It is **numerically identical** to two-pass softmax (prove the rescale telescopes).
- It needs **O(d) state, not O(T)** — no score array. This is the IO-awareness idea.
- Tiling for prefill = run this recurrence over *blocks* of keys, and over *blocks*
  of queries in the outer loop, keeping a query tile's `(m, l, acc)` in registers/L1.

### Part 2 — Why it's fast (and the honest CPU caveat)

On a **GPU**, the win is avoiding round-trips of the `N×N` matrix to HBM (keep tiles in
SRAM) — FlashAttention-1/2/3. On a **CPU**, the analogue is L1/L2 vs DRAM and never
materializing the `N×N` matrix at all. Be explicit and honest (the course ethos): at
small `T` on a CPU you may see little *speed* change from flash itself — its demonstrable
win is **memory scaling (O(N) vs O(N²)) and the algorithm**, which is ISA-independent.
Don't oversell a speedup the hardware won't give.

### Part 2b — What actually makes prefill *fast*: GEMM, not flash

This is the part students most need corrected. Prefill speed does **not** come from
flash-attention — it comes from **batching the projections into GEMM**.

Recall the roofline (Week 1, `original_plan.md` §4): decode reads each weight once to
multiply a single vector → arithmetic intensity ≈ 1 → **bandwidth-bound**. The repo's
token-by-token prefill inherits exactly this: for `T` prompt tokens it calls the per-layer
GEMV `T` times, **re-streaming every weight from DRAM `T` times**.

Batch the `T` tokens into one activation tile `X[T × hidden]` and a projection becomes a
single **GEMM** `Y = X · Wᵀ` → `[T × out]`. Now each weight is loaded once and reused
across all `T` tokens → arithmetic intensity ≈ `T` → **compute-bound for `T ≳ 100`**, and
that is precisely the regime where Apple's AMX (via `cblas_sgemm`) runs near peak. *This*
is the order-of-magnitude prefill speedup, and it is independent of how you compute
attention.

So the two wins are orthogonal and students should be able to name which is which:

| Change | Axis it moves | Mechanism |
|---|---|---|
| token-by-token → **batched GEMM** | **speed** (≈10×+ prefill) | weight reuse across T, AMX `sgemm`, intensity 1→T |
| naive → **flash** attention | **memory** (O(N²)→O(N)) | online softmax, never materialize `T×T` |

The attention block itself can also be GEMMs (`scores = Q·Kᵀ`, then `·V`), so the naive
batched path is "all GEMM + a materialized `T×T`"; the flash path keeps the GEMM
projections but tiles the attention. That is why Lab B varies *both* axes.

### Part 3 — Long context = the KV cache

- **Size.** KV cache = `2 · layers · n_kv_heads · context · head_dim · bytes`. Compute
  it for Qwen2.5-0.5B (GQA, n_kv=2) vs Qwen3-0.6B (n_kv=8) — Qwen3's larger KV is a
  deliberate quality/cost choice. Show the GiB at 32K context.
- **GQA** (already in `attention.ml`) is the first lever: KV cache shrinks by
  `n_kv/n_q`. Measure it directly — it's the cheapest long-context win and it's free in
  the repo.
- **KV quantization** — store K/V in fp16 or int8; halves/quarters the dominant
  long-context bandwidth term.
- **Sliding-window / streaming attention** — bound the cache at `W`; O(1) memory per
  step at the cost of forgetting. Discuss attention sinks (StreamingLLM).
- **Context extension** — the model was *trained* at some length; running longer needs
  RoPE rescaling: position interpolation (linear), NTK-aware, YaRN. The repo's
  `Ops.rope_tables` is where this lives. Without it, quality collapses past the trained
  length even if nothing crashes.

---

## Lab — three parts (one anchored, two as mini-project / homework)

All parts grade the same way as the rest of the course: **the reference oracle**. The
attention change must not alter the model's output — `validate.exe` /
`validate_gguf.exe` must still print `MATCH: true`. Refactoring attention while keeping
bit-stable greedy output *is* the correctness bar.

### Lab A (in-session, required) — flash-style **decode** attention

Convert `Attention.forward` from "fill a length-`T` score scratch, then `softmax`,
then weighted sum" to the **streaming online-softmax** recurrence above. Result: the
per-head score buffer disappears (O(d) state instead of O(T)).

- **Auto-grade:** `validate.exe` and `validate_gguf.exe` still `MATCH: true`.
- **Lesson:** you just implemented flash-attention's core for the decode case — and saw
  it's an O(T)→O(d) *memory* change, with little speed change at these sizes. That's the
  point.

### Lab B (mini-project) — batched **prefill**: GEMM for speed, flash for memory

Add a prefill path that processes all `T` prompt positions in one pass instead of the
token-by-token loop. Build **three** versions so the two axes (speed vs memory) are
isolated:

1. **Baseline — token-by-token** (the repo as-is). Per-layer projections are `T`
   separate GEMV calls; weights re-streamed `T` times. *Bandwidth-bound.*
2. **Batched GEMM, naive attention.** Stack the `T` tokens into `X[T × hidden]`; do every
   projection (q/k/v/o, gate/up/down, and the final logits) as one **`cblas_sgemm`**
   `Y = X·Wᵀ` (AMX). Attention materializes the `T×T` causal score matrix (`Q·Kᵀ` GEMM →
   masked row-softmax → `·V` GEMM). *Compute-bound and fast; O(T²) attention memory.*
3. **Batched GEMM, flash attention.** Same GEMM projections, but tile the attention with
   online softmax so the `T×T` matrix is never materialized. *Fast **and** O(T) memory.*

Implement the GEMM via a small `caml_qwen_sgemm` stub (see below). Causal masking in the
naive path: zero the strictly-upper triangle of the score tile before softmax; in the
flash path, skip key blocks beyond the query position.

- **Auto-grade:** all three produce prefill logits matching the token-by-token reference
  (same oracle as `validate.exe`); the flash path's peak attention scratch is
  `O(T·d + block²)`, not `O(T·T_k)`.
- **Deliverable — two plots + one paragraph:**
  - **prefill time vs `T ∈ {256, 1k, 4k, 8k}`** for all three → the **GEMM/AMX speedup**
    (1→2) is the big jump; (2→3) barely moves the clock. *Speed comes from GEMM.*
  - **peak attention memory vs `T`** for (2) and (3) → O(T²) vs O(T). *Memory comes from
    flash.*
  - Roofline reading: at which `T` does prefill cross from bandwidth- to compute-bound,
    and does the measured `sgemm` throughput approach the machine's GFLOP/s ceiling?

### Lab C (homework) — long-context decode & a mitigation

Raise `max_seq` and resize the RoPE tables; sweep context length and measure **tok/s vs
context**, showing the crossover where KV-cache reads rival weight reads (the second-order
roofline term). Then implement **one** mitigation and measure it:
- fp16 KV cache (halve KV bandwidth), **or**
- sliding-window attention (bounded cache), **or**
- (stretch) NTK/linear RoPE scaling to generate coherently past the trained length.

- **Auto-grade:** output stays within tolerance of the unmitigated run on short context;
  report the measured KV-bandwidth reduction or the max coherent context reached.

---

## Concretely, where in the repo

- `lib/attention.ml` — `Kv_cache` and `forward`; Labs A & B live here. Note the current
  module-global scratch (flagged in the audit) — Lab A removes the score scratch entirely.
- `lib/ops.ml` — `rope_tables` for Lab C context extension; `softmax_inplace` is the
  two-pass baseline students replace with the online recurrence.
- `lib/model.ml` — `max_seq`, the `pos < max_seq` guard, and the prefill loop in
  `Generate.run` that Lab B batches. Add a `forward_prefill : t -> tokens:int array ->
  Tensor.t` returning the last-position logits, with scratch tiles sized `[T × dim]`.
- `lib/stubs.c` — Lab B adds a GEMM stub alongside the existing `caml_qwen_sgemv`:
  ```c
  /* Y[M×N] = X[M×K] · W[N×K]^T  (row-major); one AMX call replaces M GEMVs. */
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
              M /*=T*/, N /*=out*/, K /*=hidden*/,
              1.0f, X, K, W, K, 0.0f, Y, N);
  ```
  (For the quant path, students either dequantize the weight tile once and `sgemm`, or
  keep GEMV per row — a good discussion of why prefill wants dense GEMM.)
- `bin/validate*.exe` — the unchanged oracle for every part.

## Readings

- FlashAttention (Dao et al., 2205.14135) and FlashAttention-2 (2307.08691) — read for
  the online-softmax recurrence and the IO/tiling argument, not the CUDA.
- "Self-attention does not need O(n²) memory" (Rabe & Staats, 2112.05682) — the cleanest
  statement of the memory result, hardware-agnostic.
- GQA (2305.13245); StreamingLLM / attention sinks (2309.17453); YaRN (2309.00071);
  position interpolation (2306.15595).
- GEMM (for Part 2b / Lab B): Salykova's CPU matmul tutorial (`salykova.github.io/matmul-cpu`)
  and the BLIS five-loop paper (Van Zee et al., IPDPS 2014); `original_plan.md` §4 "GEMM
  vs GEMV" for the prefill-is-GEMM argument.

## What's still out of scope (name it)

GPU flash-attention kernels (SRAM tiling, warp specialization, FA-3), paged-KV serving
(vLLM), and continuous batching. This session teaches the *algorithm and the memory
argument* on CPU; the GPU kernel engineering is a follow-on.
