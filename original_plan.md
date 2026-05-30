# Building a Qwen inference engine in OCaml from scratch

**Build a Q4_K-quantized Qwen3-0.6B/1.7B (or Qwen2.5-0.5B/1.5B) decoder in pure OCaml on top of `Bigarray` + Lacaml + hand-written SIMD C stubs, with `Domainslib` for parallelism and Apple Accelerate as an optional fast path.** This is the most pragmatic architecture given the OCaml ecosystem in 2026: one direct prior-art OCaml port (`jackpeck/llama2.ml`) exists, the Raven stack (`nx`/`rune`/`kaun`) is now alpha-grade and shows a viable "pure OCaml tensor primitives + transformer on top" path, and OxCaml's SIMD types let you write hot kernels without C — but the **decode-phase bottleneck is memory bandwidth, not OCaml's float overhead**, so the language choice mostly affects the small non-matmul ops (RMSNorm, softmax, RoPE), where careful `[@@noalloc]` C stubs or OxCaml `float32x8` intrinsics close the gap to C completely. The first-order performance ceiling is `bytes_per_weight × memory_bandwidth⁻¹`: on an M2 (~100 GB/s) a Qwen3-1.7B at Q4_K (~1 GB) tops out near **~100 tok/s**; on a DDR5-5600 x86 laptop, ~30–40 tok/s. Quantization, not OCaml micro-optimization, is what gets you there.

This report covers, in order: the exact algorithmic spec of small Qwen models with their non-Llama quirks; tokenizer, KV cache, and sampling; GGUF quantization math; the systems-performance roofline picture; SIMD and threading; the OCaml ecosystem state and gaps; and a five-milestone roadmap with validation at each step.

---

## 1. Qwen architecture: what differs from Llama, and what to get right

The small Qwen models you're targeting are decoder-only pre-norm transformers with **RMSNorm**, **rotary position embeddings**, **grouped-query attention**, and **SwiGLU MLPs** — the standard 2024-era recipe. Across Qwen2.5 → Qwen3 the model shrunk QKV bias and added per-head QK-norm; both decisions are stability-driven and **break drop-in Llama loaders**. The four candidate configs (all verified against the HuggingFace `config.json` files) are:

| Field | Qwen2.5-0.5B | Qwen2.5-1.5B | Qwen3-0.6B | Qwen3-1.7B |
|---|---|---|---|---|
| layers | 24 | 28 | 28 | 28 |
| hidden_size | 896 | 1536 | 1024 | 2048 |
| Q heads / KV heads | 14 / 2 | 12 / 2 | 16 / 8 | 16 / 8 |
| head_dim | 64 | 128 | **128** | 128 |
| MLP intermediate | 4864 | 8960 | 3072 | 6144 |
| rope_theta | 1e6 | 1e6 | 1e6 | 1e6 |
| QKV bias | **True** | **True** | False | False |
| **QK-Norm** | No | No | **Yes** | **Yes** |
| Tied embeddings | True | True | True | True |
| Vocab | 151,936 | 151,936 | 151,936 | 151,936 |

**Five Qwen-specific gotchas that silently corrupt outputs if missed:**

1. **Qwen2/2.5 has `bias=True` on `q_proj`, `k_proj`, `v_proj`** (but not `o_proj`); Llama has no biases anywhere. The HuggingFace `Qwen2Attention` class hard-codes this (see issue #32892). Qwen3 removes them — `attention_bias=False`.
2. **Qwen3 adds RMSNorm on each head's Q and K vectors, before RoPE** (QK-Norm, Dehghani et al. 2023). Each layer has `q_norm` and `k_norm` weight vectors of size `head_dim`, applied per-head. This is the main numerical-stability change; without it Qwen3 wouldn't have trained in bf16.
3. **`head_dim` is decoupled from `hidden_size/num_heads` in Qwen3.** Qwen3-0.6B has `hidden_size=1024`, `num_heads=16`, but `head_dim=128`, so `q_proj` outputs `16×128=2048`, not 1024. The `o_proj` is then `2048 → 1024`. Forgetting this allocates the wrong tensor shapes.
4. **All small Qwen2.5 (≤3B) and Qwen3 (≤4B) tie `lm_head.weight = embed_tokens.weight`.** Larger models don't. Do not store the matrix twice.
5. **`rope_theta = 1,000,000` for all Qwen2.5/Qwen3** (vs Llama-2's 10,000, Llama-3's 500,000), and Qwen uses HuggingFace's **`rotate_half` RoPE convention** that splits the head dim into `[first half, second half]` and pairs index `i` with `i + d/2` — *not* the paper-faithful `(i, i+1)` interleaved pairing. The two are equivalent under a permutation of the head dim, but the stored weights match HF's convention; load and apply RoPE the HF way.

### The exact forward pass (Qwen3-0.6B example, `H=1024, n_q=16, n_kv=8, d=128, I=3072, V=151936`)

```
x = embed_tokens(input_ids)                            [B, T, 1024]
for layer in 1..28:
  r = x
  x = rmsnorm(x, eps=1e-6)                              [B, T, 1024]
  q = q_proj(x); k = k_proj(x); v = v_proj(x)           [B,T,2048] [B,T,1024] [B,T,1024]
  reshape to [B,n_q,T,d] / [B,n_kv,T,d]
  q = q_norm(q); k = k_norm(k)                          # Qwen3 only, RMSNorm on last dim
  q, k = apply_rope(q, k, cos, sin)
  append k,v to KV cache  →  K, V of length T_total
  k_rep = repeat_kv(K, n_q/n_kv); v_rep = repeat_kv(V, n_q/n_kv)
  scores = (q @ k_repᵀ) / sqrt(d) + causal_mask
  attn = softmax(scores) @ v_rep                         [B, n_q, T, d]
  x = r + o_proj(reshape(attn))                         [B, T, 1024]
  r = x; x = rmsnorm(x, eps=1e-6)
  x = r + down_proj(silu(gate_proj(x)) * up_proj(x))    [B, T, 1024]
x = rmsnorm(x); logits = lm_head(x)                     [B, T, 151936]   # weight tied to embed
```

**RMSNorm formula** (no mean subtraction, no bias): `y = w · x / √(mean(x²) + ε)`, computed in fp32 then cast back. **SwiGLU**: `down(silu(gate(x)) ⊙ up(x))`, with `silu(x) = x · σ(x)`. **GQA** broadcasts K/V across `n_q/n_kv` query heads — in code, you don't materialize the repeat; you index into the smaller K/V with `head_idx // group_size`.

Authoritative references: Qwen2 (arXiv:2407.10671), Qwen2.5 (arXiv:2412.15115), Qwen3 (arXiv:2505.09388); Sebastian Raschka's from-scratch Qwen3 in Python at `github.com/rasbt/reasoning-from-scratch/blob/main/reasoning_from_scratch/qwen3.py` — the single best line-by-line architectural reference.

## 2. Tokenizer, sampling, KV cache

**Tokenizer.** Qwen uses **byte-level BPE (BBPE)** of the tiktoken/cl100k family — 151,669 learned tokens padded to 151,936. There are no UNK, BOS, SEP tokens; special tokens are `<|endoftext|>` (151643), `<|im_start|>` (151644), `<|im_end|>` (151645), plus Qwen3's `<think>`/`</think>` for reasoning mode. Chat template is ChatML: `<|im_start|>system\n…<|im_end|>\n<|im_start|>user\n…<|im_end|>\n<|im_start|>assistant\n`. A minimal from-scratch BBPE needs three pieces: the cl100k regex pre-tokenizer (uses `\p{L}`/`\p{N}` so you need a Unicode-aware regex — OCaml's `re` library handles this), the byte-level encoding step (map each byte to the canonical token piece), and a priority-queue merge loop that repeatedly applies the lowest-rank adjacent merge. ~500 lines of OCaml. Karpathy's `minbpe` is the cleanest pedagogical version. Special tokens must be split out **before** BPE so they're not subject to merging.

**KV cache.** Shape `[B, num_kv_heads, T, head_dim]` for both K and V per layer. GQA savings are dramatic for Qwen2.5 (2/14 ≈ 14% of MHA) but only 50% for Qwen3 (8/16). At bf16, full 32K context KV cache is **~400 MiB for Qwen2.5-0.5B, ~3.75 GiB for Qwen3-0.6B/1.7B** — the latter is a deliberate Qwen3 quality/cost tradeoff and matters for laptop memory budgeting. The cache is the only growing buffer in the system; allocate it once at max context length and slice into it. **Prefill** processes the full prompt in one matmul-rich pass (compute-bound for prompts > 100 tokens); **decode** is one token per step (memory-bandwidth-bound; this is what dominates latency).

**Sampling.** The canonical pipeline applies operations on raw logits in this order: repetition penalty (divide positive logits by penalty, multiply negative ones), temperature scaling, top-k mask, top-p (nucleus) mask, softmax, multinomial sample. For top-p: sort logits descending, take cumulative softmax, mask everything past the index where cumulative probability first exceeds p (keeping at least one). Qwen's recommended decoding for Qwen3 reasoning is `T=0.6, top_p=0.95`; for non-thinking, `T=0.7, top_p=0.8`. All five strategies are <50 lines of pure-OCaml code over a `float array`.

## 3. Quantization: pick Q4_K_M, understand Q8_0 and Q4_0 first

**The single most consequential design choice** in the engine. The llama.cpp quantization ecosystem has converged on K-quants (superblocks of 256 weights with double-quantized scales) as the quality/size sweet spot, with **Q4_K_M (~4.5 bpw) as the universally recommended default** — it adds only +0.05 perplexity over fp16 on Llama-7B while shrinking weights 3.5×.

**Block-level layouts you must implement** (sizes in bytes per block):

| Type | Block | Bytes | bpw | Dequant formula |
|---|---|---|---|---|
| Q8_0 | 32 | 34 | 8.5 | `w = d · q`,    d:fp16, q:int8 |
| Q4_0 | 32 | 18 | 4.5 | `w = d · (q − 8)`, q ∈ [0,15] (low nibble = idx j, high nibble = idx j+16) |
| Q4_1 | 32 | 20 | 5.0 | `w = d · q + m`, d,m:fp16, q ∈ [0,15] |
| Q4_K | 256 SB | 144 | 4.5 | `w = d · sc[b] · q − dmin · mn[b]`, with 8×6-bit scales/mins packed in 12 bytes |
| Q6_K | 256 SB | 210 | 6.56 | `w = d · scales[b] · (q − 32)`, 4+2 bit split (`ql`, `qh`) |
| Q8_K | 256 SB | 292 | 9.13 | `w = d · q`, fp32 d, used only as activation format |

**Two crucial implementation insights** from llama.cpp's `ggml-quants.c`:

1. **Weights are never materialized to fp16/fp32 in memory.** The hot loop loads a weight block (e.g. 18 B Q4_0), unpacks nibbles into int8 registers via shifts/masks, subtracts the symmetric center (e.g. `−8`), and does an int8×int8 dot product into an int32 accumulator. Only at end-of-block do you multiply by `d_weight · d_activation` (fp32) and add to the per-output-row accumulator. This is what makes quantization a memory-bandwidth win rather than a CPU-cycles loss.

2. **Activations are dynamically quantized to Q8_0 (or Q8_K for K-quants) right before each matmul.** This is what enables int8×int8 → int32 fused dot products (ARMv8.2 `vdotq_s32`, AVX-VNNI `vpdpbusd`, AVX2's slower 3-instruction `vpmaddubsw`+`vpmaddwd` sequence). The activation quantization error is negligible (you're quantizing one well-conditioned vector at a time), but the speedup is 2–4× over fp16×int8 paths.

**Asymmetric (`type-1`) vs symmetric (`type-0`)**: symmetric stores one scale per block (`w = d·q`); asymmetric adds a min/zero-point (`w = d·q + m`). Asymmetric is worth +0.5 bpw when the block's weight distribution has nonzero mean — common for MLP outputs and post-softmax activations. K-quants extend this with **double quantization**: a superblock's 8 (or 16) sub-block scales are themselves quantized to 6 bits and rescaled by one fp16 super-scale, dropping per-weight scale overhead from 0.5 to ~0.06 bpw.

**Calibration**: max-abs (Q*_0), min-max (Q*_1), or MSE-optimal search around the max-abs estimate (`make_qx_quants` in ggml-quants.c, used by K-quants). For IQ-quants (lookup-table codebooks below 4 bpw), you need an **importance matrix** captured from activations on a calibration set — these are CPU-slow due to gather-style memory access; skip for a first implementation.

**GGUF file format.** Magic `GGUF`, version 3, little-endian throughout. Header is `u32 magic | u32 version | u64 tensor_count | u64 kv_count`, then `kv_count` typed key-value metadata pairs (13 value types incl. arrays/strings), then `tensor_count` tensor info entries (`name | n_dims | dims[] | dtype | offset_in_data_section`), then padding to alignment (default 32 bytes; configurable via `general.alignment`), then bulk tensor data. The format is **mmap-designed** — tensor offsets are multiples of alignment, tensor data is contiguous. ~200 lines of OCaml using `Bytes.get_int64_le` and friends. Spec: `github.com/ggml-org/ggml/blob/master/docs/gguf.md`; canonical reader at `github.com/ggml-org/llama.cpp/blob/master/gguf-py/gguf/gguf_reader.py`.

**Safetensors** is simpler but doesn't carry tokenizer or quantization metadata: `u64 header_size | JSON header | raw tensor blob`. Use it for the unquantized HF dump from which you'll quantize, then write GGUF as your runtime format.

**Pick GGUF as your primary format** because it embeds the tokenizer (`tokenizer.ggml.*` keys), quantization blocks, and all model metadata in one file. The Q4_K_M variant of Qwen3-1.7B from `bartowski` or `lmstudio-community` on HF is your first target.

## 4. The performance picture: decode is bandwidth-bound

Apply the **roofline model** (Williams, Waterman, Patterson, CACM 2009) to LLM inference and a clear picture emerges. During prefill, weights are reused across all T prompt tokens, giving arithmetic intensity ≈ T FLOPs/byte — for T > 100 this is squarely compute-bound on any modern CPU. During decode, each step reads the entire weight tensor exactly once to multiply by a single activation vector: arithmetic intensity is **~1 FLOP/byte for fp16, ~2 for Q4** — far below the ridge point of any CPU, so **decode throughput is purely `memory_bandwidth / model_size_bytes`**.

This single fact dictates the entire optimization order. **First-order tok/s estimates** for the candidate targets:

| Hardware | Memory BW | Qwen3-1.7B @ Q4_K (~1 GB) ceiling |
|---|---|---|
| Apple M1 | 68 GB/s | ~65 tok/s |
| Apple M2 | 100 GB/s | ~95 tok/s |
| Apple M3 Pro | 150 GB/s | ~140 tok/s (note: M3 Pro narrowed the bus vs M2 Pro's 200 GB/s — regression) |
| Apple M4 | 120 GB/s | ~115 tok/s |
| DDR5-5600 laptop | ~90 GB/s | ~85 tok/s |
| LPDDR5X-7500 laptop | ~120 GB/s | ~115 tok/s |
| M2 Ultra | 800 GB/s | ~750 tok/s |

In practice, llama.cpp CPU-only reaches ~50–70% of these ceilings; a careful from-scratch engine should target 50%+. **The smaller Qwen2.5-0.5B (~250 MB at Q4) hits ~400 tok/s on M2 — at this point per-call overhead and KV-cache reads matter more than weight streaming.**

**Why quantization gives near-linear decode speedup**: bytes-per-weight is in the denominator. Going fp16 → Q4_K is a 3.5× weight-bandwidth reduction; tok/s scales similarly until the KV cache reads (which don't shrink) become a non-trivial fraction of the total bandwidth budget.

**GEMM vs GEMV**. Prefill needs full GEMM with the BLIS five-loop tiling structure: outer loops partition N (nc, L3), K (kc, packed B), and M (mc, packed A); inner loops iterate an `mr × nr` microtile kept entirely in vector registers (typically 12×4 or 14×6 on AVX-512, 8×8 on NEON), streaming a kc-deep sliver of B from L1 and Ã from L2. The microkernel is rank-1 FMA updates of the C tile. Salykova's tutorial (`salykova.github.io/matmul-cpu`) is the cleanest modern walkthrough. **Don't write this from scratch**: link Apple **Accelerate** (`cblas_sgemm`/`cblas_sgemv`) on macOS, OpenBLAS on x86, or **Justine Tunney's tinyBLAS** (`github.com/Mozilla-Ocho/llamafile`) which beats both for the small/skinny shapes LLM prefill actually hits — 30–500% faster than upstream llama.cpp on Skylake, Alder Lake, Zen 4, RPi5 per her March 2024 benchmarks (`justine.lol/matmul/`).

Decode is the opposite story: BLAS is wrong for quantized GEMV because BLAS expects fp32/fp64 inputs and re-dequantizing weights just to call `sgemv` is strictly slower than a fused dequant+dot kernel. **You hand-write one fused kernel per quantization format per ISA.** For an OCaml engine, the inventory you must ship is roughly:

- fp32 sgemm/sgemv: delegate to Accelerate or OpenBLAS via Lacaml.
- Q8_0 × Q8_0: int8 dot product using NEON `vdotq_s32` or AVX-VNNI `vpdpbusd`, with fallback `vpmaddubsw` for non-VNNI AVX2.
- Q4_0 × Q8_0: nibble unpack + sign-correct + int8 dot.
- Q4_K × Q8_K: more complex; 6-bit sub-scale unpack + per-sub-block dot.
- F16 conversion: `vcvt_f32_f16` (NEON) / `_mm256_cvtph_ps` (AVX2 with F16C, near-universal on x86).

## 5. SIMD, threading, and Apple specifics

**ARM NEON (Apple Silicon).** 32× 128-bit registers, FP16 arithmetic native since ARMv8.2-FP16 (mandatory on M1+), **`vdotq_s32` (FEAT_DotProd) is the workhorse for int8 LLM matmul** — one instruction does 16 int8×int8 multiplies and 4 group-adds into a 128-bit int32 register. ARMv8.6 adds `vbfdot_f32`/`vbfmmla_f32` (bf16 dot/matmul) and `vmmlaq_s32` (int8 2×8 × 8×2 matrix multiply); M2 and later have these. **Apple's M-series omits SVE/SVE2 entirely** — even the M4 (ARMv9.2-A) skips it. So your portable matrix-extension story on Apple is "NEON or AMX, nothing in between."

**x86 SIMD.** AVX2+FMA3 is universal (Haswell 2013+); 16× 256-bit YMM. `_mm256_fmadd_ps` is the fp32 workhorse; `_mm256_maddubs_epi16` is the pre-VNNI int8 path. **AVX-VNNI** (`_mm256_dpbusd_avx_epi32`, Alder Lake / Zen 4 client onward) is the modern int8 fast path; **AVX-512-VNNI** gives 2× width but is disabled on most Intel client chips since Alder Lake (E-cores can't run it). The asymmetric `u8×s8` shape of `vpdpbusd` requires a pre-bias (add 128 to weights, subtract correction at end of block) for symmetric int8×int8 — every llama.cpp x86 quant kernel does this.

**Apple AMX.** Undocumented matrix coprocessor with one unit per CPU cluster, reverse-engineered by Dougall Johnson (`gist.github.com/dougallj/7a75a3be1ec69ca550e7c36dc75e0d6f`, consolidated at `github.com/corsix/amx`). One outer-product instruction per cycle on a 32×32 grid of MAC units; roughly **2× NEON throughput on dense matmul**. Apple exposes it through Accelerate (`cblas_sgemm`, `cblas_hgemm`, `BNNS`) — that's your supported path. Direct AMX is research-grade only. **In practice: link `-framework Accelerate` on macOS and the AMX usage comes for free** in your fp32/fp16 prefill GEMM. For quantized formats Accelerate has no entry point and you'll fall back to hand-written NEON.

**Multithreading pattern.** For both GEMV and GEMM, **partition the output rows across threads**: each thread reads its slab of weight rows + the (shared) activation vector + writes its slab of output. No false sharing, hardware prefetcher loves the linear stride, x is replicated in each core's L1. llama.cpp does exactly this. Stop scaling threads once memory bandwidth saturates — on an M1 (68 GB/s) that's typically 4 P-cores; on Alder Lake hybrids, **pin to P-cores only**, because the lockstep barrier means one slow E-core stalls everyone. Use macOS QoS classes (`pthread_set_qos_class_self_np`) for P-core preference; on Linux use `pthread_setaffinity_np`. **Don't compose threaded BLAS with `Domainslib`** — pick one parallelism level and set `OPENBLAS_NUM_THREADS=1` if you parallelize at the OCaml layer.

## 6. The OCaml ecosystem reality in 2026

The honest assessment: **the OCaml ML ecosystem is in transition.** Owl is in maintenance (its authors formally announced "concluding" the project in 2024); the strategic effort is now **Raven** (`raven-ml.dev`, alpha3 March 2026) — an Ahrefs/Tarides-sponsored stack with `nx` (NumPy-like, multiple backends), `rune` (JAX-like autodiff via effect handlers), `kaun` (Flax-like NN, ships a GPT-2 reference), `brot` (HF tokenizers bindings), and `nx-oxcaml` (OxCaml SIMD/unboxed-types backend that "approaches the C backend in pure OCaml"). A FunOCaml 2025 workshop walks through building a transformer decoder on top of Raven (`github.com/raven-ml/funocaml-2025-llm`). **OCANNL** (`github.com/ahrefs/ocannl`) is the most architecturally ambitious from-scratch DL compiler in OCaml — its roadmap explicitly plans transformer inference in 0.7.x — but isn't there yet.

**There is exactly one direct OCaml prior art**: `jackpeck/llama2.ml`, a single-file port of Karpathy's `llama2.c` running TinyStories models. Read it cover to cover before you write a line — it's ~700 lines and shows you idiomatic OCaml patterns for transformer state, weight loading, and the autoregressive loop. **No native OCaml GGUF parser, no safetensors parser, no llama.cpp binding exists** (May 2026). These are gaps you fill — but they're small (200–500 LOC each).

**Foundational libraries for the build:**

- **`Bigarray.Array1` (float32, c_layout)** — your tensor representation. Unboxed, GC-friendly, passes to C as a raw pointer via `Caml_ba_data_val`. Avoid `Bigarray.Array2` if you can — managing strides manually with `Array1` is faster and matches C kernels' expectations.
- **Lacaml** (`github.com/mmottl/lacaml`, well-maintained, 11.x in 2025) — clean BLAS/LAPACK bindings over Bigarray, your go-to for `sgemm`/`sgemv`. Picks up OpenBLAS, MKL, or Accelerate at link time.
- **Domainslib** — `Task.parallel_for` for parallelizing across output rows or attention heads.
- **`ctypes`** for setup-path FFI, but **hand-written `external ... [@@noalloc]` C stubs** for everything in the hot loop. The dynamic ctypes path is ~150 ns/call vs ~8 ns for static stubs — irrelevant for matmul, important for tight per-token ops.
- **`re`** (Unicode-aware) for the BBPE pre-tokenizer regex.

**OCaml performance characteristics worth internalizing:**

`float` is boxed when stored in polymorphic containers; **unboxed in `float array`, `Float.Array`, all-float records, and `Bigarray`**. Hot loops over `Bigarray` with `Array1.unsafe_get`/`unsafe_set` and a Flambda 2 compiler (`5.x.x+flambda` opam switch) produce code within 1.5–3× of `-O3` C for non-BLAS work. For LLM inference, this is invisible: ~80% of cycles are inside BLAS or hand-written SIMD C stubs, where OCaml's overhead is exactly zero. The 20% surrounding code (RMSNorm, softmax, RoPE, sampling) is where Flambda 2's **Loopify** pass (which converts tail-recursive functions to allocation-free loops, `ocamlpro.com/blog/2024_04_10_the_flambda2_snippets_2/`) earns its keep — write those ops as natural OCaml `for` loops over Bigarrays and don't allocate intermediates.

**OxCaml** (Jane Street's open-source extended compiler, `oxcaml.org`) is the aggressive alternative. It adds true unboxed primitives (`float#`, `float32#`, `int64#`), unboxed records, SIMD intrinsic types (`float32x8`, `int8x16`), `[@zero_alloc]` static checks, and stack-allocation via `local_`. Anil Madhavapeddy's group built an ONNX inference engine in pure OxCaml using SIMD (`tunbury.org/2026/03/13/oxcaml-inference/`) — claimed parity with C. **The tradeoff is API instability and a smaller ecosystem.** A pragmatic choice: **stay on mainline OCaml 5.3+Flambda for the engine's structure, drop into OxCaml-style SIMD only if you outgrow C stubs.**

**C stub anatomy you'll repeat often:**

```c
CAMLprim value caml_q4k_gemv(value vW, value vX, value vY, value vDim) {
  uint8_t* W = (uint8_t*) Caml_ba_data_val(vW);
  float*   X = (float*)   Caml_ba_data_val(vX);
  float*   Y = (float*)   Caml_ba_data_val(vY);
  intnat M = Long_val(vDim);
  caml_release_runtime_system();
  /* NEON / AVX2 quantized GEMV here */
  caml_acquire_runtime_system();
  return Val_unit;
}
```

Declared in OCaml as `external q4k_gemv : ... -> unit = "caml_q4k_gemv" [@@noalloc]`. The `caml_release_runtime_system` is crucial under OCaml 5: without it, other domains stall behind your kernel, killing your Domainslib parallelism.

## 7. Recommended project structure and roadmap

The build plan below front-loads **correctness** (matching a reference implementation's logits bit-for-bit-ish) and treats every performance optimization as a separate, separately-validated step. The single most valuable habit: **dump intermediate tensors from a Python reference and compare against your OCaml engine after every change.** HuggingFace transformers in fp32 mode is the easiest reference; for quantized comparisons, llama.cpp's `--logits-file` flag dumps per-token logits.

**Directory layout:**

```
qwen_ocaml/
├── lib/
│   ├── tensor.ml          (* Bigarray helpers, shape ops, pre-allocated buffers *)
│   ├── gguf.ml            (* GGUF parser, mmap loader, metadata extraction *)
│   ├── tokenizer.ml       (* BBPE encode/decode, ChatML template *)
│   ├── quant.ml           (* Q8_0/Q4_0/Q4_K dequant + activation quantization *)
│   ├── ops.ml             (* RMSNorm, RoPE, softmax, silu — pure OCaml *)
│   ├── matmul.ml          (* dispatches to BLAS or C stubs by dtype *)
│   ├── attention.ml       (* GQA forward with KV cache *)
│   ├── model.ml           (* Qwen2/Qwen3 forward pass, layer iteration *)
│   ├── sampling.ml        (* greedy, temp, top-k, top-p, rep penalty *)
│   └── generate.ml        (* prefill + decode loop *)
├── kernels/
│   ├── matmul_f32_neon.c
│   ├── matmul_f32_avx2.c
│   ├── q4k_gemv_neon.c
│   ├── q4k_gemv_avx2.c
│   └── stubs.c            (* dispatch by ISA detection *)
├── bin/
│   ├── run.ml             (* CLI: load model, prompt, generate *)
│   └── validate.ml        (* compares against HF reference logits *)
└── test/
    └── ...                (* unit tests for each kernel + e2e *)
```

**Milestone 1 — Correct fp32 forward pass (1–2 weeks of evenings).** Pick **Qwen2.5-0.5B** (simplest: no QK-norm, GQA group 7 makes the savings visible). Load it from the HuggingFace safetensors dump using a hand-written 150-line parser. Implement every op in pure OCaml over Bigarrays: RMSNorm, RoPE (HF rotate-half convention!), GQA attention with broadcast, SwiGLU MLP, tied lm_head. Use a naive triple-loop matmul. Validate: run the same prompt through HF transformers in fp32, dump logits for the first 5 tokens of generation, compare against your OCaml output — they should match to ~1e-4 absolute. **If the first-token logits don't match, debug RoPE first** (50% of bugs), then attention biases (Qwen2 has them), then RMSNorm epsilon position. Don't move on until logits match.

**Milestone 2 — BLAS, KV cache, working chat (3–5 days).** Replace naive matmul with `Lacaml.S.gemm` and `Lacaml.S.gemv`. Add the KV cache as a pre-allocated `[layers; n_kv_heads; max_T; head_dim]` Bigarray, sliced into each step. Implement the ChatML chat template and sampling (temp + top-p). You should now get ~5–15 tok/s on a laptop for the 0.5B model. Validate: end-to-end conversation quality matches HF transformers' generations qualitatively; perplexity on a held-out text matches to 4 decimal places.

**Milestone 3 — GGUF + Q4_K_M quantization (1–2 weeks).** Write the GGUF parser. Implement Q4_K_M dequantization to fp32 first (slow but correct), then write the **fused Q4_K × Q8_K GEMV kernel** in C with NEON intrinsics (Mac) and AVX2 fallback. This is the central piece. Validate against llama.cpp: load the same Q4_K_M-quantized Qwen2.5-0.5B GGUF, generate with `temperature=0` (greedy) from the same prompt, compare token sequences. They should agree exactly for many tokens before diverging due to floating-point order-of-operations differences. **Expected 3–4× speedup over Milestone 2** purely from memory bandwidth reduction.

**Milestone 4 — Multithreading + Qwen3 (1 week).** Add `Domainslib` parallel-for across output rows in the GEMV kernels (or across attention heads in attention). Make sure to set `OPENBLAS_NUM_THREADS=1` if you're using OpenBLAS on the prefill path. Add Qwen3-0.6B support: QK-norm before RoPE, no QKV bias, decoupled head_dim. Test on a P-core-only Apple Silicon: 4 P-cores typically saturates M1/M2 base memory bandwidth; benchmark to confirm.

**Milestone 5 — Q8_0 + AVX-VNNI / NEON dotprod paths, optional AMX via Accelerate (1+ weeks).** Add Q8_0 (the "perfect" quantization, useful as a baseline and for output heads). Specialize int8 dot products by ISA via runtime detection. Optionally link Accelerate's `cblas_hgemm` for prefill on Mac — this transparently uses AMX and is 2× faster than NEON for fp16 GEMM. Profile with Instruments (Mac) or `perf stat -e cache-misses,LLC-load-misses` (Linux). **You should be at 50–70% of roofline by here.**

**Validation strategy across all milestones**. Maintain a `validate.ml` that takes a model checkpoint and a fixed prompt, runs your engine, and diffs against a pre-computed reference (HF transformers for fp32, llama.cpp for quantized). For numerical correctness use **logit-space comparison** (top-k overlap @ k=10 is more robust than absolute float diff). For sampling-equivalence use **greedy generation match** for the first N tokens. Build a small perplexity harness on WikiText-103 for the final quality check.

**What to defer or skip.** GPU (Metal/CUDA) — out of scope for a CPU toy. IQ-quants (IQ2_XXS etc.) — gather-style CPU-unfriendly; skip until you've mastered K-quants. Flash-attention — small models on short contexts don't need it; naive `O(N²)` attention is fine through 8K context. AVX-512 — disabled on most client Intel chips, the perf delta over AVX-VNNI 256-bit is small on the shapes you'll hit, and the binary-portability headaches aren't worth it for a toy.

## 8. Key references in one place

**Algorithmic.** Qwen3 paper arXiv:2505.09388; Qwen2.5 paper arXiv:2412.15115; RoPE arXiv:2104.09864; GQA arXiv:2305.13245; SwiGLU arXiv:2002.05202; RMSNorm arXiv:1910.07467; QK-Norm (Dehghani et al. 2023, PMLR v202). Sebastian Raschka's clean Qwen3 reimplementation: `github.com/rasbt/reasoning-from-scratch/blob/main/reasoning_from_scratch/qwen3.py`. Karpathy's `llama2.c`: `github.com/karpathy/llama2.c`. Karpathy's `minbpe` for tokenizer.

**Quantization & file formats.** GGUF spec: `github.com/ggml-org/ggml/blob/master/docs/gguf.md`. Reference quant kernels: `ggml/src/ggml-quants.c` and `ggml/src/ggml-cpu/quants.c` in llama.cpp. K-quants original PR: `github.com/ggml-org/llama.cpp/pull/1684`. Quality survey paper: arXiv:2601.14277. Annotated K-quant walkthrough: `haroldbenoit.com/notes/ml/llms/quantization/llama.cpp/k-quants-implementation`. Safetensors spec: `github.com/huggingface/safetensors`.

**Performance.** Roofline model: Williams/Waterman/Patterson, CACM 52(4), 2009. BLIS five-loop paper: Van Zee et al. IPDPS 2014 (`cs.utexas.edu/~flame/pubs/blis3_ipdps14.pdf`). Modern GEMM tutorial: `salykova.github.io/matmul-cpu`. Justine Tunney's `LLaMA Now Goes Faster on CPUs`: `justine.lol/matmul/`. tinyBLAS source: `github.com/Mozilla-Ocho/llamafile`. mmap weight loading: `justine.lol/mmap/`. Apple AMX reverse-engineering: `github.com/corsix/amx`. Quantized GEMV implementation walkthrough: `vijayprabhas9.github.io/gemv_optimization/`.

**OCaml ecosystem.** Raven: `github.com/raven-ml/raven`, workshop at `github.com/raven-ml/funocaml-2025-llm`. OCANNL: `github.com/ahrefs/ocannl`. Lacaml: `github.com/mmottl/lacaml`. OxCaml: `oxcaml.org`, SIMD docs at `oxcaml.org/documentation/simd/intro/`. Flambda 2 series: `ocamlpro.com/blog/2024_03_18_the_flambda2_snippets_0/`. The lone OCaml prior art: `github.com/jackpeck/llama2.ml`. C-stubs/Bigarray manual: `ocaml.org/manual/5.4/libbigarray.html`. Optimizing OCaml: `ocamlverse.net/content/optimizing_performance.html`.

## Conclusion: what this build actually teaches

A from-scratch OCaml Qwen engine is **not bottlenecked by OCaml**. The roofline analysis is brutal: decode runs at `model_bytes / bandwidth`, every byte of weight bandwidth eaten by anything that isn't a quantized weight (KV cache, activation buffers, scratch) is throughput lost, and the win from going fp16 → Q4_K is 3.5× — larger than any plausible language-level optimization you could ever extract. The interesting engineering surface is therefore (a) **getting Qwen3's QK-norm and head-dim decoupling right** so you actually load the model correctly, (b) **getting the Q4_K × Q8_K fused dequant-and-dot kernel right** in NEON and AVX2 so you actually hit memory bandwidth, and (c) **a clean OCaml structure around those hot kernels** — Bigarray-everywhere, `[@@noalloc]` C boundaries, Domainslib for row-stripe parallelism, no allocations per token. Everything else (sampling, tokenizer, KV cache, BLAS sgemm) is mechanical.

The deeper lesson, useful well beyond this project: **modern CPU LLM inference is a memory hierarchy problem disguised as a linear algebra problem.** Whether you write it in OCaml, C, or Rust matters far less than whether you understand the roofline, the GGUF block layout, and where the bytes are moving in your inner loop. OCaml is a fine host language for that understanding — and your Coq/Rocq background makes the shape-correctness invariants (which are where most "model loaded but outputs garbage" bugs hide) feel natural to encode and check.