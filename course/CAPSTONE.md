# Capstone — composing the two real decode wins

This branch (`capstone-p3-attn`) stacks the two optimizations from the speed
projects that actually moved decode throughput on a quiet machine:

- **P3 chunks** — match the matvec `dispatch_apply` chunk count to the P-cores
  (`max(1, P-cores-2)`) instead of a fixed 32, so work doesn't spill onto the
  slow E-cores. (cherry-picked from `p3-threading`)
- **attn-vectorize** — move one layer's full multi-head attention into a single
  C/NEON call (`caml_qwen_attn`), replacing the scalar OCaml loops.

Both are numerically exact / within-tolerance: `validate.exe`, `validate_gguf.exe`,
and `validate_qwen3.exe` all stay `MATCH: true`.

## The measurement (M3 Max, Q4_K_M, decode tok/s)

| config | short ctx (40 tok) | long ctx (800 tok) |
|---|---|---|
| baseline (32 chunks, scalar attn) | 24.0 | 9.1 |
| + P3 chunks only | 29.1 (+21%) | 9.0 (no change) |
| + attn-vectorize only | 26.2 (+9%) | 24.6 (2.7×) |
| **capstone (both)** | **32.5 (+35%)** | **29.9 (3.3×)** |

## Why they compose — the roofline lesson

The two changes fix **different bottlenecks**, and which one dominates depends on
context length:

- **Short context**, attention is tiny; the per-token cost is the weight matvecs.
  So **P3 chunks** is the big lever (+21%) and attention vectorization adds a
  little (+9%). Together: +35% — they stack roughly multiplicatively.

- **Long context**, the O(T)/token attention over the KV cache grows until it
  dominates. There **P3 chunks alone does nothing** (9.1→9.0): parallelizing the
  matvecs better doesn't help when the bottleneck has moved to the serial OCaml
  attention. **attn-vectorize** is what unblocks it (9.1→24.6). And once
  attention is no longer the bottleneck, the matvecs dominate again — so
  **P3 chunks now recovers its win on top** (24.6→29.9, +21%).

The capstone wins at *both* context regimes because the two levers are
orthogonal. The `+P3 only` long-context row (no change) is the whole point:
**a speedup only shows up if you fix the bottleneck that is actually binding**,
and the binding bottleneck moves as the sequence grows. Profile at the context
length you care about; don't assume the matmul kernel is always the answer.

## What did NOT make the capstone (and why)

- **P1 (NEON dequant)** / **P2 (int8 vdotq)** — correct, but ~0% on a quiet
  machine (the compiler already auto-vectorizes the dequant; the int8 dot isn't
  the bottleneck). P2's value is the correctness lesson (match ggml's exact fp32
  accumulation order). The earlier 1.2–1.4× numbers were contention artifacts.
- **P3 persistent pool** — correct but doesn't beat GCD `dispatch_apply`, and its
  spinning workers steal thermal headroom from the serial path.
- **attn head-parallelism** — once attention is C-vectorized it's no longer the
  bottleneck, so spreading heads across cores is neutral at long context and
  *slower* at short (per-layer dispatch overhead).

Three plausible parallelizations that the roofline said wouldn't help — and
measurement confirmed. The capstone keeps only the two that target a binding
bottleneck.
