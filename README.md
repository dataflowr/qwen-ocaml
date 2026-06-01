# qwen-ocaml — Efficient LLM inference, from the metal up

A from-scratch **OCaml** inference engine for **Qwen2.5 / Qwen3**, built as the
spine of a hands-on course on *what actually happens when a model runs* — and
which of the gaps to a tuned runtime (llama.cpp) are about **math** vs **memory**.

- Pure OCaml orchestration + small C/NEON kernels (Apple Silicon: NEON, FEAT_DotProd, AMX via Accelerate).
- Loads HuggingFace `safetensors` (fp32) and quantized **GGUF** (Q4_K_M: Q4_K/Q5_0/Q6_K/Q8_0).
- Every step is **graded against a reference**: HuggingFace fp32 logits, and llama.cpp greedy tokens.
- Correct first (`MATCH: true`), then fast — the two are measured separately, on purpose.

## Quick start

```sh
# OCaml 5.2 +flambda toolchain (see SETUP.md for the exact opam switch)
opam exec --switch=qwen -- dune build

# run from the repo root so relative data paths resolve (or set QWEN_DIR=/path/to/repo)
opam exec --switch=qwen -- dune exec bin/validate.exe        # Qwen2.5 fp32  -> MATCH: true
opam exec --switch=qwen -- dune exec bin/validate_gguf.exe   # quant vs llama.cpp + tok/s
opam exec --switch=qwen -- dune exec bin/run.exe -- "The capital of France is"
```

**Models are not in the repo** (gitignored — they are large). Place them under
`models/` (or point `QWEN_DIR` at a checkout that has them); see `SETUP.md` for
the exact files and how the reference data is generated.

## The course

- `course/SYLLABUS.md` — the 10-session course (M1–M5 milestones: correct fp32 → BLAS → Q4_K quant → Qwen3 → SIMD/parallel).
- `course/PROJECTS.md` — the 7 optional speed projects (P1–P7), each with a roofline hypothesis and an objective auto-grade.
- `course/EXTENSIONS.md` — how each project **plugs onto the frozen baseline**: the seam file, the `.mli` contract it must preserve, and the validator that grades it.

The seams are real: `lib/quant.mli` (the kernel boundary for P1–P4), `lib/matmul.mli`,
`lib/weights.mli`, `lib/attention.mli`. A project replaces *one* implementation behind a
stable interface; type-checking + `MATCH: true` ⇒ a correct plug-in by construction.

## Branches — the worked solutions (and honest dead-ends)

`main` is the frozen, deliberately-simple baseline. Each branch is one exploration,
measured on a quiet M3 Max against the Q4_K_M file:

| branch | what it does | decode result |
|---|---|---|
| `p1-dequant` | NEON-vectorize the fp32 dequant kernels | exact; ~0% (compiler already auto-vectorizes) |
| `p2-int8` | int8 `vdotq` for Q5_0/Q8_0, exact vs llama.cpp | exact; ~0% alone (bottleneck is elsewhere) |
| `p3-threading` | match dispatch chunks to P-cores (+ a persistent-pool stretch) | **+22%** short ctx; pool = documented dead-end |
| `attn-vectorize` | full multi-head attention in C/NEON | **+12% short / 3.1× long ctx** |
| `attn-parallel` | parallelize the per-head loop | dead-end (attention no longer the bottleneck) |
| `capstone-p3-attn` | P3 chunks **+** attn-vectorize | **+35% short / 3.3× long** — they compose |
| `capstone-int8` | the above **+** int8 GEMV for all quant types | **51 tok/s** (2.1× baseline); see `course/CAPSTONE.md` |

The throughline (this is the lesson): *an optimization only pays once it targets the
**binding** bottleneck, and the binding bottleneck moves with context length.* Three
plausible parallelizations were roofline-predicted no-ops and measurement confirmed it;
the real wins were matching dispatch to the P-cores and getting the serial OCaml
attention onto NEON. See `course/CAPSTONE.md` (on the capstone branches) for the full 2×2.

## Where it lands vs llama.cpp

On a 0.5B Q4_K_M model, the baseline ran ~24 tok/s; the stacked wins reach ~40 tok/s
exact-match and ~51 tok/s with a 95%-faithful int8 K-quant path (llama.cpp ≈ 226).
Crucially, even llama.cpp uses only ~90 of ~400 GB/s here — the remaining gap is
compute-efficiency and per-call overhead, **not** a bandwidth wall. Closing it fully
would mean reimplementing llama.cpp's kernels in C, against this engine's *teaching*
purpose. Understanding *why* that gap exists is the point.

---

Built with [Claude Code](https://claude.com/claude-code). Part of [dataflowr](https://github.com/dataflowr).
