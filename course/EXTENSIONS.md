# Final-Project Extensions — Plugging onto the Baseline

This is the **how-to** for the seven speed projects (P1–P7) in `PROJECTS.md`.
It explains the one rule that makes them work: **the baseline engine is frozen,
and your project replaces exactly one seam behind a fixed interface.** The rest
of the engine — and the auto-grader — depends on that interface staying put.

---

## The baseline is the shared, frozen code

Everything in `lib/` is the M1–M5 "spine": a correct, deliberately simple
inference engine. **You do not rewrite it.** You start from it, change one piece,
and prove with a `validate_*.exe` that you didn't break correctness — only then
do you report a speedup.

The pieces you may change are marked **SEAM** at the top of their interface file.
A seam has two halves:

- an **`.mli` contract** — the types/signatures the rest of the engine calls.
  *You must not change these.* (If you change a signature, you've forked the
  engine, not plugged into it.)
- an **`.ml` / `.c` body** — the implementation behind the contract. *This is
  what you rewrite.*

If your build still type-checks against the unchanged `.mli` **and** the
validator still prints `MATCH: true`, your plug-in is correct by construction.

---

## The seam map — where each project plugs in

| Project | Edit (the body) | Frozen contract (do not change) | Auto-grade |
|---|---|---|---|
| **P1** vectorize dequant | `lib/stubs.c` `dequant_row_*` | `lib/quant.mli` | `validate_gguf.exe` MATCH true (bit-exact) |
| **P2** int8 dot | `lib/stubs.c` `caml_qwen_qmatvec` | `lib/quant.mli` | `validate_gguf.exe` MATCH true (exact vs llama.cpp) |
| **P3** threading | `lib/stubs.c` `caml_qwen_qmatvec` dispatch | `lib/quant.mli` | `validate_gguf.exe` MATCH true (numerics untouched) |
| **P4** lower bpw | `lib/quant.ml` `row_bytes` + `lib/stubs.c` dequant case | `lib/quant.mli` | greedy top-1 vs llama.cpp on the SAME new file |
| **P5** speculative | `lib/generate.ml` + a `forward_at` helper in `lib/model.ml` | output identical to greedy | new `validate_spec.exe` token-for-token identity |
| **P6** prefill GEMM | new `forward_prefill` in `lib/model.ml` + `caml_qwen_sgemm` in `lib/stubs.c` | last-position logits match the reference | new `validate_prefill.exe` |
| **P7** KV quant | `lib/attention.ml` `Kv_cache` internals | `lib/attention.mli` | output within tolerance of fp32-KV on a fixed prompt |

Note P1/P2/P3 **all live inside `stubs.c` behind `quant.mli`** — the OCaml never
changes. P7 lives entirely behind `attention.mli`. P5/P6 extend the
`Generate`/`Model` layer and are graded by a validator you add, not by a frozen
`.mli`.

---

## The git workflow (branch per project)

The baseline is tagged `baseline`. Each project is a branch off it:

```sh
git checkout -b p1-dequant baseline      # start from the frozen baseline
# ... edit only your seam (lib/stubs.c) ...
opam exec --switch=qwen -- dune build
opam exec --switch=qwen -- dune exec bin/validate_gguf.exe   # must print MATCH: true
```

Because each project touches a different seam, two projects can be combined by
merging their branches; the only file two projects share is `stubs.c`
(P1/P2/P3/P4), so combining those is a manual merge — which is itself a good
exercise in why the kernel is the contended resource.

### The two A/B-in-one-binary projects

P4 and P7 are meant to be **compared against the baseline in the same build**, so
the engine reads them from an environment variable instead of a code change:

- **P4** — point the engine at your new GGUF: `QWEN_GGUF=models/<your>.gguf`.
  Run the baseline file and your file back-to-back, interleaved, to get a fair
  tok/s delta on the same machine load.
- **P7** — gate your fp16/int8 KV path behind an env flag (convention:
  `QWEN_KV_FP16=1`) so the default run stays the exact fp32 reference and the
  flag flips on your path. That lets one binary prove *identical output* (flag
  off vs on) and *halved KV bytes* (flag on).

---

## The roofline deliverable (25% of the grade)

The write-up is not "I got X tok/s." It is: **where did the bytes go, and did my
change move the predicted ceiling?** The honest, measured baseline (M3 Max, Q4_K_M)
is ~23 tok/s decode; llama.cpp on the same file is ~226 tok/s; the pure-bandwidth
ceiling is ~790 tok/s. The engine is **compute-bound in the per-row fp32 dequant**
(~9 GB/s effective vs ~300 GB/s available), so:

- Projects that cut **compute** in the hot kernel (P1 dequant SIMD, P2 int8 dot)
  move the number: reference results were P1 ~1.23x and P2 ~1.37x, both exact-match.
- Projects that cut **bytes** without touching the compute bottleneck (P4 lower
  bpw, P7 KV quant) are expected to show **little or no speedup on this 0.5B
  model** — and explaining *why* (you were never bandwidth-bound here) is the
  point of the project, not a failure. P4's payoff is a quality/size trade; P7's
  is ~2x context in the same memory.
- P5/P6 change *what* is computed: P6 (batched prefill GEMM) helps TTFT (2–9x,
  eroding as the prompt grows because attention stays scalar); P5 (speculative)
  only wins with high acceptance AND a batched verify, so on this engine it is
  honestly *slower* until P6's batched path backs its verify step.

State your before/after numbers, note the machine load (a shared box swings these
a lot — interleave A/B), and if you can't hit the exact-match bar, say so, fall
back to the relaxed bar in `PROJECTS.md`, and explain the first divergence.
