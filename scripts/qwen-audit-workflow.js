export const meta = {
  name: 'qwen-audit',
  description: 'Multi-expert read-only audit of the OCaml Qwen engine across 5 dimensions',
  phases: [
    { title: 'Audit', detail: 'parallel expert reviewers: C/memory-safety, transformer numerics, binary parsers, quant/dequant, OCaml quality+build' },
  ],
}

const ROOT = '/Users/lelarge/courses/Qwen_Ocaml'

const CONTEXT = [
  'PROJECT: ' + ROOT + ' — a from-scratch Qwen2.5 / Qwen3 inference engine in OCaml + C stubs.',
  'Build/run on the qwen opam switch: opam exec --switch=qwen -- dune build|exec|test.',
  'Platform: Apple M3 Max, arm64, OCaml 5.2+flambda. Apple Accelerate linked; C stubs -mcpu=native.',
  '',
  'CURRENT VERIFIED STATE (all green — do NOT re-flag these as broken without concrete evidence):',
  '  - bin/validate.exe: Qwen2.5-0.5B fp32 greedy matches HuggingFace exactly (logit diff 6e-5).',
  '  - bin/validate_qwen3.exe: Qwen3-0.6B fp32 matches HF exactly (QK-Norm, no-bias, head_dim=128).',
  '  - bin/validate_gguf.exe: Q4_K_M GGUF greedy output is char-for-char identical to llama.cpp, 24 tok/s.',
  '  - dune test: passes.',
  '',
  'KNOWN / ACCEPTED limitations — do NOT report these as new findings (mention only if you find them',
  'actually causing a correctness bug):',
  '  - Tokenizer pre-tokenizer is an ASCII-only approximation of the cl100k regex (non-English may differ).',
  '  - Attention uses a module-global scratch buffer (OCaml side is single-threaded; parallelism is only',
  '    in the C GEMV via GCD). Naive O(T^2) attention. No flash-attention.',
  '  - GGUF mmap is deliberately never closed (weights point into it for process lifetime).',
  '  - Sampling top-k/top-p are stubs; only greedy is used by the validators.',
  '  - arm64/Accelerate only; no x86 fallback yet (by design).',
  '',
  'YOUR JOB: a RIGOROUS, READ-ONLY audit. Find REAL issues: memory-safety/UB, correctness edge cases,',
  'numerical hazards, resource/lifetime bugs, integer overflow, unchecked inputs, race conditions,',
  'dead code, API misuse, and clear simplifications. DO NOT EDIT any files — report findings only.',
  'For each finding give: severity (critical|high|medium|low), file:line, a concrete description of the',
  'problem and how it manifests, and a specific suggested fix. Be concrete and cite code. Prefer a few',
  'real issues over many speculative ones. If a whole area is solid, say so briefly.',
].join('\n')

const FINDINGS_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['dimension', 'overall', 'findings'],
  properties: {
    dimension: { type: 'string' },
    overall: { type: 'string', description: 'one-paragraph assessment of this area' },
    findings: {
      type: 'array',
      items: {
        type: 'object', additionalProperties: false,
        required: ['severity', 'location', 'issue', 'fix'],
        properties: {
          severity: { type: 'string', enum: ['critical', 'high', 'medium', 'low'] },
          location: { type: 'string', description: 'file:line(s)' },
          issue: { type: 'string' },
          fix: { type: 'string' },
        },
      },
    },
  },
}

const dims = [
  {
    key: 'c-memory-safety',
    files: 'lib/stubs.c, lib/dune',
    focus: [
      'Audit the C kernels for memory safety and undefined behavior:',
      '- VLA stack usage: float scratch[cols] inside GCD blocks (cols up to 4864 -> ~19KB/frame); is that safe',
      '  on dispatch worker threads? any path with larger cols?',
      '- Bounds: do dequant_row_* / qmatvec ever read past the mmap blob? offsets are OCaml ints from a u64',
      '  GGUF field — overflow/truncation risk? stride math (cols/block)*bytes correctness.',
      '- The rule that a stub calling caml_release_runtime_system must not be [@@noalloc] (check externals).',
      '- GCD dispatch_apply correctness: data races on out[]/shared reads; exception/longjmp across the block.',
      '- half_to_float, bf16->f32, sgemv arg widths (int vs intnat), cblas usage.',
      '- Any leftover dead/experimental code, unused includes, or debug cruft.',
    ].join('\n'),
  },
  {
    key: 'transformer-numerics',
    files: 'lib/model.ml, lib/ops.ml, lib/attention.ml, lib/config.ml',
    focus: [
      'Audit the transformer math and forward pass:',
      '- RoPE (HF rotate_half) tables + application; RMSNorm (eps placement, fp32); SwiGLU; residuals.',
      '- GQA kv-head indexing, causal masking, 1/sqrt(head_dim) scale, KV-cache writes/bounds (max_seq=4096).',
      '- Qwen3 QK-Norm: per-head RMSNorm pre-RoPE, shared weight, in-place correctness; optional bias logic.',
      '- Tied lm_head via embed; decoupled head_dim (qdim vs hidden) shapes.',
      '- Any aliasing/in-place hazards in shared scratch buffers across the layer loop.',
      '- Numerical robustness (softmax stability, overflow) and correctness edge cases (pos beyond cache).',
    ].join('\n'),
  },
  {
    key: 'binary-parsers',
    files: 'lib/gguf.ml, lib/gguf.mli, lib/safetensors.ml, lib/safetensors.mli',
    focus: [
      'Audit the GGUF and safetensors parsers:',
      '- Robustness to malformed/truncated input; bounds on every read vs file size; u64->int truncation.',
      '- GGUF: skipping ALL kv value types correctly to land on the tensor table; alignment/data_start;',
      '  dims order (ne0=cols, ne1=rows); offset relative to data section; magic/version checks.',
      '- safetensors: header length, data_offsets base, the hand-written JSON scanner edge cases, dtype map.',
      '- File descriptor / mmap lifetime and error handling (open failures, missing tensors).',
    ].join('\n'),
  },
  {
    key: 'quant-dequant',
    files: 'lib/quant.ml, lib/weights.ml, lib/matmul.ml, scripts/ggml_block_layouts.md',
    focus: [
      'Audit the quantization layer and weight abstraction:',
      '- dequant block math vs scripts/ggml_block_layouts.md and ggml semantics (Q8_0/Q5_0/Q4_K/Q6_K):',
      '  Q4_K get_scale_min branch + interleave; Q5_0 5th-bit; Q6_K signed scales + center; half LE.',
      '- row_bytes / stride consistency with the C kernel; ty dispatch coverage; F32 handled.',
      '- Weights.t abstraction (dense vs quant), row vs matvec semantics, the tied-embed dual use.',
      '- Any precision concerns or shape mismatches; external signatures match the C entry points (incl. the',
      '  7-arg bytecode wrapper).',
    ].join('\n'),
  },
  {
    key: 'ocaml-quality',
    files: 'lib/tokenizer.ml, lib/generate.ml, lib/sampling.ml, lib/tensor.ml, bin/*.ml, dune-project, lib/dune, bin/dune',
    focus: [
      'Audit OCaml code quality, resource handling, and build:',
      '- Hot-loop allocations, unnecessary copies, use of unsafe_get/set vs bounds, idiomatic style.',
      '- Error handling (failwith vs results), partial functions, unused/dead code, unused deps (e.g. is',
      '  lacaml/domainslib actually used? should they be removed for simplicity?).',
      '- Tokenizer: BPE correctness/perf (merge ranks, O(n^2)), special-token handling, decode round-trip.',
      '- bin/*.ml: duplication across validators (could share), hardcoded absolute paths, CLI robustness.',
      '- dune flags/portability; -ffast-math numerical implications; overall simplicity vs the stated goal',
      '  of keeping the infra as simple as possible.',
    ].join('\n'),
  },
]

phase('Audit')
const results = await parallel(dims.map((d) => () =>
  agent(
    CONTEXT + '\n\nAUDIT DIMENSION: ' + d.key + '\nPRIMARY FILES: ' + d.files + '\n\n' + d.focus +
      '\n\nRead the files (and any others you need for context) under ' + ROOT +
      '. Return the structured findings.',
    { label: 'audit:' + d.key, phase: 'Audit', schema: FINDINGS_SCHEMA }
  )
))

return { audit: results.filter(Boolean) }
