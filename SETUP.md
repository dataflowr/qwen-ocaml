# Environment setup — status

Verified ready on Apple M3 Max (128 GB, arm64), 2026-05-29.

## Toolchain
- **opam switch `qwen`**: OCaml 5.2.0 **+flambda**, with `lacaml` 11.1.1, `domainslib` 0.5.2, `ctypes` 0.24, `re` 1.14.
  - Build: `opam exec --switch=qwen -- dune build`
  - Test:  `opam exec --switch=qwen -- dune test`  (→ "all tests passed")
- **clang 21** (arm64); kernels compiled `-O3 -ffast-math -mcpu=native` → NEON + FEAT_DotProd.
- **Apple Accelerate** linked (`-framework Accelerate`) → AMX path for fp32/fp16 GEMM.
- **llama.cpp** at `/opt/homebrew/bin` (`llama-cli`, `llama-quantize`, `llama-perplexity`) → quantized reference (Milestone 3).

## Python reference (uv)
- `.venv` (Python 3.12): torch 2.12, transformers 5.9, numpy, safetensors, huggingface_hub.
- Dump fp32 reference logits:
  `.venv/bin/python scripts/dump_hf_logits.py --model models/Qwen2.5-0.5B --prompt "..." --n 5`
  (verified: greedy → " Paris. It is the").

## Models (in models/, gitignored)
- `Qwen2.5-0.5B/` — base safetensors (942 MB) + tokenizer. **Milestone 1** target.
  config verified vs plan: 24 layers, hidden 896, 14/2 heads, head_dim 64, I=4864,
  rope_theta 1e6, tied embeddings, vocab 151936, eps 1e-6.
- `Qwen2.5-0.5B-Instruct-GGUF/...-Q4_K_M.gguf` (379 MB) — **Milestone 3** target
  (verified generating via llama-cli).

## Layout
lib/ (qwen library + stubs.c), kernels/ (SIMD kernels, M3+), bin/ (run, validate),
test/, scripts/ (HF dump), models/.

## Next: Milestone 1
Implement `Safetensors.load` (bf16→fp32), `Model.forward` (RoPE HF rotate-half,
GQA, Qwen2 QKV bias), wire `bin/validate.ml` to diff against `scripts/ref_logits.npy`.
Debug order if logits mismatch: RoPE → attention biases → RMSNorm eps position.
