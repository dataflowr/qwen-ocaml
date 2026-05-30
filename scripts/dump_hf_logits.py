#!/usr/bin/env python3
"""Dump HF transformers fp32 reference logits for Milestone-1 validation.

Run with the project's uv env:
    .venv/bin/python scripts/dump_hf_logits.py \
        --model models/Qwen2.5-0.5B --prompt "The capital of France is" --n 5

Writes scripts/ref_logits.npy with shape [n_steps, vocab] plus ref_tokens.json
(greedy continuation). Compare against bin/validate.ml (top-k@10 overlap +
max abs logit diff, target < 1e-4 vs the OCaml fp32 engine).
"""
import argparse, json
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--out", default="scripts/ref")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32)
    model.eval()

    ids = tok(args.prompt, return_tensors="pt").input_ids
    all_logits, gen = [], []
    with torch.no_grad():
        for _ in range(args.n):
            out = model(ids)
            last = out.logits[0, -1, :].float()
            all_logits.append(last.numpy())
            nxt = int(last.argmax())
            gen.append(nxt)
            ids = torch.cat([ids, torch.tensor([[nxt]])], dim=1)

    logits = np.stack(all_logits).astype(np.float32)
    np.save(args.out + "_logits.npy", logits)
    # raw little-endian f32 [n_steps * vocab] for the OCaml validator to read
    logits.tofile(args.out + "_logits.f32")
    json.dump(
        {"prompt": args.prompt, "prompt_ids": tok(args.prompt).input_ids,
         "greedy_tokens": gen, "greedy_text": tok.decode(gen),
         "n_steps": int(logits.shape[0]), "vocab": int(logits.shape[1])},
        open(args.out + "_tokens.json", "w"), indent=2)
    print("wrote", args.out + "_logits.npy", args.out + "_logits.f32",
          "and", args.out + "_tokens.json")
    print("greedy:", repr(tok.decode(gen)))


if __name__ == "__main__":
    main()
