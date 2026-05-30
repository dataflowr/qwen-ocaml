(* Model hyperparameters. Populated from GGUF metadata (or config.json during
   Milestone 1). The Qwen2.5 vs Qwen3 differences live here:
   - qkv_bias: true for Qwen2.5, false for Qwen3
   - qk_norm: false for Qwen2.5, true for Qwen3 (RMSNorm on Q/K per head, pre-RoPE)
   - head_dim is decoupled from hidden_size/n_heads in Qwen3. *)

type arch = Qwen2 | Qwen3

type t = {
  arch : arch;
  n_layers : int;
  hidden_size : int;
  n_heads : int;
  n_kv_heads : int;
  head_dim : int;
  intermediate_size : int;
  vocab_size : int;
  rope_theta : float;
  rms_eps : float;
  qkv_bias : bool;
  qk_norm : bool;
  tie_embeddings : bool;
}

(* Verified against HuggingFace config.json. *)
let qwen2_5_0_5b = {
  arch = Qwen2;
  n_layers = 24;
  hidden_size = 896;
  n_heads = 14;
  n_kv_heads = 2;
  head_dim = 64;
  intermediate_size = 4864;
  vocab_size = 151936;
  rope_theta = 1e6;
  rms_eps = 1e-6;
  qkv_bias = true;
  qk_norm = false;
  tie_embeddings = true;
}

let qwen3_0_6b = {
  arch = Qwen3;
  n_layers = 28;
  hidden_size = 1024;
  n_heads = 16;
  n_kv_heads = 8;
  head_dim = 128;
  intermediate_size = 3072;
  vocab_size = 151936;
  rope_theta = 1e6;
  rms_eps = 1e-6;
  qkv_bias = false;
  qk_norm = true;
  tie_embeddings = true;
}
