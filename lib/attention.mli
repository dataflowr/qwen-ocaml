(* GQA attention with a pre-allocated KV cache. One token-step per call.
   The model applies RMSNorm/projections/RoPE; this module owns the cache and
   the scaled-dot-product softmax over all cached keys (naive O(T) per step,
   O(T^2) over a sequence — fine for Milestone 1, <=8K context). *)

module Kv_cache : sig
  type t
  (* allocate [n_layers][n_kv_heads * max_seq * head_dim] for K and V, once. *)
  val create : Config.t -> max_seq:int -> t
  val reset : t -> unit
end

(* Attention step at sequence position [pos] (0-based) in layer [layer].
   Inputs for the current token only:
     q : length n_heads    * head_dim  (already RoPE-applied, per-head contiguous)
     k : length n_kv_heads * head_dim  (already RoPE-applied)
     v : length n_kv_heads * head_dim
   Writes k,v into the cache at [pos], then computes GQA attention against all
   keys/values at positions 0..pos (causal) into:
     out : length n_heads * head_dim   (concatenated per-head; pre o_proj)
   GQA: query head h reads kv head (h / (n_heads / n_kv_heads)). Scale 1/sqrt(head_dim). *)
val forward :
  Config.t -> Kv_cache.t -> layer:int -> pos:int ->
  q:Tensor.t -> k:Tensor.t -> v:Tensor.t -> out:Tensor.t -> unit
