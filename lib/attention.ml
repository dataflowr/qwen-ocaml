(* GQA attention with a pre-allocated KV cache. One token-step per call. *)

module Kv_cache = struct
  type t = {
    k : Tensor.t array;  (* per layer: [n_kv_heads * max_seq * head_dim] *)
    v : Tensor.t array;
    max_seq : int;
    n_kv_heads : int;
    head_dim : int;
  }

  let create (cfg : Config.t) ~max_seq : t =
    let sz = cfg.n_kv_heads * max_seq * cfg.head_dim in
    let k = Array.init cfg.n_layers (fun _ -> Tensor.zeros sz) in
    let v = Array.init cfg.n_layers (fun _ -> Tensor.zeros sz) in
    {
      k;
      v;
      max_seq;
      n_kv_heads = cfg.n_kv_heads;
      head_dim = cfg.head_dim;
    }

  let reset (c : t) : unit =
    Array.iter (fun a -> Bigarray.Array1.fill a 0.0) c.k;
    Array.iter (fun a -> Bigarray.Array1.fill a 0.0) c.v
end

(* Per-call reusable scratch for attention scores. Sized lazily up to the
   largest n_keys seen; the single buffer is reused across heads within a call
   (and across calls), so the inner loops do no heap allocation. *)
let scratch : Tensor.t ref = ref (Tensor.create 0)

let ensure_scratch n =
  if Tensor.length !scratch < n then scratch := Tensor.create n

let forward (cfg : Config.t) (cache : Kv_cache.t) ~layer ~pos
    ~(q : Tensor.t) ~(k : Tensor.t) ~(v : Tensor.t) ~(out : Tensor.t) : unit =
  let head_dim = cfg.head_dim in
  let n_heads = cfg.n_heads in
  let n_kv_heads = cfg.n_kv_heads in
  let group = n_heads / n_kv_heads in
  let max_seq = cache.Kv_cache.max_seq in
  let kc = cache.Kv_cache.k.(layer) in
  let vc = cache.Kv_cache.v.(layer) in
  let scale = 1.0 /. sqrt (float_of_int head_dim) in
  let n_keys = pos + 1 in
  ensure_scratch n_keys;
  let scores = !scratch in

  (* Write current k,v into the cache at [pos], per kv-head contiguous.
     Cache layout for kv-head kh, position p: base = (kh*max_seq + p)*head_dim. *)
  for kh = 0 to n_kv_heads - 1 do
    let src = kh * head_dim in
    let dst = (kh * max_seq + pos) * head_dim in
    for d = 0 to head_dim - 1 do
      Tensor.set kc (dst + d) (Tensor.get k (src + d));
      Tensor.set vc (dst + d) (Tensor.get v (src + d))
    done
  done;

  for h = 0 to n_heads - 1 do
    let kh = h / group in
    let q_off = h * head_dim in
    (* scores[p] = dot(q_h, k_cached[kh,p]) * scale, for p in 0..pos (causal) *)
    for p = 0 to pos do
      let k_off = (kh * max_seq + p) * head_dim in
      let acc = ref 0.0 in
      for d = 0 to head_dim - 1 do
        acc := !acc +. (Tensor.get q (q_off + d) *. Tensor.get kc (k_off + d))
      done;
      Tensor.set scores p (!acc *. scale)
    done;
    Ops.softmax_inplace scores ~off:0 n_keys;
    (* out[h] = sum_p scores[p] * v_cached[kh,p] *)
    let o_off = h * head_dim in
    for d = 0 to head_dim - 1 do
      Tensor.set out (o_off + d) 0.0
    done;
    for p = 0 to pos do
      let w = Tensor.get scores p in
      let v_off = (kh * max_seq + p) * head_dim in
      for d = 0 to head_dim - 1 do
        Tensor.set out (o_off + d)
          (Tensor.get out (o_off + d) +. (w *. Tensor.get vc (v_off + d)))
      done
    done
  done
