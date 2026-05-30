(* Pure-OCaml elementwise/normalization ops, computed in fp32 over Bigarrays.
   These are the "20% surrounding code" the plan calls out: write them as plain
   for-loops with no intermediate allocation; Flambda's loopify does the rest. *)

(* RMSNorm: y = w * x / sqrt(mean(x^2) + eps). No mean subtraction, no bias.
   x and out are length n; w is length n. *)
let rmsnorm ~eps ~(w : Tensor.t) ~(x : Tensor.t) ~(out : Tensor.t) n =
  let ss = ref 0.0 in
  for i = 0 to n - 1 do
    let v = Tensor.get x i in
    ss := !ss +. (v *. v)
  done;
  let inv = 1.0 /. sqrt (!ss /. float_of_int n +. eps) in
  for i = 0 to n - 1 do
    Tensor.set out i (Tensor.get w i *. (Tensor.get x i *. inv))
  done

(* In-place softmax over a slice [off, off+n). Numerically stable. *)
let softmax_inplace (a : Tensor.t) ~off n =
  let m = ref neg_infinity in
  for i = off to off + n - 1 do
    let v = Tensor.get a i in
    if v > !m then m := v
  done;
  let sum = ref 0.0 in
  for i = off to off + n - 1 do
    let e = exp (Tensor.get a i -. !m) in
    Tensor.set a i e;
    sum := !sum +. e
  done;
  let inv = 1.0 /. !sum in
  for i = off to off + n - 1 do
    Tensor.set a i (Tensor.get a i *. inv)
  done

let[@inline] silu x = x /. (1.0 +. exp (-.x))

(* SwiGLU gating: out = silu(gate) * up, elementwise over length n. *)
let swiglu ~(gate : Tensor.t) ~(up : Tensor.t) ~(out : Tensor.t) n =
  for i = 0 to n - 1 do
    Tensor.set out i (silu (Tensor.get gate i) *. Tensor.get up i)
  done

(* RoPE, HuggingFace rotate_half convention: split head_dim into [0..d/2) and
   [d/2..d), pair index i with i+d/2. Applies in place to one head vector of
   length head_dim at absolute position [pos]. cos/sin are precomputed tables
   of shape [max_pos, head_dim/2]. *)
let rope_inplace ~(cos : Tensor.t) ~(sin : Tensor.t) ~pos ~head_dim (v : Tensor.t) ~off =
  let half = head_dim / 2 in
  let base = pos * half in
  for i = 0 to half - 1 do
    let c = Tensor.get cos (base + i) and s = Tensor.get sin (base + i) in
    let x0 = Tensor.get v (off + i) and x1 = Tensor.get v (off + i + half) in
    Tensor.set v (off + i) ((x0 *. c) -. (x1 *. s));
    Tensor.set v (off + i + half) ((x1 *. c) +. (x0 *. s))
  done

(* Precompute RoPE cos/sin tables consistent with [rope_inplace]'s indexing
   (base = pos * half, pairing index i with i+half — HF rotate_half).
   Layout: [max_pos, head_dim/2], flat row-major, index p*half + i.
     inv_freq_i = theta ** (-. (2*i) /. head_dim)
     angle      = p *. inv_freq_i
     cos[p*half+i] = cos angle ; sin[p*half+i] = sin angle *)
let rope_tables ~head_dim ~max_pos ~theta : Tensor.t * Tensor.t =
  let half = head_dim / 2 in
  let n = max_pos * half in
  let cos_t = Tensor.create n in
  let sin_t = Tensor.create n in
  let hd = float_of_int head_dim in
  for i = 0 to half - 1 do
    let inv_freq = theta ** (-. (float_of_int (2 * i)) /. hd) in
    for p = 0 to max_pos - 1 do
      let angle = float_of_int p *. inv_freq in
      let idx = (p * half) + i in
      Tensor.set cos_t idx (cos angle);
      Tensor.set sin_t idx (sin angle)
    done
  done;
  (cos_t, sin_t)
