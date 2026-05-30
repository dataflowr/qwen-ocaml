(* Qwen2/Qwen3 forward pass: embed -> N decoder layers -> final RMSNorm -> lm_head.
   Per layer: RMSNorm, QKV (+bias if Qwen2), optional QK-norm (Qwen3) pre-RoPE,
   RoPE, GQA attention w/ KV cache, residual; RMSNorm, SwiGLU MLP, residual.
   lm_head weight tied to embed_tokens.

   Weights are Weights.t (dtype-agnostic): the fp32 Safetensors path uses
   Weights.dense, the quantized GGUF path uses Weights.quant. Both share one
   forward. Norms and biases stay plain F32 Tensor.t. *)

type layer = {
  input_ln : Tensor.t;          (* [hidden] *)
  wq : Weights.t;               (* [n_heads*head_dim, hidden] *)
  bq : Tensor.t option;         (* [n_heads*head_dim]; None for Qwen3 *)
  wk : Weights.t;               (* [n_kv_heads*head_dim, hidden] *)
  bk : Tensor.t option;         (* [n_kv_heads*head_dim]; None for Qwen3 *)
  wv : Weights.t;               (* [n_kv_heads*head_dim, hidden] *)
  bv : Tensor.t option;         (* [n_kv_heads*head_dim]; None for Qwen3 *)
  wo : Weights.t;               (* [hidden, n_heads*head_dim] (no bias) *)
  q_norm : Tensor.t option;     (* [head_dim]; Qwen3 QK-Norm on q, pre-RoPE *)
  k_norm : Tensor.t option;     (* [head_dim]; Qwen3 QK-Norm on k, pre-RoPE *)
  post_ln : Tensor.t;           (* [hidden] *)
  wgate : Weights.t;            (* [inter, hidden] *)
  wup : Weights.t;              (* [inter, hidden] *)
  wdown : Weights.t;            (* [hidden, inter] *)
}

type t = {
  cfg : Config.t;
  embed : Weights.t;            (* [vocab, hidden]; tied lm_head *)
  final_ln : Tensor.t;          (* [hidden] *)
  layers : layer array;
  cos : Tensor.t;
  sin : Tensor.t;
  kv : Attention.Kv_cache.t;
  (* scratch buffers, reused across tokens and layers *)
  x : Tensor.t;                 (* hidden *)
  h : Tensor.t;                 (* hidden *)
  q : Tensor.t;                 (* n_heads*head_dim *)
  k : Tensor.t;                 (* n_kv_heads*head_dim *)
  v : Tensor.t;                 (* n_kv_heads*head_dim *)
  attn : Tensor.t;              (* n_heads*head_dim *)
  o : Tensor.t;                 (* hidden *)
  g : Tensor.t;                 (* inter *)
  u : Tensor.t;                 (* inter *)
  s : Tensor.t;                 (* inter *)
  d : Tensor.t;                 (* hidden *)
  logits : Tensor.t;            (* vocab *)
}

let max_seq = 4096

(* shared scratch + rope + kv + record assembly, given Weights.t per tensor. *)
let assemble (cfg : Config.t) ~embed ~final_ln ~layers : t =
  let qdim = cfg.n_heads * cfg.head_dim in
  let kvdim = cfg.n_kv_heads * cfg.head_dim in
  let cos, sin =
    Ops.rope_tables ~head_dim:cfg.head_dim ~max_pos:max_seq ~theta:cfg.rope_theta
  in
  let kv = Attention.Kv_cache.create cfg ~max_seq in
  {
    cfg; embed; final_ln; layers; cos; sin; kv;
    x = Tensor.create cfg.hidden_size;
    h = Tensor.create cfg.hidden_size;
    q = Tensor.create qdim;
    k = Tensor.create kvdim;
    v = Tensor.create kvdim;
    attn = Tensor.create qdim;
    o = Tensor.create cfg.hidden_size;
    g = Tensor.create cfg.intermediate_size;
    u = Tensor.create cfg.intermediate_size;
    s = Tensor.create cfg.intermediate_size;
    d = Tensor.create cfg.hidden_size;
    logits = Tensor.create cfg.vocab_size;
  }

let create (cfg : Config.t) (path : string) : t =
  let st = Safetensors.open_file path in
  let g name = Safetensors.tensor st name in
  let qdim = cfg.n_heads * cfg.head_dim in
  let kvdim = cfg.n_kv_heads * cfg.head_dim in
  let hidden = cfg.hidden_size in
  let inter = cfg.intermediate_size in
  let dense w ~rows ~cols = Weights.dense w ~rows ~cols in
  (* Qwen2 has q/k/v bias; Qwen3 has none. Qwen3 has q/k RMSNorm; Qwen2 none. *)
  let bias p name = if cfg.qkv_bias then Some (g (p ^ name)) else None in
  let norm p name = if cfg.qk_norm then Some (g (p ^ name)) else None in
  let layers =
    Array.init cfg.n_layers (fun l ->
      let p = Printf.sprintf "model.layers.%d." l in
      {
        input_ln = g (p ^ "input_layernorm.weight");
        wq = dense (g (p ^ "self_attn.q_proj.weight")) ~rows:qdim ~cols:hidden;
        bq = bias p "self_attn.q_proj.bias";
        wk = dense (g (p ^ "self_attn.k_proj.weight")) ~rows:kvdim ~cols:hidden;
        bk = bias p "self_attn.k_proj.bias";
        wv = dense (g (p ^ "self_attn.v_proj.weight")) ~rows:kvdim ~cols:hidden;
        bv = bias p "self_attn.v_proj.bias";
        wo = dense (g (p ^ "self_attn.o_proj.weight")) ~rows:hidden ~cols:qdim;
        q_norm = norm p "self_attn.q_norm.weight";
        k_norm = norm p "self_attn.k_norm.weight";
        post_ln = g (p ^ "post_attention_layernorm.weight");
        wgate = dense (g (p ^ "mlp.gate_proj.weight")) ~rows:inter ~cols:hidden;
        wup = dense (g (p ^ "mlp.up_proj.weight")) ~rows:inter ~cols:hidden;
        wdown = dense (g (p ^ "mlp.down_proj.weight")) ~rows:hidden ~cols:inter;
      })
  in
  let embed =
    dense (g "model.embed_tokens.weight") ~rows:cfg.vocab_size ~cols:hidden
  in
  let final_ln = g "model.norm.weight" in
  Safetensors.close st;
  assemble cfg ~embed ~final_ln ~layers

(* Build a plain F32 Tensor.t from a GGUF F32 tensor (norms, biases). *)
let gguf_f32 (gg : Gguf.t) (name : string) : Tensor.t =
  let info = Gguf.info gg name in
  let blob = Gguf.blob gg in
  let off = Gguf.data_start gg + info.offset in
  let n = Array.fold_left ( * ) 1 info.dims in
  let dst = Tensor.create n in
  (* ty=0 (F32): dequant_row is a straight copy. *)
  Quant.dequant_row blob off info.ty dst n;
  dst

let create_gguf (cfg : Config.t) (path : string) : t =
  let gg = Gguf.open_file path in
  let blob = Gguf.blob gg in
  let data_start = Gguf.data_start gg in
  (* quantized/any 2D weight -> Weights.quant; rows=dims.(1), cols=dims.(0). *)
  let w name =
    let info = Gguf.info gg name in
    let off = data_start + info.offset in
    let cols = info.dims.(0) and rows = info.dims.(1) in
    Weights.quant blob ~off ~ty:info.ty ~rows ~cols
  in
  let layers =
    Array.init cfg.n_layers (fun l ->
      let p = Printf.sprintf "blk.%d." l in
      {
        input_ln = gguf_f32 gg (p ^ "attn_norm.weight");
        wq = w (p ^ "attn_q.weight");
        bq = (if cfg.qkv_bias then Some (gguf_f32 gg (p ^ "attn_q.bias")) else None);
        wk = w (p ^ "attn_k.weight");
        bk = (if cfg.qkv_bias then Some (gguf_f32 gg (p ^ "attn_k.bias")) else None);
        wv = w (p ^ "attn_v.weight");
        bv = (if cfg.qkv_bias then Some (gguf_f32 gg (p ^ "attn_v.bias")) else None);
        wo = w (p ^ "attn_output.weight");
        q_norm = (if cfg.qk_norm then Some (gguf_f32 gg (p ^ "attn_q_norm.weight")) else None);
        k_norm = (if cfg.qk_norm then Some (gguf_f32 gg (p ^ "attn_k_norm.weight")) else None);
        post_ln = gguf_f32 gg (p ^ "ffn_norm.weight");
        wgate = w (p ^ "ffn_gate.weight");
        wup = w (p ^ "ffn_up.weight");
        wdown = w (p ^ "ffn_down.weight");
      })
  in
  (* token_embd is the tied lm_head: same Weights.quant for embed row lookup
     AND final logits matvec. *)
  let embed = w "token_embd.weight" in
  let final_ln = gguf_f32 gg "output_norm.weight" in
  assemble cfg ~embed ~final_ln ~layers
  (* keep gg / blob alive: closing would munmap the data the Weights point at,
     so we deliberately do NOT close gg here (mmap freed at process exit). *)

let add_bias_opt ~(out : Tensor.t) ~(bias : Tensor.t option) n =
  match bias with
  | None -> ()
  | Some b ->
    for i = 0 to n - 1 do
      Tensor.set out i (Tensor.get out i +. Tensor.get b i)
    done

(* Qwen3 QK-Norm: RMSNorm (weight [head_dim], shared across heads) applied to
   each head's vector in place, before RoPE. No-op when [w] is None. *)
let qk_norm_inplace ~eps ~(w : Tensor.t option) (t : Tensor.t) ~n_heads ~head_dim =
  match w with
  | None -> ()
  | Some w ->
    for hh = 0 to n_heads - 1 do
      let s = Tensor.row ~cols:head_dim t hh in
      Ops.rmsnorm ~eps ~w ~x:s ~out:s head_dim
    done

let forward (m : t) ~(token : int) ~(pos : int) : Tensor.t =
  let cfg = m.cfg in
  let hidden = cfg.hidden_size in
  let inter = cfg.intermediate_size in
  let head_dim = cfg.head_dim in
  let eps = cfg.rms_eps in
  if pos < 0 || pos >= max_seq then
    invalid_arg (Printf.sprintf "Model.forward: pos %d outside [0,%d)" pos max_seq);
  if token < 0 || token >= cfg.vocab_size then
    invalid_arg (Printf.sprintf "Model.forward: token %d outside vocab" token);
  (* x = embed row[token] *)
  m.embed.row token m.x;
  for l = 0 to cfg.n_layers - 1 do
    let ly = m.layers.(l) in
    (* attention block *)
    Ops.rmsnorm ~eps ~w:ly.input_ln ~x:m.x ~out:m.h hidden;
    ly.wq.matvec m.q m.h;
    add_bias_opt ~out:m.q ~bias:ly.bq (Tensor.length m.q);
    ly.wk.matvec m.k m.h;
    add_bias_opt ~out:m.k ~bias:ly.bk (Tensor.length m.k);
    ly.wv.matvec m.v m.h;
    add_bias_opt ~out:m.v ~bias:ly.bv (Tensor.length m.v);
    (* Qwen3 QK-Norm (per head, pre-RoPE); no-op for Qwen2 *)
    qk_norm_inplace ~eps ~w:ly.q_norm m.q ~n_heads:cfg.n_heads ~head_dim;
    qk_norm_inplace ~eps ~w:ly.k_norm m.k ~n_heads:cfg.n_kv_heads ~head_dim;
    (* RoPE per head *)
    for hh = 0 to cfg.n_heads - 1 do
      Ops.rope_inplace ~cos:m.cos ~sin:m.sin ~pos ~head_dim m.q ~off:(hh * head_dim)
    done;
    for hh = 0 to cfg.n_kv_heads - 1 do
      Ops.rope_inplace ~cos:m.cos ~sin:m.sin ~pos ~head_dim m.k ~off:(hh * head_dim)
    done;
    Attention.forward cfg m.kv ~layer:l ~pos ~q:m.q ~k:m.k ~v:m.v ~out:m.attn;
    (* o_proj (no bias), residual *)
    ly.wo.matvec m.o m.attn;
    for i = 0 to hidden - 1 do
      Tensor.set m.x i (Tensor.get m.x i +. Tensor.get m.o i)
    done;
    (* MLP block *)
    Ops.rmsnorm ~eps ~w:ly.post_ln ~x:m.x ~out:m.h hidden;
    ly.wgate.matvec m.g m.h;
    ly.wup.matvec m.u m.h;
    Ops.swiglu ~gate:m.g ~up:m.u ~out:m.s inter;
    ly.wdown.matvec m.d m.s;
    for i = 0 to hidden - 1 do
      Tensor.set m.x i (Tensor.get m.x i +. Tensor.get m.d i)
    done
  done;
  Ops.rmsnorm ~eps ~w:m.final_ln ~x:m.x ~out:m.h hidden;
  (* tied lm_head: logits = embed @ h, embed is [vocab, hidden] *)
  m.embed.matvec m.logits m.h;
  m.logits
