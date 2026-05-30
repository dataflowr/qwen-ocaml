(* Prefill (full prompt, compute-bound) + decode loop (one token/step,
   bandwidth-bound). Calls Model.forward, then greedy argmax, streams tokens. *)

let argmax (a : Tensor.t) n =
  let best = ref 0 and bv = ref (Tensor.get a 0) in
  for i = 1 to n - 1 do
    let v = Tensor.get a i in
    if v > !bv then (bv := v; best := i)
  done;
  !best

let run (m : Model.t) ~(prompt_ids : int list) ~(max_tokens : int) : int list =
  let cfg = m.Model.cfg in
  let vocab = cfg.vocab_size in
  let prompt = Array.of_list prompt_ids in
  let plen = Array.length prompt in
  if plen = 0 then invalid_arg "Generate.run: empty prompt";
  let pos = ref 0 in
  let last = ref (Tensor.create 0) in
  (* prefill: feed each prompt token at its position *)
  for i = 0 to plen - 1 do
    last := Model.forward m ~token:prompt.(i) ~pos:!pos;
    incr pos
  done;
  (* greedy decode *)
  let out = ref [] in
  let next = ref (argmax !last vocab) in
  for _ = 1 to max_tokens do
    out := !next :: !out;
    last := Model.forward m ~token:!next ~pos:!pos;
    incr pos;
    next := argmax !last vocab
  done;
  List.rev !out
