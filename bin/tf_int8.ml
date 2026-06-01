(* Teacher-forced top-1 agreement of the active GGUF decode path vs the llama.cpp
   greedy reference tokens (same prompt as validate_gguf). Feeds the REFERENCE
   tokens at every step and checks whether the engine's argmax equals the next
   reference token -- the honest faithfulness metric for a lossy int8 kernel
   (free-running diverges at the first near-tie and compounds). *)

let gguf_path =
  Qwen.Paths.resolve "models/Qwen2.5-0.5B-Instruct-GGUF/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf"

let prompt_ids = [785; 6722; 315; 9625; 374]

(* llama.cpp greedy reference (== our fp32 path), captured with QWEN_KQ_FP32=1 *)
let ref_tokens =
  [12095;13;1084;374;279;7772;3283;304;4505;323;279;4843;7772;3283;304;279;1879;
   13;1084;374;7407;304;279;9806;315;9625;11;389;279;18494;13648;315;279;37685;
   15029;13;1084;374;30083;389]

let argmax (a : Qwen.Tensor.t) n =
  let best = ref 0 and bv = ref (Qwen.Tensor.get a 0) in
  for i = 1 to n - 1 do
    let v = Qwen.Tensor.get a i in
    if v > !bv then (bv := v; best := i)
  done;
  !best

let () =
  let cfg = Qwen.Config.qwen2_5_0_5b in
  let m = Qwen.Model.create_gguf cfg gguf_path in
  let vocab = cfg.vocab_size in
  let pos = ref 0 in
  let logits = ref (Qwen.Tensor.create 0) in
  List.iter (fun t -> logits := Qwen.Model.forward m ~token:t ~pos:!pos; incr pos) prompt_ids;
  let agree = ref 0 and total = ref 0 and first_miss = ref (-1) in
  List.iteri (fun i t ->
    let pred = argmax !logits vocab in
    incr total;
    if pred = t then incr agree
    else if !first_miss < 0 then first_miss := i;
    logits := Qwen.Model.forward m ~token:t ~pos:!pos; incr pos
  ) ref_tokens;
  Printf.printf "teacher-forced top-1 agreement: %d/%d (%.1f%%); first miss at step %d\n%!"
    !agree !total (100. *. float_of_int !agree /. float_of_int !total) !first_miss
