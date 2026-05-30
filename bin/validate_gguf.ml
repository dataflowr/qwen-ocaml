(* Load the quantized GGUF via Model.create_gguf, greedy-decode ~40 tokens from
   prompt "The capital of France is" (prompt_ids [785;6722;315;9625;374]),
   detokenize, and compare against the llama.cpp greedy reference text in
   scripts/llama_ref_gguf.txt. Times load + tok/s. *)

let gguf_path =
  "/Users/lelarge/courses/Qwen_Ocaml/models/Qwen2.5-0.5B-Instruct-GGUF/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf"
let tokenizer_path = "/Users/lelarge/courses/Qwen_Ocaml/models/Qwen2.5-0.5B/tokenizer.json"
let ref_path = "/Users/lelarge/courses/Qwen_Ocaml/scripts/llama_ref_gguf.txt"

let prompt_ids = [785; 6722; 315; 9625; 374]
let prompt = "The capital of France is"
let n_gen = 40

let argmax (a : Qwen.Tensor.t) n =
  let best = ref 0 and bv = ref (Qwen.Tensor.get a 0) in
  for i = 1 to n - 1 do
    let v = Qwen.Tensor.get a i in
    if v > !bv then (bv := v; best := i)
  done;
  !best

let read_file path =
  let ic = open_in_bin path in
  let n = in_channel_length ic in
  let s = really_input_string ic n in
  close_in ic; s

let () =
  let cfg = Qwen.Config.qwen2_5_0_5b in
  Printf.printf "loading GGUF model...\n%!";
  let t0 = Unix.gettimeofday () in
  let m = Qwen.Model.create_gguf cfg gguf_path in
  let tok = Qwen.Tokenizer.load tokenizer_path in
  let t1 = Unix.gettimeofday () in
  let load_s = t1 -. t0 in
  Printf.printf "[load %.2fs]\n%!" load_s;
  (* prefill *)
  let prompt_arr = Array.of_list prompt_ids in
  let plen = Array.length prompt_arr in
  let pos = ref 0 in
  let last = ref (Qwen.Tensor.create 0) in
  for i = 0 to plen - 1 do
    last := Qwen.Model.forward m ~token:prompt_arr.(i) ~pos:!pos;
    incr pos
  done;
  (* greedy decode n_gen tokens *)
  let t2 = Unix.gettimeofday () in
  let got = ref [] in
  let next = ref (argmax !last cfg.vocab_size) in
  for _ = 1 to n_gen do
    got := !next :: !got;
    last := Qwen.Model.forward m ~token:!next ~pos:!pos;
    incr pos;
    next := argmax !last cfg.vocab_size
  done;
  let t3 = Unix.gettimeofday () in
  let got = List.rev !got in
  let tok_per_s = float_of_int n_gen /. (t3 -. t2) in
  let gen_text = Qwen.Tokenizer.decode tok (Array.of_list got) in
  let full = prompt ^ gen_text in
  Printf.printf "generated ids: [%s]\n%!"
    (String.concat ";" (List.map string_of_int got));
  Printf.printf "GOT : %s\n%!" full;
  let reference = String.trim (read_file ref_path) in
  Printf.printf "REF : %s\n%!" reference;
  Printf.printf "[generated %d tok in %.2fs = %.1f tok/s]\n%!"
    n_gen (t3 -. t2) tok_per_s;
  (* compare on the leading ~30 generated tokens (prefix match of generated
     text against the reference's continuation). *)
  let ref_cont =
    let pl = String.length prompt in
    if String.length reference >= pl
       && String.sub reference 0 pl = prompt
    then String.sub reference pl (String.length reference - pl)
    else reference
  in
  let g = String.trim gen_text and r = String.trim ref_cont in
  let n_cmp = min (min (String.length g) (String.length r)) 130 in
  let prefix_match =
    n_cmp > 0 && String.sub g 0 n_cmp = String.sub r 0 n_cmp
  in
  Printf.printf "MATCH (leading ~30 tok / %d chars): %b\n%!" n_cmp prefix_match
