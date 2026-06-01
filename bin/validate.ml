(* Compare engine logits/tokens against a precomputed reference:
   - scripts/ref_tokens.json : prompt_ids, greedy_tokens, n_steps, vocab
   - scripts/ref_logits.f32   : raw LE float32, n_steps*vocab *)

let model_path = Qwen.Paths.resolve "models/Qwen2.5-0.5B/model.safetensors"
let tokens_json = Qwen.Paths.resolve "scripts/ref_tokens.json"
let logits_f32 = Qwen.Paths.resolve "scripts/ref_logits.f32"

(* hand parse: extract int list following "key" *)
let read_file path =
  let ic = open_in_bin path in
  let n = in_channel_length ic in
  let s = really_input_string ic n in
  close_in ic; s

let parse_int_list s key =
  let re = Re.compile (Re.Perl.re (key ^ "\"\\s*:\\s*\\[([^]]*)\\]")) in
  match Re.exec_opt re s with
  | None -> failwith ("key not found: " ^ key)
  | Some g ->
    let body = Re.Group.get g 1 in
    String.split_on_char ',' body
    |> List.filter_map (fun t ->
         let t = String.trim t in
         if t = "" then None else Some (int_of_string t))

let parse_int s key =
  let re = Re.compile (Re.Perl.re (key ^ "\"\\s*:\\s*([0-9]+)")) in
  match Re.exec_opt re s with
  | None -> failwith ("key not found: " ^ key)
  | Some g -> int_of_string (Re.Group.get g 1)

let load_ref_logits path n =
  let s = read_file path in
  let b = Bytes.of_string s in
  let a = Array.make n 0.0 in
  for i = 0 to n - 1 do
    let bits = Bytes.get_int32_le b (i * 4) in
    a.(i) <- Int32.float_of_bits bits
  done;
  a

let () =
  let js = read_file tokens_json in
  let prompt_ids = parse_int_list js "prompt_ids" in
  let greedy = parse_int_list js "greedy_tokens" in
  let n_steps = parse_int js "n_steps" in
  let vocab = parse_int js "vocab" in
  Printf.printf "prompt_ids: [%s]\n%!"
    (String.concat ";" (List.map string_of_int prompt_ids));
  Printf.printf "loading model...\n%!";
  let m = Qwen.Model.create Qwen.Config.qwen2_5_0_5b model_path in
  (* prefill, capturing step-0 logits = logits after last prompt token *)
  let prompt = Array.of_list prompt_ids in
  let plen = Array.length prompt in
  let pos = ref 0 in
  let step0 = ref (Qwen.Tensor.create 0) in
  for i = 0 to plen - 1 do
    let lg = Qwen.Model.forward m ~token:prompt.(i) ~pos:!pos in
    if i = plen - 1 then step0 := Qwen.Tensor.copy lg;
    incr pos
  done;
  (* compare step-0 logits *)
  let ref_logits = load_ref_logits logits_f32 (n_steps * vocab) in
  let maxdiff = ref 0.0 in
  for j = 0 to vocab - 1 do
    let d = abs_float (Qwen.Tensor.get !step0 j -. ref_logits.(j)) in
    if d > !maxdiff then maxdiff := d
  done;
  Printf.printf "step-0 max abs logit diff: %g\n%!" !maxdiff;
  (* our step-0 argmax *)
  let argmax (a : Qwen.Tensor.t) n =
    let best = ref 0 and bv = ref (Qwen.Tensor.get a 0) in
    for i = 1 to n - 1 do
      let v = Qwen.Tensor.get a i in
      if v > !bv then (bv := v; best := i)
    done; !best
  in
  Printf.printf "step-0 argmax got=%d expected=%d\n%!" (argmax !step0 vocab)
    (List.nth greedy 0);
  (* now greedy decode continuing from prefill state *)
  let got = ref [] in
  let next = ref (argmax !step0 vocab) in
  for _ = 1 to n_steps do
    got := !next :: !got;
    let lg = Qwen.Model.forward m ~token:!next ~pos:!pos in
    incr pos;
    next := argmax lg vocab
  done;
  let got = List.rev !got in
  Printf.printf "greedy got     : [%s]\n%!"
    (String.concat ";" (List.map string_of_int got));
  Printf.printf "greedy expected: [%s]\n%!"
    (String.concat ";" (List.map string_of_int greedy));
  let matches = (got = greedy) in
  Printf.printf "MATCH: %b\n%!" matches
