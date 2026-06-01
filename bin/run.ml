(* CLI: load model, encode prompt, generate. *)
let model_dir = Qwen.Paths.resolve "models/Qwen2.5-0.5B"

let () =
  let prompt =
    if Array.length Sys.argv > 1 then
      String.concat " " (Array.to_list (Array.sub Sys.argv 1 (Array.length Sys.argv - 1)))
    else "The capital of France is"
  in
  let max_tokens =
    match Sys.getenv_opt "MAX_TOKENS" with Some s -> int_of_string s | None -> 32
  in
  let cfg = Qwen.Config.qwen2_5_0_5b in
  Printf.eprintf "loading model...\n%!";
  let t0 = Unix.gettimeofday () in
  let m =
    match Sys.getenv_opt "QWEN_GGUF" with
    | Some gguf when gguf <> "" -> Qwen.Model.create_gguf cfg gguf
    | _ -> Qwen.Model.create cfg (Filename.concat model_dir "model.safetensors")
  in
  let tok = Qwen.Tokenizer.load (Filename.concat model_dir "tokenizer.json") in
  let t1 = Unix.gettimeofday () in
  let ids = Array.to_list (Qwen.Tokenizer.encode tok prompt) in
  Printf.eprintf "prompt ids: [%s]\n%!"
    (String.concat ";" (List.map string_of_int ids));
  let gen = Qwen.Generate.run m ~prompt_ids:ids ~max_tokens in
  let t2 = Unix.gettimeofday () in
  let text = Qwen.Tokenizer.decode tok (Array.of_list gen) in
  Printf.printf "%s%s\n%!" prompt text;
  let ntok = List.length gen in
  Printf.eprintf "\n[load %.2fs | generated %d tok in %.2fs = %.1f tok/s]\n%!"
    (t1 -. t0) ntok (t2 -. t1)
    (float_of_int ntok /. (t2 -. t1))
