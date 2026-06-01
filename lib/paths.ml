(* Resolve data files (models/, scripts/) relative to the repo root so the
   engine is portable. Default root is the current directory -- run the
   executables from the repo root, e.g.

     opam exec --switch=qwen -- dune exec bin/validate_gguf.exe

   Override the root with the QWEN_DIR environment variable if you run from
   elsewhere or keep the large model files outside the checkout:

     QWEN_DIR=/path/to/qwen-ocaml dune exec bin/run.exe *)

let root =
  match Sys.getenv_opt "QWEN_DIR" with
  | Some d when d <> "" -> d
  | _ -> "."

(* resolve a repo-relative path (e.g. "models/Qwen2.5-0.5B/tokenizer.json"). *)
let resolve (rel : string) : string = Filename.concat root rel
