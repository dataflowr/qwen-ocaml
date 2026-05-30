(* Byte-level BPE (GPT-2/cl100k family) loaded from a HuggingFace tokenizer.json.
   Special/added tokens are split out before BPE. ChatML helpers for chat use. *)

type t

(* load vocab + merges + added_tokens from a tokenizer.json path. *)
val load : string -> t

val encode : t -> string -> int array
val decode : t -> int array -> string

val eos_id : t -> int        (* <|endoftext|> = 151643 *)
val im_start_id : t -> int   (* <|im_start|> = 151644 *)
val im_end_id : t -> int     (* <|im_end|>   = 151645 *)

(* Wrap a user message in the ChatML template, primed for assistant generation:
   <|im_start|>user\n{msg}<|im_end|>\n<|im_start|>assistant\n *)
val encode_chat : t -> string -> int array
