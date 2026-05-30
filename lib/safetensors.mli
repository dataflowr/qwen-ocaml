(* Safetensors reader. File layout:
     u64 header_len (LE) | JSON header | raw tensor blob
   JSON maps name -> { "dtype", "shape":[..], "data_offsets":[start,end] }
   (offsets are relative to the start of the blob). Qwen2.5 weights are BF16;
   tensor returns them materialized as row-major float32. *)

type t

(* mmap (or read) the file and parse the header. *)
val open_file : string -> t

val names : t -> string list
val mem : t -> string -> bool

(* logical shape, e.g. [|896; 896|] for a [out;in] Linear weight. *)
val shape : t -> string -> int array

(* materialized row-major float32 copy of the named tensor (BF16/F16/F32 -> F32). *)
val tensor : t -> string -> Tensor.t

val close : t -> unit
