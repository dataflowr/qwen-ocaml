(* GGUF v3 reader (little-endian). Header:
     u32 magic "GGUF" | u32 version | u64 tensor_count | u64 kv_count
   then kv_count metadata pairs, then tensor_count tensor-info entries
   (name | u32 n_dims | u64 dims[] | u32 ggml_type | u64 offset), then padding
   to general.alignment (default 32), then the tensor data section.
   tensor offsets are relative to the start of the data section. *)

type tensor_info = {
  ty : int;          (* ggml type id: F32=0,Q5_0=6,Q8_0=8,Q4_K=12,Q6_K=14,... *)
  dims : int array;  (* ggml order: dims.(0)=ne0=cols(in), dims.(1)=ne1=rows(out) *)
  offset : int;      (* byte offset within the data section *)
}

type blob = (int, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t

type t

val open_file : string -> t

val names : t -> string list
val mem : t -> string -> bool
val info : t -> string -> tensor_info

(* whole-file mmap; absolute byte offset of a tensor = data_start t + info.offset *)
val blob : t -> blob
val data_start : t -> int

val metadata_int : t -> string -> int option
val metadata_float : t -> string -> float option
val metadata_string : t -> string -> string option

val close : t -> unit
