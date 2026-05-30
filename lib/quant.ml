(* ggml quant dequant + fused GEMV kernels (C stubs in stubs.c).
   Supports ggml types: F32(0), Q5_0(6), Q8_0(8), Q4_K(12), Q6_K(14).

   Block math ported verbatim from llama.cpp ggml-quants.c
   (see scripts/ggml_block_layouts.md). Verified against the python `gguf`
   reference dequantizer to ~5e-11 for Q5_0/Q8_0/Q4_K/Q6_K. *)

(* Same concrete type as Gguf.blob (whole-file mmap); kept local so this
   module typechecks with no dependency on Gguf. *)
type blob = (int, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t

(* dequant_row blob byte_off ty dst n:
   dequant n weights of ggml type [ty] starting at absolute byte offset
   [byte_off] in [blob] into dst.[0 .. n-1]. [n] must be a multiple of the
   block element count for [ty] (32 for Q5_0/Q8_0, 256 for Q4_K/Q6_K). *)
external dequant_row : blob -> int -> int -> Tensor.t -> int -> unit
  = "caml_qwen_dequant_row"

(* qmatvec blob base_off ty x out rows cols:
   out.[r] = dot(dequant(row r), x.[0 .. cols-1]) for r in 0 .. rows-1.
   Rows are contiguous starting at [base_off]; per-row byte stride is
   (cols/block_elems)*block_bytes. F32 dispatches to Accelerate sgemv. *)
external qmatvec : blob -> int -> int -> Tensor.t -> Tensor.t -> int -> int -> unit
  = "caml_qwen_qmatvec_bc" "caml_qwen_qmatvec"

(* per-row byte stride for a 2D weight row of [cols] elements of ggml type [ty]. *)
let row_bytes ~ty ~cols =
  (* require cols to be an exact multiple of the block element count; a
     non-multiple would silently drop the tail weights in the kernel. *)
  let exact blk bytes =
    if cols mod blk <> 0 then
      invalid_arg
        (Printf.sprintf "Quant.row_bytes: cols %d not a multiple of block %d" cols blk);
    (cols / blk) * bytes
  in
  match ty with
  | 0  -> cols * 4               (* F32:  4 bytes / elem    *)
  | 8  -> exact 32 34            (* Q8_0: 34 bytes / 32     *)
  | 6  -> exact 32 22            (* Q5_0: 22 bytes / 32     *)
  | 12 -> exact 256 144          (* Q4_K: 144 bytes / 256   *)
  | 14 -> exact 256 210          (* Q6_K: 210 bytes / 256   *)
  | _  -> invalid_arg "Quant.row_bytes: unsupported ggml type"
