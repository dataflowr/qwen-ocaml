(* ============================================================================
   SEAM: quantized kernel boundary.  Projects P1, P2, P3, P4 plug in HERE.

   This signature is the CONTRACT. A project may change the IMPLEMENTATION
   behind it (the C in lib/stubs.c, and for P4 the [row_bytes] dispatch in
   quant.ml) but must NOT change these types or signatures — the rest of the
   engine, and every validator, depends on exactly this surface.

     P1 (vectorize dequant)  -> rewrite dequant_row_* in stubs.c
     P2 (int8 dot)           -> rewrite caml_qwen_qmatvec in stubs.c
     P3 (threading)          -> rewrite caml_qwen_qmatvec dispatch in stubs.c
     P4 (lower bpw)          -> add a ggml type case to [row_bytes] + a C dequant case

   Grade: validate_gguf.exe must stay MATCH: true (P1/P3 bit-exact, P2 exact
   vs llama.cpp). Numbers and the layout notes live in scripts/ggml_block_layouts.md.
   ============================================================================ *)

(* Whole-file GGUF mmap (same concrete type as Gguf.blob; kept local so this
   module typechecks with no dependency on Gguf). *)
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

(* per-row byte stride for a 2D weight row of [cols] elements of ggml type [ty].
   P4: add your new ggml type here (and the matching dequant case in stubs.c). *)
val row_bytes : ty:int -> cols:int -> int
