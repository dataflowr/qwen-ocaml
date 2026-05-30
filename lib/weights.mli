(* ============================================================================
   SEAM: the dtype-agnostic weight.  This is the spine that lets ONE
   Model.forward serve both the fp32 (Safetensors) and quantized (GGUF) paths.

   A weight is just two closures + its shape:
     matvec out x : out = W . x   (out has [rows], x has [cols])
     row i dst    : dst.[0..cols) = W[i,:]   (dequant/copy one row)

   The two constructors choose the backend: [dense] -> Matmul (fp32),
   [quant] -> Quant (the kernel seam). New backends (a new dtype, a fused
   path) add a constructor here without touching Model.forward.
   ============================================================================ *)

type t = {
  matvec : Tensor.t -> Tensor.t -> unit;  (* matvec out x : out = W . x *)
  row    : int -> Tensor.t -> unit;       (* row i dst : dst.[0..cols) = W[i,:] *)
  rows   : int;
  cols   : int;
}

(* dense fp32 weight backed by a [rows*cols] row-major Tensor.t. *)
val dense : Tensor.t -> rows:int -> cols:int -> t

(* quantized GGUF weight: rows contiguous from absolute byte offset [off] in
   [blob], per-row stride = Quant.row_bytes ~ty ~cols. *)
val quant : Quant.blob -> off:int -> ty:int -> rows:int -> cols:int -> t
