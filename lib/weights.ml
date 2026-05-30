(* Dtype-agnostic weight: a 2D [rows x cols] matrix that can be either dense
   fp32 (Safetensors path) or a quantized GGUF tensor. Exposes:
     - matvec out x : out = W . x        (out has [rows], x has [cols])
     - row i dst    : dequant/copy row i into dst.[0 .. cols-1]
   so Model.forward can share one path for both fp32 and quantized weights. *)

type t = {
  matvec : Tensor.t -> Tensor.t -> unit;  (* matvec out x : out = W . x *)
  row    : int -> Tensor.t -> unit;       (* row i dst : dst.[0..cols) = W[i,:] *)
  rows   : int;
  cols   : int;
}

(* dense fp32 weight backed by a [rows*cols] Tensor.t (row-major). *)
let dense (w : Tensor.t) ~rows ~cols : t =
  let matvec out x = Matmul.matvec ~w ~x ~out ~rows ~cols in
  let row i dst =
    let base = i * cols in
    for j = 0 to cols - 1 do
      Tensor.set dst j (Tensor.get w (base + j))
    done
  in
  { matvec; row; rows; cols }

(* quantized GGUF weight: rows are contiguous starting at absolute byte offset
   [off] in [blob], per-row stride = Quant.row_bytes ~ty ~cols. *)
let quant (blob : Quant.blob) ~off ~ty ~rows ~cols : t =
  let matvec out x = Quant.qmatvec blob off ty x out rows cols in
  let stride = Quant.row_bytes ~ty ~cols in
  let row i dst = Quant.dequant_row blob (off + i * stride) ty dst cols in
  { matvec; row; rows; cols }
