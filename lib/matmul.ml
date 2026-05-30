(* Matmul dispatch. fp32 GEMV delegates to Accelerate cblas_sgemv (AMX-backed)
   via the C stub; quantized GEMV uses the fused dequant+dot C kernels (Quant). *)

(* sanity link to the C stub (used by the kernel unit tests) *)
external sdot : Tensor.t -> Tensor.t -> int -> float = "caml_qwen_sdot"

(* fp32 GEMV via Accelerate cblas_sgemv (AMX-backed). y = W*x, W row-major
   [rows x cols]. Same signature as matvec_naive — drop-in replacement. *)
external sgemv : Tensor.t -> Tensor.t -> Tensor.t -> int -> int -> unit
  = "caml_qwen_sgemv"

let matvec ~(w : Tensor.t) ~(x : Tensor.t) ~(out : Tensor.t) ~rows ~cols =
  sgemv w x out rows cols
