(* ============================================================================
   SEAM: dense fp32 GEMV.  The M2 "BLAS is the floor" baseline lives here.

   matvec is a drop-in for a naive triple loop: y = W . x, W row-major
   [rows x cols]. The baseline delegates to Accelerate cblas_sgemv (AMX-backed).
   A project that wants to explore the fp32 path (custom NEON GEMV, blocking,
   etc.) replaces the BODY of matmul.ml while keeping this signature.
   ============================================================================ *)

(* fp32 dot of two Tensors over the first [n] elements (AMX cblas_sdot).
   Exposed for kernel unit tests. *)
external sdot : Tensor.t -> Tensor.t -> int -> float = "caml_qwen_sdot"

(* fp32 GEMV via Accelerate cblas_sgemv: out = W . x, W row-major [rows x cols]. *)
external sgemv : Tensor.t -> Tensor.t -> Tensor.t -> int -> int -> unit
  = "caml_qwen_sgemv"

(* matvec ~w ~x ~out ~rows ~cols : out = W . x. The engine's fp32 matvec. *)
val matvec :
  w:Tensor.t -> x:Tensor.t -> out:Tensor.t -> rows:int -> cols:int -> unit
