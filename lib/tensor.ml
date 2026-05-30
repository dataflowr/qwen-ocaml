(* Tensor representation: flat float32 Bigarray.Array1 (c_layout) + explicit
   shape/stride bookkeeping. Array1 (not Array2) keeps memory contiguous and
   matches what the C/SIMD kernels expect (raw pointer via Caml_ba_data_val). *)

open Bigarray

type t = (float, float32_elt, c_layout) Array1.t

let create n : t = Array1.create Float32 c_layout n
let zeros n : t =
  let a = create n in
  Array1.fill a 0.0;
  a

let length (a : t) = Array1.dim a

(* unchecked accessors for hot loops *)
let[@inline] get (a : t) i = Array1.unsafe_get a i
let[@inline] set (a : t) i v = Array1.unsafe_set a i v

let copy (a : t) : t =
  let b = create (length a) in
  Array1.blit a b;
  b

(* view a [rows*cols] buffer's row r as a fresh sub-Array1 (no copy) *)
let row ~cols (a : t) r : t = Array1.sub a (r * cols) cols

let of_float_array (xs : float array) : t =
  let n = Array.length xs in
  let a = create n in
  for i = 0 to n - 1 do
    set a i xs.(i)
  done;
  a
