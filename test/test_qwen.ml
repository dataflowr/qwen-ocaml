(* Per-kernel unit tests. Start with ops (RMSNorm/softmax/silu) and the C stub,
   since those are exercisable before model load exists. *)

let approx a b = Float.abs (a -. b) < 1e-5

let test_softmax () =
  let a = Qwen.Tensor.of_float_array [| 1.0; 2.0; 3.0 |] in
  Qwen.Ops.softmax_inplace a ~off:0 3;
  let sum = Qwen.Tensor.get a 0 +. Qwen.Tensor.get a 1 +. Qwen.Tensor.get a 2 in
  assert (approx sum 1.0);
  assert (Qwen.Tensor.get a 2 > Qwen.Tensor.get a 0)

let test_sdot () =
  let a = Qwen.Tensor.of_float_array [| 1.0; 2.0; 3.0 |] in
  let b = Qwen.Tensor.of_float_array [| 4.0; 5.0; 6.0 |] in
  assert (approx (Qwen.Matmul.sdot a b 3) 32.0)

let test_rmsnorm () =
  let x = Qwen.Tensor.of_float_array [| 1.0; 2.0; 3.0; 4.0 |] in
  let w = Qwen.Tensor.of_float_array [| 1.0; 1.0; 1.0; 1.0 |] in
  let out = Qwen.Tensor.zeros 4 in
  Qwen.Ops.rmsnorm ~eps:1e-6 ~w ~x ~out 4;
  (* rms of [1,2,3,4] = sqrt(7.5); check out[3] ~ 4/sqrt(7.5) *)
  assert (approx (Qwen.Tensor.get out 3) (4.0 /. sqrt 7.5))

let () =
  test_softmax ();
  test_sdot ();
  test_rmsnorm ();
  print_endline "all tests passed"
