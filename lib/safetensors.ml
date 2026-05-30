(* Safetensors reader. File layout:
     u64 header_len (LE) | JSON header | raw tensor blob
   JSON maps name -> { "dtype", "shape":[..], "data_offsets":[start,end] }
   (offsets are relative to the start of the blob). Qwen2.5 weights are BF16;
   tensor returns them materialized as row-major float32. *)

open Bigarray

type dtype = BF16 | F16 | F32 | F64 | I64 | I32 | I16 | I8 | U8 | BOOL

type entry = {
  dtype : dtype;
  shape : int array;
  begin_off : int; (* relative to start of blob *)
  end_off : int;
}

type t = {
  fd : Unix.file_descr;
  data : (int, int8_unsigned_elt, c_layout) Array1.t;
  blob_start : int; (* 8 + header_len: absolute file offset of blob *)
  entries : (string, entry) Hashtbl.t;
  order : string list; (* names in header order, excluding __metadata__ *)
}

(* ----- tiny JSON scanner -------------------------------------------------- *)

exception Parse_error of string

type parser_state = { s : string; mutable pos : int; len : int }

let mk_parser s = { s; pos = 0; len = String.length s }

let peek p = if p.pos < p.len then p.s.[p.pos] else '\000'

let advance p = p.pos <- p.pos + 1

let skip_ws p =
  let continue = ref true in
  while !continue && p.pos < p.len do
    match p.s.[p.pos] with
    | ' ' | '\t' | '\n' | '\r' -> advance p
    | _ -> continue := false
  done

let expect p c =
  skip_ws p;
  if peek p <> c then
    raise (Parse_error (Printf.sprintf "expected '%c' at %d" c p.pos));
  advance p

let parse_string p =
  skip_ws p;
  if peek p <> '"' then raise (Parse_error (Printf.sprintf "expected string at %d" p.pos));
  advance p;
  let buf = Buffer.create 32 in
  let continue = ref true in
  while !continue do
    if p.pos >= p.len then raise (Parse_error "unterminated string");
    let c = p.s.[p.pos] in
    advance p;
    match c with
    | '"' -> continue := false
    | '\\' ->
      if p.pos >= p.len then raise (Parse_error "bad escape");
      let e = p.s.[p.pos] in
      advance p;
      (match e with
       | '"' -> Buffer.add_char buf '"'
       | '\\' -> Buffer.add_char buf '\\'
       | '/' -> Buffer.add_char buf '/'
       | 'n' -> Buffer.add_char buf '\n'
       | 't' -> Buffer.add_char buf '\t'
       | 'r' -> Buffer.add_char buf '\r'
       | 'b' -> Buffer.add_char buf '\b'
       | 'f' -> Buffer.add_char buf '\012'
       | 'u' ->
         if p.pos + 4 > p.len then raise (Parse_error "bad \\u");
         let hex = String.sub p.s p.pos 4 in
         p.pos <- p.pos + 4;
         let code = int_of_string ("0x" ^ hex) in
         if code < 128 then Buffer.add_char buf (Char.chr code)
         else Buffer.add_char buf '?'
       | _ -> raise (Parse_error "unknown escape"))
    | _ -> Buffer.add_char buf c
  done;
  Buffer.contents buf

let parse_int_array p =
  expect p '[';
  skip_ws p;
  let acc = ref [] in
  if peek p = ']' then advance p
  else begin
    let continue = ref true in
    while !continue do
      skip_ws p;
      let start = p.pos in
      if peek p = '-' then advance p;
      while p.pos < p.len && (let c = p.s.[p.pos] in c >= '0' && c <= '9') do
        advance p
      done;
      let num = String.sub p.s start (p.pos - start) in
      acc := int_of_string num :: !acc;
      skip_ws p;
      (match peek p with
       | ',' -> advance p
       | ']' -> advance p; continue := false
       | _ -> raise (Parse_error (Printf.sprintf "bad array at %d" p.pos)))
    done
  end;
  List.rev !acc

let rec skip_value p =
  skip_ws p;
  match peek p with
  | '"' -> ignore (parse_string p)
  | '{' ->
    advance p;
    skip_ws p;
    if peek p = '}' then advance p
    else begin
      let continue = ref true in
      while !continue do
        ignore (parse_string p);
        expect p ':';
        skip_value p;
        skip_ws p;
        (match peek p with
         | ',' -> advance p
         | '}' -> advance p; continue := false
         | _ -> raise (Parse_error "bad object"))
      done
    end
  | '[' ->
    advance p;
    skip_ws p;
    if peek p = ']' then advance p
    else begin
      let continue = ref true in
      while !continue do
        skip_value p;
        skip_ws p;
        (match peek p with
         | ',' -> advance p
         | ']' -> advance p; continue := false
         | _ -> raise (Parse_error "bad array"))
      done
    end
  | _ ->
    let continue = ref true in
    while !continue && p.pos < p.len do
      (match p.s.[p.pos] with
       | ',' | '}' | ']' | ' ' | '\t' | '\n' | '\r' -> continue := false
       | _ -> advance p)
    done

let dtype_of_string = function
  | "BF16" -> BF16
  | "F16" | "FP16" -> F16
  | "F32" | "FP32" -> F32
  | "F64" -> F64
  | "I64" -> I64
  | "I32" -> I32
  | "I16" -> I16
  | "I8" -> I8
  | "U8" -> U8
  | "BOOL" -> BOOL
  | s -> raise (Parse_error ("unsupported dtype: " ^ s))

let parse_entry p =
  expect p '{';
  skip_ws p;
  let dtype = ref None in
  let shape = ref [||] in
  let offs = ref (0, 0) in
  if peek p = '}' then advance p
  else begin
    let continue = ref true in
    while !continue do
      let key = parse_string p in
      expect p ':';
      (match key with
       | "dtype" -> dtype := Some (dtype_of_string (parse_string p))
       | "shape" -> shape := Array.of_list (parse_int_array p)
       | "data_offsets" ->
         (match parse_int_array p with
          | [ a; b ] -> offs := (a, b)
          | _ -> raise (Parse_error "data_offsets must have 2 elems"))
       | _ -> skip_value p);
      skip_ws p;
      (match peek p with
       | ',' -> advance p
       | '}' -> advance p; continue := false
       | _ -> raise (Parse_error "bad entry object"))
    done
  end;
  let dtype = match !dtype with Some d -> d | None -> raise (Parse_error "missing dtype") in
  let (b, e) = !offs in
  { dtype; shape = !shape; begin_off = b; end_off = e }

let parse_header json =
  let p = mk_parser json in
  expect p '{';
  let tbl = Hashtbl.create 512 in
  let order = ref [] in
  skip_ws p;
  if peek p = '}' then advance p
  else begin
    let continue = ref true in
    while !continue do
      let key = parse_string p in
      expect p ':';
      if key = "__metadata__" then skip_value p
      else begin
        let entry = parse_entry p in
        Hashtbl.replace tbl key entry;
        order := key :: !order
      end;
      skip_ws p;
      (match peek p with
       | ',' -> advance p
       | '}' -> advance p; continue := false
       | _ -> raise (Parse_error "bad top-level object"))
    done
  end;
  (tbl, List.rev !order)

let read_header_len data =
  let v = ref 0L in
  for i = 7 downto 0 do
    v := Int64.add (Int64.shift_left !v 8) (Int64.of_int (Array1.unsafe_get data i))
  done;
  Int64.to_int !v

let open_file path =
  let fd = Unix.openfile path [ Unix.O_RDONLY ] 0 in
  let data =
    array1_of_genarray
      (Unix.map_file fd Int8_unsigned c_layout false [| -1 |])
  in
  let header_len = read_header_len data in
  let blob_start = 8 + header_len in
  let json = Bytes.create header_len in
  for i = 0 to header_len - 1 do
    Bytes.unsafe_set json i (Char.unsafe_chr (Array1.unsafe_get data (8 + i)))
  done;
  let entries, order = parse_header (Bytes.unsafe_to_string json) in
  { fd; data; blob_start; entries; order }

let names t = t.order

let mem t name = Hashtbl.mem t.entries name

let find t name =
  match Hashtbl.find_opt t.entries name with
  | Some e -> e
  | None -> failwith ("safetensors: tensor not found: " ^ name)

let shape t name = (find t name).shape

let[@inline] u16 data off =
  Array1.unsafe_get data off lor (Array1.unsafe_get data (off + 1) lsl 8)

let[@inline] u32 data off =
  let b0 = Array1.unsafe_get data off in
  let b1 = Array1.unsafe_get data (off + 1) in
  let b2 = Array1.unsafe_get data (off + 2) in
  let b3 = Array1.unsafe_get data (off + 3) in
  Int32.logor
    (Int32.logor (Int32.of_int b0) (Int32.shift_left (Int32.of_int b1) 8))
    (Int32.logor (Int32.shift_left (Int32.of_int b2) 16)
       (Int32.shift_left (Int32.of_int b3) 24))

(* bulk bf16 -> f32 in C: src is the mmap'd byte array, off the byte offset of
   the tensor's first element, dst the destination f32 buffer, n the count. *)
external bf16_to_f32_bulk :
  (int, int8_unsigned_elt, c_layout) Array1.t -> int -> Tensor.t -> int -> unit
  = "caml_qwen_bf16_to_f32"

let f16_to_f32 h =
  let sign = (h lsr 15) land 1 in
  let exp = (h lsr 10) land 0x1f in
  let frac = h land 0x3ff in
  let bits =
    if exp = 0 then
      if frac = 0 then Int32.shift_left (Int32.of_int sign) 31
      else begin
        let e = ref (-1) in
        let f = ref frac in
        while !f land 0x400 = 0 do
          f := !f lsl 1;
          decr e
        done;
        let f = !f land 0x3ff in
        let exp32 = 127 - 15 + !e + 2 in
        Int32.logor
          (Int32.shift_left (Int32.of_int sign) 31)
          (Int32.logor (Int32.shift_left (Int32.of_int exp32) 23)
             (Int32.shift_left (Int32.of_int f) 13))
      end
    else if exp = 0x1f then
      Int32.logor
        (Int32.shift_left (Int32.of_int sign) 31)
        (Int32.logor (Int32.shift_left 0xffl 23)
           (Int32.shift_left (Int32.of_int frac) 13))
    else
      let exp32 = exp - 15 + 127 in
      Int32.logor
        (Int32.shift_left (Int32.of_int sign) 31)
        (Int32.logor (Int32.shift_left (Int32.of_int exp32) 23)
           (Int32.shift_left (Int32.of_int frac) 13))
  in
  Int32.float_of_bits bits

let numel shape = Array.fold_left ( * ) 1 shape

let tensor t name : Tensor.t =
  let e = find t name in
  let n = numel e.shape in
  let out = Tensor.create n in
  let base = t.blob_start + e.begin_off in
  let data = t.data in
  (match e.dtype with
   | BF16 -> bf16_to_f32_bulk data base out n
   | F16 ->
     for i = 0 to n - 1 do
       let bits = u16 data (base + (i * 2)) in
       Tensor.set out i (f16_to_f32 bits)
     done
   | F32 ->
     for i = 0 to n - 1 do
       let bits = u32 data (base + (i * 4)) in
       Tensor.set out i (Int32.float_of_bits bits)
     done
   | _ -> failwith ("safetensors: unsupported dtype for tensor " ^ name));
  out

let close t =
  (try Unix.close t.fd with _ -> ())
