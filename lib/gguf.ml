(* GGUF v3 reader (little-endian). *)

open Bigarray

type tensor_info = {
  ty : int;
  dims : int array;
  offset : int;
}

type blob = (int, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t

(* metadata scalar values we keep for metadata_* queries *)
type meta =
  | MInt of int
  | MFloat of float
  | MString of string
  | MBool of bool
  | MOther

type t = {
  fd : Unix.file_descr;
  data : blob;
  data_start : int;
  infos : (string, tensor_info) Hashtbl.t;
  order : string list; (* tensor names in file order *)
  meta : (string, meta) Hashtbl.t;
}

(* ----- little-endian primitive readers over the int8 mmap ----------------- *)

let u8 (d : blob) pos = Array1.unsafe_get d pos

let u16 d pos = u8 d pos lor (u8 d (pos + 1) lsl 8)

let u32 d pos =
  let a = u8 d pos in
  let b = u8 d (pos + 1) in
  let c = u8 d (pos + 2) in
  let e = u8 d (pos + 3) in
  a lor (b lsl 8) lor (c lsl 16) lor (e lsl 24)

(* u64 read into an OCaml int. File offsets/dims here fit in 63 bits. *)
let u64 d pos =
  let lo = u32 d pos in
  let hi = u32 d (pos + 4) in
  lo lor (hi lsl 32)

let i32 d pos =
  let v = u32 d pos in
  if v land 0x80000000 <> 0 then v - (1 lsl 32) else v

let i64 d pos = u64 d pos (* assume fits *)

let i16 d pos =
  let v = u16 d pos in
  if v land 0x8000 <> 0 then v - (1 lsl 16) else v

let i8 d pos =
  let v = u8 d pos in
  if v land 0x80 <> 0 then v - 256 else v

let f32 d pos =
  let v = u32 d pos in
  Int32.float_of_bits (Int32.logand (Int32.of_int v) 0xFFFFFFFFl)

let f64 d pos =
  let lo = u32 d pos in
  let hi = u32 d (pos + 4) in
  let bits =
    Int64.logor
      (Int64.logand (Int64.of_int lo) 0xFFFFFFFFL)
      (Int64.shift_left (Int64.logand (Int64.of_int hi) 0xFFFFFFFFL) 32)
  in
  Int64.float_of_bits bits

let str d pos =
  (* u64 len + bytes; returns (string, next_pos) *)
  let len = u64 d pos in
  let b = Bytes.create len in
  for i = 0 to len - 1 do
    Bytes.unsafe_set b i (Char.unsafe_chr (u8 d (pos + 8 + i)))
  done;
  (Bytes.unsafe_to_string b, pos + 8 + len)

(* size in bytes of a scalar metadata value type (not array/string) *)
let scalar_size = function
  | 0 -> 1 (* u8 *)
  | 1 -> 1 (* i8 *)
  | 2 -> 2 (* u16 *)
  | 3 -> 2 (* i16 *)
  | 4 -> 4 (* u32 *)
  | 5 -> 4 (* i32 *)
  | 6 -> 4 (* f32 *)
  | 7 -> 1 (* bool *)
  | 10 -> 8 (* u64 *)
  | 11 -> 8 (* i64 *)
  | 12 -> 8 (* f64 *)
  | t -> failwith (Printf.sprintf "gguf: bad scalar value type %d" t)

(* read a scalar metadata value of [ty] at [pos]; returns (meta, next_pos) *)
let read_scalar d ty pos =
  match ty with
  | 0 -> (MInt (u8 d pos), pos + 1)
  | 1 -> (MInt (i8 d pos), pos + 1)
  | 2 -> (MInt (u16 d pos), pos + 2)
  | 3 -> (MInt (i16 d pos), pos + 2)
  | 4 -> (MInt (u32 d pos), pos + 4)
  | 5 -> (MInt (i32 d pos), pos + 4)
  | 6 -> (MFloat (f32 d pos), pos + 4)
  | 7 -> (MBool (u8 d pos <> 0), pos + 1)
  | 8 ->
      let s, next = str d pos in
      (MString s, next)
  | 10 -> (MInt (u64 d pos), pos + 8)
  | 11 -> (MInt (i64 d pos), pos + 8)
  | 12 -> (MFloat (f64 d pos), pos + 8)
  | t -> failwith (Printf.sprintf "gguf: bad value type %d" t)

(* parse (and skip) a full metadata value of [ty] at [pos]; returns
   (meta_for_scalar, next_pos). For arrays returns MOther. *)
let rec read_value d ty pos =
  match ty with
  | 9 ->
      (* array: u32 elem_type, u64 count, then count elements *)
      let elem_ty = u32 d pos in
      let count = u64 d (pos + 4) in
      let p = ref (pos + 12) in
      if elem_ty = 8 then
        for _ = 1 to count do
          let _, next = str d !p in
          p := next
        done
      else if elem_ty = 9 then
        for _ = 1 to count do
          let _, next = read_value d 9 !p in
          p := next
        done
      else begin
        let sz = scalar_size elem_ty in
        p := !p + (sz * count)
      end;
      (MOther, !p)
  | _ -> read_scalar d ty pos

let align_up x a = ((x + a - 1) / a) * a

let open_file path =
  let fd = Unix.openfile path [ Unix.O_RDONLY ] 0 in
  let data : blob =
    array1_of_genarray (Unix.map_file fd Int8_unsigned c_layout false [| -1 |])
  in
  (* header *)
  if Array1.dim data < 24 then failwith "gguf: file too small for header";
  let magic = u32 data 0 in
  if magic <> 0x46554747 (* "GGUF" LE *) then
    failwith "gguf: bad magic";
  let version = u32 data 4 in
  (* this parser implements v3 (u64 counts); v1/v2 used u32 counts. *)
  if version <> 3 then
    failwith (Printf.sprintf "gguf: unsupported version %d (only v3)" version);
  let tensor_count = u64 data 8 in
  let kv_count = u64 data 16 in
  let pos = ref 24 in
  let meta = Hashtbl.create 256 in
  for _ = 1 to kv_count do
    let key, p1 = str data !pos in
    let vty = u32 data p1 in
    let m, p2 = read_value data vty (p1 + 4) in
    Hashtbl.replace meta key m;
    pos := p2
  done;
  (* tensor infos *)
  let infos = Hashtbl.create 512 in
  let order = ref [] in
  for _ = 1 to tensor_count do
    let name, p1 = str data !pos in
    let n_dims = u32 data p1 in
    let p = ref (p1 + 4) in
    let dims = Array.make n_dims 0 in
    for i = 0 to n_dims - 1 do
      dims.(i) <- u64 data !p;
      p := !p + 8
    done;
    let ty = u32 data !p in
    let offset = u64 data (!p + 4) in
    pos := !p + 12;
    Hashtbl.replace infos name { ty; dims; offset };
    order := name :: !order
  done;
  let alignment =
    match Hashtbl.find_opt meta "general.alignment" with
    | Some (MInt n) when n > 0 -> n
    | _ -> 32
  in
  let data_start = align_up !pos alignment in
  { fd; data; data_start; infos; order = List.rev !order; meta }

let names t = t.order
let mem t name = Hashtbl.mem t.infos name
let info t name = Hashtbl.find t.infos name
let blob t = t.data
let data_start t = t.data_start

let metadata_int t key =
  match Hashtbl.find_opt t.meta key with
  | Some (MInt n) -> Some n
  | Some (MFloat f) -> Some (int_of_float f)
  | Some (MBool b) -> Some (if b then 1 else 0)
  | _ -> None

let metadata_float t key =
  match Hashtbl.find_opt t.meta key with
  | Some (MFloat f) -> Some f
  | Some (MInt n) -> Some (float_of_int n)
  | _ -> None

let metadata_string t key =
  match Hashtbl.find_opt t.meta key with
  | Some (MString s) -> Some s
  | _ -> None

let close t = (try Unix.close t.fd with _ -> ())
