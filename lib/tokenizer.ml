(* Byte-level BPE (GPT-2/cl100k family) loaded from a HuggingFace tokenizer.json.
   Special/added tokens are split out before BPE. ChatML helpers for chat use. *)

type t = {
  vocab : (string, int) Hashtbl.t;          (* token-string (unicode chars) -> id *)
  id_to_tok : (int, string) Hashtbl.t;      (* id -> token-string *)
  ranks : (string, int) Hashtbl.t;          (* "A B" merge -> rank (lower=better) *)
  specials : (string * int) list;           (* content -> id, longest-first *)
  byte_to_uni : string array;               (* 0..255 -> unicode char (utf8 string) *)
  uni_to_byte : (string, int) Hashtbl.t;    (* unicode char (utf8) -> byte *)
  eos_id : int;
  im_start_id : int;
  im_end_id : int;
}

(* ---------- GPT-2 bytes_to_unicode ---------- *)

(* Encode a single Unicode code point to a UTF-8 string. *)
let utf8_of_cp cp =
  let b = Buffer.create 4 in
  if cp < 0x80 then Buffer.add_char b (Char.chr cp)
  else if cp < 0x800 then begin
    Buffer.add_char b (Char.chr (0xC0 lor (cp lsr 6)));
    Buffer.add_char b (Char.chr (0x80 lor (cp land 0x3F)))
  end else if cp < 0x10000 then begin
    Buffer.add_char b (Char.chr (0xE0 lor (cp lsr 12)));
    Buffer.add_char b (Char.chr (0x80 lor ((cp lsr 6) land 0x3F)));
    Buffer.add_char b (Char.chr (0x80 lor (cp land 0x3F)))
  end else begin
    Buffer.add_char b (Char.chr (0xF0 lor (cp lsr 18)));
    Buffer.add_char b (Char.chr (0x80 lor ((cp lsr 12) land 0x3F)));
    Buffer.add_char b (Char.chr (0x80 lor ((cp lsr 6) land 0x3F)));
    Buffer.add_char b (Char.chr (0x80 lor (cp land 0x3F)))
  end;
  Buffer.contents b

(* Build the GPT-2 byte<->unicode mapping.
   "printable" bytes (33..126, 161..172, 174..255) map to themselves;
   the remaining bytes map to U+0100, U+0101, ... in order. *)
let build_byte_unicode () =
  let byte_to_uni = Array.make 256 "" in
  let uni_to_byte = Hashtbl.create 512 in
  let is_printable b =
    (b >= 0x21 && b <= 0x7E) || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF)
  in
  let n = ref 0 in
  for b = 0 to 255 do
    let cp =
      if is_printable b then b
      else begin
        let c = 0x100 + !n in
        incr n;
        c
      end
    in
    let s = utf8_of_cp cp in
    byte_to_uni.(b) <- s;
    Hashtbl.replace uni_to_byte s b
  done;
  (byte_to_uni, uni_to_byte)

(* ---------- minimal JSON scanning ---------- *)

(* We only need a few pieces from tokenizer.json:
     model.vocab    : object  token(JSON-string) -> int
     model.merges   : array of JSON-strings "A B"
     added_tokens   : array of objects with "id" and "content"
   We avoid a full JSON parser; a focused scanner suffices. *)

(* Read whole file. *)
let read_file path =
  let ic = open_in_bin path in
  let n = in_channel_length ic in
  let s = really_input_string ic n in
  close_in ic;
  s

(* Decode a JSON string literal starting at position i (s.[i] = '"').
   Returns (decoded_string, index_after_closing_quote).
   Handles \uXXXX (including surrogate pairs) and the standard escapes. *)
let parse_json_string s i =
  assert (s.[i] = '"');
  let buf = Buffer.create 16 in
  let j = ref (i + 1) in
  let n = String.length s in
  let hex4 k =
    int_of_string ("0x" ^ String.sub s k 4)
  in
  while s.[!j] <> '"' do
    let c = s.[!j] in
    if c = '\\' then begin
      let e = s.[!j + 1] in
      (match e with
       | '"' -> Buffer.add_char buf '"'; j := !j + 2
       | '\\' -> Buffer.add_char buf '\\'; j := !j + 2
       | '/' -> Buffer.add_char buf '/'; j := !j + 2
       | 'b' -> Buffer.add_char buf '\b'; j := !j + 2
       | 'f' -> Buffer.add_char buf '\012'; j := !j + 2
       | 'n' -> Buffer.add_char buf '\n'; j := !j + 2
       | 'r' -> Buffer.add_char buf '\r'; j := !j + 2
       | 't' -> Buffer.add_char buf '\t'; j := !j + 2
       | 'u' ->
         let cp = hex4 (!j + 2) in
         if cp >= 0xD800 && cp <= 0xDBFF
            && !j + 6 < n && s.[!j + 6] = '\\' && s.[!j + 7] = 'u' then begin
           let lo = hex4 (!j + 8) in
           let combined = 0x10000 + ((cp - 0xD800) lsl 10) + (lo - 0xDC00) in
           Buffer.add_string buf (utf8_of_cp combined);
           j := !j + 12
         end else begin
           Buffer.add_string buf (utf8_of_cp cp);
           j := !j + 6
         end
       | _ -> Buffer.add_char buf e; j := !j + 2)
    end else begin
      Buffer.add_char buf c;
      incr j
    end
  done;
  (Buffer.contents buf, !j + 1)

(* Find the position just after the next occurrence of [key] (a quoted key
   literally written as "key") starting from [from]. Returns the index of the
   first char after the colon following the key. *)
let find_key s key from =
  (* search for the substring "\"key\"" *)
  let pat = "\"" ^ key ^ "\"" in
  let idx =
    let plen = String.length pat in
    let slen = String.length s in
    let rec loop i =
      if i + plen > slen then -1
      else if String.sub s i plen = pat then i
      else loop (i + 1)
    in
    loop from
  in
  if idx < 0 then failwith ("key not found: " ^ key);
  (* skip to colon *)
  let j = ref (idx + String.length pat) in
  while s.[!j] <> ':' do incr j done;
  incr j;
  (* skip whitespace *)
  while s.[!j] = ' ' || s.[!j] = '\n' || s.[!j] = '\t' || s.[!j] = '\r' do incr j done;
  !j

(* Skip whitespace. *)
let skip_ws s j =
  let j = ref j in
  while s.[!j] = ' ' || s.[!j] = '\n' || s.[!j] = '\t' || s.[!j] = '\r' do incr j done;
  !j

(* Parse an integer starting at j. Returns (int, next_index). *)
let parse_int s j =
  let j = skip_ws s j in
  let start = j in
  let j = ref j in
  if s.[!j] = '-' then incr j;
  while s.[!j] >= '0' && s.[!j] <= '9' do incr j done;
  (int_of_string (String.sub s start (!j - start)), !j)

(* Parse the model.vocab object: { "tok": id, ... }. Position j points at '{'. *)
let parse_vocab_object s j vocab id_to_tok =
  let j = skip_ws s j in
  assert (s.[j] = '{');
  let j = ref (j + 1) in
  let continue = ref true in
  let jj = skip_ws s !j in
  if s.[jj] = '}' then (j := jj + 1; continue := false);
  while !continue do
    let jj = skip_ws s !j in
    let (tok, after) = parse_json_string s jj in
    let after = skip_ws s after in
    assert (s.[after] = ':');
    let (id, after2) = parse_int s (after + 1) in
    Hashtbl.replace vocab tok id;
    Hashtbl.replace id_to_tok id tok;
    let after2 = skip_ws s after2 in
    (match s.[after2] with
     | ',' -> j := after2 + 1
     | '}' -> j := after2 + 1; continue := false
     | c -> failwith (Printf.sprintf "vocab parse: unexpected '%c'" c))
  done;
  !j

(* Parse the model.merges array of strings. Position j points at '['. *)
let parse_merges_array s j ranks =
  let j = skip_ws s j in
  assert (s.[j] = '[');
  let j = ref (j + 1) in
  let rank = ref 0 in
  let continue = ref true in
  let jj = skip_ws s !j in
  if s.[jj] = ']' then (j := jj + 1; continue := false);
  while !continue do
    let jj = skip_ws s !j in
    let (merge, after) = parse_json_string s jj in
    Hashtbl.replace ranks merge !rank;
    incr rank;
    let after = skip_ws s after in
    (match s.[after] with
     | ',' -> j := after + 1
     | ']' -> j := after + 1; continue := false
     | c -> failwith (Printf.sprintf "merges parse: unexpected '%c'" c))
  done;
  !j

(* Parse added_tokens array: [ {"id":N,"content":"..", ...}, ... ].
   Returns list of (content, id). Position j points at '['. *)
let parse_added_tokens s j =
  let j = skip_ws s j in
  assert (s.[j] = '[');
  let j = ref (j + 1) in
  let acc = ref [] in
  let continue = ref true in
  let jj = skip_ws s !j in
  if s.[jj] = ']' then (j := jj + 1; continue := false);
  while !continue do
    (* find this object's bounds, then locate id and content keys within. *)
    let obj_start = skip_ws s !j in
    assert (s.[obj_start] = '{');
    (* find matching closing brace *)
    let depth = ref 0 in
    let k = ref obj_start in
    let in_str = ref false in
    let stop = ref false in
    while not !stop do
      let c = s.[!k] in
      if !in_str then begin
        if c = '\\' then incr k
        else if c = '"' then in_str := false
      end else begin
        if c = '"' then in_str := true
        else if c = '{' then incr depth
        else if c = '}' then begin decr depth; if !depth = 0 then stop := true end
      end;
      incr k
    done;
    let obj_end = !k in (* index just past '}' *)
    let obj = String.sub s obj_start (obj_end - obj_start) in
    let id_pos = find_key obj "id" 0 in
    let (id, _) = parse_int obj id_pos in
    let content_pos = find_key obj "content" 0 in
    let (content, _) = parse_json_string obj content_pos in
    acc := (content, id) :: !acc;
    let after = skip_ws s obj_end in
    (match s.[after] with
     | ',' -> j := after + 1
     | ']' -> j := after + 1; continue := false
     | c -> failwith (Printf.sprintf "added_tokens parse: unexpected '%c'" c))
  done;
  List.rev !acc

(* ---------- load ---------- *)

let load path =
  let s = read_file path in
  let vocab = Hashtbl.create 200000 in
  let id_to_tok = Hashtbl.create 200000 in
  let ranks = Hashtbl.create 200000 in
  (* model.vocab: find the "model" object then "vocab" within it.
     Since "vocab" is unique enough as a key, search globally. *)
  let vpos = find_key s "vocab" 0 in
  let _ = parse_vocab_object s vpos vocab id_to_tok in
  let mpos = find_key s "merges" 0 in
  let _ = parse_merges_array s mpos ranks in
  let added = parse_added_tokens s (find_key s "added_tokens" 0) in
  (* register specials in vocab/id_to_tok too so decode works. *)
  List.iter (fun (content, id) ->
    Hashtbl.replace vocab content id;
    Hashtbl.replace id_to_tok id content) added;
  (* sort specials longest-first for greedy splitting. *)
  let specials =
    List.sort (fun (a, _) (b, _) -> compare (String.length b) (String.length a)) added
  in
  let (byte_to_uni, uni_to_byte) = build_byte_unicode () in
  let find name = try List.assoc name added with Not_found -> -1 in
  {
    vocab; id_to_tok; ranks; specials; byte_to_uni; uni_to_byte;
    eos_id = find "<|endoftext|>";
    im_start_id = find "<|im_start|>";
    im_end_id = find "<|im_end|>";
  }

let eos_id t = t.eos_id
let im_start_id t = t.im_start_id
let im_end_id t = t.im_end_id

(* ---------- pre-tokenizer (ASCII approximation of cl100k Split) ---------- *)

(* Regex (case-insensitive contractions):
     (?i:'s|'t|'re|'ve|'m|'ll|'d)
   | [^\r\n\p{L}\p{N}]?\p{L}+
   | \p{N}                      (cl100k uses \p{N}{1,3}; HF Qwen uses \p{N})
   | ?[^\s\p{L}\p{N}]+[\r\n]*
   | \s*[\r\n]+
   | \s+(?!\S)
   | \s+
   ASCII: \p{L} -> [A-Za-z], \p{N} -> [0-9]. *)

let is_letter c = (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z')
let is_digit c = c >= '0' && c <= '9'
let is_ws c = c = ' ' || c = '\t' || c = '\n' || c = '\r' || c = '\012' || c = '\011'
let is_nl c = c = '\n' || c = '\r'
let is_alnum c = is_letter c || is_digit c

(* Match a contraction at position i (lowercased). Returns length matched or 0. *)
let match_contraction s i n =
  if i >= n || s.[i] <> '\'' then 0
  else begin
    let lc c = if c >= 'A' && c <= 'Z' then Char.chr (Char.code c + 32) else c in
    let c1 = if i + 1 < n then lc s.[i + 1] else '\000' in
    let c2 = if i + 2 < n then lc s.[i + 2] else '\000' in
    match c1 with
    | 's' | 't' | 'm' | 'd' -> 2
    | 'r' when c2 = 'e' -> 3
    | 'v' when c2 = 'e' -> 3
    | 'l' when c2 = 'l' -> 3
    | _ -> 0
  end

(* Split a chunk of text (with no special tokens) into pre-tokens.
   Implements the regex alternation in priority order at each position:
     1. contraction
     2. [^\r\n\p{L}\p{N}]? \p{L}+
     3. \p{N}
     4. \ ?[^\s\p{L}\p{N}]+[\r\n]*
     5. \s*[\r\n]+
     6. \s+(?!\S)
     7. \s+ *)
let pretokenize s =
  let n = String.length s in
  let out = ref [] in
  let i = ref 0 in
  while !i < n do
    let start = !i in
    let c = s.[!i] in
    let cl = match_contraction s !i n in
    if cl > 0 then
      (* 1. contraction *)
      i := !i + cl
    else if is_letter c then begin
      (* 2 (no prefix). \p{L}+ *)
      while !i < n && is_letter s.[!i] do incr i done
    end
    else if (not (is_nl c)) && (not (is_alnum c))
            && !i + 1 < n && is_letter s.[!i + 1] then begin
      (* 2 (with prefix). [^\r\n\p{L}\p{N}]? \p{L}+
         The optional prefix char may even be whitespace (regex only excludes \r\n). *)
      incr i;
      while !i < n && is_letter s.[!i] do incr i done
    end
    else if is_digit c then
      (* 3. \p{N} (single digit; Qwen pattern has no {1,3}) *)
      incr i
    else begin
      (* Determine start for alternative 4: optional single space then a run of
         [^\s\p{L}\p{N}]+. *)
      let p = if c = ' ' then !i + 1 else !i in
      if p < n && (let d = s.[p] in not (is_ws d) && not (is_alnum d)) then begin
        (* 4. \ ?[^\s\p{L}\p{N}]+[\r\n]* *)
        i := p;
        while !i < n && (let d = s.[!i] in not (is_ws d) && not (is_alnum d)) do incr i done;
        while !i < n && is_nl s.[!i] do incr i done
      end
      else begin
        (* whitespace-only alternatives 5/6/7. c is whitespace here. *)
        let ws_start = !i in
        while !i < n && is_ws s.[!i] do incr i done;
        let ws_end = !i in
        let last_nl = ref (-1) in
        for k = ws_start to ws_end - 1 do if is_nl s.[k] then last_nl := k done;
        if !last_nl >= 0 then begin
          (* 5. \s*[\r\n]+ : greedy ws ending at the last newline in the run.
             Anything after the last newline (trailing spaces) is re-scanned. *)
          i := !last_nl + 1
        end
        else if ws_end < n then
          (* followed by non-ws (loop stopped before EOF, no newline):
             6. \s+(?!\S) cannot consume the final space, so leave one for the
             next token's optional-space prefix. *)
          i := if ws_end - ws_start > 1 then ws_end - 1 else ws_end
          (* if only a single space and it's followed by non-ws, alt 6 fails
             entirely and alt 7 (\s+) takes the single space. ws_end already. *)
        else
          (* trailing whitespace at EOF: 7. \s+ keeps all. *)
          i := ws_end
      end
    end;
    if !i = start then incr i; (* safety: never stall *)
    out := String.sub s start (!i - start) :: !out
  done;
  List.rev !out

(* ---------- BPE ---------- *)

(* Apply BPE merges to a list of single-char (unicode) symbol strings.
   Returns the merged list of token strings. *)
let bpe t (symbols : string list) =
  match symbols with
  | [] | [_] -> symbols
  | _ ->
    let arr = Array.of_list symbols in
    let len = ref (Array.length arr) in
    let continue = ref true in
    while !continue && !len > 1 do
      (* find the adjacent pair with the lowest rank. *)
      let best_rank = ref max_int in
      let best_i = ref (-1) in
      for k = 0 to !len - 2 do
        let pair = arr.(k) ^ " " ^ arr.(k + 1) in
        (match Hashtbl.find_opt t.ranks pair with
         | Some r when r < !best_rank -> best_rank := r; best_i := k
         | _ -> ())
      done;
      if !best_i < 0 then continue := false
      else begin
        (* merge arr.(best_i) and arr.(best_i+1) *)
        let bi = !best_i in
        arr.(bi) <- arr.(bi) ^ arr.(bi + 1);
        (* shift left *)
        for k = bi + 1 to !len - 2 do arr.(k) <- arr.(k + 1) done;
        decr len
      end
    done;
    Array.to_list (Array.sub arr 0 !len)

(* Convert a raw byte chunk into its list of unicode-char symbol strings. *)
let bytes_to_symbols t (chunk : string) =
  let n = String.length chunk in
  let acc = ref [] in
  for i = n - 1 downto 0 do
    let b = Char.code chunk.[i] in
    acc := t.byte_to_uni.(b) :: !acc
  done;
  !acc

(* Encode a chunk that contains NO special tokens. *)
let encode_ordinary t (text : string) ids =
  let pretoks = pretokenize text in
  List.iter (fun pt ->
    let syms = bytes_to_symbols t pt in
    let merged = bpe t syms in
    List.iter (fun tok ->
      match Hashtbl.find_opt t.vocab tok with
      | Some id -> ids := id :: !ids
      | None ->
        (* fall back: should not happen for byte-level vocab, but emit per-char *)
        failwith ("token not in vocab: " ^ tok)) merged) pretoks

(* Find the earliest special-token occurrence in s starting at [from].
   Returns Some (pos, content, id) or None. *)
let next_special t s from =
  let n = String.length s in
  let best = ref None in
  List.iter (fun (content, id) ->
    let clen = String.length content in
    (* search for content from `from` *)
    let rec loop i =
      if i + clen > n then ()
      else if String.sub s i clen = content then begin
        (match !best with
         | Some (bp, _, _) when bp <= i -> ()
         | _ -> best := Some (i, content, id))
      end
      else loop (i + 1)
    in
    loop from) t.specials;
  !best

let encode t (text : string) =
  let ids = ref [] in
  let n = String.length text in
  let pos = ref 0 in
  let continue = ref true in
  while !continue do
    match next_special t text !pos with
    | Some (sp, content, id) ->
      if sp > !pos then encode_ordinary t (String.sub text !pos (sp - !pos)) ids;
      ids := id :: !ids;
      pos := sp + String.length content
    | None ->
      if !pos < n then encode_ordinary t (String.sub text !pos (n - !pos)) ids;
      continue := false
  done;
  Array.of_list (List.rev !ids)

(* ---------- decode ---------- *)

(* Decode one token string of unicode chars back to raw bytes.
   We iterate the utf8 string char-by-char (variable width) and map each
   unicode char to its byte. *)
let decode_token t (tok : string) (buf : Buffer.t) =
  let n = String.length tok in
  let i = ref 0 in
  while !i < n do
    let c = Char.code tok.[!i] in
    let w =
      if c < 0x80 then 1
      else if c < 0xE0 then 2
      else if c < 0xF0 then 3
      else 4
    in
    let ch = String.sub tok !i w in
    (match Hashtbl.find_opt t.uni_to_byte ch with
     | Some b -> Buffer.add_char buf (Char.chr b)
     | None -> Buffer.add_string buf ch); (* unknown: pass through raw *)
    i := !i + w
  done

let decode t (ids : int array) =
  let buf = Buffer.create 256 in
  Array.iter (fun id ->
    match Hashtbl.find_opt t.id_to_tok id with
    | Some tok ->
      (* specials decode to their literal content (their bytes round-trip too,
         since their unicode chars are ASCII and map to themselves). *)
      decode_token t tok buf
    | None -> ()) ids;
  Buffer.contents buf

(* ---------- chat template ---------- *)

let encode_chat t (msg : string) =
  let s =
    "<|im_start|>user\n" ^ msg ^ "<|im_end|>\n<|im_start|>assistant\n"
  in
  encode t s
