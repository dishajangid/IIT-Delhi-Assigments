open Ast
open Parser
open Lexing

let parse_with_error lexbuf =
  try
    Parser.program Lexer.token lexbuf
  with
  | Parsing.Parse_error ->
      let pos = lexbuf.Lexing.lex_curr_p in
      let line = pos.Lexing.pos_lnum in
      let col = pos.Lexing.pos_cnum - pos.Lexing.pos_bol in
      Printf.printf "Syntax error at line %d, column %d\n" line col;
      exit 1

(* Function to parse input from a Prolog file *)
let parse_pl_file filename =
  let chan = open_in filename in
  let lexbuf = Lexing.from_channel chan in
  try
    let ast = parse_with_error lexbuf in
    close_in chan;
    ast
  with
  | Exit ->
      close_in chan;
      failwith "Parsing failed"

(* Sample Prolog file *)
let filename = "input.pl"
(* Parse the input *)
let clauses = parse_pl_file filename