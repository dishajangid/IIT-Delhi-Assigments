open Lexer
open Parser
open Interpreter

(* Custom exceptions for parsing and interpretation errors *)
exception ParsingError of string
exception InterpreterError of string

(* Function to parse with error handling *)
let parse_with_error lexbuf =
  try
    Parser.program Lexer.read lexbuf
  with
  | Parsing.Parse_error ->
      let pos = lexbuf.Lexing.lex_curr_p in
      let line = pos.Lexing.pos_lnum in
      let col = pos.Lexing.pos_cnum - pos.Lexing.pos_bol in
      raise (ParsingError (Printf.sprintf "Syntax error at line %d, column %d" line col))

(* Function to parse a Prolog goal with error handling *)
let parse_goal_with_error line =
  try
    Parser.goal Lexer.read (Lexing.from_string line)
  with
  | Parsing.Parse_error ->
      let pos = Lexing.dummy_pos in
      let line = pos.Lexing.pos_lnum in
      let col = pos.Lexing.pos_cnum - pos.Lexing.pos_bol in
      raise (ParsingError (Printf.sprintf "Syntax error at line %d, column %d" line col))

(* Function to parse a Prolog file with error handling *)
let parse_pl_file filename =
  let chan = open_in filename in
  let lexbuf = Lexing.from_channel chan in
  try
    let ast = parse_with_error lexbuf in
    close_in chan;
    ast
  with
  | ParsingError msg ->
      close_in chan;
      failwith ("Parsing failed: " ^ msg)

(* Main program *)
let () =
  if Array.length Sys.argv < 2 then begin
    print_string "Input file not provided.\nExiting...\n";
    exit 0;
  end;

  if Array.length Sys.argv > 2 then begin
    print_string "Too many arguments.\nExiting...\n";
    exit 0;
  end;

  let filename = Sys.argv.(1) in
  let init_prog = parse_pl_file filename in
  let _ = checkProgram init_prog in
  let prog = modifyInitialProg init_prog 1 in

  print_string "Program loaded successfully\n";

  try
    while true do
      print_string "?- ";
      let line = read_line () in
      if line = "quit." then exit 0
      else
        try
          let g = parse_goal_with_error line in
          match interpret_goal prog g with
          | (true, _) -> print_string "true.\n"
          | (false, _) -> print_string "false.\n"
        with
        | ParsingError msg ->
            Printf.printf "Parsing error: %s\n" msg
        | InterpreterError msg ->
            Printf.printf "Interpreter error: %s\n" msg
    done
  with
  | End_of_file ->
      print_string "\n% halt\n"
