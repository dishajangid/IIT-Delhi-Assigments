(* lexer.mll *)

{
  open Parser
  open Lexing
}
(* ocamlyacc -v  parser.mly 
    export OCAMLRUNPARAM='p'
*)

let digit = ['0' - '9']
let int = '-'? digit+
let letter = ['a' - 'z' 'A' - 'Z'] 
let alphanumeric = digit | letter
let variable = ('_'|['A'-'Z']) (alphanumeric | '_')*
let constant = "\"" letter (alphanumeric | '_')* "\""
let functor_symbol = ['a' - 'z']['a' - 'z' 'A' - 'Z' '_' '1'-'9']* 
let predicate_symbol = letter (alphanumeric | '_')*
(* let identifier = ('_'|['a'-'z']) (alphanumeric | '_')* *)

rule token = parse
  | [' ' '\t']+  { token lexbuf }         (* Skip whitespace *)
  | ['\n']+          {Lexing.new_line lexbuf; token lexbuf}
  | '('               { LPAREN }
  | ')'               { RPAREN }
  | '['               { LSQUARE }
  | ']'               { RSQUARE }
  | '|'               {PIPE}
  | ';'                {SEMICOLON}
  | '='                {EQUAL '='}
  | ','               { COMMA }
  | '_'               {UNDERSCORE '_'}
  | '.'               { DOT }
  | ":-"              { ASSIGN }
  | "fail"             { KEYWORD "fail"}
  | "intT"             {KEYWORD "intT" }
  | "boolT"             {KEYWORD "boolT"}
  | "list"             {KEYWORD "list"}
  | "true"              {KEYWORD "true"} 
  | "false"              {KEYWORD "false"} 
  | '!'                {CUT '!'}
  | variable as v     { VAR v }
  | int as i          { INT (int_of_string(i))}
  | constant as c     { CONST c }
  | functor_symbol as f { FUNC f }
  | "?-"                {GOAL}
  (* | identifier as id   {IDENT id} *)
  (* | predicate_symbol as p { PRED p } *)
  | eof               { EOF }
  | _                 { failwith ("Unexpected character: " ^ Lexing.lexeme lexbuf) }
