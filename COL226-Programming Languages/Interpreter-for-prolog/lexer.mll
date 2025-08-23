{
  open Parser;;
  exception InvalidToken of char ;;
  open Lexing;;
}

let alpha_num = ['A'-'Z' 'a'-'z' '0'-'9' '_']
let var = ['A'-'Z'](alpha_num*)
let cons = ['a'-'z'](alpha_num*) | ("\"" [^ '\"']+ "\"")
let sp = [' ' '\t']+
let number = '0'|['1'-'9']['0'-'9']*

rule read = parse
    eof                   {EOF}
  | sp                    {read lexbuf}
  | '\n'                  { Lexing.new_line lexbuf; read lexbuf } (* Update line number *)
  | var as v              {VAR(v)}
  | cons as c             {CONS(c)} 
  | number as n           {NUM(int_of_string n)}
  | '('                   {LPAREN}
  | ')'                   {RPAREN}
  | '['                   {LB}
  | ']'                   {RB}
  | ','                   {COMMA}
  | ':'                   {COLON}
  | ';'                   {SEMICOLON}
  | '='                   {EQ}
  | '+'                   {PLUS}
  | '-'                   {MINUS}
  | '*'                   {MUL}
  | '/'                   {DIV}
  | '>'                   {GREATERTHAN}
  | '<'                   {SMALLERTHAN}
  | "\\="                 {NOT_EQ}
  | '|'                   {PIPE}
  | '!'                   {CUT}
  | '.'                   {DOT}
  | ":-"                  {ASSIGN}
  | '_'                   {UNDERSCORE}
  | '%'                   {single_line_comment lexbuf}
  | "/*"                  {multi_line_comment 0 lexbuf}
  | _ as s                {raise (InvalidToken s)}

and single_line_comment = parse
  | '\n'                  { Lexing.new_line lexbuf; read lexbuf } (* Update line number *)
  | _                     {single_line_comment lexbuf}
  | eof                   {EOF} (* Handle end of file within a comment *)

and multi_line_comment depth = parse
  | "*/"                  {if depth = 0 then read lexbuf else multi_line_comment (depth-1) lexbuf}
  | "/*"                  {multi_line_comment (depth+1) lexbuf}
  | '\n'                  { Lexing.new_line lexbuf; multi_line_comment depth lexbuf } (* Update line number *)
  | _                     {multi_line_comment depth lexbuf}
  | eof                   {failwith "Syntax error: Unterminated /* comment"} (* Handle end of file within a comment *)
