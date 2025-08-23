%{
    open Ast
    open List
    open Printf
%}
// %token <string> LPAREN RPAREN LSQUARE RSQUARE COMMA DOT ASSIGN VAR CONST FUNC INT EOF

%token <string> VAR CONST FUNC PRED KEYWORD 
%token <int> INT
%token<char> CUT UNDERSCORE EQUAL  
%token LPAREN RPAREN COMMA DOT EOF RSQUARE LSQUARE ASSIGN GOAL  PIPE SEMICOLON COLON

%start program
%type <Ast.clause_list> program
%type <Ast.atomic_formula list> atomic_formula_list

%%

program:
   clause_list EOF { Clause_list ($1) }
   | clause_list goal DOT EOF { Prog ($1, $2)}

clause_list:
    clause DOT{[$1] }
  | clause_list clause DOT {$2 :: $1 }
  // | clause_list goal DOT {Prog ($1, $2)}
  
clause:
    fact { Fact ($1) }
  | rule {  Rule ($1) }
  // | rule {Truth($1)}

fact:
    atomic_formula  {Fact ($1) }

rule:
    atomic_formula ASSIGN atomic_formula_list  { Rule ($1, $3) }
    // | atomic_formula ASSIGN CUT {Printf.printf ":::atomic formula rule truth:::\n"; Truth($1, $3)}

atomic_formula:
    FUNC LPAREN term_list RPAREN { Atm_form ($1, $3) }
    | CUT {Cut($1)}
    | KEYWORD {Key_word ($1)}
    | FUNC {Identifier ($1)}
    | VAR {Var($1)}
    | LSQUARE term PIPE term RSQUARE {Non_Empty}
    | atomic_formula EQUAL atomic_formula {Non_Empty}
    | atomic_formula SEMICOLON atomic_formula {Non_Empty}
    // | EQUAL {Printf.printf "Identified correct "; Equal($1)}
    // | SEMICOLON {Semicolon($1)}
    

atomic_formula_list:
    atomic_formula { [$1]}
  | atomic_formula COMMA atomic_formula_list {$1 :: $3} 
  | atomic_formula SEMICOLON atomic_formula_list {$1 :: $3} 

term_list:
    term {[$1] }
  | term COMMA term_list {$1 :: $3 }
  | term SEMICOLON term_list {$1 :: $3 }

term:
    VAR { Var ($1) }
  | CONST { Const ($1) }
  | INT  { Int ($1)}
  | FUNC LPAREN term_list RPAREN {Func ($1, $3) }
  | LSQUARE RSQUARE {Empty }
  | LSQUARE term RSQUARE { Non_Empty }
  | LPAREN term RPAREN { Non_Empty }
  | LSQUARE term PIPE term RSQUARE { Non_Empty }
  | term EQUAL term {Non_Empty}
  | term SEMICOLON term {Non_Empty}
  // | LSQUARE term PIPE UNDERSCORE RSQUARE { Non_Empty }
  | UNDERSCORE {Underscore($1) }
  | FUNC {Identifier ($1)}
  | term COMMA term      {MoreTerm ($1 , $3)}
  
  | KEYWORD {Key_word($1)}
  // | EQUAL {Printf.printf "Identified correct "; Equal($1)}
  // | SEMICOLON {Semicolon($1)}

goal:
    GOAL atomic_formula {[$2] }
  // | goal COMMA atomic_formula { match $1 with Goal lst -> Goal ($3 :: lst) }
    | GOAL atomic_formula COMMA goal {$2 :: $4 }