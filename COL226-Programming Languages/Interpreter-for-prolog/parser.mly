%{
    open Interpreter;;
%}

%token <string> VAR CONS
%token <int> NUM
%token LPAREN RPAREN LB RB COMMA EQ NOT_EQ DOT CUT ASSIGN PIPE PLUS MINUS 
        MUL DIV GREATERTHAN SMALLERTHAN  UNDERSCORE COLON SEMICOLON EOF

%left COMMA
%nonassoc EQ PIPE SMALLERTHAN GREATERTHAN
%left PLUS MINUS
%left MUL DIV
%nonassoc DOT

%start program goal
%type <Interpreter.program> program
%type <Interpreter.goal> goal
%%

program:
    EOF                                 {[]}
  | clause_list EOF                     {$1}
;

clause_list:
    clause                              {[$1]}
  | clause clause_list                  {($1)::$2}
;

clause:
    atomic_formula DOT                           {Fact(Head($1))}
  | atomic_formula ASSIGN atomic_formula_list DOT            {Rule(Head($1), Body($3))}
;

goal:
    atomic_formula_list DOT                      {Goal($1)}
;

atomic_formula_list:
    atomic_formula                                {[$1]}
  | atomic_formula COMMA atomic_formula_list                {($1)::$3}
;

atomic_formula:
    /* LPAREN atomic_formula RPAREN                          {$2} */
  | CONS                                {Atm_form($1, [])}
  | CONS LPAREN term_list RPAREN                {Atm_form($1, $3)}
  | term EQ term                        {Atm_form("_eq", [$1; $3])}
  | term NOT_EQ term                    {Atm_form("_not_eq", [$1; $3])}
  | term SMALLERTHAN term                        {Atm_form("<", [$1; $3])}
  | term GREATERTHAN term                        {Atm_form(">", [$1; $3])}
  | CUT                                 {Atm_form("_cut", [])}
;

term_list:
    term                                {[$1]}
  | term COMMA term_list                {($1)::$3}
;

term:
    LPAREN term RPAREN                  {$2}
  | VAR                                 {Var($1)}
  | UNDERSCORE                          {Var("_")}
  | CONS                                {Node($1, [])}
  | NUM                                 {Num($1)}
  | CONS LPAREN term_list RPAREN                {Node($1, $3)}
  | term PLUS term                      {Node("+", [$1; $3])}
  | term MINUS term                     {Node("-", [$1; $3])}
  | term MUL term                      {Node("*", [$1; $3])}
  | term DIV term                       {Node("/", [$1; $3])}
  | LB RB                               {Node("_empty_list", [])}
  | LB list_body RB                     {$2}
;

list_body:
    term                                 {Node("_list", [$1; Node("_empty_list", [])])}
  | term COMMA list_body                 {Node("_list", [$1; $3])}
  | term PIPE term                       {Node("_list", [$1; $3])}
;