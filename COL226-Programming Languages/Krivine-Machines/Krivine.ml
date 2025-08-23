exception KrivineError of string

type opcode =
  | LDN of int
  | LDB of bool
  | LOOKUP of string
  | PLUS
  | TIMES
  | AND
  | OR
  | NOT
  | EQ
  | GT
  | PROJ of int
  | TUPLE
  | BIND of string
  | COND of (opcode list) * (opcode list)
  | APP
  | MKCLOS of string * (opcode list)
  | RET

type value =
  | N of int
  | B of bool
  | Closure of exp * (string * value) list
  | Mkclosure of string * (opcode list) * (string * value) list
  | TupleVal of (value list)
  | Unit

and exp =
  | V of string
  | Bl of bool
  | Plus of exp * exp
  | Times of exp * exp
  | And of exp * exp
  | Or of exp * exp
  | Not of exp
  | Eq of exp * exp
  | Gt of exp * exp
  | IfTE of exp * exp * exp
  | Abs of string * exp
  | App of exp * exp
  | Tuple of exp list
  | Proj of int * exp
  | Let of string * exp * exp
  | Num of int
  | UnitExp

and stack = value list
and dump = (stack * (string * value) list * opcode list) list

let rec show_exp = function
  | V x -> x
  | Bl b -> string_of_bool b
  | Num n -> string_of_int n
  | Plus (e1, e2) -> "(" ^ show_exp e1 ^ " + " ^ show_exp e2 ^ ")"
  | Times (e1, e2) -> "(" ^ show_exp e1 ^ " * " ^ show_exp e2 ^ ")"
  | And (e1, e2) -> "(" ^ show_exp e1 ^ " && " ^ show_exp e2 ^ ")"
  | Or (e1, e2) -> "(" ^ show_exp e1 ^ " || " ^ show_exp e2 ^ ")"
  | Not e -> "not " ^ show_exp e
  | Eq (e1, e2) -> "(" ^ show_exp e1 ^ " = " ^ show_exp e2 ^ ")"
  | Gt (e1, e2) -> "(" ^ show_exp e1 ^ " > " ^ show_exp e2 ^ ")"
  | IfTE (e0, e1, e2) -> "(if " ^ show_exp e0 ^ " then " ^ show_exp e1 ^ " else " ^ show_exp e2 ^ ")"
  | Abs (x, e) -> "(\\ " ^ x ^ " -> " ^ show_exp e ^ ")"
  | App (e1, e2) -> "(" ^ show_exp e1 ^ " " ^ show_exp e2 ^ ")"
  | Tuple exprs -> "(" ^ String.concat ", " (List.map show_exp exprs) ^ ")"
  | Let (x, e1, e2) -> "let " ^ x ^ " = " ^ show_exp e1 ^ " in " ^ show_exp e2
  | UnitExp -> "unit"

let rec show_opcodes opcodes =
  let opcode_to_string = function
    | LDN n -> "LDN " ^ string_of_int n
    | LDB b -> "LDB " ^ string_of_bool b
    | LOOKUP x -> "LOOKUP " ^ x
    | PLUS -> "PLUS"
    | TIMES -> "TIMES"
    | AND -> "AND"
    | OR -> "OR"
    | NOT -> "NOT"
    | EQ -> "EQ"
    | GT -> "GT"
    | TUPLE -> "TUPLE"
    | COND (c1, c2) -> "COND (" ^ show_opcodes c1 ^ ", " ^ show_opcodes c2 ^ ")"
    | APP -> "APP"
    | MKCLOS (x, c) -> "MKCLOS (" ^ x ^ ", " ^ show_opcodes c ^ ")"
    | RET -> "RET"
    | PROJ i -> "PROJ " ^ string_of_int i
    | BIND x -> "BIND " ^ x
  in
  String.concat "; " (List.map opcode_to_string opcodes)

let rec string_of_value = function
  | N n -> string_of_int n
  | B b -> string_of_bool b
  | TupleVal vs -> " ------tuple------ "
  (* | TupleVal vs ->
    let value_strings = List.map string_of_value vs in
    String.concat "; " value_strings  | TupleVal vs -> "Tuple(" ^ String.concat ", " (List.map string_of_value vs) ^ ")" *)
  | Unit -> "unit"

let rec take n lst =
  if n <= 0 then []
  else match lst with
       | [] -> []
       | hd :: tl -> hd :: take (n - 1) tl

let rec compile (e: exp) : opcode list =
  match e with
  | Num n -> [LDN n]
  | Bl b -> [LDB b]
  | V x -> [LOOKUP x]
  | Plus (e1, e2) -> compile e1 @ compile e2 @ [PLUS]
  | Times (e1, e2) -> compile e1 @ compile e2 @ [TIMES]
  | And (e1, e2) -> compile e1 @ compile e2 @ [AND]
  | Or (e1, e2) -> compile e1 @ compile e2 @ [OR]
  | Not e -> compile e @ [NOT]
  | Eq (e1, e2) -> compile e1 @ compile e2 @ [EQ]
  | Gt (e1, e2) -> compile e1 @ compile e2 @ [GT]
  | IfTE (e0, e1, e2) -> compile e0 @ [COND (compile e1, compile e2)]
  | Abs (x, e1) -> [MKCLOS (x, compile e1 @ [RET])]
  | App (e1, e2) -> (compile e1) @ (compile e2) @ [APP]
  
  
  (* | Tuple exprs ->  List.flatten (List.map compile exprs) @ [LDN (List.length exprs)] @ [TUPLE] *)
  (* | Proj (i, e) -> compile e @ [PROJ i] *)
  | Tuple exprs ->
    let compiled_exprs = List.flatten (List.map compile exprs) in
    let num_exprs = List.length exprs in
    compiled_exprs @ [LDN num_exprs; TUPLE]
  | Proj (i, e) -> compile e @ [PROJ i]

  | Let (x, e1, e2) ->
      compile e1 @ 
      [BIND x] @ 
      compile e2
  | UnitExp -> []

let rec stkmc (s: stack) (g: (string * value) list) (c: opcode list) (d: dump) : value =
  match s, g, c, d with
  | TupleVal vs :: s', _, [], _ ->
      TupleVal vs
  | Closure (Bl v, []) :: s', _, [], _ ->
      B v
  | Closure (Num v, []) :: s', _, [], _ ->
      N v
  | v :: _, _, [], _ -> v
  | s, g, LDN n :: c', d -> stkmc (Closure (Num n, []) :: s) g c' d
  | s, g, LDB b :: c', d -> stkmc (Closure (Bl b, []) :: s) g c' d
  | s, g, LOOKUP x :: c', d ->
      let v = List.assoc x g in
      stkmc (v :: s) g c' d
  | Closure (Num n1, _) :: Closure (Num n2, _) :: s', _, PLUS :: c', _ -> stkmc (Closure (Num (n1 + n2), []) :: s') g c' d
  | Closure (Num n1, _) :: Closure (Num n2, _) :: s', _, TIMES :: c', _ -> stkmc (Closure (Num (n1 * n2), []) :: s') g c' d
  | Closure (Bl b1, _) :: Closure (Bl b2, _) :: s', _, AND :: c', _ -> stkmc (Closure (Bl (b1 && b2), []) :: s') g c' d
  | Closure (Bl b1, _) :: Closure (Bl b2, _) :: s', _, OR :: c', _ -> stkmc (Closure (Bl (b1 || b2), []) :: s') g c' d
  | Closure (Bl b1, _) :: s', _, NOT :: c', _ -> stkmc (Closure (Bl (not b1), []) :: s') g c' d
  | Closure (Num n1, []) :: Closure (Num n2, []) :: s', _, EQ :: c', _ -> stkmc (Closure (Bl (n1 = n2), []) :: s') g c' d
  | Closure (Num n2, []) :: Closure (Num n1, []) :: s', _, GT :: c', _ -> stkmc (Closure (Bl (n1 > n2), []) :: s') g c' d
 
  | s, g, COND (c1, c2) :: c', d ->
    let cond = match s with
      | Closure (Bl b, []) :: _ -> b
      | _ -> false
    in
    if cond then
      stkmc s g (c1 @ c') d
    else
      stkmc s g (c2 @ c') d

  | s, g, MKCLOS (x, c') :: c'', d -> 
      stkmc (Mkclosure (x, c', g) :: s) g c'' d
  | a :: Mkclosure (x, c', g') :: s, g, APP :: c'', d -> stkmc [] ((x, a) :: g') c' ((s, g, c'') :: d)
  | v :: s', g, RET :: _, (s'', g', c'') :: d' -> stkmc (v :: s'') g' c'' d'

  | Closure(Num n, []) :: s', g, TUPLE :: c', d ->
    Printf.printf "\nin tuple\n";
    let tuple = TupleVal (List.rev (take n s')) in
    
    stkmc (tuple :: s') g c' d;
    

  | TupleVal vs :: s', _, PROJ i :: c', _ ->
    (* Printf.printf "\ninproj\n"; *)
    let idx = i - 1 in
    if idx >= 0 && idx < List.length vs then
      let proj_value = List.nth vs idx in
      (* Printf.printf "\nout of proj\n"; *)
      Printf.printf "\n%s\n", (string_of_value proj_value);
      stkmc (proj_value :: s') g c' d
    else
      raise (KrivineError "Projection index out of range")
 
  
  | s, g, BIND x :: c', d ->
      stkmc s ((x, List.hd s) :: g) c' d
  | _, _, _, _ -> raise (KrivineError "Invalid state")


let eval_expr (e: exp) : value =
  stkmc [] [] (compile e) []

(* Test cases for different expressions *)

(* let _ = *)
  (* Test arithmetic expressions *)
  (* let expr1 = Plus (Num 3, Num 4) in (* 3 + 4 *)
  assert (eval_expr expr1 = N 7); *)

  (* Test boolean expressions *)
  (* let expr3 = IfTE (Bl true, Num 10, Num 20) in (* if true then 10 else 20 *)
  assert (eval_expr expr3 = N 10); *)

  (* let expr4 = Not (Bl false) in (* not false *)
  assert (eval_expr expr4 = B true); *)

  (* Test function application *)
  (* let expr5 = App (Abs ("x", Plus (V "x", Num 3)), Num 7) in (* (fun x -> x + 3) 7 *)
  assert (eval_expr expr5 = N 10); *)

  (* let expr6 = Let ("x", Num 5, Plus (V "x", Num 3)) in (* let x = 5 in x + 3 *)
  assert (eval_expr expr6 = N 8); *)

  (* Test conditional with comparison *)
  (* let expr9 = IfTE (Gt (Num 10, Num 5), Bl true, Bl false) in (* if 10 > 5 then true else false *)
  assert (eval_expr expr9 = B true); *)

  (* Print results *)
  (* print_endline ("Expr 1: " ^ show_exp expr1 ^ " = " ^ string_of_value (eval_expr expr1)); *)
  (* print_endline ("Expr 3: " ^ show_exp expr3 ^ " = " ^ string_of_value (eval_expr expr3)); *)
  (* print_endline ("Expr 4: " ^ show_exp expr4 ^ " = " ^ string_of_value (eval_expr expr4)); *)
  (* print_endline ("Expr 5: " ^ show_exp expr5 ^ " = " ^ string_of_value (eval_expr expr5)); *)
  (* print_endline ("Expr 6: " ^ show_exp expr6 ^ " = " ^ string_of_value (eval_expr expr6)); *)
  (* print_endline ("Expr 9: " ^ show_exp expr9 ^ " = " ^ string_of_value (eval_expr expr9)); *)

(* Define test expressions *)
(* Evaluate an expression and print the result *)
let evaluate_expression expr =
  let result = eval_expr expr in
  Printf.printf "Expression: %s\n" (show_exp expr);
  Printf.printf "Result: %s\n\n" (string_of_value result)

(* Example expressions to evaluate *)

let expr1 = Plus (Num 3, Num 4)  
let expr2 = IfTE (Bl true, Num 10, Num 20)  
let expr3 = Not (Bl false)  
let expr4 = App (Abs ("x", Plus (V "x", Num 3)), Num 7)  
let expr5 = Let ("x", Num 5, Plus (V "x", Num 3))  
let expr6 = Tuple [Num 1; Num 7; Num 5] 


let expr7 = Proj (2, expr6)  
(* let expr8 = IfTE (Gt (Num 10, Num 5), Bl true, Bl false)   *)

(* Evaluate each expression and print the results *)
(* let () =  *)
  (* evaluate_expression expr1;
  evaluate_expression expr2;
  evaluate_expression expr3;
  evaluate_expression expr4; *)
  (* evaluate_expression expr5; *)
  (* evaluate_expression expr6; *)
  (* evaluate_expression expr7; *)
  (* evaluate_expression expr8; *)

