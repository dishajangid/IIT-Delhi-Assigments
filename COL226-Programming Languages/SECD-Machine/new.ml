exception SECDError of string

type pattern = 
  | PNum of int 
  | PBl of bool
  | PTuple of pattern list

type opcode = LDN of int | LDB of bool | LOOKUP of string | PLUS | TIMES | AND | OR | NOT | EQ | GT 
            | PROJ of int | CASE of (pattern * (opcode list)) list | TUPLE | BIND of string
            | COND of (opcode list) * (opcode list) | APP | MKCLOS of string*(opcode list) | RET;;

(* type environment = (stfring * value) list *)

type value =
  | N of int  (* Numeric value *)
  | B of bool (* Boolean value *)
  | Closure of string * (opcode list) * (string * value) list
  | TupleVal of (value list)
  | Unit

type exp = Num of int | Bl of bool | V of string | Plus of exp*exp | Times of exp*exp | And of exp*exp | Or of exp*exp | Not of exp 
        | Eq of exp*exp | Gt of exp*exp | IfTE of exp*exp*exp | Abs of string*exp | App of exp*exp | Tuple of exp list | Proj of int * exp 
        | Case of exp * ((pattern * exp) list)  | Let of string * exp * exp | Unit | Match of exp * ((pattern * exp) list)


type stack = value list
type dump = (stack * (string * value) list * (opcode list)) list


(* helper functions  *)
let rec show_exp = function
  | Num n -> string_of_int n
  | Bl b -> string_of_bool b
  | V x -> x
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
  | Proj (i, e) -> "Proj(" ^ string_of_int i ^ ", " ^show_exp e ^ ")"
  
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
    | COND (c1, c2) -> "COND (" ^ show_opcodes c1 ^ ", " ^ show_opcodes c2 ^ ")"
    | APP -> "APP"
    | MKCLOS (x, c) -> "MKCLOS (" ^ x ^ ", " ^ show_opcodes c ^ ")"
    | RET -> "RET"
    | TUPLE -> "TUPLE"
    | PROJ i -> "PROJ" ^ string_of_int i
    | BIND x -> "BIND " ^ x 

  in
  String.concat "; " (List.map opcode_to_string opcodes)


let rec match_pattern (pat: pattern) (v: value) (acc: 'a) (env: (string * value) list) : 'a =
  match pat, v with
  | PNum n, N m -> if n = m then acc else raise (SECDError "Pattern matching failed")
  | PBl b1, B b2 -> if b1 = b2 then acc else raise (SECDError "Pattern matching failed")
  | PTuple pats, TupleVal vs -> List.fold_left2 (fun acc p v -> match_pattern p v acc env) acc pats vs
  | _ -> raise (SECDError "Pattern matching failed")


  let string_of_value = function
  | N n -> string_of_int n
  | B b -> string_of_bool b
  | Closure (x, _, _) -> "Closure(" ^ x ^ ")"
  | TupleVal vs -> "tuple val"
  
  | _ -> raise (SECDError "string_of_value match failed")



  let rec take n lst =
    if n <= 0 then []
    else match lst with
         | [] -> []
         | hd :: tl -> hd :: take (n - 1) tl



let rec eval (e: exp) (rho: (string * value) list -> value) (env: (string * value) list) : value =

  match e with
  | Num n -> N n
  | Bl b -> B b 
  | V x -> rho env
  | Plus (e1, e2) -> let N n1 = (eval e1 rho env) and N n2 = (eval e2 rho env) in N (n1 + n2)
  | Times (e1, e2) -> let N n1 = (eval e1 rho env) and N n2 = (eval e2 rho env) in N (n1 * n2)
  | And (e1, e2) -> let B n1 = (eval e1 rho env) and B n2 = (eval e2 rho env) in B (n1 && n2)
  | Or (e1, e2) -> let B n1 = (eval e1 rho env) and B n2 = (eval e2 rho env) in B (n1 || n2)
  | Not e1 -> let B b1 = (eval e1 rho env) in B (not b1)
  | Eq (e1, e2) -> let N n1 = (eval e1 rho env) and N n2 = (eval e2 rho env) in B (n1 = n2)
  | Gt (e1, e2) -> let N n1 = (eval e1 rho env) and N n2 = (eval e2 rho env) in B (n1 > n2)
  | IfTE (e0, e1, e2) -> let B b0 = (eval e0 rho env) in if b0 then (eval e1 rho env) else (eval e2 rho env)
  | Tuple exprs -> TupleVal (List.map (fun e -> eval e rho env) exprs)
  | Proj (i, e) -> (match eval e rho env with | TupleVal vals -> List.nth vals (i-1) | _ -> raise (Failure "Projection from non-tuple"))
  | Let (x, e1, e2) ->
    let v = eval e1 rho env in
    let new_env = (x, v) :: env in
    eval e2 rho new_env
  | Unit -> Unit
  | _ -> raise (SECDError "eval match failed")

let rec compile (e: exp) : opcode list =
  let rec compile_case (p, exp) = compile exp @ [CASE [(p, compile exp)]] in
  match e with 
  | Num n -> [LDN n]
  | Bl b -> [LDB b]
  | V x -> [LOOKUP x]
  | Plus (e1, e2) -> (compile e1) @ (compile e2) @ [PLUS]
  | Times (e1, e2) -> (compile e1) @ (compile e2) @ [TIMES]
  | And (e1, e2) -> (compile e1) @ (compile e2) @ [AND]
  | Or (e1, e2) -> (compile e1) @ (compile e2) @ [OR]
  | Not e1 -> (compile e1) @ [NOT]
  | Eq (e1, e2) -> (compile e1) @ (compile e2) @ [EQ]
  | Gt (e1, e2) -> (compile e1) @ (compile e2) @ [GT]
  | IfTE (e0, e1, e2) -> (compile e0) @ [COND (compile e1, compile e2)]
  | App (e1, e2) -> (compile e1) @ (compile e2) @ [APP]
  | Abs (x, e1) -> [MKCLOS (x, (compile e1) @ [RET])]
  | Tuple exprs ->  List.flatten (List.map compile exprs) @ [LDN (List.length exprs)] @ [TUPLE]
  | Proj (i, e) -> compile e @ [PROJ i]
  | Let (x, e1, e2) ->
    compile e1 @      (* Compile the binding expression *)
    [BIND x] @
    compile e2        (* Compile the body expression *)
  | Match (scrutinee, cases) ->
    let compiled_scrutinee = compile scrutinee in
    let compiled_cases = List.map (fun (pat, exp) -> (pat, compile exp)) cases in
    let compiled_branches = List.map (fun (pat, exp) -> (pat, compile exp)) cases in
    compiled_scrutinee @ [CASE compiled_branches]

  | _ -> raise (SECDError "compile match failed")
  
let rec stkmc (s: stack) (g: (string * value) list) (c: opcode list) (d: dump) : value =
  match s, g, c, d with
  (* | v :: _, _ ,[], _ -> v (*No more opcodes, return top*) *)
  
  | TupleVal vs :: _, _, [], _ ->
    (* Print the tuple elements individually *)
    let print_tuple_element = function
      | N n -> Printf.printf "%d " n
      | B b -> Printf.printf "%b " b
      | _ -> Printf.printf "Invalid"
    in
    Printf.printf "Final Stack: Tuple(";
    List.iter print_tuple_element vs;
    print_endline ")";
    TupleVal vs (* Return the tuple *)

  | v :: _, _, [], _ ->
    (* Print the whole stack *)
    Printf.printf "Final Stack: %s\n" (String.concat "; " (List.map string_of_value (List.rev (s))));
    v (* Return the top of the stack *)

  | s,_, LDN n :: c', _ -> stkmc (N n :: s) g c' d
  | s,_, LDB b :: c', _ -> stkmc (B b :: s) g c' d
  | s, _, LOOKUP x :: c', _ -> stkmc ((List.assoc x g) :: s) g c' d
  | N n2 :: N n1 :: s', _, PLUS :: c', _ -> stkmc (N (n1 + n2) :: s') g c' d
  | N n2 :: N n1 :: s', _, TIMES :: c', _ -> stkmc (N (n1 * n2) :: s') g c' d
  | B b2 :: B b1 :: s', _, AND :: c', _ -> stkmc (B (b1 && b2) :: s') g c' d
  | B b2 :: B b1 :: s', _, OR :: c', _ -> stkmc (B (b1 || b2) :: s') g c' d
  | B b1 :: s', _, NOT :: c', _ -> stkmc (B (not b1) :: s') g c' d
  | N n2 :: N n1 :: s', _, EQ :: c', _ -> stkmc (B (n1 = n2) :: s') g c' d
  | N n2 :: N n1 :: s', _, GT :: c', _ -> stkmc (B (n1 > n2) :: s') g c' d
  | B true :: s', _, COND (c1, c2) :: c', _ -> stkmc s' g (c1 @ c') d
  | B false :: s', _, COND (c1, c2) :: c', _ -> stkmc s' g (c2 @ c') d
  | s, g, (MKCLOS (x, c') :: c''), d -> stkmc (Closure (x, c',g)::s) g c'' d
  | a:: Closure (x,c',g')::s, g, APP :: c'', d -> stkmc [] ((x,a)::g') c' ((s,g,c'')::d)
  | a :: s', g'', RET :: c', (s, g, c'') :: d -> stkmc (a :: s) g c'' d

  | N n :: s', g, TUPLE :: c', d ->
    let tuple = TupleVal (List.rev (take (n) s')) in
    stkmc (tuple :: s') g c' d (* Push the constructed tuple onto the stack *)

  (* | TupleVal vs :: s', g, PROJ i :: c', d ->
    let projected_value = List.nth vs (i - 1) in
    stkmc (projected_value :: s') g c' d *)


  | TupleVal vs :: s', g, PROJ i :: c', d ->
    let projected_value = List.nth vs (i - 1) in
    let rec drop_n n lst =
      if n <= 0 then lst
      else match lst with
            | _ :: tl -> drop_n (n - 1) tl
            | [] -> []
    in
    let rest_of_stack = drop_n (List.length vs) s' in
    stkmc (projected_value :: rest_of_stack) g c' d

  | s, g, BIND x :: c', d ->
    let v = List.hd s in
    let new_g = (x, v) :: g in
    stkmc (List.tl s) new_g c' d
    
  (* | s, g, CASE branches :: c', d ->
    let rec find_matching_branch = function
      | [] -> raise (SECDError "No matching pattern")
      | (pat, exp) :: rest ->
          try
            let scrutinee = match s with
                              | [] -> raise (SECDError "Empty stack")
                              | hd :: _ -> hd
            in
            let match_exp = eval pat (fun _ -> scrutinee) g in
            (match match_exp with
            | TupleVal vs ->
              (match pat with
              | PTuple pats ->
                  let env' = List.combine pats vs @ g in
                  stkmc s env' exp d
              | _ -> raise (SECDError "Invalid pattern"))
          
            | _ -> raise (SECDError "Invalid pattern"))
          with SECDError _ ->
            find_matching_branch rest
    in
    find_matching_branch branches *)
  
    


  | s ,e , c, d -> 
      Printf.printf "Stack: %s\n" (String.concat "; " (List.map string_of_value s));
      Printf.printf "Environment: %s\n" (String.concat ", " (List.map (fun (var, value) -> var ^ "=" ^ string_of_value value) e));
      Printf.printf "Opcodes: %s\n" (show_opcodes c);
      Printf.printf "Dump: %s\n" (String.concat "; " (List.map (fun (s', e', c') -> "(" ^ (String.concat "; " (List.map string_of_value s')) ^ ", " ^ (String.concat ", " (List.map (fun (var, value) -> var ^ "=" ^ string_of_value value) e')) ^ ", " ^ (show_opcodes c') ^ ")") d));
      raise (SECDError "did not match")  
;;
(*test 1*)



  (* let test_with_env_and_vars () =

    let exp = App (Abs ("x", IfTE (V "x", Num 1, Num 0)), Bl true) in
    (* let exp = App (Abs ("x", V "x"), IfTE (Bl false, Num 2, Num 3)) in *)
    (* let exp = App (Abs ("x", IfTE (V "x", Num 1, Num 0)), IfTE (Bl true, Bl false, Bl true)) in *)
      
    Printf.printf "Expression: %s\n" (show_exp exp);
    let compiled = compile exp in
    Printf.printf "Compiled opcodes: %s\n" (show_opcodes compiled);
    let result = stkmc [] [("x", N 6); ("y", N 3)] compiled [] in
    match result with
    | N n -> Printf.printf "Result: %d\n" n
    | _ -> print_endline "Invalid result"
  ;;
  
  test_with_env_and_vars ();; *)




   
  (* let rho = [("x", N 4)] ;;
   let abs1 = Abs("x", Plus(V "x" ,Num 5));;
   let app1 = App( abs1, Num 10);;
   let c = compile app1;;
   let x = stkmc [] rho c [] ;; *)



(*test 2 *)
   (* let test_boolean_operations () =
    (* Define the expression: (true && false) || (true || false) *)
    let exp = Or (And (Bl true, Bl false), Or (Bl true, Bl false)) in
    let compiled = compile exp in
    let result = stkmc [] [] compiled [] in
    match result with
    | B b -> Printf.printf "Result: %b\n" b
    | _ -> print_endline "Invalid result"
  ;;
  
  test_boolean_operations (); *)



  
  
  (* let test_conditional_expression () =
    (* Define the expression: if (3 > 2) then 5 else 10 *)
    let exp = IfTE (Gt (Num 3, Num 2), Num 5, Num 10) in
    let compiled = compile exp in
    let result = stkmc [] [] compiled [] in
    match result with
    | N n -> Printf.printf "Result: %d\n" n
    | _ -> print_endline "Invalid result"
  ;;
  
  test_conditional_expression (); *)


  
  (* let test_tuple_opcode () =
    let exp = Tuple [Num 1; Num 23; Num 32] in
    let compiled = compile exp in
    Printf.printf "Compiled opcodes: %s\n" (show_opcodes compiled);
    let result = stkmc [] [] compiled [] in
    match result with
    | TupleVal values ->
      Printf.printf "Result: ";
      List.iter (fun value -> Printf.printf "%s " (string_of_value value)) values;
      print_endline ""
    | _ -> print_endline "Invalid result";;
  
  test_tuple_opcode ();; *)
  


  (* let test_projection_opcode () =
    let proj_exp = Proj (4, Tuple [Num 1; Num 5; Num 4; Num 3]) in
    let proj_compiled = compile proj_exp in
    let result = stkmc [] [] proj_compiled [] in
  
    match result with
    | N n ->
      Printf.printf "Result of projection: %d\n" n
    | _ ->
      print_endline "Invalid result"
  ;;
  
  test_projection_opcode (); *)
   

  let test_let_expression () =
    (* let exp = Let ("x", Num 5, Plus (V "x", Num 10)) in *)
    let exp =
      Let ("x", Num 5,
        Plus (V "x", 
          Let ("y", Num 24,
            Times (V "y", Plus (V "x", Num 10))
          )
        )
      ) in
    
    (* Compile the expression *)
    let compiled = compile exp in
    (* Print the compiled opcode list *)
    Printf.printf "Compiled opcodes: %s\n" (show_opcodes compiled);
    (* Execute the compiled opcode list *)
    let result = stkmc [] [] compiled [] in
    match result with
    | N n -> Printf.printf "Result: %d\n" n
    | _ -> print_endline "Invalid result";;
  
  (* Test the let expression *)
  test_let_expression ();
  