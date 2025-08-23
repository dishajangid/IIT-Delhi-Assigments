type variable = string
type symbol = string
type signature = (symbol * int) list
type term = Var of variable | Num of int | Node of symbol * (term list)
type atomic_formula = Atm_form of symbol * (term list)
type head = Head of atomic_formula
type body = Body of atomic_formula list
type clause = Fact of head | Rule of head * body
type program = clause list
type goal = Goal of atomic_formula list
type substitution = (variable * term) list

exception NOT_UNIFIABLE
exception NotFound
exception InvalidProgram
exception NotPossible

let exists x ys =
  List.exists (fun z -> x = z) ys

let union l1 l2 =
  List.fold_right (fun x acc -> if exists x acc then acc else x :: acc) l1 l2
  
let checkProgram prog =
  List.for_all (function
    | Fact (Head a) | Rule (Head (a), _) ->
        (match a with
        | Atm_form ("_eq", _) | Atm_form ("_not_eq", _) | Atm_form ("_cut", _)
          | Atm_form (">", _) | Atm_form ("<", _) -> raise InvalidProgram
        | _ -> true)
  ) prog

let rec modifyTerm (i:int) (t:term): term = match t with
    Var(v) -> Var((string_of_int i) ^ v)
  | Node(s, l) -> Node(s, List.map (modifyTerm i) l)
  | Num _ -> t
;;

let rec modifyAtom (i:int) (a:atomic_formula): atomic_formula = match a with
  Atm_form(s, l) -> Atm_form(s, List.map (modifyTerm i) l)
;;

let rec modifyClause (cl:clause) (i:int): clause = match cl with
    Fact(Head(a)) -> Fact(Head(modifyAtom i a))
  | Rule(Head(a), Body(l)) -> Rule(Head(modifyAtom i a), Body(List.map (modifyAtom i) l))
;;


let rec modifyInitialProg prog i =
  List.map (fun cl -> modifyClause cl i) prog
;;
let rec modifyProg2 prog (Atm_form (s, _)) =
  List.map (fun cl ->
    match cl with
    | Fact (Head (Atm_form (s', _))) | Rule (Head (Atm_form (s', _)), _) ->
        if s = s' then modifyClause cl 0 else cl
  ) prog
;;

let rec vars_term (t:term): variable list =
  match t with
      Var(v) -> [v]
    | Node(s, l) -> List.fold_left union [] (List.map vars_term l)
    | Num _ -> []
;;

let vars_atom (Atm_form(s, l): atomic_formula): variable list = vars_term (Node(s, l))
;;

let rec vars_goal (Goal(g): goal): variable list = List.fold_left union [] (List.map vars_atom g)
;;

let rec subst (substitution : substitution) (term : term) : term =
  match term with
  | Var x ->
      (match List.assoc_opt x substitution with
      | Some t -> t
      | None -> Var x)
  | Num _ -> term
  | Node (s, l) ->
      let l' = List.map (subst substitution) l in
      Node (s, l')
  ;;

let rec subst_atom (s : substitution) (Atm_form (s', l) : atomic_formula) : atomic_formula =
  let subst_term_list = List.map (subst s) l in
  Atm_form (s', subst_term_list)
;;  


let rec variableInTerm (v:variable) (t:term): bool =
  match t with
      Var(x) -> x = v
    | Node(s, l) -> List.fold_left (||) false (List.map (variableInTerm v) l)
    | _ -> false
;;

let rec compose (s1 : substitution) (s2 : substitution) : substitution =
  let apply_subst (x, t) =
    let t' = subst s2 t in
    (x, t')
  in
  let s1' = List.map apply_subst s1 in
  let s2_not_in_s1 =
    List.filter (fun (x, _) -> not (List.exists (fun (y, _) -> x = y) s1')) s2
  in
  s1' @ s2_not_in_s1
;;

let rec mgu_term (t1:term) (t2:term): substitution =
  match (t1, t2) with
      (Var(x), Var(y)) -> if x = y then []
                      else [(x, Var(y))]
    | (Var(x), Node(_, _)) -> if variableInTerm x t2 then raise NOT_UNIFIABLE
                            else [(x, t2)]
    | (Node(_, _), Var(y)) -> if variableInTerm y t1 then raise NOT_UNIFIABLE
                            else [(y, t1)]
    | (Num(n1), Num(n2)) -> if n1 = n2 then [] else raise NOT_UNIFIABLE
    | (Num(n1), Var(x)) -> [(x, t1)]
    | (Var(x), Num(n2)) -> [(x, t2)] 
    | (Node(s1, l1), Node(s2, l2)) ->
        if s1 <> s2 || (List.length l1 <> List.length l2) then raise NOT_UNIFIABLE
        else
          let f s tt = compose s (mgu_term (subst s (fst tt)) (subst s (snd tt))) in
          List.fold_left f [] (List.combine l1 l2)
    | _ -> raise NOT_UNIFIABLE
;;

let mgu_atom (Atm_form(s1, l1): atomic_formula) (Atm_form(s2, l2): atomic_formula): substitution = mgu_term (Node(s1, l1)) (Node(s2, l2))
;;



let rec print_term_list (tl:term list) = match tl with
    [] -> Printf.printf ""
  | [t] -> print_term t
  | t::tls -> (
      print_term t;
      Printf.printf ",";
      print_term_list tls;
    )

and print_list_body (t:term) = match t with
    Node("_empty_list", []) -> Printf.printf ""
  | Node("_list", [t1; Node("_empty_list", [])]) -> print_term t1
  | Node("_list", [t1; t2]) -> (
      print_term t1;
      Printf.printf ",";
      print_list_body t2;
    )
  | _ -> raise NotPossible

and print_term (t:term) = match t with
    Var(v) -> Printf.printf " %s " v
  | Node("_empty_list", []) -> Printf.printf " [] "
  | Node(s, []) -> Printf.printf " %s " s
  | Node("_list", _) -> (
      Printf.printf " [";
      print_list_body t;
      Printf.printf "] ";
    )
  | Node(s, l) -> (
      Printf.printf " %s ( " s;
      print_term_list l;
      Printf.printf " ) ";
    )
  | Num(n) -> Printf.printf " %d " n
;;

let rec getSolution (unif:substitution) (vars:variable list) = match vars with
    [] -> []
  | v::vs ->
      let rec occurs l = match l with
          [] -> raise NotFound
        | x::xs -> if (fst x) = v then x
                    else occurs xs
      in
      try (occurs unif)::getSolution unif vs
      with NotFound -> getSolution unif vs
;;

let get1char () =
  let termio = Unix.tcgetattr Unix.stdin in
  let () = Unix.tcsetattr Unix.stdin Unix.TCSADRAIN
          { termio with Unix.c_icanon = false } in
  let res = input_char stdin in
  Unix.tcsetattr Unix.stdin Unix.TCSADRAIN termio;
  res

let rec printSolution (unif:substitution) = match unif with
    [] -> Printf.printf "true. "
  | [(v, t)] -> (
      Printf.printf "%s =" v;
      print_term t;
    )
  | (v, t)::xs -> (
      Printf.printf "%s =" v;
      print_term t;
      Printf.printf ", ";
      printSolution xs;
    )
;;

let solve_atom_atom (a1:atomic_formula) (a2:atomic_formula) (unif:substitution): substitution =
  compose unif (mgu_atom (subst_atom unif a1) (subst_atom unif a2))
;;

let solve_term_term (t1:term) (t2:term) (unif:substitution): substitution =
  compose unif (mgu_term (subst unif t1) (subst unif t2))
;;

let rec simplify_term (t:term): term = match t with
    Num(_) -> t
  | Node("+", [t1; t2]) -> (
      match ((simplify_term t1), (simplify_term t2)) with
          (Num(n1), Num(n2)) -> Num(n1 + n2)
        | _ -> raise NOT_UNIFIABLE
    )
  | Node("-", [t1; t2]) -> (
      match ((simplify_term t1), (simplify_term t2)) with
          (Num(n1), Num(n2)) -> Num(n1 - n2)
        | _ -> raise NOT_UNIFIABLE
    )
  | Node("*", [t1; t2]) -> (
      match ((simplify_term t1), (simplify_term t2)) with
          (Num(n1), Num(n2)) -> Num(n1 * n2)
        | _ -> raise NOT_UNIFIABLE
    )
  | Node("/", [t1; t2]) -> (
      match ((simplify_term t1), (simplify_term t2)) with
          (Num(n1), Num(n2)) -> Num(n1 / n2)
        | _ -> raise NOT_UNIFIABLE
      )
  | _ -> t
;;

let eval (a:atomic_formula) (unif:substitution): substitution = match a with
    Atm_form("_eq", [t1; t2])
  | Atm_form("_not_eq", [t1; t2]) -> compose unif (mgu_term (simplify_term (subst unif t1)) (simplify_term (subst unif t2)))
  | Atm_form(">", [t1; t2]) -> (
        match (simplify_term (subst unif t1), simplify_term (subst unif t2)) with
            (Num(n1), Num(n2)) -> if n1 > n2 then unif else raise NOT_UNIFIABLE
          | _ -> raise NOT_UNIFIABLE
    )
  | Atm_form("<", [t1; t2]) -> (
      match (simplify_term (subst unif t1), simplify_term (subst unif t2)) with
          (Num(n1), Num(n2)) -> if n1 < n2 then unif else raise NOT_UNIFIABLE
        | _ -> raise NOT_UNIFIABLE
    )
  | _ -> unif
;;

let rec solve_goal (prog:program) (g:goal) (unif:substitution) (vars:variable list): (bool * substitution) =
  match g with
      Goal([]) -> (
        printSolution (getSolution unif vars);
        flush stdout;
        let choice = ref (get1char()) in
        while(!choice <> '.' && !choice <> ';') do
          Printf.printf "\nUnknown Action: %c \nAction? " (!choice);
          flush stdout;
          choice := get1char();
        done;
        Printf.printf "\n";
        if !choice = '.' then (true, [])
        else (false, [])
      )
    | Goal(a::gs) -> match a with
          Atm_form("_eq", _) | Atm_form(">", _) | Atm_form("<", _) -> (
            try solve_goal prog (Goal(gs)) (eval a unif) vars
            with NOT_UNIFIABLE -> (false, [])
          )
        | Atm_form("_not_eq", _) -> (
            try (false, eval a unif)
            with NOT_UNIFIABLE -> solve_goal prog (Goal(gs)) unif vars
          )
        | Atm_form("_cut", _) -> let _ = solve_goal prog (Goal(gs)) unif vars in (true, [])
        | _ ->
          let new_prog = modifyProg2 prog a in
          let rec iter prog' = match prog' with
              [] -> (false, [])
            | cl::ps -> match cl with
                Fact(Head(a')) -> (
                  try
                    let u = (solve_atom_atom a' a unif) in
                    match (solve_goal new_prog (Goal(gs)) u vars) with
                        (true, u') -> (true, u')
                      | _ -> iter ps
                  with NOT_UNIFIABLE -> iter ps
                )
              | Rule(Head(a'), Body(al)) -> (
                  try
                    let u = (solve_atom_atom a' a unif) in
                    match (solve_goal new_prog (Goal(al @ gs)) u vars) with
                        (true, u') -> (true, u')
                      | _ -> iter ps
                  with NOT_UNIFIABLE -> iter ps
                )
        in iter prog
;;

let interpret_goal (prog:program) (g:goal) = solve_goal prog g [] (vars_goal g)
;;