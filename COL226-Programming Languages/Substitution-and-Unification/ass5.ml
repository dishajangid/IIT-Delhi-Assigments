
type symbol = string;;
type signature = (symbol* int) list;;  (*each sumbol is associated with an arity*)

type tree = V of string | C of {node: symbol ; children: tree list};;

let check_sig (sig_list : signature): bool = 
  let rec check_duplicates = function
    | [] -> false
    | (x,_) :: xs -> List.exists (fun (s, _) -> s = x) xs || check_duplicates xs
  in 
  let rec non_negative = function
    | [] -> true
    | (_, arity) :: rest -> arity >= 0 && non_negative rest
  in 
  not (check_duplicates sig_list) && non_negative sig_list;;

(* let rec wftree (signature: signature) (tree: tree) : bool = 
   *)
(* let signature1 = [("+", 2) ; ("-", 2) ; ("$", 0) ; ("not", -1)];; *)
(* let check_s = check_sig signature1;;
Printf.printf "Check signature 1 : %b\n" check_s;; *)

let rec ht (tree : tree) : int =
  match tree with
  | V _ -> 1
  | C { children } -> 1 + List.fold_left (fun acc t -> max acc (ht t)) 0 children;;

let rec sz (tree: tree) : int = 
  match tree with
  | V _ -> 1
  | C {children} -> 1 + List.fold_left (fun acc t -> acc + sz t) 0 children;;



(* let tree =
  C {
    node = "a";
    children = [
      C { node = "+"; children = [C {node = "e"; children = [V "x" ; V "y"]};V "f"]};
      V "1";
      C { node = "-"; children = [V "1"; V "4"]}
    ]
  };; *)



(* let height = ht tree ;;
Printf.printf "Height of the tree: %d\n" height;;
let size = sz tree ;;
Printf.printf "Size of the tree: %d\n" size;; *)

let is_variable (s : string) : bool =
  if s = "" then false  (* Empty string is not considered a variable *)
  else if (s.[0] >= 'a' && s.[0] <= 'z') then true 
  else false;;


let rec vars (tree: tree) : string list = 
  let rec collect_vars acc = function
    | V s -> if is_variable s then s :: acc else acc
    | C {children} -> List.fold_left collect_vars acc children
  in
  List.sort_uniq compare (collect_vars [] tree);;

  

(* let lst_var = vars tree;; *)

(* Printf.printf "List of variables: ";
List.iter (fun v -> Printf.printf "%s, " v) lst_var;
Printf.printf "\n"; *)

let rec mirror (tree : tree) : tree = 
  match tree with
  | V v -> V v 
  | C {node ; children } -> C {node; children = List.rev(List.map mirror children) }

(* let new_tree = mirror tree;; *)



type substitution = (string*tree) list;;

let check_subst (subst_x : substitution) : bool = 
  let rec check_duplicates = function
    | [] -> false
    | (x, _):: xs -> List.exists (fun (s, _) -> s = x) xs || check_duplicates xs
  in
  not (check_duplicates subst_x);;


(* let substitution_1 = [("x", C {node = "a" ; children = []}); ("y", V "b")] *)
(* let substitution_2 = [("x", V "c"); ("z", C {node = "d" ; children = [V "e"]})] *)


let rec subst (tree: tree) (subst_x : substitution) : tree = 
  match tree with
  | V v -> 
    (match find_subst_tree v subst_x with
    | Some t -> t
    | None -> V v)
  | C {node; children} -> 
    C {node; children = List.map (fun child -> subst child subst_x) children }

and find_subst_tree (var: string) (subst_x: substitution) : tree option =
    match subst_x with
    | [] -> None
    | (v,t) :: rest -> 
        if v = var then Some t
        else find_subst_tree var rest;;

(* let subs_tree = find_subst_tree "x" substitution_1;; *)
(* let subs_new_tree = subst tree substitution_1;; *)

let compose_subst (s1 : substitution) (s2 : substitution) : substitution =
  let apply_subst_single subst_x (var, tree) =
    match List.assoc_opt var s2 with
    | Some t -> (var, t)
    | None -> (var, subst tree s2)
  in
    let apply_s1 = List.map (apply_subst_single s2) s1 in
      let s2_not_in_s1 = List.filter (fun (var, _) -> not (List.mem_assoc var s1)) s2 in
        apply_s1 @ s2_not_in_s1
;;


(* let comp_subs1  = compose_subst substitution_1 substitution_2;; *)
(* let comp_subs2  = compose_subst substitution_2 substitution_1;; *)

let rec occurs (x : string) (t : tree) : bool =
  match t with
  | V y -> x = y
  | C { node = _; children = s } -> List.exists (occurs x) s

exception NOT_UNIFIABLE;;

let rec mgu (s : tree) (t : tree) : substitution =
  match (s, t) with
  | (V x, V y) -> if x = y then [] else [(x, t)]
  | (C { node = f; children = sc }, C { node = g; children = tc }) ->
      if f = g && List.length sc = List.length tc
      then my_unify (List.combine sc tc)
      else raise NOT_UNIFIABLE
  | ((V x, t) | (t, V x)) ->
      if occurs x t
      then raise NOT_UNIFIABLE
      else [(x, t)]

and my_unify (s : (tree * tree) list) : substitution =
  match s with
  | [] -> []
  | (x, y) :: t ->
      let t2 = my_unify t in
      let t1 = mgu (subst x t2) (subst y t2) in
      t1 @ t2
  ;;



(* let un_mirr = mgu (mirror tree1) (mirror tree2) ;;  *)


(*  
let tree1 = V "x";;
let tree2 = V "x";; 

let tree3 = V "x";;
let tree4 = V "y";;

let tree5 = V "x";;
let tree6 = C { node = "f"; children = [V "y"; V "z"] };;

let tree7 = C { node = "f"; children = [V "y"; V "z"] };;
let tree8 = V "x";;

let tree9 = C { node = "f"; children = [V "x"; V "y"] };;
let tree10 = C { node = "f"; children = [V "x"; V "y"] };;

let tree11 = C { node = "f"; children = [V "x"; V "y"] };;
let tree12 = C { node = "f"; children = [V "y"; V "z"] };;

let tree13 = C { node = "f"; children = [C { node = "g"; children = [V "x"] }; V "y"] };;
let tree14 = C { node = "f"; children = [C { node = "g"; children = [V "z"] }; V "y"] };;

let tree15 = C { node = "h"; children = [V "x"] };;
let tree16 = C { node = "g"; children = [V "y"] };;

let tree17 = C { node = "h"; children = [V "z"] };;
let tree18 = C { node = "i"; children = [tree15; tree16] };;
let tree19 = C { node = "j"; children = [tree15; tree16; tree17] };;
let tree20 = C { node = "i"; children = [tree17; tree16] };;
let tree21 = C { node = "l"; children = [tree18; tree19] };;


let un_ = mgu tree20 tree18 ;; 
 *)
