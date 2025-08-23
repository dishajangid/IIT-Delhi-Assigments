(* ast.ml *)

type variable = string
type constant = string
type function_symbol = string
(* type predicate_symbol = string *)
(* type keyword = string *)
type identifier = string
type keyword = string




type term =
  | Var of variable
  | Const of constant
  | Int of int
  | Func of function_symbol * (term list)
  | Empty 
  | Key_word of keyword
  | Underscore of char
  | Identifier of identifier
  | Non_Empty 
  | MoreTerm of term * term 
  (* | Equal of char
  | Semicolon of char *)
  
type term_list = Terms of term list

type atomic_formula = 
      Atm_form of function_symbol * (term list)
      | Underscore of char
      | Cut of char
      | Key_word of keyword
      | Var of variable
      | Identifier of identifier
      | Non_Empty
      (* | Equal of char
      | Semicolon of char *)

(* type atomic_formula_list = Atm_formulas of atomic_formula list *)

type fact = Fact of atomic_formula

type rule = 
    Rule of atomic_formula * (atomic_formula list)
  (* | Truth of atomic_formula * char *)

type clause =
  | Fact of fact
  | Rule of rule
  (* | Truth of rule *)

type goal = atomic_formula list

type clause_list = Clause_list of clause list
  | Prog of (clause list) * goal
