run:
	ocamllex lexer.mll
	ocamlyacc -v parser.mly
	export OCAMLRUNPARAM='p'
	ocamlc -c ast.ml
	ocamlc -o a ast.ml parser.mli parser.ml lexer.ml main.ml
