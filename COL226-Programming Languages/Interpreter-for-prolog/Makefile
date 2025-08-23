run:
	ocamlc -c interpreter.ml
	ocamlyacc -v parser.mly
	ocamlc -c parser.mli
	ocamlc -c parser.ml
	ocamllex lexer.mll
	ocamlc -c lexer.ml
	ocamlc -c main.ml
	ocamlfind ocamlc -linkpkg -package unix -o a interpreter.cmo parser.cmo lexer.cmo main.cmo  



# Clean up the generated files (executables and object files)
clean:
	rm -f *.cmo *.cmi parser.ml parser.mli lexer.ml *.output
