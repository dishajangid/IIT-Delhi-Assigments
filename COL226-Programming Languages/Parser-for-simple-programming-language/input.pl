mem(_,[]):-fail.
mem(X,[X|_]):-!.
mem(X,[_|R]):-mem(X,R).


hastype(_, N, intT):- integer(N).
hastype(_, T, boolT):- T = true; T = false.
hastype(_, N, ListT) :- N = [_|_], hastype(_, N, ListT).



hastype(G, X, T):- mem((X,T), G).


hastype(G, add(E1,E2), intT):- hastype(G, E1, intT), hastype(G, E2, intT) .
hastype(G, sub(E1,E2), intT):- hastype(G, E1, intT), hastype(G, E2, intT) .
hastype(G, mul(E1,E2), intT):- hastype(G, E1, intT), hastype(G, E2, intT) .
hastype(G, div(E1,E2), floatT):- hastype(G, E1, intT), hastype(G, E2, intT).
hastype(G, div(E1,E2), floatT):- hastype(G, E1, floatT), hastype(G, E2, floatT).
hastype(G, mod(E1,E2), intT):- hastype(G, E1, intT), hastype(G, E2, intT).
hastype(G, if(Cond, Then, Else), T):- hastype(G, Cond, boolT), hastype(G, Then, T), hastype(G, Else, T).
hastype(G, if(true, Then, Else), T):-  hastype(G, Then, T).
hastype(G, if(false, Then, Else), T):-  hastype(G, Else, T).


hastype(G, greater(E1,E2), boolT):- hastype(G, E1, intT), hastype(G, E2, intT) .
hastype(G, smaller(E1,E2), boolT):- hastype(G, E1, intT), hastype(G, E2, intT) .
hastype(G, and(E1,E2), boolT):- hastype(G, E1, boolT), hastype(G, E2, boolT).
hastype(G, or(E1,E2), boolT):- hastype(G, E1, boolT), hastype(G, E2, boolT) .
hastype(G, equal(E1,E2), boolT):- hastype(G, E1, intT), hastype(G, E2, intT) .
hastype(G, not(E1), boolT):- hastype(G, E1, boolT).


