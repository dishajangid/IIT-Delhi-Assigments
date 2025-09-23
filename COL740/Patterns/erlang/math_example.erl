%% File: math_example.erl
%% To compile: c(math_example).
%% To run: math_example:start().

-module(math_example).
-export([start/0, add/2, compute/0]).

%% A helper function that performs addition
add(A, B) ->
    A + B.

%% Another function that calls add/2
compute() ->
    X = 5,
    Y = 7,
    Result = add(X, Y),
    io:format("The sum of ~p and ~p is ~p~n", [X, Y, Result]),
    Result.

%% Entry point
start() ->
    compute().

