-module(future).
-export([async/1, await/2, start/0]).

async(Fun) when is_function(Fun, 0) ->
    Ref    = make_ref(),
    Parent = self(),
    Pid = spawn(fun() ->
        Result = Fun(),
        Parent ! {Ref, Result}
    end),
    {future, Ref, Pid}.

await({future, Ref, _Pid}, Timeout) ->
    receive
        {Ref, Value} -> {ok, Value}
    after Timeout ->
        timeout
    end.

start() ->
    F = async(fun() -> timer:sleep(200), 42 end),
    io:format("await 100 ms -> ~p~n", [await(F, 100)]),
    io:format("await 500 ms -> ~p~n", [await(F, 500)]).
