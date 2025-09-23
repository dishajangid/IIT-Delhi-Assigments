%% file: counter_server.erl
-module(counter_server).
-export([start/0, inc/0, get/0]).
-export([loop/1]).

start() ->
    Pid = spawn(?MODULE, loop, [0]),
    register(counter, Pid),
    ok.

%% client APIs (send and wait for reply)
inc() ->
    counter ! {inc, self()},
    receive {ok, New} -> New end.

get() ->
    counter ! {get, self()},
    receive {ok, Val} -> Val end.

%% the server owns the state
loop(N) ->
    receive
        {inc, From} ->
            New = N + 1,
            From ! {ok, New},
            loop(New);
        {get, From} ->
            From ! {ok, N},
            loop(N)
    end.

