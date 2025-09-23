-module(conc_hello2).
-export([start/0, greeter/0]).

start() ->
    % Spawn three greeter processes
    Pids = [spawn(conc_hello2, greeter, []) || _ <- lists:seq(1,5)],

    % Send each greeter a hello message
    lists:foreach(fun(Pid) ->
        Pid ! {self(), "Hello, World!"}
    end, Pids),

    % Collect all replies
    gather_replies(Pids).

gather_replies([]) -> ok;
gather_replies([Pid | Rest]) ->

    receive
        {Pid, Reply} ->
            io:format("Got reply from ~p: ~s~n", [Pid, Reply])
    end,
    gather_replies(Rest).

greeter() ->
    receive
        {From, Msg} ->
            io:format("Greeter (~p) received: ~s~n", [self(), Msg]),
            From ! {self(), "Hi there!"}
    end.

