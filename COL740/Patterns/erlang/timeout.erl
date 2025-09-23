-module(timeout).
-export([start/0, server/0]).

start() ->
    Pid = spawn(?MODULE, server, []),
    %% ask server to reply
    Pid ! {self(), hello},
    %% wait up to 1000 ms
    receive
        {Pid, Reply} ->
            io:format("Got reply: ~p~n", [Reply])
    after 1000 ->
        io:format("No reply within 1 second!~n")
    end.

server() ->
    receive
        {From, Msg} ->
            %% pretend to be slow
            timer:sleep(1500),
            From ! {self(), {ok, Msg}},
            server()
    end.

