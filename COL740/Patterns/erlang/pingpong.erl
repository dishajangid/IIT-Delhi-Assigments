-module(pingpong).
-export([start/0, ping/1, pong/0]).

start() ->
    % Spawn the pong process
    PongPid = spawn(pingpong, pong, []),
    % Spawn the ping process and tell it who to talk to
    spawn(pingpong, ping, [PongPid]).

ping(PongPid) ->
    % Send first message
    PongPid ! {ping, self()},
    % Enter loop to keep talking
    ping_loop(PongPid).

ping_loop(PongPid) ->
    receive
        {pong, PongPid} ->
            io:format("Ping received pong~n"),
            timer:sleep(1000), % small delay
            PongPid ! {ping, self()},
            ping_loop(PongPid)
    end.

pong() ->
    receive
        {ping, From} ->
            io:format("Pong received ping~n"),
            From ! {pong, self()},
            pong()  % stay alive and keep responding
    end.

