-module(conc_hello).
-export([start/0, greeter/0]).

start() ->
    % Spawn a new process running the greeter function
    Pid = spawn(conc_hello, greeter, []),
    % Send a message to that process
    Pid ! {self(), "Hello, World!"},
    % Wait to receive the reply
    receive
        {Pid, Reply} ->
            io:format("Got reply: ~s~n", [Reply])
    end.

greeter() ->
    receive
        {From, Msg} ->
            io:format("Greeting received: ~s~n", [Msg]),
            % Send a reply back
            From ! {self(), "Hi there!"}
    end.

