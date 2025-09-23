%% File: crash_demo.erl
-module(crash_demo).
-export([start/0, crash/0, worker/0, init/0]).

start() ->
    spawn(?MODULE, init, []).

init() ->
    process_flag(trap_exit, true),
    %% Start worker and link to it
    Worker = spawn_link(?MODULE, worker, []),
    register(worker, Worker),
    monitor_loop().

monitor_loop() ->
    receive
        {'EXIT', _From, _Reason} ->
            io:format("Worker crashed, restarting...~n"),
            New = spawn_link(?MODULE, worker, []),
            register(worker, New),
            monitor_loop();
        _ ->
            monitor_loop()
    end.

worker() ->
    io:format("Worker started.~n"),
    receive
        crash ->
            exit(bad_thing);
        Msg ->
            io:format("Worker got: ~p~n", [Msg]),
            worker()
    end.

crash() ->
    worker ! crash.
