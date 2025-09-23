-module(prodcons).
-export([start/0, producer/2, consumer/0]).

start() ->
    Consumer = spawn(prodcons, consumer, []),
    spawn(prodcons, producer, [Consumer, 1]),
    spawn(prodcons, producer, [Consumer, 2]).

producer(Consumer, Id) ->
    lists:foreach(
      fun(N) ->
          Msg = io_lib:format("Item ~p from producer ~p", [N, Id]),
          Consumer ! {self(), Msg},
          timer:sleep(500)
      end,
      lists:seq(1,5)).

consumer() ->
    receive
        {From, Msg} ->
            io:format("Consumer got: ~s~n", [Msg]),
            From ! ack,
            consumer()
    end.

