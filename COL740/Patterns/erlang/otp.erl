-module(otp).
-behaviour(gen_server).

%% API
-export([start_link/0, inc/0, get/0]).

%% gen_server callbacks
-export([init/1, handle_call/3, handle_cast/2]).

%%%% API FUNCTIONS %%%%

start_link() ->
    %% Start the gen_server, register it under the module name
    gen_server:start_link({local, ?MODULE}, ?MODULE, 0, []).

inc() ->
    gen_server:cast(?MODULE, inc).

get() ->
    gen_server:call(?MODULE, get).

%%%% CALLBACKS %%%%

init(Initial) ->
    {ok, Initial}.   %% initial state = 0

handle_call(get, _From, State) ->
    {reply, State, State}.   %% reply with current count

handle_cast(inc, State) ->
    {noreply, State + 1}.    %% increment state

