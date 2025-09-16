#include <iostream>
#include <memory>

// Forward declaration
class State;

// Context
class Context {
    std::unique_ptr<State> state_;
public:
    explicit Context(std::unique_ptr<State> s) : state_(std::move(s)) {}
    void setState(std::unique_ptr<State> s) { state_ = std::move(s); }
    void request(); // delegate to current state
};

// State (abstract)
class State {
public:
    virtual ~State() = default;
    virtual void handle(Context& ctx) = 0;
};

// Concrete States
class OnState : public State {
public:
    void handle(Context& ctx) override;
};

class OffState : public State {
public:
    void handle(Context& ctx) override;
};

// Implement transitions
void OnState::handle(Context& ctx) {
    std::cout << "Light is ON → switching OFF\n";
    ctx.setState(std::make_unique<OffState>());
}

void OffState::handle(Context& ctx) {
    std::cout << "Light is OFF → switching ON\n";
    ctx.setState(std::make_unique<OnState>());
}

// Context::request definition
void Context::request() {
    state_->handle(*this);
}

// Client
int main() {
    Context light(std::make_unique<OffState>());  // start OFF
    light.request();  // OFF → ON
    light.request();  // ON → OFF
    light.request();  // OFF → ON
}

