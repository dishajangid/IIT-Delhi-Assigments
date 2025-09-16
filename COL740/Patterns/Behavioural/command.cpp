#include <iostream>
#include <memory>
#include <utility>

// Command
class Command {
public:
    virtual ~Command() = default;
    virtual void execute() = 0;
};

// Receiver
class Light {
public:
    void on()  { std::cout << "Light ON\n"; }
    void off() { std::cout << "Light OFF\n"; }
};

// Concrete Commands
class LightOn : public Command {
    Light& light_;
public:
    explicit LightOn(Light& l) : light_(l) {}
    void execute() override { light_.on(); }
};

class LightOff : public Command {
    Light& light_;
public:
    explicit LightOff(Light& l) : light_(l) {}
    void execute() override { light_.off(); }
};

// Invoker
class Button {
    std::unique_ptr<Command> cmd_;
public:
    explicit Button(std::unique_ptr<Command> c) : cmd_(std::move(c)) {}
    void set(std::unique_ptr<Command> c) { cmd_ = std::move(c); }
    void press() { if (cmd_) cmd_->execute(); }
};

// Client
int main() {
    Light light;

    Button btn(std::make_unique<LightOn>(light)); // bind "on"
    btn.press();                                   // -> Light ON

    btn.set(std::make_unique<LightOff>(light));   // swap to "off"
    btn.press();                                   // -> Light OFF
}

