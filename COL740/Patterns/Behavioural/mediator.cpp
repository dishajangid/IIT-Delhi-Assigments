#include <iostream>
#include <string>
#include <vector>

// Forward declare
class Mediator;

// Colleague
class Colleague {
protected:
    Mediator* mediator_;
    std::string name_;
public:
    Colleague(Mediator* m, std::string name) : mediator_(m), name_(std::move(name)) {}
    virtual ~Colleague() = default;
    virtual void send(const std::string& msg) = 0;
    virtual void receive(const std::string& msg, const std::string& from) = 0;
    const std::string& name() const { return name_; }
};

// Mediator
class Mediator {
public:
    virtual ~Mediator() = default;
    virtual void add(Colleague* c) = 0;
    virtual void relay(const std::string& msg, Colleague* sender) = 0;
};

// Concrete Mediator
class ChatMediator : public Mediator {
    std::vector<Colleague*> members_;
public:
    void add(Colleague* c) override { members_.push_back(c); }
    void relay(const std::string& msg, Colleague* sender) override {
        for (auto* m : members_) if (m != sender) m->receive(msg, sender->name());
    }
};

// Concrete Colleague
class User : public Colleague {
public:
    using Colleague::Colleague;
    void send(const std::string& msg) override { mediator_->relay(msg, this); }
    void receive(const std::string& msg, const std::string& from) override {
        std::cout << name_ << " got from " << from << ": " << msg << "\n";
    }
};

// Client
int main() {
    ChatMediator chat;
    User alice(&chat, "Alice");
    User bob(&chat, "Bob");
    User charlie(&chat, "Charlie");

    chat.add(&alice);
    chat.add(&bob);
    chat.add(&charlie);

    alice.send("Hello!");
    bob.send("Hi Alice!");
}

