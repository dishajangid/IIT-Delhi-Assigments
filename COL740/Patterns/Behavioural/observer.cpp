#include <iostream>
#include <string>
#include <vector>

// Forward declare
class Observer;

// Subject
class Subject {
    std::vector<Observer*> observers_;
public:
    void attach(Observer* o) { observers_.push_back(o); }
    void notify(const std::string& msg);
};

// Observer
class Observer {
public:
    virtual ~Observer() = default;
    virtual void update(const std::string& msg) = 0;
};

// Concrete Observer
class User : public Observer {
    std::string name_;
public:
    explicit User(std::string n) : name_(std::move(n)) {}
    void update(const std::string& msg) override {
        std::cout << name_ << " received: " << msg << "\n";
    }
};

// Subject method definition (after Observer is declared)
void Subject::notify(const std::string& msg) {
    for (auto* o : observers_) o->update(msg);
}

// Client
int main() {
    Subject newsAgency;

    User alice("Alice");
    User bob("Bob");

    newsAgency.attach(&alice);
    newsAgency.attach(&bob);

    newsAgency.notify("Breaking News: Observer Pattern Works!");
}

