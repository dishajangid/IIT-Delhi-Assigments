#include <iostream>

// Handler
class Handler {
protected:
    Handler* next_ = nullptr;
public:
    virtual ~Handler() = default;
    void setNext(Handler* n) { next_ = n; }
    virtual bool handle(int severity) {
        return next_ ? next_->handle(severity) : false; // default: pass along
    }
};

// Concrete Handlers
class SimpleHandler : public Handler {
public:
    bool handle(int severity) override {
        if (severity <= 1) { std::cout << "Handled by SimpleHandler\n"; return true; }
        return Handler::handle(severity);
    }
};

class AdvancedHandler : public Handler {
public:
    bool handle(int severity) override {
        if (severity <= 2) { std::cout << "Handled by AdvancedHandler\n"; return true; }
        return Handler::handle(severity);
    }
};

class ExpertHandler : public Handler {
public:
    bool handle(int severity) override {
        if (severity <= 3) { std::cout << "Handled by ExpertHandler\n"; return true; }
        return Handler::handle(severity);
    }
};

// Client
int main() {
    SimpleHandler  h1;
    AdvancedHandler h2;
    ExpertHandler   h3;

    h1.setNext(&h2);
    h2.setNext(&h3);

    for (int s : {1, 2, 3, 4}) {
        if (!h1.handle(s)) {
            std::cout << "No handler for severity " << s << std::endl << std::endl ;
        }
    }
}

