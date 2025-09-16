#include <iostream>
#include <memory>

// Strategy interface
class Strategy {
public:
    virtual ~Strategy() = default;
    virtual int compute(int a, int b) const = 0;
};

// Concrete Strategies
class Add : public Strategy {
public:
    int compute(int a, int b) const override { return a + b; }
};

class Multiply : public Strategy {
public:
    int compute(int a, int b) const override { return a * b; }
};

// Context
class Calculator {
    std::unique_ptr<Strategy> strat_;
public:
    explicit Calculator(std::unique_ptr<Strategy> s) : strat_(std::move(s)) {}
    void setStrategy(std::unique_ptr<Strategy> s) { strat_ = std::move(s); }
    int run(int a, int b) const { return strat_->compute(a, b); }
};

// Client
int main() {
    Calculator calc(std::make_unique<Add>());
    std::cout << "Add: " << calc.run(3, 4) << "\n";        // 7

    calc.setStrategy(std::make_unique<Multiply>());
    std::cout << "Multiply: " << calc.run(3, 4) << "\n";   // 12
}
