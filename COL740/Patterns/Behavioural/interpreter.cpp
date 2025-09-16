#include <iostream>
#include <memory>
#include <string>

// Abstract Expression
class Expression {
public:
    virtual ~Expression() = default;
    virtual bool interpret(const std::string& context) const = 0;
};

// Terminal Expression
class TerminalExpression : public Expression {
    std::string token_;
public:
    explicit TerminalExpression(std::string token) : token_(std::move(token)) {}
    bool interpret(const std::string& context) const override {
        return context.find(token_) != std::string::npos;
    }
};

// Nonterminal: Or
class OrExpression : public Expression {
    std::unique_ptr<Expression> left_, right_;
public:
    OrExpression(std::unique_ptr<Expression> l, std::unique_ptr<Expression> r)
        : left_(std::move(l)), right_(std::move(r)) {}
    bool interpret(const std::string& context) const override {
        return left_->interpret(context) || right_->interpret(context);
    }
};

// Nonterminal: And
class AndExpression : public Expression {
    std::unique_ptr<Expression> left_, right_;
public:
    AndExpression(std::unique_ptr<Expression> l, std::unique_ptr<Expression> r)
        : left_(std::move(l)), right_(std::move(r)) {}
    bool interpret(const std::string& context) const override {
        return left_->interpret(context) && right_->interpret(context);
    }
};

// Client
int main() {
    // Grammar: isFriend := "Alice" OR "Bob"
    auto isFriend = std::make_unique<OrExpression>(
        std::make_unique<TerminalExpression>("Alice"),
        std::make_unique<TerminalExpression>("Bob")
    );

    // Grammar: isCouple := "Charlie" AND "Diana"
    auto isCouple = std::make_unique<AndExpression>(
        std::make_unique<TerminalExpression>("Charlie"),
        std::make_unique<TerminalExpression>("Diana")
    );

    std::string s1 = "Alice and Bob are here";
    std::string s2 = "Charlie loves Diana";
    std::string s3 = "Charlie and Alice";

    std::cout << std::boolalpha << s1 << " -> isFriend? " << isFriend->interpret(s1) << "\n";
    std::cout << std::boolalpha << s2 << " -> isCouple? " << isCouple->interpret(s2) << "\n";
    std::cout << std::boolalpha << s3 << " -> isCouple? " << isCouple->interpret(s3) << "\n";
}

