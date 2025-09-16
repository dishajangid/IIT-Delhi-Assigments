#include <iostream>
#include <vector>

// Forward declarations
class Circle;
class Square;

// Visitor interface
class Visitor {
public:
    virtual ~Visitor() = default;
    virtual void visit(Circle& c) = 0;
    virtual void visit(Square& s) = 0;
};

// Element interface
class Shape {
public:
    virtual ~Shape() = default;
    virtual void accept(Visitor& v) = 0;
};

// Concrete Elements
class Circle : public Shape {
public:
    void accept(Visitor& v) override { v.visit(*this); }
};

class Square : public Shape {
public:
    void accept(Visitor& v) override { v.visit(*this); }
};

// Concrete Visitor
class DrawVisitor : public Visitor {
public:
    void visit(Circle&) override { std::cout << "Drawing Circle\n"; }
    void visit(Square&) override { std::cout << "Drawing Square\n"; }
};

class AreaVisitor : public Visitor {
public:
    void visit(Circle&) override { std::cout << "Circle area = πr^2 (demo)\n"; }
    void visit(Square&) override { std::cout << "Square area = side^2 (demo)\n"; }
};

// Client
int main() {
    std::vector<Shape*> shapes = { new Circle(), new Square() };

    DrawVisitor draw;
    AreaVisitor area;

    std::cout << "--- Drawing ---\n";
    for (auto* s : shapes) s->accept(draw);

    std::cout << "--- Area Calculation ---\n";
    for (auto* s : shapes) s->accept(area);

    for (auto* s : shapes) delete s;
}

