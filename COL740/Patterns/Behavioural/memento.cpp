#include <iostream>
#include <string>
#include <vector>

// Memento
class Memento {
    std::string state;
public:
    explicit Memento(std::string s) : state(std::move(s)) {}
    std::string getState() const { return state; }
};

// Originator
class Editor {
    std::string text;
public:
    void type(const std::string& words) { text += words; }
    Memento save() const { return Memento(text); }
    void restore(const Memento& m) { text = m.getState(); }
    void show() const { std::cout << text << "\n"; }
};

// Caretaker
class History {
    std::vector<Memento> snapshots;
public:
    void push(const Memento& m) { snapshots.push_back(m); }
    Memento pop() {
        Memento m = snapshots.back();
        snapshots.pop_back();
        return m;
    }
};

// Client
int main() {
    Editor editor;
    History history;

    editor.type("Hello ");
    history.push(editor.save());  // save #1

    editor.type("World!");
    history.push(editor.save());  // save #2

    editor.show(); // Hello World!

    editor.restore(history.pop()); // undo to last save
    editor.show(); // Hello World!

    editor.restore(history.pop()); // undo again
    editor.show(); // Hello 
}

