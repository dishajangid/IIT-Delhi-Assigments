#include <iostream>

// Abstract Class
class Game {
public:
    virtual ~Game() = default;

    // Template method defines the skeleton of the algorithm
    void play() {
        start();
        playTurn();
        end();
    }

protected:
    virtual void start() = 0;
    virtual void playTurn() = 0;
    virtual void end() = 0;
};

// Concrete Class 1
class Chess : public Game {
protected:
    void start() override { std::cout << "Chess started.\n"; }
    void playTurn() override { std::cout << "Chess move played.\n"; }
    void end() override { std::cout << "Chess ended.\n"; }
};

// Concrete Class 2
class Soccer : public Game {
protected:
    void start() override { std::cout << "Soccer match kicked off.\n"; }
    void playTurn() override { std::cout << "Goal attempt!\n"; }
    void end() override { std::cout << "Soccer match finished.\n"; }
};

// Client
int main() {
    Chess chess;
    Soccer soccer;

    std::cout << "--- Playing Chess ---\n";
    chess.play();

    std::cout << "--- Playing Soccer ---\n";
    soccer.play();
}

