#include <iostream>
#include <future>
#include <thread>

int main() {
    std::promise<int> p;
    std::future<int> f = p.get_future();

    std::thread([&p] {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        p.set_value(42);
    }).detach();

    std::cout << "Doing work in main...\n";
    std::cout << "Result: " << f.get() << "\n";
}

