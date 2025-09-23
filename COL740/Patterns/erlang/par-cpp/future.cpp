#include <iostream>
#include <future>
#include <thread>

int main() {
    // Launch async task
    std::future<int> f = std::async(std::launch::async, [] {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        return 42;
    });

    // Do something else
    std::cout << "Doing work in main...\n";

    // Wait and get result
    int result = f.get();
    std::cout << "Result: " << result << "\n";
}
