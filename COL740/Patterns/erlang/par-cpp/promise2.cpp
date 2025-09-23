#include <iostream>
#include <future>
#include <thread>
#include <chrono>
#include <vector>
#include <stdexcept>

// Worker: computes a sum; fails if it sees a negative number.
void compute_sum(std::promise<int> p, std::vector<int> data) {
    try {
        int sum = 0;
        for (int x : data) {
            if (x < 0) throw std::runtime_error("negative input encountered");
            sum += x;
            std::this_thread::sleep_for(std::chrono::milliseconds(80)); // simulate work
        }
        p.set_value(sum); // success path
    } catch (...) {
        p.set_exception(std::current_exception()); // propagate the error
    }
}

int main() {
    // -------- Task A (success) --------
    std::promise<int> p1;
    std::future<int> f1 = p1.get_future();
    std::thread t1(compute_sum, std::move(p1), std::vector<int>{1,2,3,4,5});

    // -------- Task B (will fail) --------
    std::promise<int> p2;
    std::future<int> f2 = p2.get_future();
    std::thread t2(compute_sum, std::move(p2), std::vector<int>{10, -1, 20}); // <- negative triggers error

    // Main thread can do other work here...
    std::cout << "Main is doing other work...\n";

    // Consume results (or errors)
    try {
        std::cout << "Sum A = " << f1.get() << "\n";
    } catch (const std::exception& e) {
        std::cout << "Task A failed: " << e.what() << "\n";
    }

    try {
        std::cout << "Sum B = " << f2.get() << "\n";
    } catch (const std::exception& e) {
        std::cout << "Task B failed: " << e.what() << "\n";
    }

    t1.join();
    t2.join();
}

