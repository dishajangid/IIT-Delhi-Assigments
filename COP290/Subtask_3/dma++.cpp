#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <iomanip>
#include <ctime>
#include <algorithm>
#include <cmath>

using namespace std;

struct StockData {
    std::string date;
    std::string direction;  // Buy, Sell, Hold
    int quantity;
    double price;
};

double calculateER(const std::vector<double>& prices, int n) {
    double sumAbsChange = 0.0;
    double priceChange = prices[n] - prices[0];
    for (int i = 1; i < n; ++i) {
        sumAbsChange += std::abs(prices[i] - prices[i - 1]);
    }
    if (sumAbsChange == 0) return 0.0;
    return std::abs(priceChange) / sumAbsChange;
}

double calculateSF(double SF_prev, double ER, double c1, double c2) {
    double numerator = 2 * ER;
    double denominator = 1 + c2;
    double SF = SF_prev + c1 * (numerator / denominator - 1);
    return SF;
}

int main(int argc, char* argv[]) {
    if (argc != 11) {
        std::cerr << "Usage: " << argv[0] << " strategy symbol n x p max_hold_days c1 c2 start_date end_date" << std::endl;
        return 1;
    }

    std::string strategy = argv[1];
    std::string symbol = argv[2];
    int n = std::stoi(argv[3]);
    int x = std::stoi(argv[4]);
    float p = std::stof(argv[5]);
    int max_hold_days = std::stoi(argv[6]);
    double c1 = std::stod(argv[7]);
    double c2 = std::stod(argv[8]);
    std::string start_date = argv[9];
    std::string end_date = argv[10];

    std::ifstream input_file("archive/" + symbol + ".txt");
    if (!input_file.is_open()) {
        std::cerr << "Error: Unable to open input file" << std::endl;
        return 1;
    }

    std::string line;
    std::vector<StockData> stock_data;
    StockData data;
    bool date_set = false;

    while (std::getline(input_file, line)) {
        std::istringstream iss(line);
        std::string key, value;
        iss >> key >> value;

        if (key == "DATE:") {
            // Set the date for the current StockData object
            data.date = value;
            date_set = true;
        }
        else if (key == "CLOSE:") {
            // Set the price for the current StockData object
            data.price = std::stod(value);

            // If both date and price are set, push the StockData object into the vector
            if (date_set) {
                stock_data.push_back(data);
                data = StockData(); // Create a new StockData object for the next pair of DATE and CLOSE lines
                date_set = false; // Reset the flag
            }
        }
    }

    // Handle non-trading days
    std::vector<std::string> trading_days;
    for (const auto& data : stock_data) {
        trading_days.push_back(data.date);
    }

    // Find the last trading day
    std::string last_trading_day = end_date;
    if (std::find(trading_days.begin(), trading_days.end(), last_trading_day) == trading_days.end()) {
        last_trading_day = trading_days.back();
    }

    // Implement DMA++ strategy
    std::vector<int> signals;
    std::vector<double> prices(n + 1, 0.0); // stores prices for calculating ER
    double SF = 0.5; // Smoothing Factor (SF0)
    double AMA_prev = stock_data[0].price; // Adaptive Moving Average (AMA0)
    for (size_t i = 0; i < stock_data.size(); ++i) {
        if (stock_data[i].date >= start_date && stock_data[i].date <= last_trading_day) {
            // Calculate Efficiency Ratio (ER)
            for (int j = 0; j <= n; ++j) {
                prices[j] = stock_data[i + j].price;
            }
            double ER = calculateER(prices, n);

            // Calculate Smoothing Factor (SF)
            SF = calculateSF(SF, ER, c1, c2);

            // Calculate Adaptive Moving Average (AMA)
            double AMA = AMA_prev + SF * (stock_data[i].price - AMA_prev);

            // Determine buy/sell signal
            if ((stock_data[i].price - AMA) >= (p / 100.0) * AMA) {
                signals.push_back(1); // Buy
            } else if ((AMA - stock_data[i].price) >= (p / 100.0) * AMA) {
                signals.push_back(-1); // Sell
            } else {
                signals.push_back(0); // Hold
            }

            AMA_prev = AMA; // Update AMA for next iteration
        }
    }


    std::tm start_date_tm = {};
    std::istringstream start_date_stream(start_date);
    start_date_stream >> std::get_time(&start_date_tm, "%Y-%m-%d");

    // Add one day
    start_date_tm.tm_mday -= 1;
    std::mktime(&start_date_tm);

    // Convert the updated date back to a string
    std::ostringstream start_date_updated_stream;
    start_date_updated_stream << std::put_time(&start_date_tm, "%Y-%m-%d");
    start_date = start_date_updated_stream.str();





    // Filter data for start_date to end_date
    std::vector<StockData> result_data;
    for (const auto& data : stock_data) {
        if (data.date >= start_date && data.date <= last_trading_day) {
            result_data.push_back(data);
        }
    }

    // Calculate daily cashflow
    std::ofstream cashflow_file("daily_cashflow.csv");
    cashflow_file << "Date,Cashflow\n";
    double cashflow = 0.0;
    int a = signals.size();
    int i = a - 1;
    int days_held = 0; // Counter for tracking days held
    while (i >= 0) {
        // Increment days held if position is held
        if (signals[i] != 0) {
            days_held++;
        }
        // Check if max_hold_days is reached and forcefully close position
        if (days_held >= max_hold_days) {
            signals[i] = -signals[i]; // Reverse signal to close position
            days_held = 0; // Reset days held counter
        }
        cashflow += signals[i] * (result_data[i].price);
        cashflow_file << result_data[i].date << "," << -1 * cashflow << "\n";
        i--;
    }
    cashflow_file.close();

    // Write to order statistics.csv
    std::ofstream order_file("order_statistics.csv");
    order_file << "Date,Direction,Quantity,Price\n";
    for (size_t i = 0; i < signals.size(); ++i) {
        if (signals[i] != 0) {
            order_file << result_data[i].date << "," << (signals[i] == 1 ? "BUY" : "SELL") << ",1," << result_data[i+1].price << "\n";
        }
    }
    order_file.close();

    return 0;
}
