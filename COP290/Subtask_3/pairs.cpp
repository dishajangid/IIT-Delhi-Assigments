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
    double price;
    std::string direction;  // Buy, Sell, Hold
    int quantity;
};

// Function to convert date format from yyyy-mm-dd to dd/mm/yyyy
std::string convertDateFormat(const std::string& inputDate) {
    std::tm tm = {};
    std::istringstream ss(inputDate);
    ss >> std::get_time(&tm, "%Y-%m-%d");

    if (ss.fail()) {
        // Handle parsing error if needed
        return inputDate;
    }

    std::ostringstream oss;
    oss << std::put_time(&tm, "%d/%m/%Y");
    return oss.str();
}



int main(int argc, char* argv[]) {

    std::string strategy = argv[1];
    std::string symbol1 = argv[2];
    std::string symbol2 = argv[3];
    int x = std::stoi(argv[4]);
    int n = std::stoi(argv[5]);
    float threshold = std::stof(argv[6]);
    std::string start_date = argv[7];
    std::string end_date = argv[8];

    std::ifstream input_file1(symbol1 + ".txt");
    std::ifstream input_file2(symbol2 + ".txt");

    if (!input_file1.is_open() || !input_file2.is_open()) {
        std::cerr << "Error: Unable to open input file(s)" << std::endl;
        return 1;
    }

    std::string line;
    std::vector<StockData> stock_data1;
    std::vector<StockData> stock_data2;
    StockData data1;
    StockData data2;
    bool date_set=false;

    while (std::getline(input_file1, line)) {
        std::istringstream iss(line);
        std::string key, value;
        iss >> key >> value;

        if (key == "DATE:") {
            // Set the date for the current StockData object
            data1.date = value;
            date_set = true;
        }
        else if (key == "CLOSE:") {
            // Set the price for the current StockData object
            data1.price = std::stod(value);

            // If both date and price are set, push the StockData object into the vector
            if (date_set) {
                stock_data1.push_back(data1);
                data1 = StockData(); // Create a new StockData object for the next pair of DATE and CLOSE lines
                date_set = false; // Reset the flag
            }
        }
      }


    while (std::getline(input_file2, line)) {
      std::istringstream iss(line);
      std::string key, value;
      iss >> key >> value;

      if (key == "DATE:") {
          // Set the date for the current StockData object
          data2.date = value;
          date_set = true;
      }
      else if (key == "CLOSE:") {
          // Set the price for the current StockData object
          data2.price = std::stod(value);

          // If both date and price are set, push the StockData object into the vector
          if (date_set) {
              stock_data2.push_back(data2);
              data2 = StockData(); // Create a new StockData object for the next pair of DATE and CLOSE lines
              date_set = false; // Reset the flag
          }
      }
    }



    if (stock_data1.size() != stock_data2.size()) {
        std::cerr << "Error: Unequal number of data points for the given stock pair." << std::endl;
        return 1;
    }

    // Parse the start date
     std::istringstream start_stream(start_date);
     int start_day, start_month, start_year;
     char discard;
     start_stream >> start_day >> discard >> start_month >> discard >> start_year;

     // Parse the end date
     std::istringstream end_stream(end_date);
     int end_day, end_month, end_year;
     end_stream >> end_day >> discard >> end_month >> discard >> end_year;

     // Convert to yyyy-mm-dd format
     std::ostringstream start_formatted;
     start_formatted << std::setw(4) << std::setfill('0') << start_year << "-"
                    << std::setw(2) << std::setfill('0') << start_month << "-"
                    << std::setw(2) << std::setfill('0') << start_day;
     start_date = start_formatted.str();

     std::ostringstream end_formatted;
     end_formatted << std::setw(4) << std::setfill('0') << end_year << "-"
                  << std::setw(2) << std::setfill('0') << end_month << "-"
                  << std::setw(2) << std::setfill('0') << end_day;
     end_date = end_formatted.str();


    // Handle non-trading days
    std::vector<std::string> trading_days;
    for (const auto& data : stock_data1) {
        trading_days.push_back(data.date);
    }

    // Find the last trading day
    std::string last_trading_day = end_date;
    if (std::find(trading_days.begin(), trading_days.end(), last_trading_day) == trading_days.end()) {
        last_trading_day = trading_days.back();
    }

    int f1=0;
    int f2=0;


    std::vector<int> signals1;
    std::vector<int> signals2;
    int z = 0;

    for(z=0; z<stock_data1.size(); z++){
      if(stock_data1[z].date >= start_date) break;
    }

    // z has index of from where to start loop

    for (size_t i = 0; i < z; ++i) {
          signals1.push_back(0);
          signals2.push_back(0);
    }

    double spread = 0.0;
    double rm = 0.0;
    double rsd = 0.0;
    double zs = 0.0;


    for (size_t i = z; i <stock_data1.size() ; ++i) {

        spread = stock_data1[i].price - stock_data1[i].price;
        rm = 0.0;
        rsd = 0.0;
        zs = 0.0;

        for (int j = 1; j <= n; ++j) {

            rm += stock_data1[i-j].price - stock_data2[i-j].price;

        }

        rm/=n;
        cout<< rm<<endl;

        for(int j = 1; j <= n; ++j) {

            rsd += ((stock_data1[i-j].price - stock_data2[i-j].price) - rm) * ((stock_data1[i-j].price - stock_data2[i-j].price) - rm);
            cout<< rsd <<" ";
        }

        rsd/=n;
        rsd = std:: sqrt(rsd);
        zs = (spread - rm) / rsd;

        if (zs > threshold && f2<x) {
            signals1.push_back(-1);
            signals2.push_back(1);  // Buy
            f2+=1;
            f1-=1;

        } else if (zs < -threshold && f2>-x) {
          signals2.push_back(-1);
          signals1.push_back(1);  // Buy
          f1+=1;
          f2-=1;

        } else if(zs > -threshold && zs < threshold) {
            signals1.push_back(0);  // Hold
            signals2.push_back(0);  // Hold
        }


      }


      // Calculate daily cashflow
      std::ofstream cashflow_file("daily_cashflow.csv");
      cashflow_file << "Date,Cashflow\n";
      double cashflow = 0.0;
      int i = z;

      double k, l = 0.0;
      double l1 = 0.0;
      double l2 = 0.0;
      l1 = stock_data1.back().price;
      l2 = stock_data2.back().price;

      while(i<signals1.size()) {
          cashflow -= signals1[i] * (stock_data1[i].price) + signals2[i] * (stock_data2[i].price );//- result_data[i].price);
          if(i == signals1.size() - 1){
            k = cashflow;
          }
          cashflow_file << convertDateFormat(stock_data1[i].date) << "," << cashflow << "\n";
          i++;
      }
      cashflow_file.close();

      l1 = f1*l1;
      l2 = f2*l2;
      l = l1+l2+k;


      // Calculate daily cashflow
      std::ofstream fpl_file("final_pnl.txt");
      fpl_file << l;

      // Write to order statistics.csv
      std::ofstream order_file1("order_statistics_1.csv");
      order_file1 << "Date,Order_dir,Quantity,Price\n";

      for (size_t i = z; i < signals1.size(); ++i) {
          // cout<<signals[i]<<":::";
          if (signals1[i] != 0) {
              order_file1 << convertDateFormat(stock_data1[i].date) << "," << (signals1[i] == 1 ? "BUY" : "SELL") << ",1," << stock_data1[i].price << "\n";
          }
      }
      order_file1.close();

      // Write to order statistics.csv
      std::ofstream order_file2("order_statistics_2.csv");
      order_file2 << "Date,Order_dir,Quantity,Price\n";

      for (size_t i = z; i < signals2.size(); ++i) {
          // cout<<signals[i]<<":::";
          if (signals2[i] != 0) {
              order_file2 << convertDateFormat(stock_data2[i].date) << "," << (signals2[i] == 1 ? "BUY" : "SELL") << ",1," << stock_data2[i].price << "\n";
          }
      }
      order_file2.close();

      // Add code to execute Python plotting scripts
      std::string plot_command = "python plots.py";
      int plot_result = std::system(plot_command.c_str());
  return 0;
  }
