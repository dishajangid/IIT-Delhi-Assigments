#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <iomanip>
#include <cmath>

struct StockData {
    std::string date;
    double high;
    double low;
    double prev_close;
    double price;
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
  std::string symbol = argv[2];
  int x = std::stoi(argv[3]);
  int n = std::stoi(argv[4]);
  double adx_threshold = std::stod(argv[5]);
  std::string start_date = argv[6];
  std::string end_date = argv[7];

    std::ifstream input_file(symbol + ".txt");
    if (!input_file.is_open()) {
        std::cerr << "Error: Unable to open input file" << std::endl;
        return 1;
    }

    std::string line;
    std::vector<StockData> stock_data;
    StockData data;

    while (std::getline(input_file, line)) {
        std::istringstream iss(line);
        std::string key, value;
        iss >> key >> value;

        if (key == "DATE:") {
            data.date = value;
        } else if (key == "HIGH:") {
            data.high = std::stod(value);
        } else if (key == "LOW:") {
            data.low = std::stod(value);
        } else if (key == "CLOSE:") {
            data.price = std::stod(value);
        }else if (key == "PREV._CLOSE:") {
            data.prev_close = std::stod(value);
            stock_data.push_back(data);
        }
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
    for (const auto& data : stock_data) {
        trading_days.push_back(data.date);
    }

    // Find the last trading day
    std::string last_trading_day = end_date;
    if (std::find(trading_days.begin(), trading_days.end(), last_trading_day) == trading_days.end()) {
        last_trading_day = trading_days.back();
    }

    std::vector<int> signals;
    int z = 0;

    for(z=0; z<stock_data.size(); z++){
      if(stock_data[z].date >= start_date) break;
    }

    //z has index of from where to start loop

    for (size_t i = 0; i < z; ++i) {
          signals.push_back(0);
    }

    // m stores j-num_past_days
    double d1 =0.0;
    double d2 = 0.0;
    double tr = 0.0;
    double e =0.0;
    double al =0.0;
    al = 2/(n+1);
    double di1 =0.0;
    double di2 = 0.0;
    double dx = 0.0;
    double adx = 0.0;
    int f=0;


    for (size_t i = z; i < stock_data.size(); ++i) {

          d1 = 0.0;
          d2 = 0.0;
          tr = std::max(stock_data[i].high - stock_data[i].low, std::max(stock_data[i].high - stock_data[i-1].price, stock_data[i].low - stock_data[i-1].price));
          if(i == z) e = tr;
          else e = (1-al)* e + al*tr;

          d1 += std::max(0.0, stock_data[i].high - stock_data[i - 1].high);

          d2 += std::max(0.0, stock_data[i - 1].low - stock_data[i].low);

          if(i == z) di1 = d1/e;
          else di1 = (1-al)* e + al*(d1/e);

          if(i == z) di2 = d2/e;
          else di2 = (1-al)* e + al*(d2/e);

          if((di1+di2) == 0) dx = 0;

          else dx = (di1 - di2) / (di1 + di2) * 100;

          if(i == z) adx = dx;
          else adx = (1-al)* adx + al*dx;

          // Generate buy/sell signals based on RSI thresholds
          if (adx > adx_threshold && f<x) {
              signals.push_back(1); // Buy
              f+=1;
          } else if (adx < adx_threshold && f>-x) {
              signals.push_back(-1); // Sell
              f-=1;
          } else {
              signals.push_back(0); // Hold
          }

          }


          // Calculate daily cashflow
          std::ofstream cashflow_file("daily_cashflow.csv");
          cashflow_file << "Date,Cashflow\n";
          double cashflow = 0.0;
          int i = z;

          double k, l = 0.0;
          l = stock_data.back().price;

          while(i<signals.size()) {
              cashflow -= signals[i] * (stock_data[i].price );//- result_data[i].price);
              if(i == signals.size() - 1){
                k = cashflow;
              }
              cashflow_file << convertDateFormat(stock_data[i].date) << "," << cashflow << "\n";
              i++;
          }
          cashflow_file.close();

          l = f*l + k;


          // Calculate daily cashflow
          std::ofstream fpl_file("final_pnl.txt");
          fpl_file << l;

          // Write to order statistics.csv
          std::ofstream order_file("order_statistics.csv");
          order_file << "Date,Order_dir,Quantity,Price\n";

          for (size_t i = z; i < signals.size(); ++i) {
              // cout<<signals[i]<<":::";
              if (signals[i] != 0) {
                  order_file << convertDateFormat(stock_data[i].date) << "," << (signals[i] == 1 ? "BUY" : "SELL") << ",1," << stock_data[i].price << "\n";
              }
          }
          order_file.close();

          // Add code to execute Python plotting scripts
          std::string plot_command = "python plots.py";
          int plot_result = std::system(plot_command.c_str());
      return 0;
      }
