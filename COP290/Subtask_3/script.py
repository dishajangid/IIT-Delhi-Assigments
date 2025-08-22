import os
import sys
from datetime import datetime
from datetime import timedelta, date
import pandas as pd
from jugaad_data.nse import stock_df

# Check if the correct number of command-line arguments are provided
if len(sys.argv) < 4:
    print("Usage: python script.py <stock_symbol> <start_date> <end_date>")
    sys.exit(1)

import sys

strategy = sys.argv[1]
start_date_str = sys.argv[2]
end_date_str = sys.argv[3]
n= None
stock_symbol1 = None
stock_symbol2 = None
stock_symbol = None

if strategy == "MACD":
    stock_symbol = sys.argv[4]

if strategy == "PAIRS":
    stock_symbol1 = sys.argv[4]
    stock_symbol2 = sys.argv[5]
    n = int(sys.argv[6])

if strategy != "PAIRS" and strategy != "MACD":
    stock_symbol = sys.argv[4]
    n = int(sys.argv[5])

# Convert start and end dates to datetime objects
start_date = datetime.strptime(start_date_str, '%d/%m/%Y').date()
end_date = datetime.strptime(end_date_str, '%d/%m/%Y').date()


if strategy != "MACD":
    start_date=start_date - timedelta(2*n)

if strategy == "PAIRS":
    # Write data for the specified stock into a text file in the current directory
    df1 = stock_df(symbol=stock_symbol1, from_date=start_date, to_date=end_date)

    # Replace spaces in column names with underscores
    df1.columns = df1.columns.str.replace(' ', '_')

    # Change the file extension to .txt and write the data in the specified format
    filename1 = f'{stock_symbol1}.txt'
    with open(filename1, 'w') as txt_file:
        for index in reversed(df1.index):  # Reverse the order of the index (dates)
            for key, value in df1.loc[index].items():
                txt_file.write(f"{key}: {value}\n")
            txt_file.write("\n")

    # Write data for the specified stock into a text file in the current directory
    df2= stock_df(symbol=stock_symbol2, from_date=start_date, to_date=end_date)

    # Replace spaces in column names with underscores
    df2.columns = df2.columns.str.replace(' ', '_')

    # Change the file extension to .txt and write the data in the specified format
    filename = f'{stock_symbol2}.txt'
    with open(filename, 'w') as txt_file:
        for index in reversed(df2.index):  # Reverse the order of the index (dates)
            for key, value in df2.loc[index].items():
                txt_file.write(f"{key}: {value}\n")
            txt_file.write("\n")

else:
    # Write data for the specified stock into a text file in the current directory
    df = stock_df(symbol=stock_symbol, from_date=start_date, to_date=end_date)

    # Replace spaces in column names with underscores
    df.columns = df.columns.str.replace(' ', '_')

    # Change the file extension to .txt and write the data in the specified format
    filename = f'{stock_symbol}.txt'
    with open(filename, 'w') as txt_file:
        for index in reversed(df.index):  # Reverse the order of the index (dates)
            for key, value in df.loc[index].items():
                txt_file.write(f"{key}: {value}\n")
            txt_file.write("\n")

print(f"Jugaad data for {stock_symbol} from {start_date_str} to {end_date_str} has been written to a text file in the current directory.")
