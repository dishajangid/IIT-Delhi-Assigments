import pandas as pd
import h5py
from datetime import datetime,date, timedelta
from jugaad_data.nse import stock_df
import time
import os
import sys
import matplotlib.pyplot as plt
import sqlite3
#import msgpack

sym = sys.argv[1]
years = int(sys.argv[2])

#start_date = date(datetime.today().year - years, datetime.today().month, datetime.today().date)
start_date = datetime.today().date() - timedelta(days=365 * years)
today_date = datetime.today().date()

df = stock_df(symbol = sym, from_date = start_date, to_date = today_date, series = "EQ")

# Function to write df to different file formats and benchmark the time and size
def write_and_benchmark_df(df, symbol, file_format):
    start_time = time.time()
    
    if file_format == 'csv':
        df.to_csv(f'{symbol}.csv', index=False)
    elif file_format == 'txt':
        df.to_csv(f'{symbol}.txt', sep='\t', index=False)
    elif file_format == 'pkl':
        df.to_pickle(f'{symbol}.pkl')
    elif file_format == 'parquet':
        df.to_parquet(f'{symbol}.parquet', index=False)
    elif file_format == 'json':
        df.to_json(f'{symbol}.json', orient = 'records', lines = True)
    elif file_format == 'h5':
        df.to_hdf(f'{symbol}.h5', key='data', mode='w')
    elif file_format == 'xlsx':
        df.to_excel(f'{symbol}.xlsx', index=False)
    elif file_format == 'db':
        df.to_sql(f'{symbol}', sqlite3.connect(f'{symbol}.db'), index=False)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    file_size = os.path.getsize(f'{symbol}.{file_format}')
    
    return elapsed_time, file_size

# List of columns to include in the df
columns = ['DATE', 'OPEN', 'CLOSE', 'HIGH', 'LOW', 'LTP', 'VOLUME', 'VALUE', 'NO OF TRADES']


# Benchmark results dictionary
results = {'symbol': [], 'file_format': [], 'time_taken': [], 'file_size': []}

# Loop through each stock symbol


    
    # Select specific columns
df_selected = df[columns]
    
    # Loop through each file format
for file_format in ['csv', 'txt', 'parquet','json', 'h5','pkl','xlsx','db']:#,'binary']:
    time_taken, file_size = write_and_benchmark_df(df_selected, sym, file_format)
        
        # Store results in the dictionary
    results['symbol'].append(sym)
    results['file_format'].append(file_format)
    results['time_taken'].append(time_taken)
    results['file_size'].append(file_size)

# Create a dfFrame from the results dictionary
results_df = pd.DataFrame(results)

# Plotting the comparison graph
plt.figure(figsize=(10, 6))
for file_format in results_df['file_format'].unique():
    format_df = results_df[results_df['file_format'] == file_format]
    plt.scatter(format_df['file_size'], format_df['time_taken'], label=file_format)

plt.title('File Size vs Time Taken for Different File Formats')
plt.xlabel('File Size (Bytes)')
plt.ylabel('Time Taken (Seconds)')
plt.legend()
plt.show()

