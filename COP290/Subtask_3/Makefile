# Makefile for fetching stock data using script.py and running basic.cpp or dma.cpp

# Define the targets
all: install fetch_data run_analysis

# Define the Python interpreter
PYTHON = python3

# Target to install dependencies from requirements.txt
install:
	$(PYTHON) -m pip install -r requirements.txt

# Target to fetch stock data using script.py
fetch_data:
	$(PYTHON) script.py $(strategy) $(start_date) $(end_date) $(symbol1) $(symbol2) $(symbol) $(n) 

# Target to run analysis based on the selected strategy
run_analysis: fetch_data
ifeq ($(strategy), BASIC)
	g++ basic.cpp -o stock_analysis -std=c++11
	./stock_analysis $(strategy) $(symbol) $(n) $(x) $(start_date) $(end_date)
else ifeq ($(strategy), DMA)
	g++ dma.cpp -o stock_analysis -std=c++11
	./stock_analysis $(strategy) $(symbol) $(n) $(x) $(p) $(start_date) $(end_date)
else ifeq ($(strategy), MACD)
	g++ macd.cpp -o stock_analysis -std=c++11
	./stock_analysis $(strategy) $(symbol) $(x) $(start_date) $(end_date)
else ifeq ($(strategy), RSI)
	g++ rsi.cpp -o stock_analysis -std=c++11
	./stock_analysis $(strategy) $(symbol) $(x) $(n) $(oversold_threshold) $(overbought_threshold) $(start_date) $(end_date)
else ifeq ($(strategy), PAIRS)
	g++ pairs.cpp -o stock_analysis -std=c++11
	./stock_analysis $(strategy) $(symbol1) $(symbol2) $(x) $(n) $(threshold) $(start_date) $(end_date)
else ifeq ($(strategy), ADX)
	g++ ADX.cpp -o stock_analysis -std=c++11
	./stock_analysis $(strategy) $(symbol) $(x) $(n) $(adx_threshold) $(start_date) $(end_date)
else
	@echo "Invalid strategy. Please use 'strategy=basic' or 'strategy=dma'."
	@exit 1
endif

# Target to clean generated files
clean:
	rm -f stock_analysis
	rm -f *.txt
	rm -f daily_cashflow.csv
	rm -f order_statistics.csv

# Usage message
help:
	@echo "Usage:"
	@echo "  make strategy=<strategy> symbol=<symbol> n=<n> x=<x> start_date=<start_date> end_date=<end_date>"
	@echo "  make clean         # Clean generated files"
	@echo "  make help          # Show this message"

# PHONY targets to ensure they are always executed
.PHONY: all install fetch_data run_analysis clean help
