import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import os

def fetch_stock_data(symbol, start_date, end_date):
    """
    使用yfinance获取指定股票的历史数据
    """
    stock = yf.Ticker(symbol)
    df = stock.history(start=start_date, end=end_date)
    return df

def save_data(df, symbol, directory):
    """
    将数据保存为CSV文件
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = f"{symbol}_data.csv"
    filepath = os.path.join(directory, filename)
    df.to_csv(filepath)
    print(f"Data saved to {filepath}")

def main():
    # 设置参数
    symbol = "AAPL"  # 以苹果公司为例
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)  # 获取过去一年的数据
    
    # 获取数据
    df = fetch_stock_data(symbol, start_date, end_date)
    
    # 保存数据
    save_data(df, symbol, "data/raw")

if __name__ == "__main__":
    main()
