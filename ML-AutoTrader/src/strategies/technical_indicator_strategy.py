import pandas as pd
import numpy as np

def calculate_indicators(df):
    # 计算简单移动平均线 (SMA)
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_30'] = df['Close'].rolling(window=30).mean()
    
    # 计算相对强弱指标 (RSI)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    return df

def generate_signals(df):
    df['Signal'] = 0  # 0 表示不操作，1 表示买入，-1 表示卖出
    
    # 当短期SMA上穿长期SMA时买入
    df.loc[(df['SMA_10'] > df['SMA_30']) & (df['SMA_10'].shift(1) <= df['SMA_30'].shift(1)), 'Signal'] = 1
    
    # 当短期SMA下穿长期SMA时卖出
    df.loc[(df['SMA_10'] < df['SMA_30']) & (df['SMA_10'].shift(1) >= df['SMA_30'].shift(1)), 'Signal'] = -1
    
    # 当RSI大于70时卖出（超买）
    df.loc[df['RSI'] > 70, 'Signal'] = -1
    
    # 当RSI小于30时买入（超卖）
    df.loc[df['RSI'] < 30, 'Signal'] = 1
    
    return df

def backtest_strategy(df, initial_capital=10000):
    df['Position'] = df['Signal'].replace(to_replace=0, method='ffill')
    df['Returns'] = df['Close'].pct_change()
    df['Strategy_Returns'] = df['Position'].shift(1) * df['Returns']
    df['Cumulative_Returns'] = (1 + df['Strategy_Returns']).cumprod()
    df['Equity_Curve'] = initial_capital * df['Cumulative_Returns']
    
    total_return = df['Equity_Curve'].iloc[-1] / initial_capital - 1
    sharpe_ratio = np.sqrt(252) * df['Strategy_Returns'].mean() / df['Strategy_Returns'].std()
    
    return df, total_return, sharpe_ratio

def main():
    # 加载数据
    df = pd.read_csv('data/processed/AAPL_featured.csv', index_col='Date', parse_dates=True)
    
    # 计算指标
    df = calculate_indicators(df)
    
    # 生成交易信号
    df = generate_signals(df)
    
    # 回测策略
    results, total_return, sharpe_ratio = backtest_strategy(df)
    
    print(f"Technical Indicator Strategy:")
    print(f"Total Return: {total_return:.2%}")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    
    # 保存结果
    results.to_csv('data/technical_indicator_results.csv')
    print("Results saved to data/technical_indicator_results.csv")

if __name__ == "__main__":
    main()