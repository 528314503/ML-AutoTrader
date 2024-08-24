import pandas as pd
import numpy as np

def load_data(filepath):
    """
    加载CSV文件中的数据
    """
    return pd.read_csv(filepath, index_col='Date', parse_dates=True)

def calculate_technical_indicators(df):
    """
    计算技术指标
    """
    # 计算移动平均线
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_30'] = df['Close'].rolling(window=30).mean()
    
    # 计算相对强弱指标 (RSI)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # 计算布林带
    df['BB_middle'] = df['Close'].rolling(window=20).mean()
    df['BB_upper'] = df['BB_middle'] + 2 * df['Close'].rolling(window=20).std()
    df['BB_lower'] = df['BB_middle'] - 2 * df['Close'].rolling(window=20).std()
    
    return df

def prepare_features(df):
    """
    准备特征
    """
    df['Returns'] = df['Close'].pct_change()
    df['Target'] = np.where(df['Returns'].shift(-1) > 0, 1, 0)  # 预测下一天的涨跌
    
    # 删除包含NaN的行
    df.dropna(inplace=True)
    
    return df

def main():
    # 加载数据
    df = load_data('data/raw/AAPL_data.csv')
    
    # 计算技术指标
    df = calculate_technical_indicators(df)
    
    # 准备特征
    df = prepare_features(df)
    
    # 保存处理后的数据
    df.to_csv('data/processed/AAPL_processed.csv')
    print("Processed data saved to data/processed/AAPL_processed.csv")

if __name__ == "__main__":
    main()