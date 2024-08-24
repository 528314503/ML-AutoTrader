import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_processed_data(filepath):
    """
    加载处理过的数据
    """
    return pd.read_csv(filepath, index_col='Date', parse_dates=True)

def create_lag_features(df, columns, lag_periods):
    """
    创建滞后特征
    """
    for column in columns:
        for lag in lag_periods:
            df[f'{column}_lag_{lag}'] = df[column].shift(lag)
    return df

def create_rolling_features(df, columns, windows):
    """
    创建滚动窗口特征
    """
    for column in columns:
        for window in windows:
            df[f'{column}_rolling_mean_{window}'] = df[column].rolling(window=window).mean()
            df[f'{column}_rolling_std_{window}'] = df[column].rolling(window=window).std()
    return df

def normalize_features(df):
    """
    标准化特征
    """
    scaler = StandardScaler()
    columns_to_normalize = df.columns.drop(['Target'])
    df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])
    return df

def main():
    # 加载处理过的数据
    df = load_processed_data('data/processed/AAPL_processed.csv')
    
    # 创建滞后特征
    lag_columns = ['Close', 'Volume', 'Returns', 'RSI']
    lag_periods = [1, 2, 3, 5]
    df = create_lag_features(df, lag_columns, lag_periods)
    
    # 创建滚动窗口特征
    rolling_columns = ['Close', 'Volume', 'Returns']
    rolling_windows = [5, 10, 20]
    df = create_rolling_features(df, rolling_columns, rolling_windows)
    
    # 标准化特征
    df = normalize_features(df)
    
    # 删除包含NaN的行
    df.dropna(inplace=True)
    
    # 保存特征工程后的数据
    df.to_csv('data/processed/AAPL_featured.csv')
    print("Featured data saved to data/processed/AAPL_featured.csv")

if __name__ == "__main__":
    main()