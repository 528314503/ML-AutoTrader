import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

def load_data(filepath):
    return pd.read_csv(filepath, index_col='Date', parse_dates=True)

def prepare_features(df):
    # 使用与训练时相同的特征工程步骤
    # 这里简化处理，实际应用中应该与训练时保持一致
    df['Returns'] = df['Close'].pct_change()
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_30'] = df['Close'].rolling(window=30).mean()
    
    # 删除NaN值
    df.dropna(inplace=True)
    
    # 标准化特征
    scaler = StandardScaler()
    features = ['Returns', 'SMA_10', 'SMA_30', 'Volume']
    df[features] = scaler.fit_transform(df[features])
    
    return df

def backtest(df, model, initial_capital=10000):
    # 使用模型进行预测
    X = df.drop(['Close', 'Open', 'High', 'Low'], axis=1)
    df['Predicted'] = model.predict(X)
    
    # 模拟交易
    df['Position'] = np.where(df['Predicted'] > 0.5, 1, 0)  # 1表示买入，0表示卖出
    df['Strategy_Returns'] = df['Position'].shift(1) * df['Returns']
    df['Cumulative_Returns'] = (1 + df['Strategy_Returns']).cumprod()
    df['Equity_Curve'] = initial_capital * df['Cumulative_Returns']
    
    # 计算性能指标
    total_return = df['Equity_Curve'].iloc[-1] / initial_capital - 1
    sharpe_ratio = np.sqrt(252) * df['Strategy_Returns'].mean() / df['Strategy_Returns'].std()
    
    return df, total_return, sharpe_ratio

def main():
    # 加载数据
    df = load_data('data/processed/AAPL_featured.csv')
    
    # 准备特征
    df = prepare_features(df)
    
    # 加载模型
    model = joblib.load('models/random_forest_model.joblib')
    
    # 进行回测
    results, total_return, sharpe_ratio = backtest(df, model)
    
    print(f"Total Return: {total_return:.2%}")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    
    # 保存回测结果
    results.to_csv('data/backtest_results.csv')
    print("Backtest results saved to data/backtest_results.csv")

if __name__ == "__main__":
    main()