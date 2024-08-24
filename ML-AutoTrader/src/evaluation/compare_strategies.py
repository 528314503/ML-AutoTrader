import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import StandardScaler

def load_data(filepath):
    return pd.read_csv(filepath, index_col='Date', parse_dates=True)

def load_model(model_path, scaler_path):
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

def prepare_features(df):
    features = ['SMA_10', 'SMA_30', 'RSI', 'Returns', 'Volume']
    return df[features]

def backtest_model(df, model, scaler, model_name):
    X = prepare_features(df)
    X_scaled = scaler.transform(X)
    df['Predicted'] = model.predict(X_scaled)
    
    df['Position'] = df['Predicted'].shift(1)
    df['Strategy_Returns'] = df['Position'] * df['Returns']
    df['Cumulative_Returns'] = (1 + df['Strategy_Returns']).cumprod()
    df[f'{model_name}_Equity_Curve'] = 10000 * df['Cumulative_Returns']
    
    total_return = df[f'{model_name}_Equity_Curve'].iloc[-1] / 10000 - 1
    sharpe_ratio = np.sqrt(252) * df['Strategy_Returns'].mean() / df['Strategy_Returns'].std()
    
    return df, total_return, sharpe_ratio

def plot_equity_curves(df, strategies):
    plt.figure(figsize=(12, 6))
    for strategy in strategies:
        plt.plot(df.index, df[f'{strategy}_Equity_Curve'], label=strategy)
    plt.plot(df.index, 10000 * (1 + df['Returns']).cumprod(), label='Buy and Hold')
    plt.title('Equity Curves of Different Strategies')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value')
    plt.legend()
    plt.savefig('reports/figures/equity_curves_comparison.png')
    plt.close()

def main():
    # 加载数据
    df = load_data('data/processed/AAPL_featured.csv')
    
    # 加载模型
    random_forest_model = joblib.load('models/random_forest_model.joblib')
    svm_model, svm_scaler = load_model('models/svm_model.joblib', 'models/svm_scaler.joblib')
    xgboost_model, xgboost_scaler = load_model('models/xgboost_model.joblib', 'models/xgboost_scaler.joblib')
    
    # 回测模型
    df, rf_return, rf_sharpe = backtest_model(df, random_forest_model, StandardScaler(), 'RandomForest')
    df, svm_return, svm_sharpe = backtest_model(df, svm_model, svm_scaler, 'SVM')
    df, xgb_return, xgb_sharpe = backtest_model(df, xgboost_model, xgboost_scaler, 'XGBoost')
    
    # 加载技术指标策略结果
    tech_df = pd.read_csv('data/technical_indicator_results.csv', index_col='Date', parse_dates=True)
    df['Technical_Equity_Curve'] = tech_df['Equity_Curve']
    tech_return = tech_df['Equity_Curve'].iloc[-1] / 10000 - 1
    tech_sharpe = np.sqrt(252) * tech_df['Strategy_Returns'].mean() / tech_df['Strategy_Returns'].std()
    
    # 打印结果
    print(f"Random Forest - Total Return: {rf_return:.2%}, Sharpe Ratio: {rf_sharpe:.2f}")
    print(f"SVM - Total Return: {svm_return:.2%}, Sharpe Ratio: {svm_sharpe:.2f}")
    print(f"XGBoost - Total Return: {xgb_return:.2%}, Sharpe Ratio: {xgb_sharpe:.2f}")
    print(f"Technical Indicator - Total Return: {tech_return:.2%}, Sharpe Ratio: {tech_sharpe:.2f}")