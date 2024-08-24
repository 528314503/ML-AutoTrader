import pandas as pd
import numpy as np

class RiskManager:
    def __init__(self, max_position_size=0.1, stop_loss_pct=0.02, take_profit_pct=0.05):
        self.max_position_size = max_position_size
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
    
    def calculate_position_size(self, capital, current_price):
        max_shares = int(capital * self.max_position_size / current_price)
        return max_shares
    
    def apply_risk_management(self, df, initial_capital=10000):
        df['Position_Size'] = df.apply(lambda row: self.calculate_position_size(initial_capital, row['Close']), axis=1)
        
        df['Stop_Loss'] = df['Close'] * (1 - self.stop_loss_pct)
        df['Take_Profit'] = df['Close'] * (1 + self.take_profit_pct)
        
        df['Actual_Position'] = 0
        current_position = 0
        entry_price = 0
        
        for i in range(1, len(df)):
            if df['Predicted'].iloc[i] > 0.5 and current_position == 0:
                # Enter long position
                current_position = df['Position_Size'].iloc[i]
                entry_price = df['Close'].iloc[i]
            elif (df['Close'].iloc[i] <= df['Stop_Loss'].iloc[i-1] or 
                  df['Close'].iloc[i] >= df['Take_Profit'].iloc[i-1]) and current_position > 0:
                # Exit position
                current_position = 0
                entry_price = 0
            
            df['Actual_Position'].iloc[i] = current_position
        
        df['Strategy_Returns'] = df['Actual_Position'].shift(1) * df['Returns']
        df['Cumulative_Returns'] = (1 + df['Strategy_Returns']).cumprod()
        df['Equity_Curve'] = initial_capital * df['Cumulative_Returns']
        
        return df

def main():
    # 加载回测结果
    df = pd.read_csv('data/backtest_results.csv', index_col='Date', parse_dates=True)
    
    # 应用风险管理
    risk_manager = RiskManager()
    results_with_risk_management = risk_manager.apply_risk_management(df)
    
    # 计算新的性能指标
    total_return = results_with_risk_management['Equity_Curve'].iloc[-1] / 10000 - 1
    sharpe_ratio = np.sqrt(252) * results_with_risk_management['Strategy_Returns'].mean() / results_with_risk_management['Strategy_Returns'].std()
    
    print(f"Total Return (with risk management): {total_return:.2%}")
    print(f"Sharpe Ratio (with risk management): {sharpe_ratio:.2f}")
    
    # 保存结果
    results_with_risk_management.to_csv('data/results_with_risk_management.csv')
    print("Results with risk management saved to data/results_with_risk_management.csv")

if __name__ == "__main__":
    main()