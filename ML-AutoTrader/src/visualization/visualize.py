import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import joblib
import numpy as np
def load_data(filepath):
    """
    加载预测结果数据
    """
    return pd.read_csv(filepath, index_col='Date', parse_dates=True)

def plot_price_and_predictions(df):
    """
    绘制股票价格和预测结果
    """
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['Close'], label='Close Price')
    plt.scatter(df.index[df['Predicted'] == 1], df['Close'][df['Predicted'] == 1], 
                color='green', label='Predicted Up', marker='^')
    plt.scatter(df.index[df['Predicted'] == 0], df['Close'][df['Predicted'] == 0], 
                color='red', label='Predicted Down', marker='v')
    plt.title('Stock Price and Predictions')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.savefig('reports/figures/price_and_predictions.png')
    plt.close()

def plot_confusion_matrix(y_true, y_pred):
    """
    绘制混淆矩阵
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('reports/figures/confusion_matrix.png')
    plt.close()

def plot_feature_importance(model, feature_names):
    """
    绘制特征重要性
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(12, 8))
    plt.title("Feature Importances")
    plt.bar(range(len(importances)), importances[indices])
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
    plt.tight_layout()
    plt.savefig('reports/figures/feature_importance.png')
    plt.close()

def main():
    # 加载预测结果数据
    df = load_data('data/predictions/AAPL_predictions.csv')
    
    # 绘制股票价格和预测结果
    plot_price_and_predictions(df)
    
    # 绘制混淆矩阵
    plot_confusion_matrix(df['Actual'], df['Predicted'])
    
    # 加载模型并绘制特征重要性
    model = joblib.load('models/random_forest_model.joblib')
    feature_names = df.columns.drop(['Actual', 'Predicted'])
    plot_feature_importance(model, feature_names)
    
    print("Visualizations saved in reports/figures/")

if __name__ == "__main__":
    main()