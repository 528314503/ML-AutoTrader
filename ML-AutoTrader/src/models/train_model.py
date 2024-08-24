import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

def load_featured_data(filepath):
    """
    加载特征工程后的数据
    """
    return pd.read_csv(filepath, index_col='Date', parse_dates=True)

def prepare_data_for_training(df):
    """
    准备用于训练的数据
    """
    X = df.drop('Target', axis=1)
    y = df['Target']
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_random_forest(X_train, y_train):
    """
    训练随机森林模型
    """
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """
    评估模型性能
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

def save_model(model, filepath):
    """
    保存训练好的模型
    """
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")

def main():
    # 加载特征工程后的数据
    df = load_featured_data('data/processed/AAPL_featured.csv')
    
    # 准备训练数据
    X_train, X_test, y_train, y_test = prepare_data_for_training(df)
    
    # 训练模型
    model = train_random_forest(X_train, y_train)
    
    # 评估模型
    evaluate_model(model, X_test, y_test)
    
    # 保存模型
    save_model(model, 'models/random_forest_model.joblib')

if __name__ == "__main__":
    main()