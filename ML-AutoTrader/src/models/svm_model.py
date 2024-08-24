import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

def load_data(filepath):
    return pd.read_csv(filepath, index_col='Date', parse_dates=True)

def prepare_features(df):
    features = ['SMA_10', 'SMA_30', 'RSI', 'Returns', 'Volume']
    X = df[features]
    y = np.where(df['Returns'].shift(-1) > 0, 1, 0)  # 预测下一天的涨跌
    
    # 删除NaN值
    X = X.dropna()
    y = y[X.index]
    
    return X, y

def train_svm_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = SVC(kernel='rbf', C=1.0, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"SVM Model Accuracy: {accuracy:.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return model, scaler

def main():
    # 加载数据
    df = load_data('data/processed/AAPL_featured.csv')
    
    # 准备特征
    X, y = prepare_features(df)
    
    # 训练模型
    model, scaler = train_svm_model(X, y)
    
    # 保存模型和 scaler
    joblib.dump(model, 'models/svm_model.joblib')
    joblib.dump(scaler, 'models/svm_scaler.joblib')
    print("SVM model and scaler saved to models/ directory")

if __name__ == "__main__":
    main()