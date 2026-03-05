import xgboost as xgb

class AIModel:
    def __init__(self):
        # Tham số tối ưu cho dữ liệu chứng khoán có độ nhiễu cao
        self.model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='binary:logistic',
            random_state=42
        )
        self.features = ['returns', 'volatility', 'z_score', 'macd_hist', 'hurst']

    def train(self, df):
        # Gán nhãn: 1 nếu 3 ngày tới tăng > 1.5% (đủ bù phí thuế)
        df['target'] = (df['close'].shift(-3) > df['close'] * 1.015).astype(int)
        df = df.dropna()
        
        X = df[self.features]
        y = df['target']
        self.model.fit(X, y)
        
    def predict_prob(self, df):
        return self.model.predict_proba(df[self.features])[:, 1]