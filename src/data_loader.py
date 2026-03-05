import xgboost as xgb

class AIModel:
    def __init__(self):
        self.model = xgb.XGBClassifier(
            n_estimators=200, 
            max_depth=5, 
            learning_rate=0.03,
            subsample=0.8, 
            colsample_bytree=0.8, 
            objective='binary:logistic',
            random_state=42
        )
        # Bổ sung 'vn_returns' và 'market_corr'
        self.features = [
            'returns', 'volatility', 'z_score', 'macd_hist', 'hurst',
            'adl_zscore', 'price_to_vwap', 'vn_returns', 'market_corr'
        ]

    def train(self, df):
        df['target'] = (df['close'].shift(-3) > df['close'] * 1.015).astype(int)
        df = df.dropna()
        self.model.fit(df[self.features], df['target'])
        
    def predict_prob(self, df):
        return self.model.predict_proba(df[self.features])[:, 1]