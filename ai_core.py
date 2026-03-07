import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV # Thêm thư viện Dò tìm tự động

def build_features(df):
    df = df.copy()
    
    # --- 1. CÁC CHỈ BÁO NỀN TẢNG ---
    df['returns'] = np.log(df['close'] / df['close'].shift(1))
    df['volatility'] = df['returns'].rolling(window=21).std() * np.sqrt(252)
    
    ma20 = df['close'].rolling(window=20).mean()
    std20 = df['close'].rolling(window=20).std()
    df['z_score'] = (df['close'] - ma20) / std20
    
    ma50 = df['close'].rolling(window=50).mean()
    df['price_to_ma50'] = (df['close'] - ma50) / ma50
    ma200 = df['close'].rolling(window=200).mean()
    df['price_to_ma200'] = (df['close'] - ma200) / ma200
    
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9) 
    df['rsi_14'] = 100 - (100 / (1 + rs))

    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()
    df['macd_hist'] = macd_line - macd_signal
    
    def get_hurst(ts):
        lags = range(2, 20)
        tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
        try: return np.polyfit(np.log(lags), np.log(tau), 1)[0] * 2.0
        except: return 0.5
    df['hurst'] = df['close'].rolling(window=100).apply(get_hurst, raw=True)
    
    high_low_diff = df['high'] - df['low']
    high_low_diff = high_low_diff.replace(0, 0.001) 
    mfm = ((df['close'] - df['low']) - (df['high'] - df['close'])) / high_low_diff
    df['adl'] = (mfm * df['volume']).cumsum()
    df['adl_zscore'] = (df['adl'] - df['adl'].rolling(20).mean()) / df['adl'].rolling(20).std()
    
    tp_v = ((df['high'] + df['low'] + df['close']) / 3) * df['volume']
    df['vwap_14'] = tp_v.rolling(window=14).sum() / df['volume'].rolling(window=14).sum()
    df['price_to_vwap'] = (df['close'] - df['vwap_14']) / df['vwap_14']

    # --- 2. BOLLINGER BANDS ---
    df['bb_upper'] = ma20 + (std20 * 2)
    df['bb_lower'] = ma20 - (std20 * 2)
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / ma20  
    df['bb_pct_b'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-9) 

    # --- 3. ATR ---
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df['atr_14'] = true_range.rolling(14).mean()
    df['atr_ratio'] = df['atr_14'] / df['close']

    # --- 4. CHAIKIN MONEY FLOW ---
    money_flow_mult = ((df['close'] - df['low']) - (df['high'] - df['close'])) / high_low_diff
    money_flow_vol = money_flow_mult * df['volume']
    df['cmf_20'] = money_flow_vol.rolling(20).sum() / (df['volume'].rolling(20).sum() + 1e-9)

    # --- 5. SEASONALITY ---
    if 'date' in df.columns:
        df['month'] = df['date'].dt.month
        df['day_of_week'] = df['date'].dt.dayofweek
    else:
        df['month'] = df.index.month
        df['day_of_week'] = df.index.dayofweek

    return df.dropna()

class AIModel:
    def __init__(self):
        # 1. Khởi tạo "Phôi não" trống
        self.base_model = XGBClassifier(
            objective='binary:logistic', 
            random_state=42,
            eval_metric='logloss'
        )
        
        # 2. Khai báo không gian tiến hóa (Tủ quần áo thông số)
        self.param_grid = {
            'n_estimators': [100, 200, 300, 400],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 0.9]
        }
        
        # Mô hình chính thức sẽ được cấp sau khi thi tuyển
        self.model = None 
        
        self.features = [
            'returns', 'volatility', 'z_score', 'macd_hist', 'hurst', 'adl_zscore', 
            'price_to_vwap', 'price_to_ma50', 'price_to_ma200', 'rsi_14',
            'bb_width', 'bb_pct_b', 'atr_ratio', 'cmf_20', 'month', 'day_of_week'
        ]
    
    def train(self, df):
        df['target'] = (df['close'].shift(-3) > df['close'] * 1.015).astype(int)
        df = df.dropna()
        
        X = df[self.features]
        y = df['target']
        
        # 3. MỞ HỘP TÌM KIẾM TỰ ĐỘNG (HYPERPARAMETER TUNING)
        # Random chọn ra 5 tổ hợp xuất sắc nhất để thi đấu (tối ưu tốc độ web)
        search = RandomizedSearchCV(
            estimator=self.base_model,
            param_distributions=self.param_grid,
            n_iter=5,        # Số lần bốc thăm thử nghiệm
            cv=3,            # Cross-validation (thi đấu 3 vòng)
            scoring='accuracy',
            random_state=42,
            n_jobs=-1        # Tận dụng tối đa CPU của máy chủ
        )
        
        # Bắt đầu thi tuyển
        search.fit(X, y)
        
        # 4. Gắn bộ não đạt giải Nhất làm bộ não chính thức để trade
        self.model = search.best_estimator_
        
    def predict_prob(self, df):
        # Dùng bộ não đã tinh chỉnh tối ưu nhất để đưa ra tỷ lệ %
        return self.model.predict_proba(df[self.features])[:, 1]