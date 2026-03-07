import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
import streamlit as st # Thêm để hiển thị thông báo lỗi nếu cần

def build_features(df):
    if len(df) < 205: # Bảo vệ: Cần ít nhất 200 phiên cho MA200 + 5 phiên dự phòng
        return pd.DataFrame() # Trả về DF rỗng để hàm train biết mà bỏ qua
        
    df = df.copy()
    
    # --- 1. CÁC CHỈ BÁO NỀN TẢNG ---
    df['returns'] = np.log(df['close'] / df['close'].shift(1))
    df['volatility'] = df['returns'].rolling(window=21).std() * np.sqrt(252)
    
    ma20 = df['close'].rolling(window=20).mean()
    std20 = df['close'].rolling(window=20).std().replace(0, 0.001) # Tránh chia cho 0
    df['z_score'] = (df['close'] - ma20) / std20
    
    ma50 = df['close'].rolling(window=50).mean()
    df['price_to_ma50'] = (df['close'] - ma50) / (ma50 + 1e-9)
    ma200 = df['close'].rolling(window=200).mean()
    df['price_to_ma200'] = (df['close'] - ma200) / (ma200 + 1e-9)
    
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
        try:
            lags = range(2, 20)
            # Thêm kiểm tra std để tránh log(0)
            stds = [np.std(np.subtract(ts[lag:], ts[:-lag])) for lag in lags]
            stds = [s if s > 0 else 1e-9 for s in stds]
            tau = [np.sqrt(s) for s in stds]
            return np.polyfit(np.log(lags), np.log(tau), 1)[0] * 2.0
        except:
            return 0.5
            
    df['hurst'] = df['close'].rolling(window=100).apply(get_hurst, raw=True)
    
    high_low_diff = (df['high'] - df['low']).replace(0, 0.001) 
    mfm = ((df['close'] - df['low']) - (df['high'] - df['close'])) / high_low_diff
    df['adl'] = (mfm * df['volume']).cumsum()
    
    adl_std = df['adl'].rolling(20).std().replace(0, 0.001)
    df['adl_zscore'] = (df['adl'] - df['adl'].rolling(20).mean()) / adl_std
    
    vol_sum = df['volume'].rolling(window=14).sum().replace(0, 1)
    tp_v = ((df['high'] + df['low'] + df['close']) / 3) * df['volume']
    df['vwap_14'] = tp_v.rolling(window=14).sum() / vol_sum
    df['price_to_vwap'] = (df['close'] - df['vwap_14']) / (df['vwap_14'] + 1e-9)

    # --- 2. BOLLINGER BANDS ---
    df['bb_upper'] = ma20 + (std20 * 2)
    df['bb_lower'] = ma20 - (std20 * 2)
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / (ma20 + 1e-9)  
    df['bb_pct_b'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-9) 

    # --- 3. ATR ---
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df['atr_14'] = true_range.rolling(14).mean()
    df['atr_ratio'] = df['atr_14'] / (df['close'] + 1e-9)

    # --- 4. CHAIKIN MONEY FLOW ---
    money_flow_vol = mfm * df['volume']
    df['cmf_20'] = money_flow_vol.rolling(20).sum() / (df['volume'].rolling(20).sum() + 1e-9)

    # --- 5. SEASONALITY ---
    if isinstance(df.index, pd.DatetimeIndex):
        df['month'] = df.index.month
        df['day_of_week'] = df.index.dayofweek
    elif 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df['month'] = df['date'].dt.month
        df['day_of_week'] = df['date'].dt.dayofweek
    else:
        # Fallback nếu không có date
        df['month'] = 1
        df['day_of_week'] = 1

    # --- 6. MULTI-TIMEFRAME ---
    w_ema12 = df['close'].ewm(span=60, adjust=False).mean()
    w_ema26 = df['close'].ewm(span=130, adjust=False).mean()
    df['weekly_macd'] = w_ema12 - w_ema26
    df['weekly_macd_signal'] = df['weekly_macd'].ewm(span=45, adjust=False).mean()
    df['weekly_macd_hist'] = df['weekly_macd'] - df['weekly_macd_signal']
    df['mtf_trend_up'] = (df['weekly_macd_hist'] > 0).astype(int)
    df['momentum_4w'] = df['close'] / df['close'].shift(20).replace(0, np.nan) - 1

    # Thay thế Inf bằng NaN và drop
    return df.replace([np.inf, -np.inf], np.nan).dropna()

class AIModel:
    def __init__(self):
        self.base_model = XGBClassifier(
            objective='binary:logistic', 
            random_state=42,
            eval_metric='logloss',
            n_jobs=1 # Tránh xung đột tài nguyên trên Streamlit Cloud
        )
        
        self.param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 5],
            'learning_rate': [0.05, 0.1],
            'subsample': [0.8],
            'colsample_bytree': [0.8]
        }
        self.model = None 
        
        self.features = [
            'returns', 'volatility', 'z_score', 'macd_hist', 'hurst', 'adl_zscore', 
            'price_to_vwap', 'price_to_ma50', 'price_to_ma200', 'rsi_14',
            'bb_width', 'bb_pct_b', 'atr_ratio', 'cmf_20', 'month', 'day_of_week',
            'weekly_macd_hist', 'mtf_trend_up', 'momentum_4w'
        ]
    
    def train(self, df):
        if df.empty or len(df) < 30: # Kiểm tra dữ liệu đầu vào
            return False
            
        df = df.copy()
        df['target'] = (df['close'].shift(-3) > df['close'] * 1.015).astype(int)
        df = df.dropna()
        
        if len(df) < 20: # Sau khi tạo target và dropna, kiểm tra lại lần nữa
            return False
            
        X = df[self.features]
        y = df['target']
        
        # Kiểm tra nếu y chỉ có 1 class (ví dụ toàn 0 hoặc toàn 1)
        if y.nunique() < 2:
            return False

        try:
            search = RandomizedSearchCV(
                estimator=self.base_model,
                param_distributions=self.param_grid,
                n_iter=3, # Giảm số lần thử để chạy nhanh hơn khi quét Top 10
                cv=2,     # Giảm CV xuống 2 để chịu được các mã ít dữ liệu
                scoring='accuracy',
                random_state=42,
                n_jobs=-1        
            )
            search.fit(X, y)
            self.model = search.best_estimator_
            return True
        except Exception as e:
            st.warning(f"Không thể huấn luyện mô hình: {str(e)}")
            return False
        
    def predict_prob(self, df):
        if self.model is None:
            return np.zeros(len(df))
        return self.model.predict_proba(df[self.features])[:, 1]