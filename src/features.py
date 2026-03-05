import numpy as np
import pandas as pd

def build_features(df):
    df = df.copy()
    
    # 1. Log Returns: Đảm bảo tính dừng cho chuỗi thời gian
    df['returns'] = np.log(df['close'] / df['close'].shift(1))
    
    # 2. Realized Volatility (21 phiên ~ 1 tháng)
    df['volatility'] = df['returns'].rolling(window=21).std() * np.sqrt(252)
    
    # 3. Z-Score (Normalization): $z = \frac{x - \mu}{\sigma}$
    ma20 = df['close'].rolling(window=20).mean()
    std20 = df['close'].rolling(window=20).std()
    df['z_score'] = (df['close'] - ma20) / std20
    
    # 4. Bollinger Bands (Biên độ xác suất)
    df['bb_upper'] = ma20 + (2 * std20)
    df['bb_lower'] = ma20 - (2 * std20)
    
    # 5. MACD (Động lượng hội tụ/phân kỳ)
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd_line'] = ema12 - ema26
    df['macd_signal'] = df['macd_line'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd_line'] - df['macd_signal']

    # 6. Hurst Exponent (Ước tính tính xu hướng)
    def get_hurst(ts):
        lags = range(2, 20)
        tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
        return np.polyfit(np.log(lags), np.log(tau), 1)[0] * 2.0

    df['hurst'] = df['close'].rolling(window=100).apply(get_hurst, raw=True)
    
    return df.dropna()