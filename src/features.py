import numpy as np
import pandas as pd

def build_features(df):
    df = df.copy()
    
    # 1. Biến đổi Giá cơ bản
    df['returns'] = np.log(df['close'] / df['close'].shift(1))
    df['volatility'] = df['returns'].rolling(window=21).std() * np.sqrt(252)
    
    # 2. Z-Score & MACD
    ma20 = df['close'].rolling(window=20).mean()
    std20 = df['close'].rolling(window=20).std()
    df['z_score'] = (df['close'] - ma20) / std20
    
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()
    df['macd_hist'] = macd_line - macd_signal

    # 3. Hurst Exponent (Xu hướng)
    def get_hurst(ts):
        lags = range(2, 20)
        tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
        return np.polyfit(np.log(lags), np.log(tau), 1)[0] * 2.0

    df['hurst'] = df['close'].rolling(window=100).apply(get_hurst, raw=True)
    
    # ---------------------------------------------------
    # BẢN NÂNG CẤP: TOÁN VI MÔ & DÒNG TIỀN
    # ---------------------------------------------------
    # 4. ADL (Accumulation/Distribution Line)
    # Xử lý lỗi chia cho 0 nếu High == Low (chống nhiễu)
    high_low_diff = df['high'] - df['low']
    high_low_diff = high_low_diff.replace(0, 0.001) 
    
    mfm = ((df['close'] - df['low']) - (df['high'] - df['close'])) / high_low_diff
    mfv = mfm * df['volume']
    df['adl'] = mfv.cumsum()
    
    # Chuẩn hóa ADL (Z-score của ADL) để AI dễ tiêu hóa
    df['adl_zscore'] = (df['adl'] - df['adl'].rolling(20).mean()) / df['adl'].rolling(20).std()

    # 5. VWAP (Volume Weighted Average Price) rolling 14 ngày
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    tp_v = typical_price * df['volume']
    df['vwap_14'] = tp_v.rolling(window=14).sum() / df['volume'].rolling(window=14).sum()
    
    # Khoảng cách giá so với giá vốn trung bình (tính bằng %)
    df['price_to_vwap'] = (df['close'] - df['vwap_14']) / df['vwap_14']
    
    return df.dropna()