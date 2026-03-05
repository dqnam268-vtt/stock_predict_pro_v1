import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

class DataLoader:
    def get_data(self, symbol, days=1095):
        yf_symbol = symbol if symbol.endswith(".VN") else f"{symbol}.VN"
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # 1. Tải dữ liệu Cổ phiếu
        ticker = yf.Ticker(yf_symbol)
        df = ticker.history(start=start_date, end=end_date)
        
        if df.empty: return pd.DataFrame()
            
        df.reset_index(inplace=True)
        df.columns = [c.lower() for c in df.columns]
        if 'date' in df.columns and df['date'].dt.tz is not None:
            df['date'] = df['date'].dt.tz_localize(None)
            
        # 2. Tải dữ liệu Thị trường chung (VN-Index)
        vn_ticker = yf.Ticker("^VNINDEX")
        vn_df = vn_ticker.history(start=start_date, end=end_date)
        
        if not vn_df.empty:
            vn_df.reset_index(inplace=True)
            vn_df.columns = [c.lower() for c in vn_df.columns]
            if 'date' in vn_df.columns and vn_df['date'].dt.tz is not None:
                vn_df['date'] = vn_df['date'].dt.tz_localize(None)
            
            # Chỉ lấy cột ngày và giá đóng cửa của VN-Index
            vn_df = vn_df[['date', 'close']].rename(columns={'close': 'vn_close'})
            
            # Ghép (Merge) dữ liệu thị trường vào cổ phiếu
            df = pd.merge(df, vn_df, on='date', how='left')
            df['vn_close'] = df['vn_close'].ffill()
        else:
            df['vn_close'] = 1000 # Giá trị dự phòng
            
        return df