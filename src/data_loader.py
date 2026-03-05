import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

class DataLoader:
    def get_data(self, symbol, days=1095):
        yf_symbol = symbol if symbol.endswith(".VN") else f"{symbol}.VN"
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        ticker = yf.Ticker(yf_symbol)
        df = ticker.history(start=start_date, end=end_date)
        
        if df.empty: return pd.DataFrame()
            
        df.reset_index(inplace=True)
        df.columns = [c.lower() for c in df.columns] # Đưa về chữ thường
        
        if 'date' in df.columns and df['date'].dt.tz is not None:
            df['date'] = df['date'].dt.tz_localize(None)
            
        return df