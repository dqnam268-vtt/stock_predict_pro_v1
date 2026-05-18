import streamlit st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from xgboost import XGBRegressor 
import yfinance as yf
from datetime import datetime, timedelta
import sys
import os
import requests
import time
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import google.generativeai as genai

# KẾT NỐI MODULE BỘ NÃO VĨ MÔ
from ai_core import build_features, AIModel

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# ==========================================
# CẤU TRÚC DANH MỤC NHÓM NGÀNH
# ==========================================
INDUSTRIES = {
    "🏦 Ngân hàng": ["VCB", "BID", "CTG", "MBB", "TCB", "VPB", "ACB", "STB", "SHB", "HDB"],
    "📈 Chứng khoán": ["SSI", "VND", "HCM", "VCI", "VIX", "SHS", "MBS", "FTS", "BSI"],
    "🏢 Bất động sản & KCN": ["VHM", "VIC", "VRE", "NVL", "DIG", "DXG", "KBC", "PDR", "IDC", "SZC"],
    "🏗️ Thép & Xây dựng": ["HPG", "HSG", "NKG", "HT1", "BCC", "VCG", "CTD", "HBC"],
    "🛒 Bán lẻ & Công nghệ": ["FPT", "MWG", "PNJ", "DGW", "FRT", "CMG"],
    "🛢️ Dầu khí & Năng lượng": ["GAS", "PVD", "PVS", "BSR", "POW", "PLX", "NT2"],
    "🚢 Cảng biển & Thủy sản": ["HAH", "GMD", "VSC", "VHC", "ANV", "FMC"]
}

# ==========================================
# THAM SỐ CHIẾN THUẬT T+2 SNIPER CHÍNH XÁC
# ==========================================
T2_PROB_THRESHOLD = 0.60  # Xác suất tối thiểu 60% cho T+2 短期
T2_TAKE_PROFIT = 0.045    # Chốt lời mục tiêu +4.5%
T2_STOP_LOSS = 0.030      # Cắt lỗ nghiêm ngặt -3.0%

if 'sent_9h05' not in st.session_state: st.session_state['sent_9h05'] = None
if 'sent_13h05' not in st.session_state: st.session_state['sent_13h05'] = None
if 'sent_15h05' not in st.session_state: st.session_state['sent_15h05'] = None

def send_telegram_alert(bot_token, chat_id, message):
    if not bot_token or not chat_id: return False
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {'chat_id': chat_id, 'text': message, 'parse_mode': 'Markdown'}
    try: return requests.post(url, data=payload).status_code == 200
    except: return False

# ==========================================
# PHẦN 1: KHO DỮ LIỆU CLOUD CHO UI
# ==========================================
class CloudDataLoader:
    def __init__(self):
        self.db = None
        try:
            self.sheet_id = st.secrets["SHEET_ID"]
            creds_dict = dict(st.secrets["gcp_service_account"])
            scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
            creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
            self.client = gspread.authorize(creds)
            self.db = self.client.open_by_key(self.sheet_id)
        except Exception as e:
            st.error(f"🚨 LỖI API GOOGLE: {str(e)}")

    def download_yf(self, yf_symbol, start, end):
        df = pd.DataFrame()
        for attempt in range(4):
            try:
                ticker = yf.Ticker(yf_symbol)
                df = ticker.history(start=start, end=end)
                if not df.empty: break
            except: time.sleep(1)
        if df.empty: return pd.DataFrame()
        df.reset_index(inplace=True)
        df.columns = [c.lower() for c in df.columns]
        if 'date' in df.columns and df['date'].dt.tz is not None:
            df['date'] = df['date'].dt.tz_localize(None)
            
        invalid_rows = (df['close'] <= 100) | (df['close'] > 2000000)
        df.loc[invalid_rows, 'close'] = np.nan
        df['close'].ffill(inplace=True)
        df.dropna(subset=['close'], inplace=True)
        return df

    def get_data(self, symbol, days=3650):
        yf_symbol = symbol if symbol.endswith(".VN") else f"{symbol}.VN"
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        if self.db is None: return self.download_yf(yf_symbol, start_date, end_date)
        try:
            worksheet = self.db.worksheet(symbol)
            data = worksheet.get_all_records()
            df = pd.DataFrame(data)
            if not df.empty: df['date'] = pd.to_datetime(df['date'])
        except Exception:
            return self.download_yf(yf_symbol, start_date, end_date)

        if df.empty:
            df = self.download_yf(yf_symbol, start_date, end_date)
            if not df.empty:
                df_save = df[['date', 'open', 'high', 'low', 'close', 'volume']].copy()
                df_save['date'] = df_save['date'].dt.strftime('%Y-%m-%d')
                try:
                    worksheet.clear()
                    worksheet.append_rows([df_save.columns.values.tolist()] + df_save.values.tolist())
                except: pass
        else:
            last_date = df['date'].max()
            if end_date.date() > last_date.date() and end_date.weekday() < 5:
                new_start = last_date + timedelta(days=1)
                new_df = self.download_yf(yf_symbol, new_start, end_date)
                if not new_df.empty:
                    df_save = new_df[['date', 'open', 'high', 'low', 'close', 'volume']].copy()
                    df_save['date'] = df_save['date'].dt.strftime('%Y-%m-%d')
                    try:
                        worksheet.append_rows(df_save.values.tolist())
                        df = pd.concat([df, new_df]).drop_duplicates(subset=['date'], keep='last').reset_index(drop=True)
                    except: pass
        return df

    def save_leaderboard(self, df_leaderboard):
        if self.db is None: return False
        try:
            worksheet = self.db.worksheet("Top_10_Leaderboard")
        except:
            try: worksheet = self.db.add_worksheet(title="Top_10_Leaderboard", rows="50", cols="10")
            except: return False
        try:
            worksheet.clear()
            worksheet.append_rows([df_leaderboard.columns.values.tolist()] + df_leaderboard.values.tolist())
            return True
        except: return False

    def load_leaderboard(self):
        if self.db is None: return pd.DataFrame()
        try:
            worksheet = self.db.worksheet("Top_10_Leaderboard")
            data = worksheet.get_all_records()
            return pd.DataFrame(data)
        except: return pd.DataFrame()

@st.cache_data(ttl=3600, show_spinner=False)
def analyze_symbol(symbol, future_days):
    df = CloudDataLoader().get_data(symbol)
    if df is None or df.empty or len(df) < 50: return None
    
    df_feat = build_features(df)
    model = AIModel()
    model.train(df_feat)
    all_probs = model.predict_prob(df_feat)
    prob = all_probs[-1]
    
    df_reg = df[['close']].copy()
    for i in range(1, 6): df_reg[f'lag_{i}'] = df_reg['close'].shift(i)
    df_reg = df_reg.dropna()
    features_reg = [f'lag_{5}', f'lag_{4}', f'lag_{3}', f'lag_{2}', f'lag_{1}']
    
    X_adapt = df_reg[features_reg]
    y_adapt = df_reg['close']
    reg_model_adapt = XGBRegressor(n_estimators=150, max_depth=4, learning_rate=0.05, random_state=99)
    reg_model_adapt.fit(X_adapt, y_adapt)
    
    current_price_baseline = df['close'].iloc[-1]
    max_bound = current_price_baseline * 1.15
    min_bound = current_price_baseline * 0.85
    
    future_preds_adapt = []
    current_lags_adapt = df['close'].iloc[-5:].values.tolist()
    for _ in range(future_days):
        pred = reg_model_adapt.predict(np.array([current_lags_adapt]))[0]
        pred = float(np.clip(pred, min_bound, max_bound))
        future_preds_adapt.append(pred)
        current_lags_adapt.pop(0)
        current_lags_adapt.append(pred)
        
    return {'df': df, 'df_feat': df_feat, 'prob': prob, 'all_probs': all_probs, 'future_preds_adapt': future_preds_adapt, 'features_count': len(model.features), 'data_rows': len(df_feat)}

def run_advanced_backtest(df_bt, nav):
    fee = 0.0015         
    stop_loss = -0.04    
    take_profit = 0.06   
    capital = nav
    in_position = False
    entry_price = 0
    shares = 0
    days_held = 0
    winning_trades = 0
    total_trades = 0
    equity_curve = []
    buy_hold_curve = []
    
    if len(df_bt) == 0:
        df_bt['strategy_equity'] = nav
        df_bt['bnh_equity'] = nav
        return df_bt, 0, 0
        
    initial_price = df_bt['close'].iloc[0]
    bnh_shares = (nav * (1 - fee)) / initial_price
    
    for index, row in df_bt.iterrows():
        current_price = row['close']
        prob = row['prob']
        if in_position:
            days_held += 1
            unrealized_return = (current_price - entry_price) / entry_price
            if days_held >= 3:
                if unrealized_return <= stop_loss or unrealized_return >= take_profit or prob < 0.48:
                    capital = shares * current_price * (1 - fee) 
                    total_trades += 1
                    if (current_price * (1 - fee)) > (entry_price * (1 + fee)): winning_trades += 1
                    in_position = False
                    shares = 0
                    entry_price = 0
                    days_held = 0
        if not in_position:
            if prob > 0.55: 
                in_position = True
                entry_price = current_price
                investable_capital = capital * (1 - fee) 
                shares = investable_capital / entry_price
                days_held = 0
        if in_position: daily_equity = shares * current_price
        else: daily_equity = capital
            
        equity_curve.append(daily_equity)
        buy_hold_curve.append(bnh_shares * current_price)
        
    df_bt['strategy_equity'] = equity_curve
    df_bt['bnh_equity'] = buy_hold_curve
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
    return df_bt, win_rate, total_trades

# ==========================================
# QUÉT TỔNG THỂ VÀ TỐI ƯU T+2 (BẢN VÁ LỖI KẾT NỐI MA TRẬN BẤT TỬ)
# ==========================================
def get_bulk_report(mode="standard", status_element=None):
    all_tickers = [tic for sublist in INDUSTRIES.values() for tic in sublist]
    all_results = []
    
    if status_element: 
        status_element.warning(f"📡 Đang kết nối Trạm vũ trụ Yahoo... Quét sỉ hệ thống {mode.upper()} chống chặn IP.")
    
    try:
        yf_symbols = [s if s.endswith(".VN") else f"{s}.VN" for s in all_tickers]
        # Sử dụng group_by="ticker" gom 50 mã về 1 Request duy nhất, bẻ gãy hoàn toàn cơ chế quét bot của Yahoo Finance
        bulk_data = yf.download(yf_symbols, period="2y", group_by="ticker", progress=False, threads=True)
    except Exception as e:
        return f"⚠️ Lỗi kết nối trực tiếp máy chủ Yahoo: {str(e)}. Thầy vui lòng quét lại sau vài giây."

    for sym in all_tickers:
        yf_s = sym if sym.endswith(".VN") else f"{sym}.VN"
        try:
            if yf_s not in bulk_data.columns.levels[0]: 
                continue
            df_ticker = bulk_data[yf_s].dropna(subset=['Close'])
            if len(df_ticker) < 205: 
                continue
                
            df = df_ticker.copy()
            df.reset_index(inplace=True)
            df.columns = [c.lower() for c in df.columns]
            
            if 'date' not in df.columns and 'datetime' in df.columns:
                df.rename(columns={'datetime': 'date'}, inplace=True)
            
            df_feat = build_features(df)
            if df_feat.empty: 
                continue
                
            model = AIModel()
            if not model.train(df_feat): 
                continue
                
            prob = model.predict_prob(df_feat)[-1]
            
            tp, sl = (0.06, 0.04) if mode == "standard" else (T2_TAKE_PROFIT, T2_STOP_LOSS)
            kelly = prob - ((1 - prob) / (tp / sl))
            
            all_results.append({
                "sym": sym, 
                "buy": df['close'].iloc[-1], 
                "prob": prob, 
                "kelly": kelly * 100
            })
        except:
            continue

    if status_element: 
        status_element.empty()
        
    if not all_results: 
        return "⚠️ Yahoo Finance đang nghẽn mạng cục bộ. Thầy vui lòng nhấn lại lệnh Quét sau 10 giây nhé!"

    df_all = pd.DataFrame(all_results).sort_values(by=["prob", "kelly"], ascending=False)
    threshold = 0.55 if mode == "standard" else T2_PROB_THRESHOLD
    df_qualified = df_all[df_all['prob'] >= threshold]

    if not df_qualified.empty:
        df_res = df_qualified.head(10)
        title = "🎯 TOP 10 CỔ PHIẾU TỐT NHẤT T+5" if mode == "standard" else "⚡ DANH MỤC T+2 SNIPER (>60%)"
        msg = f"*{title}*\n\n"
    else:
        # TỰ ĐỘNG CHUYỂN DỊCH THEO Ý THẦY: Nếu không có mã nào đạt chuẩn, nhả ngay Top 3 tiềm năng nhất
        df_res = df_all.head(3)
        title = f"⚠️ TOP 3 MÃ TIỀM NĂNG NHẤT KHUNG {mode.upper()}"
        msg = f"*{title} (Dù chưa đạt chuẩn {threshold*100}%)*\n"
        msg += "_Thị trường chung hiện tại đang rủi ro, lực bán tháo áp đảo. Thầy nên ưu tiên đưa 3 mã khỏe nhất này vào Watchlist:_\n\n"

    for _, row in df_res.iterrows():
        icon = "✅" if row['prob'] >= threshold else "🟡"
        msg += f"{icon} *{row['sym']}* | Giá: {row['buy']:,.0f}đ | Win: {row['prob']*100:.1f}% | Kelly: {row['kelly']:.1f}%\n"
        
    return msg

# ------------------------------------------------------------------
# BỘ CHUYỂN ĐỔI CHỈ BÁO THÀNH THANG ĐIỂM 100 (19 TIÊU CHÍ)
# ------------------------------------------------------------------
def safe_score(val):
    if np.isnan(val) or np.isinf(val): return 50
    return min(100, max(0, int(val)))

def get_19_criteria_scores(row, prob, df):
    scores = {}
    row_dict = row.iloc[0].to_dict()
    
    scores["1. XU HƯỚNG: Vị thế so với MA50 (Trung hạn)"] = safe_score((row_dict.get('price_to_ma50', 0) + 0.1) * 500)
    scores["2. XU HƯỚNG: Vị thế so với MA200 (Dài hạn)"] = safe_score((row_dict.get('price_to_ma200', 0) + 0.2) * 250)
    scores["3. DÒNG TIỀN: Vị thế so với VWAP (Khớp lệnh)"] = safe_score((row_dict.get('price_to_vwap', 0) + 0.05) * 1000)
    
    scores["4. ĐỘNG LƯỢNG: Sức mạnh RSI 14"] = safe_score(row_dict.get('rsi_14', 50))
    scores["5. ĐỘNG LƯỢNG: Xung lực MACD Histogram"] = safe_score(50 + row_dict.get('macd_hist', 0) * 5000)
    scores["6. ĐỘT BIẾN: Chỉ số Z-Score (Gia tốc giá)"] = safe_score(50 + row_dict.get('z_score', 0) * 20)
    
    scores["7. DÒNG TIỀN: Áp lực Mua/Bán (Chaikin CMF)"] = safe_score(50 + row_dict.get('cmf_20', 0) * 200)
    scores["8. DÒNG TIỀN: Mức độ Tích lũy (ADL Z-Score)"] = safe_score(50 + row_dict.get('adl_zscore', 0) * 20)
    
    vol_ratio = df['volume'].iloc[-1] / (df['volume'].rolling(20).mean().iloc[-1] + 1) if len(df) > 20 else 1
    scores["9. DÒNG TIỀN: Đột biến Khối lượng (Volume)"] = safe_score(50 + (vol_ratio - 1) * 25)
    
    scores["10. BĂNG TẦN: Vị trí dải Bollinger (%B)"] = safe_score(row_dict.get('bb_pct_b', 0.5) * 100)
    scores["11. BĂNG TẦN: Độ nén dải (Bollinger Width)"] = safe_score(100 - row_dict.get('bb_width', 0.1) * 500) 
    scores["12. RỦI RO: Tốc độ biến động (ATR Ratio)"] = safe_score(100 - row_dict.get('atr_ratio', 1) * 50) 
    scores["13. RỦI RO: Mức độ nhiễu loạn (Volatility)"] = safe_score(100 - row_dict.get('volatility', 0) * 1000)
    
    scores["14. ĐA KHUNG: Sức mạnh Tuần (Weekly MACD)"] = safe_score(50 + row_dict.get('weekly_macd_hist', 0) * 5000)
    scores["15. ĐA KHUNG: Quán tính 4 Tuần (Momentum)"] = safe_score(50 + row_dict.get('momentum_4w', 0) * 200)
    
    scores["16. THỐNG KÊ: Tính bền vững xu hướng (Hurst)"] = safe_score(row_dict.get('hurst', 0.5) * 100)
    scores["17. THỐNG KÊ: Tỷ suất lợi nhuận (Returns)"] = safe_score(50 + row_dict.get('returns', 0) * 1000)
    scores["18. CƠ BẢN: Lịch sử Cổ tức/Chia tách"] = 100 if (row_dict.get('dividends', 0) > 0 or row_dict.get('stock_splits', 0) > 0) else 50
    
    scores["19. TỔNG HỢP: Điểm AI Định lượng (XGBoost)"] = safe_score(prob * 100)
    
    return scores

# ==========================================
# PHẦN 3: GIAO DIỆN APP (UI)
# ==========================================
st.set_page_config(page_title="AI Quant - Bảng Điều Khiển", layout="wide")

with st.sidebar:
    st.header("🤖 NamY AI Sniper")
    try:
        bot_token = st.secrets["TELEGRAM_TOKEN"]
        chat_id = st.secrets["TELEGRAM_CHAT_ID"]
        st.success("✅ Đã kết nối khóa Telegram!")
    except:
        bot_token = st.text_input("🔑 Telegram Bot Token:", type="password")
        chat_id = st.text_input("💬 Telegram Chat ID:")
        
    try:
        gemini_api_key = st.secrets["GEMINI_API_KEY"]
        st.success("✅ Đã kết nối khóa Gemini AI!")
    except:
        gemini_api_key = st.text_input("🧠 Gemini API Key (Tùy chọn):", type="password")
        
    if gemini_api_key:
        genai.configure(api_key=gemini_api_key)
    
    st.markdown("---")
    st.subheader("⚡ CHIẾN THUẬT T+2 SNIPER")
    if st.button("🚀 QUÉT 3 MÃ T+2 TỐT NHẤT", type="primary", use_container_width=True):
        status_t2 = st.empty()
        report_t2 = get_bulk_report(mode="t2", status_element=status_t2)
        st.markdown(report_t2)
        send_telegram_alert(bot_token, chat_id, report_t2)

    st.markdown("---")
    st.header("⚙️ Chế độ Tự Động T+5")
    auto_bot = st.toggle("📡 Bật Auto-Bot T+5 (Báo cáo Định kỳ)", value=False)
    st.caption("AI tự chạy ngầm gửi Báo cáo Top 10 toàn TT vào lúc: 9h05, 13h05 và 15h05.")
    
    st.write("")
    if st.button("🚀 GỬI BÁO CÁO TOP 10 T+5 NGAY", use_container_width=True):
        if bot_token and chat_id:
            status_text = st.empty() 
            report_msg = get_bulk_report(mode="standard", status_element=status_text)
            
            status_text.warning("✅ *Đang tổng hợp tín hiệu và bắn qua Telegram...*")
            full_message = f"⚡ *BÁO CÁO NHANH THEO YÊU CẦU (THỦ CÔNG)* ⚡\n\n{report_msg}"
            
            if send_telegram_alert(bot_token, chat_id, full_message):
                status_text.success("🎉 Đã bắn báo cáo Top 10 qua Telegram thành công!")
            else:
                status_text.error("Gửi thất bại. Hãy kiểm tra lại cấu hình Telegram.")
        else:
            st.error("Thầy vui lòng nhập Token và Chat ID trước nhé!")
            
    st.markdown("---")
    st.caption("✨ Tối ưu & Phát triển bởi **NamY**")
    
st.title("Hệ thống Dự báo AI Quant")

if st.button("🔄 Xóa Nhớ Đệm & Cập nhật Dữ liệu Mới Nhất", use_container_width=True):
    st.cache_data.clear()
    st.success("Đã xóa bộ nhớ đệm. Chờ lệnh quét mới để AI nạp lại bộ dữ liệu Vĩ mô!")

col_s1, col_s2, col_s3, col_s4 = st.columns(4)
with col_s1:
    selected_sector = st.selectbox("📊 Chọn Nhóm Ngành:", list(INDUSTRIES.keys()))
    current_tickers = INDUSTRIES[selected_sector]
with col_s2:
    symbol = st.selectbox("🎯 Chọn Mã (Chi tiết Tab 1,2):", current_tickers)
with col_s3:
    timeframe = st.selectbox("🔙 Dò Cực trị:", ["Theo Tuần", "Theo Tháng", "Theo Quý", "Theo Năm"], index=1)
with col_s4:
    future_horizon = st.selectbox("🔮 Dự báo Tương lai:", ["1 Tuần tới", "1 Tháng tới", "3 Tháng tới"], index=1)

col_nav1, col_nav2 = st.columns([1, 3])
with col_nav1:
    nav = st.number_input("💵 Vốn Đầu Tư (VNĐ):", min_value=1000000, value=100000000, step=10000000, format="%d")
with col_nav2:
    st.write("") 
    show_candle = st.toggle("🕯️ Biểu đồ Nến Nhật", value=False)

if "Tuần" in future_horizon: future_days = 5
elif "3 Tháng" in future_horizon: future_days = 63
else: future_days = 21

bt_days_dict = {"1 Tháng qua": 21, "3 Tháng qua": 63, "6 Tháng qua": 126, "1 Năm qua": 252, "3 Năm qua": 750, "Toàn bộ lịch sử (10 Năm)": 2500}

with st.spinner(f"Đang đọc dữ liệu từ Excel cho mã {symbol}..."):
    result = analyze_symbol(symbol, future_days)

if result is not None:
    df = result['df']
    df_feat = result['df_feat']
    prob = result['prob']
    all_probs = result['all_probs']
    future_preds_adapt = result['future_preds_adapt']
    
    latest_row = df_feat.tail(1)
    current_price = latest_row['close'].values[0]
    price_to_vwap = latest_row['price_to_vwap'].values[0]
    adl_zscore = latest_row['adl_zscore'].values[0]
    
    mtf_trend = 1
    if 'mtf_trend_up' in latest_row.columns:
        mtf_trend = latest_row['mtf_trend_up'].values[0]
        
    last_date = df['date'].iloc[-1]
    future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=future_days)

    future_min_idx = int(np.argmin(future_preds_adapt))
    buy_date_chart = future_dates[future_min_idx]
    buy_price_chart = future_preds_adapt[future_min_idx]
    
    can_sell_chart = False
    sell_date_chart = None
    sell_price_chart = None
    chart_profit_pct = 0
    
    if future_min_idx + 1 < len(future_preds_adapt):
        valid_sell_slice = future_preds_adapt[future_min_idx + 1:]
        offset_idx = int(np.argmax(valid_sell_slice))
        future_max_idx = future_min_idx + 1 + offset_idx
        
        sell_date_chart = future_dates[future_max_idx]
        sell_price_chart = future_preds_adapt[future_max_idx]
        chart_profit_pct = (sell_price_chart - buy_price_chart) / buy_price_chart * 100
        can_sell_chart = True

    profit_expectation = 0.06
    loss_expectation = 0.04
    win_loss_ratio = profit_expectation / loss_expectation
    kelly_pct = max(0, (prob - ((1 - prob) / win_loss_ratio))) * 100 
    shares_to_buy = int((nav * (kelly_pct / 100)) / current_price) if current_price > 0 else 0
    
    # GỌI HÀM CHẤM 19 ĐIỂM
    feature_scores = get_19_criteria_scores(latest_row, prob, df)
    overall_score = int(prob * 100)

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🔮 Dự báo Chi tiết", "📊 Kỷ luật Thực chiến", "🏆 Radar Tín Hiệu", "📈 Xếp Hạng Ngành", "🧠 Tình trạng AI"])
    
    with tab1:
        col1, col2 = st.columns([1, 2.8])
        with col1:
            st.info("💡 Điểm Tổng Hợp AI (Thang 100)")
            
            st.markdown(f"<h1 style='text-align: center; color: {'#00CC00' if overall_score >= 55 else '#FF0000'}; font-size: 60px;'>{overall_score}</h1>", unsafe_allow_html=True)
            st.progress(overall_score / 100)
            st.write("---")
            
            st.write(f"- **VWAP:** {'Tích cực' if price_to_vwap > 0 else 'Tiêu cực'}")
            st.write(f"- **ADL:** {'Gom hàng' if adl_zscore > 0 else 'Xả hàng'}")
            
            if mtf_trend == 1:
                st.write("- **Khung Tuần:** Đồng thuận Tăng 📈")
            else:
                st.write("- **Khung Tuần:** Đang rủi ro (Nên cẩn trọng) ⚠️")

            st.write("---")
            st.write("🎯 **DỰ PHÓNG CỰC TRỊ:**")
            st.write(f"- 🟢 **MUA:** {buy_date_chart.strftime('%d/%m/%Y')} (~ {buy_price_chart:,.0f}đ)")
            if can_sell_chart:
                st.write(f"- 🔴 **BÁN:** {sell_date_chart.strftime('%d/%m/%Y')} (~ {sell_price_chart:,.0f}đ)")
                st.write(f"*(Biên độ: +{chart_profit_pct:.1f}%)*")

        with col2:
            st.subheader(f"Biểu đồ Đa chiều - {symbol}")
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, subplot_titles=("", ""), row_width=[0.25, 0.75])
            df_plot = df.iloc[-150:]
            
            if show_candle:
                fig.add_trace(go.Candlestick(x=df_plot['date'], open=df_plot['open'], high=df_plot['high'], low=df_plot['low'], close=df_plot['close'], name='Nến', increasing_line_color='#00CC00', decreasing_line_color='#FF0000'), row=1, col=1)
            else:
                fig.add_trace(go.Scatter(x=df_plot['date'], y=df_plot['close'], mode='lines', name='Giá thực tế', line=dict(color='#1f77b4', width=2)), row=1, col=1)
            
            fig.add_trace(go.Scatter(x=future_dates, y=future_preds_adapt, mode='lines', name='AI Dự đoán Cực trị', line=dict(color='magenta', width=2.5, dash='dash')), row=1, col=1)
            fig.add_trace(go.Scatter(x=[buy_date_chart], y=[buy_price_chart], mode='markers', name='Đáy MUA dự kiến', marker=dict(color='lime', symbol='triangle-up', size=16, line=dict(color='black', width=1))), row=1, col=1)
            
            if can_sell_chart:
                fig.add_trace(go.Scatter(x=[sell_date_chart], y=[sell_price_chart], mode='markers', name='Đỉnh BÁN dự kiến', marker=dict(color='red', symbol='triangle-down', size=16, line=dict(color='black', width=1))), row=1, col=1)
                fig.add_trace(go.Scatter(x=[buy_date_chart, sell_date_chart], y=[buy_price_chart, sell_price_chart], mode='lines', name='Biên lợi nhuận', line=dict(color='green', width=1.5, dash='dot')), row=1, col=1)

            volume_colors = ['#00CC00' if row['close'] >= row['open'] else '#FF0000' for _, row in df_plot.iterrows()]
            fig.add_trace(go.Bar(x=df_plot['date'], y=df_plot['volume'], marker_color=volume_colors, name='Volume'), row=2, col=1)
            
            fig.update_layout(hovermode="x unified", margin=dict(l=0, r=0, t=30, b=0), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), dragmode="pan")
            fig.update_layout(xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True})

        if kelly_pct > 0 and prob >= 0.55:
            st.success(f"**🟢 ĐẠT CHUẨN:** {symbol} - Đề xuất vào {kelly_pct:.1f}% Vốn ({shares_to_buy:,} CP). Lãi kỳ vọng +6% / Cắt lỗ -4%.")
        else:
            st.warning(f"**⚠️ CHƯA ĐẠT CHUẨN:** {symbol} - Điểm AI chưa đủ 55 hoặc rủi ro ẩn. Khuyến nghị đứng ngoài.")
            
        st.markdown("---")
        
        col_gemini, col_tele = st.columns(2)
        with col_gemini:
            st.subheader("🧠 Hỏi Chuyên gia Gemini")
            if st.button("💬 Phân tích mã " + symbol, use_container_width=True):
                if not gemini_api_key:
                    st.error("⚠️ Thầy cần nhập Gemini API Key ở thanh menu bên trái trước!")
                else:
                    with st.spinner("Gemini đang dò tìm máy chủ và đọc dữ liệu..."):
                        try:
                            model_ai = genai.GenerativeModel('gemini-1.5-flash-latest')
                            prompt = f"""
                            Đóng vai một chuyên gia giao dịch định lượng (Quant Trader). Hãy viết 1 đoạn nhận xét ngắn gọn (khoảng 3-4 câu) bằng tiếng Việt cho cổ phiếu {symbol}.
                            Dữ liệu thuật toán hiện tại:
                            - Điểm sức mạnh AI (Xác suất tăng): {overall_score}/100.
                            - Cường độ dòng tiền (ADL): {'Dương (Đang gom hàng)' if adl_zscore > 0 else 'Âm (Đang xả hàng)'}.
                            - Chỉ báo giá trị thực (VWAP): {'Tốt' if price_to_vwap > 0 else 'Xấu'}.
                            - Xu hướng khung tuần: {'Tăng' if mtf_trend == 1 else 'Rủi ro'}.
                            - Tỷ trọng vốn khuyến nghị (Kelly): {kelly_pct:.1f}%.
                            Kết luận dứt khoát: Có nên mua hay không? (Nên mua nếu Điểm AI >= 55 và Kelly > 0). Văn phong chuyên nghiệp, lạnh lùng, dứt khoát.
                            """
                            response = model_ai.generate_content(prompt)
                            st.session_state[f'gemini_comment_{symbol}'] = response.text
                        except Exception as e:
                            st.error(f"Lỗi API Gemini: {str(e)}")
                            
            if f'gemini_comment_{symbol}' in st.session_state:
                st.info(st.session_state[f'gemini_comment_{symbol}'])

        with col_tele:
            st.subheader("✈️ Bắn tín hiệu cá nhân")
            if st.button(f"📲 Gửi Phím hàng mã {symbol} qua Telegram", use_container_width=True, type="secondary"):
                if bot_token and chat_id:
                    gemini_text = st.session_state.get(f'gemini_comment_{symbol}', "Chưa có nhận định từ Gemini.")
                    status_icon = "🟢 MUA" if (kelly_pct > 0 and prob >= 0.55) else "⚠️ ĐỨNG NGOÀI"
                    
                    single_msg = f"🔍 *PHÂN TÍCH ĐỘC LẬP: {symbol}* ({status_icon})\n\n"
                    single_msg += f"- Điểm Sức mạnh AI: *{overall_score}/100*\n"
                    single_msg += f"- Giá hiện tại: {current_price:,.0f}đ\n"
                    single_msg += f"- Điểm vào lệnh cực trị (Dự kiến): {buy_price_chart:,.0f}đ\n"
                    single_msg += f"- Điểm chốt lời cực trị (Dự kiến): {sell_price_chart:,.0f}đ\n"
                    single_msg += f"- Phân bổ Vốn (Kelly): {kelly_pct:.1f}%\n\n"
                    single_msg += f"🤖 *Chuyên gia Gemini nhận định:*\n_{gemini_text}_"
                    
                    if send_telegram_alert(bot_token, chat_id, single_msg):
                        st.success("Đã bắn báo cáo mã này thành công!")
                    else:
                        st.error("Gửi thất bại. Check lại API Telegram.")
                else:
                    st.error("Chưa nhập thông tin Telegram!")

        st.write("")
        with st.expander("🔎 XEM BẢNG CHẤM ĐIỂM CHI TIẾT 19 TIÊU CHÍ VĨ MÔ (Thang 100)"):
            st.caption("Thuật toán đã dịch các chỉ báo kỹ thuật thô thành ngôn ngữ Giao dịch Định lượng. Điểm được chuẩn hóa từ 0 đến 100 (100 = Cực kỳ Tích cực/An toàn, 0 = Cực kỳ Tiêu cực/Rủi ro).")
            
            score_cols = st.columns(3)
            for i, (feat_name, f_score) in enumerate(feature_scores.items()):
                col_idx = i % 3
                with score_cols[col_idx]:
                    st.markdown(f"**{feat_name}**: {f_score}/100")
                    st.progress(f_score / 100)

    with tab2:
        st.subheader(f"Mô phỏng Đánh tiền Thật (Đã trừ Phí 0.15% & Thuế) - Mã {symbol}")
        bt_timeframe_single = st.selectbox("⏳ Chọn chu kỳ kiểm tra:", list(bt_days_dict.keys()), index=1, key="bt_single")
        bt_days_single = bt_days_dict[bt_timeframe_single]
        
        bt_days_actual_single = min(bt_days_single, len(df_feat))
        bt_df_current = df_feat.tail(bt_days_actual_single).copy()
        bt_df_current['prob'] = all_probs[-bt_days_actual_single:]
        
        bt_df_current, win_rate_single, total_trades_single = run_advanced_backtest(bt_df_current, nav)
        
        fig_bt = go.Figure()
        fig_bt.add_trace(go.Scatter(x=bt_df_current['date'], y=bt_df_current['strategy_equity'], mode='lines', name='Vốn Đánh Theo AI', line=dict(color='magenta', width=2.5)))
        fig_bt.add_trace(go.Scatter(x=bt_df_current['date'], y=bt_df_current['bnh_equity'], mode='lines', name='Vốn Mua & Giữ', line=dict(color='gray', width=1.5, dash='dot')))
        fig_bt.add_hline(y=nav, line_dash="dash", line_color="red", annotation_text="Vốn Ban Đầu", annotation_position="bottom right")

        fig_bt.update_layout(yaxis_title="Tổng Tài Sản Net (VND)", hovermode="x unified", margin=dict(l=0, r=0, t=30, b=0), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig_bt, use_container_width=True)
        
        profit_vnd_single = bt_df_current['strategy_equity'].iloc[-1] - nav
        
        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric(f"Lãi/Lỗ Thực tế ({bt_timeframe_single})", f"{profit_vnd_single:,.0f} đ")
        col_m2.metric("Tỷ lệ Win (Sau Thuế Phí)", f"{win_rate_single:.1f}%")
        col_m3.metric("Tần suất Giao dịch", f"{total_trades_single} Lệnh")

    with tab3:
        st.subheader("🏆 Radar Tín Hiệu & Báo Cáo Telegram (Lọc Top 5 Giao diện)")
        col_btn1, col_btn2 = st.columns(2)
        run_scan = False
        scan_mode = "sector"
        
        with col_btn1:
            if st.button(f"🔍 Quét & Tìm Top 5 Ngành {selected_sector}", type="primary", use_container_width=True):
                run_scan = True; scan_mode = "sector"
        with col_btn2:
            if st.button("🌍 Quét Toàn Bộ TT (Lọc Top 5 Cực phẩm)", type="primary", use_container_width=True):
                run_scan = True; scan_mode = "all"
                
        if run_scan:
            target_tickers = current_tickers if scan_mode == "sector" else [tic for sublist in INDUSTRIES.values() for tic in sublist]
            progress_bar = st.progress(0)
            radar_results_ui = []
            
            for i, sym in enumerate(target_tickers):
                res = analyze_symbol(sym, future_days)
                if not res: continue
                scan_prob = res['prob']
                cur_price = res['df_feat']['close'].iloc[-1]
                scan_kelly = max(0, (scan_prob - ((1-scan_prob)/(0.06/0.04)))) * 100
                        
                if scan_kelly > 0 and scan_prob >= 0.55:
                    radar_results_ui.append({"Mã CP": sym, "Xác suất Tăng": scan_prob, "Tỷ trọng Vốn (Kelly)": scan_kelly / 100, "Giá Canh Mua": cur_price})
                progress_bar.progress((i + 1) / len(target_tickers))
                
            progress_bar.empty()
            if radar_results_ui:
                radar_df_ui = pd.DataFrame(radar_results_ui).sort_values(by="Tỷ trọng Vốn (Kelly)", ascending=False).reset_index(drop=True)
                st.dataframe(radar_df_ui.style.format({"Xác suất Tăng": "{:.1%}", "Tỷ trọng Vốn (Kelly)": "{:.1%}", "Giá Canh Mua": "{:,.0f} đ"}).background_gradient(subset=["Xác suất Tăng", "Tỷ trọng Vốn (Kelly)"], cmap="Greens"), use_container_width=True, height=400)
            else:
                st.warning("⚠️ Không có mã nào đạt chuẩn Mua trong nhóm này!")

    with tab4:
        st.subheader(f"📈 Bảng Xếp Hạng Kỷ Luật Thực Chiến: Toàn Thị Trường")
        bt_timeframe_all = st.selectbox("⏳ Chọn chu kỳ Backtest:", list(bt_days_dict.keys()), index=1, key="bt_all")
        bt_days_all = bt_days_dict[bt_timeframe_all]
        
        col_btn_t4_1, col_btn_t4_2, col_btn_t4_3 = st.columns(3)
        with col_btn_t4_1:
            btn_rank_sector = st.button("🔄 Xếp Hạng Nhóm Ngành", type="secondary", use_container_width=True)
        with col_btn_t4_2:
            btn_view_top10 = st.button("⚡ Xem Bảng Phong Thần", type="primary", use_container_width=True)
        with col_btn_t4_3:
            btn_update_top10 = st.button("⚙️ Cập nhật Bảng (Quét 50 mã)", type="secondary", use_container_width=True)

        if btn_rank_sector:
            with st.spinner("Đang chạy Backtest nâng cao từng mã trong ngành..."):
                all_bt_results = []
                bt_progress = st.progress(0)
                for idx, sym in enumerate(current_tickers):
                    res_bt = analyze_symbol(sym, future_days)
                    if not res_bt: continue
                    df_f_bt = res_bt['df_feat']
                    bt_days_actual = min(bt_days_all, len(df_f_bt))
                    bt_df = df_f_bt.tail(bt_days_actual).copy()
                    bt_df['prob'] = res_bt['all_probs'][-bt_days_actual:]
                    
                    bt_df, win_rate_pct, total_tr = run_advanced_backtest(bt_df, nav)
                    final_equity = bt_df['strategy_equity'].iloc[-1]
                    profit_pct = (final_equity / nav - 1) 
                    bnh_profit_pct = (bt_df['bnh_equity'].iloc[-1] / nav - 1) 
                    roll_max = bt_df['strategy_equity'].cummax()
                    max_dd = (bt_df['strategy_equity'] / roll_max - 1).min() 
                    
                    all_bt_results.append({
                        "Mã CP": sym, 
                        "Lãi ròng AI": profit_pct, 
                        "So với Mua ôm": profit_pct - bnh_profit_pct, 
                        "Win Rate": win_rate_pct / 100, 
                        "Số lệnh": total_tr,
                        "Drawdown": max_dd
                    })
                    bt_progress.progress((idx + 1) / len(current_tickers))
                bt_progress.empty()
            if all_bt_results:
                df_bt_all = pd.DataFrame(all_bt_results).sort_values(by="Lãi ròng AI", ascending=False).reset_index(drop=True)
                st.dataframe(df_bt_all.style.format({
                    "Lãi ròng AI": "{:+.2%}", "So với Mua ôm": "{:+.2%}", 
                    "Win Rate": "{:.1%}", "Drawdown": "{:.1%}"
                }).background_gradient(subset=["Lãi ròng AI", "Win Rate"], cmap="RdYlGn"), use_container_width=True)

        if btn_view_top10:
            with st.spinner("Đang kéo dữ liệu từ Đám mây..."):
                loader = CloudDataLoader()
                df_top10 = loader.load_leaderboard()
                if not df_top10.empty:
                    try:
                        for col in ["Lãi ròng AI", "Tỷ lệ Thắng", "Giá Canh Mua", "Kelly Mua Mới"]:
                            if col in df_top10.columns:
                                df_top10[col] = df_top10[col].astype(str).str.replace("'", "").str.replace(",", ".").astype(float)
                        
                        valid_buys = df_top10[df_top10['Kelly Mua Mới'] > 0]
                        valid_count = len(valid_buys)
                        
                        st.markdown("---")
                        if valid_count > 0:
                            best_sym = valid_buys.iloc[0]['Mã CP']
                            st.success(f"🎯 **ĐÁNH GIÁ TỔNG QUAN:** Dòng tiền **TÍCH CỰC**. Có **{valid_count}/10 mã** lọt vào điểm mua an toàn. Đứng đầu sóng đang là **{best_sym}**.")
                        else:
                            st.error("⚠️ **ĐÁNH GIÁ TỔNG QUAN:** Thị trường **RỦI RO CAO**. Khuyến nghị: **ÔM TIỀN MẶT ĐỨNG NGOÀI** và thêm Top 10 này vào Danh sách theo dõi!")
                        st.markdown("---")
                        st.dataframe(df_top10.style.format({
                            "Lãi ròng AI": "{:+.2%}", 
                            "Tỷ lệ Thắng": "{:.1%}", 
                            "Giá Canh Mua": "{:,.0f} đ", 
                            "Kelly Mua Mới": "{:.1%}"
                        }).background_gradient(subset=["Lãi ròng AI"], cmap="RdYlGn"), use_container_width=True)
                    except Exception as e:
                        st.error("⚠️ Định dạng cũ bị lỗi. Bấm 'Cập nhật Bảng' để AI ghi đè dữ liệu mới lên Sheet nhé!")
                        st.dataframe(df_top10, use_container_width=True)
                else:
                    st.warning("Bảng Phong Thần chưa có dữ liệu. Bấm nút 'Cập nhật Bảng' trước nhé!")

        if btn_update_top10:
            with st.spinner("Đang cày xới 50 mã..."):
                all_top10_results = []
                all_tickers_list = [tic for sublist in INDUSTRIES.values() for tic in sublist]
                bt_progress = st.progress(0)
                
                for idx, sym in enumerate(all_tickers_list):
                    res_bt = analyze_symbol(sym, future_days)
                    if not res_bt: continue
                    
                    df_f_bt = res_bt['df_feat']
                    bt_days_actual = min(bt_days_all, len(df_f_bt))
                    bt_df = df_f_bt.tail(bt_days_actual).copy()
                    bt_df['prob'] = res_bt['all_probs'][-bt_days_actual:]
                    
                    bt_df, win_rate_pct, total_tr = run_advanced_backtest(bt_df, nav)
                    profit_pct = (bt_df['strategy_equity'].iloc[-1] / nav - 1) 
                    
                    scan_prob = res_bt['prob']
                    cur_price = res_bt['df_feat']['close'].iloc[-1]
                    scan_kelly = max(0, (scan_prob - ((1-scan_prob)/(0.06/0.04)))) 

                    all_top10_results.append({
                        "Mã CP": sym, 
                        "Lãi ròng AI": profit_pct, 
                        "Tỷ lệ Thắng": win_rate_pct / 100,
                        "Giá Canh Mua": cur_price,
                        "Kelly Mua Mới": scan_kelly
                    })
                    bt_progress.progress((idx + 1) / len(all_tickers_list))
                bt_progress.empty()
                
                if all_top10_results:
                    df_top10 = pd.DataFrame(all_top10_results).sort_values(by="Lãi ròng AI", ascending=False).head(10).reset_index(drop=True)
                    df_top10_save = df_top10.copy()
                    for col in ["Lãi ròng AI", "Tỷ lệ Thắng", "Giá Canh Mua", "Kelly Mua Mới"]:
                        df_top10_save[col] = df_top10_save[col].apply(lambda x: f"'{x}")
                    
                    loader = CloudDataLoader()
                    loader.save_leaderboard(df_top10_save)
                    valid_buys = df_top10[df_top10['Kelly Mua Mới'] > 0]
                    valid_count = len(valid_buys)
                    
                    st.markdown("---")
                    if valid_count > 0:
                        best_sym = valid_buys.iloc[0]['Mã CP']
                        st.success(f"🎯 **ĐÁNH GIÁ TỔNG QUAN:** Dòng tiền **TÍCH CỰC**. Có **{valid_count}/10 mã** lọt vào điểm mua an toàn. Đứng đầu sóng đang là **{best_sym}**.")
                    else:
                        st.error("⚠️ **ĐÁNH GIÁ TỔNG QUAN:** Thị trường **RỦI RO CAO**. Khuyến nghị: **ÔM TIỀN MẶT ĐỨNG NGOÀI**!")
                    st.markdown("---")
                    st.dataframe(df_top10.style.format({
                        "Lãi ròng AI": "{:+.2%}", 
                        "Tỷ lệ Thắng": "{:.1%}", 
                        "Giá Canh Mua": "{:,.0f} đ", 
                        "Kelly Mua Mới": "{:.1%}"
                    }).background_gradient(subset=["Lãi ròng AI"], cmap="RdYlGn"), use_container_width=True)
                    
    with tab5:
        st.subheader("🧠 Trạng thái Đào tạo & Kho dữ liệu")
        col_ai1, col_ai2, col_ai3 = st.columns(3)
        col_ai1.metric("Thuật toán (AI Core)", "XGBoost 2.0 (Học sâu)")
        col_ai2.metric("Dữ liệu Lịch sử Đã nạp", f"Tối đa ({result['data_rows']} nến/mã)")
        col_ai3.metric("Bộ Đặc trưng (Features)", f"{result['features_count']} chỉ báo Vĩ mô")
        st.info("💡 **Hệ thống Kiểm tra & Huấn luyện Liên tục:** Đã vá xong lỗi MultiIndex của yfinance.")

# ==========================================
# BỘ NÃO CHẠY NGẦM (AUTO-BOT: 9h05, 13h05, 15h05)
# ==========================================
if auto_bot and bot_token and chat_id:
    vn_time = datetime.utcnow() + timedelta(hours=7)
    today_str = vn_time.strftime("%Y-%m-%d")
    
    if vn_time.weekday() < 5: 
        is_9h05 = (vn_time.hour == 9 and 5 <= vn_time.minute <= 20)
        is_13h05 = (vn_time.hour == 13 and 5 <= vn_time.minute <= 20)
        is_15h05 = (vn_time.hour == 15 and 5 <= vn_time.minute <= 20)

        trigger_9 = is_9h05 and st.session_state['sent_9h05'] != today_str
        trigger_13 = is_13h05 and st.session_state['sent_13h05'] != today_str
        trigger_15 = is_15h05 and st.session_state['sent_15h05'] != today_str

        if trigger_9 or trigger_13 or trigger_15:
            if trigger_9: session_name = "SÁNG (9h05)"
            elif trigger_13: session_name = "CHIỀU (13h05)"
            else: session_name = "TỔNG KẾT (15h05)"
            
            report_msg = get_bulk_report(mode="standard") 
            full_msg = f"🔔 *BÁO CÁO ĐỊNH KỲ: PHIÊN {session_name}* ({vn_time.strftime('%d/%m')})\n\n{report_msg}"
            send_telegram_alert(bot_token, chat_id, full_msg)
            
            if trigger_9: st.session_state['sent_9h05'] = today_str
            if trigger_13: st.session_state['sent_13h05'] = today_str
            if trigger_15: st.session_state['sent_15h05'] = today_str