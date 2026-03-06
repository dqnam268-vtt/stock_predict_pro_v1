import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from xgboost import XGBRegressor, XGBClassifier
import yfinance as yf
from datetime import datetime, timedelta
import sys
import os
import requests

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# ==========================================
# HÀM GỬI CẢNH BÁO TELEGRAM
# ==========================================
def send_telegram_alert(bot_token, chat_id, message):
    if not bot_token or not chat_id: return False
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {'chat_id': chat_id, 'text': message, 'parse_mode': 'Markdown'}
    try: return requests.post(url, data=payload).status_code == 200
    except: return False

# ==========================================
# PHẦN 1: CÁC MODULE TOÁN HỌC & DỮ LIỆU
# ==========================================
class DataLoader:
    def get_data(self, symbol, days=1095):
        yf_symbol = symbol if symbol.endswith(".VN") else f"{symbol}.VN"
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        ticker = yf.Ticker(yf_symbol)
        df = ticker.history(start=start_date, end=end_date)
        if df.empty: return pd.DataFrame()
            
        df.reset_index(inplace=True)
        df.columns = [c.lower() for c in df.columns]
        if 'date' in df.columns and df['date'].dt.tz is not None:
            df['date'] = df['date'].dt.tz_localize(None)
            
        vn_ticker = yf.Ticker("^VNINDEX")
        vn_df = vn_ticker.history(start=start_date, end=end_date)
        if not vn_df.empty:
            vn_df.reset_index(inplace=True)
            vn_df.columns = [c.lower() for c in vn_df.columns]
            if 'date' in vn_df.columns and vn_df['date'].dt.tz is not None:
                vn_df['date'] = vn_df['date'].dt.tz_localize(None)
            vn_df = vn_df[['date', 'close']].rename(columns={'close': 'vn_close'})
            df = pd.merge(df, vn_df, on='date', how='left')
            df['vn_close'] = df['vn_close'].ffill()
        else:
            df['vn_close'] = 1000 
        return df

def build_features(df):
    df = df.copy()
    df['returns'] = np.log(df['close'] / df['close'].shift(1))
    df['volatility'] = df['returns'].rolling(window=21).std() * np.sqrt(252)
    ma20 = df['close'].rolling(window=20).mean()
    std20 = df['close'].rolling(window=20).std()
    df['z_score'] = (df['close'] - ma20) / std20
    
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()
    df['macd_hist'] = macd_line - macd_signal

    def get_hurst(ts):
        lags = range(2, 20)
        tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
        return np.polyfit(np.log(lags), np.log(tau), 1)[0] * 2.0
    df['hurst'] = df['close'].rolling(window=100).apply(get_hurst, raw=True)
    
    high_low_diff = df['high'] - df['low']
    high_low_diff = high_low_diff.replace(0, 0.001) 
    mfm = ((df['close'] - df['low']) - (df['high'] - df['close'])) / high_low_diff
    df['adl'] = (mfm * df['volume']).cumsum()
    df['adl_zscore'] = (df['adl'] - df['adl'].rolling(20).mean()) / df['adl'].rolling(20).std()

    tp_v = ((df['high'] + df['low'] + df['close']) / 3) * df['volume']
    df['vwap_14'] = tp_v.rolling(window=14).sum() / df['volume'].rolling(window=14).sum()
    df['price_to_vwap'] = (df['close'] - df['vwap_14']) / df['vwap_14']
    
    if 'vn_close' in df.columns:
        df['vn_returns'] = np.log(df['vn_close'] / df['vn_close'].shift(1))
        df['market_corr'] = df['returns'].rolling(window=21).corr(df['vn_returns'])
        df['market_corr'] = df['market_corr'].fillna(0)
    else:
        df['vn_returns'] = 0; df['market_corr'] = 0
    return df.dropna()

class AIModel:
    def __init__(self):
        self.model = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8, objective='binary:logistic', random_state=42)
        self.features = ['returns', 'volatility', 'z_score', 'macd_hist', 'hurst', 'adl_zscore', 'price_to_vwap', 'vn_returns', 'market_corr']
    def train(self, df):
        df['target'] = (df['close'].shift(-3) > df['close'] * 1.015).astype(int)
        df = df.dropna()
        self.model.fit(df[self.features], df['target'])
    def predict_prob(self, df):
        return self.model.predict_proba(df[self.features])[:, 1]

@st.cache_data(ttl=900, show_spinner=False)
def analyze_symbol(symbol, future_days):
    df = DataLoader().get_data(symbol)
    if df.empty or len(df) < 50: return None
    
    df_feat = build_features(df)
    
    model = AIModel()
    model.train(df_feat)
    prob = model.predict_prob(df_feat.tail(1))[0]
    
    df_reg = df[['close']].copy()
    for i in range(1, 6): df_reg[f'lag_{i}'] = df_reg['close'].shift(i)
    df_reg = df_reg.dropna()
    features_reg = [f'lag_{5}', f'lag_{4}', f'lag_{3}', f'lag_{2}', f'lag_{1}']
    
    X_adapt = df_reg[features_reg]
    y_adapt = df_reg['close']
    reg_model_adapt = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=99)
    reg_model_adapt.fit(X_adapt, y_adapt)
    
    future_preds_adapt = []
    current_lags_adapt = df['close'].iloc[-5:].values.tolist()
    for _ in range(future_days):
        pred = reg_model_adapt.predict(np.array([current_lags_adapt]))[0]
        future_preds_adapt.append(float(pred))
        current_lags_adapt.pop(0)
        current_lags_adapt.append(float(pred))
        
    return {'df': df, 'df_feat': df_feat, 'prob': prob, 'future_preds_adapt': future_preds_adapt}

# ==========================================
# PHẦN 2: GIAO DIỆN APP (UI)
# ==========================================
st.set_page_config(page_title="AI Quant - Cảnh Báo", layout="wide")

with st.sidebar:
    st.header("🤖 Cài đặt Telegram Bot")
    try:
        bot_token = st.secrets["TELEGRAM_TOKEN"]
        chat_id = st.secrets["TELEGRAM_CHAT_ID"]
        st.success("✅ Đã kết nối khóa bảo mật Streamlit!")
    except:
        bot_token = st.text_input("🔑 Telegram Bot Token:", type="password")
        chat_id = st.text_input("💬 Chat ID của bạn:")
    
    if st.button("🔔 Gửi tin nhắn Test", use_container_width=True):
        if bot_token and chat_id:
            if send_telegram_alert(bot_token, chat_id, "✅ *Tuyệt vời!*\nHệ thống AI Quant đang hoạt động tốt."):
                st.success("Đã gửi tin nhắn test!")
            else: st.error("Gửi thất bại.")
        else: st.warning("Vui lòng cấu hình Bot.")
    st.markdown("---")
    
st.title("📈 Hệ thống Dự báo Định lượng (AI Quant)")

if st.button("🔄 Cập nhật Dữ liệu Real-time (Xóa bộ nhớ đệm)", use_container_width=True):
    st.cache_data.clear()
    st.success("Đã tải lại dữ liệu mới nhất. Radar sẽ quét lại từ đầu!")

col_sel1, col_sel2, col_sel3 = st.columns(3)
with col_sel1:
    tickers = ["GAS", "HT1", "VCB", "MBB", "BID", "SSI", "VND", "HCM", "FPT", "VIX"]
    symbol = st.selectbox("🎯 Chọn mã cổ phiếu (Xem chi tiết):", tickers)
with col_sel2:
    timeframe = st.selectbox("🔙 Dò tìm Cực trị:", ["Theo Tuần (5 phiên)", "Theo Tháng (21 phiên)", "Theo Quý (63 phiên)", "Theo Năm (252 phiên)"], index=1)
with col_sel3:
    future_horizon = st.selectbox("🔮 AI Dự báo Tương lai:", ["1 Tuần tới (5 phiên)", "1 Tháng tới (21 phiên)", "3 Tháng tới (63 phiên)"], index=1)

nav = st.number_input("💵 Nhập Tổng Vốn Đầu Tư (VNĐ):", min_value=1000000, value=100000000, step=10000000, format="%d")
show_candle = st.toggle("🕯️ Hiển thị Biểu đồ Nến Nhật", value=False)

window_dict = {"Theo Tuần (5 phiên)": 5, "Theo Tháng (21 phiên)": 21, "Theo Quý (63 phiên)": 63, "Theo Năm (252 phiên)": 252}
window = window_dict[timeframe]

if "Tuần" in future_horizon: future_days = 5
elif "3 Tháng" in future_horizon: future_days = 63
else: future_days = 21

with st.spinner(f"Đang phân tích {symbol}..."):
    result = analyze_symbol(symbol, future_days)

if result is not None:
    df = result['df']
    df_feat = result['df_feat']
    prob = result['prob']
    future_preds_adapt = result['future_preds_adapt']
    
    latest_row = df_feat.tail(1)
    current_price = latest_row['close'].values[0]
    price_to_vwap = latest_row['price_to_vwap'].values[0]
    adl_zscore = latest_row['adl_zscore'].values[0]
    
    if 'market_corr' in latest_row.columns and latest_row['vn_close'].values[0] != 1000:
        market_corr = latest_row['market_corr'].values[0]
        if market_corr > 0.6: corr_status = f"Đồng pha VN-Index ({market_corr:.2f})"
        elif market_corr < -0.3: corr_status = f"Ngược pha VN-Index ({market_corr:.2f})"
        else: corr_status = f"Ít phụ thuộc ({market_corr:.2f})"
    else: corr_status = "⚠️ Lỗi VN-Index"
        
    last_date = df['date'].iloc[-1]
    future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=future_days)

    future_min_idx = int(np.argmin(future_preds_adapt))
    buy_date = future_dates[future_min_idx]
    buy_price = future_preds_adapt[future_min_idx]
    
    can_sell_T3 = False
    profit_pct = 0
    if future_min_idx + 3 < len(future_preds_adapt):
        valid_sell_slice = future_preds_adapt[future_min_idx + 3:]
        offset_idx = int(np.argmax(valid_sell_slice))
        future_max_idx = future_min_idx + 3 + offset_idx
        sell_date = future_dates[future_max_idx]
        sell_price = future_preds_adapt[future_max_idx]
        profit_pct = (sell_price - buy_price) / buy_price * 100
        can_sell_T3 = True

    kelly_pct = 0
    if can_sell_T3 and profit_pct > 0:
        b = profit_pct / 5.0
        if b > 0: kelly_pct = max(0, (prob - ((1 - prob) / b)) / 2) * 100 

    shares_to_buy = int((nav * (kelly_pct / 100)) / buy_price) if buy_price > 0 else 0

    tab1, tab2, tab3 = st.tabs(["🔮 Dự báo Chi tiết", "📊 Backtest", "🏆 Radar Quét Toàn Thị Trường"])
    
    with tab1:
        col1, col2 = st.columns([1, 2.8])
        with col1:
            st.info("💡 Tín hiệu AI & Dòng tiền")
            st.metric("Xác suất tăng (3 phiên tới)", f"{prob*100:.1f}%")
            st.write("---")
            st.write(f"- **VWAP:** {'Tích cực' if price_to_vwap > 0 else 'Tiêu cực'}")
            st.write(f"- **ADL:** {'Gom hàng' if adl_zscore > 0 else 'Xả hàng'}")
            st.write(f"- **Tương quan:** {corr_status}")

        with col2:
            st.subheader(f"Biểu đồ Đa chiều - {symbol}")
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, subplot_titles=("", ""), row_width=[0.25, 0.75])
            df_plot = df.iloc[-150:]
            
            if show_candle:
                fig.add_trace(go.Candlestick(x=df_plot['date'], open=df_plot['open'], high=df_plot['high'], low=df_plot['low'], close=df_plot['close'], name='Nến', increasing_line_color='#00CC00', decreasing_line_color='#FF0000'), row=1, col=1)
            else:
                fig.add_trace(go.Scatter(x=df_plot['date'], y=df_plot['close'], mode='lines', name='Giá thực tế', line=dict(color='#1f77b4', width=2)), row=1, col=1)
            
            fig.add_trace(go.Scatter(x=future_dates, y=future_preds_adapt, mode='lines', name='AI Dự báo', line=dict(color='magenta', width=2.5, dash='dash')), row=1, col=1)
            fig.add_trace(go.Scatter(x=[buy_date], y=[buy_price], mode='markers', name='MUA', marker=dict(color='lime', symbol='triangle-up', size=16, line=dict(color='black', width=1))), row=1, col=1)
            
            if can_sell_T3:
                fig.add_trace(go.Scatter(x=[sell_date], y=[sell_price], mode='markers', name='BÁN', marker=dict(color='red', symbol='triangle-down', size=16, line=dict(color='black', width=1))), row=1, col=1)
                fig.add_trace(go.Scatter(x=[buy_date, sell_date], y=[buy_price, sell_price], mode='lines', name='Kỳ vọng', line=dict(color='green', width=1.5, dash='dot')), row=1, col=1)

            volume_colors = ['#00CC00' if row['close'] >= row['open'] else '#FF0000' for _, row in df_plot.iterrows()]
            fig.add_trace(go.Bar(x=df_plot['date'], y=df_plot['volume'], marker_color=volume_colors, name='Volume'), row=2, col=1)
            
            fig.update_layout(hovermode="x unified", margin=dict(l=0, r=0, t=30, b=0), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), dragmode="pan")
            fig.update_layout(xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True})

        st.success(f"**Bản ghi nhớ:** {symbol} - Khuyến nghị đi vốn: {kelly_pct:.1f}% ({shares_to_buy:,} CP)")

    with tab2: st.subheader("Tính năng Backtest (Đang bảo trì để ưu tiên tốc độ Bot)")

    # ------------------------------------
    # TAB 3: RADAR QUÉT & CẢNH BÁO TỰ ĐỘNG
    # ------------------------------------
    with tab3:
        st.subheader("🏆 Radar Toàn Thị Trường & Cảnh Báo Telegram")
        st.write("Radar sẽ đánh giá cả 10 mã và gửi báo cáo phân loại (Nên Mua / Nên Đứng Ngoài) thẳng về điện thoại.")
        
        if st.button("🚀 Quét Toàn Bộ & Báo cáo Bot"):
            progress_bar = st.progress(0)
            radar_results = []
            
            # Khởi tạo 2 danh sách riêng biệt
            good_stocks = []
            bad_stocks = []
            
            for i, sym in enumerate(tickers):
                res = analyze_symbol(sym, future_days)
                if not res: continue
                
                scan_prob = res['prob']
                scan_preds = res['future_preds_adapt']
                
                min_idx = int(np.argmin(scan_preds))
                buy_p = scan_preds[min_idx]
                
                scan_profit = 0
                if min_idx + 3 < len(scan_preds):
                    scan_profit = (max(scan_preds[min_idx + 3:]) - buy_p) / buy_p * 100
                    
                scan_kelly = 0
                if scan_profit > 0 and (scan_profit / 5.0) > 0:
                    scan_kelly = max(0, (scan_prob - ((1-scan_prob)/(scan_profit/5.0))) / 2) * 100
                        
                radar_results.append({
                    "Mã CP": sym, "Xác suất Tăng": scan_prob,
                    "Tỷ trọng Vốn (Kelly)": scan_kelly / 100, "Kỳ vọng T+3": scan_profit / 100,
                    "Giá Canh Mua": buy_p
                })
                
                # PHÂN LOẠI MÃ CHO BÁO CÁO TELEGRAM
                if scan_kelly > 0:
                    good_stocks.append(f"✅ *{sym}* | Đợi Mua: {buy_p:,.0f}đ | Kỳ vọng: +{scan_profit:.2f}% | Kelly: {scan_kelly:.1f}% vốn")
                else:
                    bad_stocks.append(f"➖ _{sym}_ | TT Xấu / Đứng ngoài quan sát")
                
                progress_bar.progress((i + 1) / len(tickers))
                
            progress_bar.empty()
            
            if radar_results:
                radar_df = pd.DataFrame(radar_results).sort_values(by="Tỷ trọng Vốn (Kelly)", ascending=False).reset_index(drop=True)
                st.dataframe(radar_df.style.format({"Xác suất Tăng": "{:.1%}", "Tỷ trọng Vốn (Kelly)": "{:.1%}", "Kỳ vọng T+3": "{:+.2%}", "Giá Canh Mua": "{:,.0f} đ"}).background_gradient(subset=["Xác suất Tăng", "Tỷ trọng Vốn (Kelly)"], cmap="Greens"), use_container_width=True, height=400)
                
                # GỬI BÁO CÁO TỔNG HỢP QUA TELEGRAM
                if bot_token and chat_id:
                    final_msg = "🏆 *BÁO CÁO RADAR AI QUANT* 🏆\n\n"
                    
                    if len(good_stocks) > 0:
                        final_msg += "🎯 *DANH MỤC ĐẠT CHUẨN MUA:*\n"
                        final_msg += "\n".join(good_stocks) + "\n\n"
                    else:
                        final_msg += "⚠️ *KHÔNG CÓ MÃ ĐẠT CHUẨN MUA.*\n(Thị trường rủi ro, nên cầm tiền mặt)\n\n"
                    
                    final_msg += "📊 *TRẠNG THÁI CÁC MÃ CÒN LẠI:*\n"
                    final_msg += "\n".join(bad_stocks)
                    
                    send_telegram_alert(bot_token, chat_id, final_msg)
                    st.toast("Đã gửi báo cáo tổng hợp chi tiết qua Telegram!", icon="✈️")
                else:
                    st.info("Quét hoàn tất. Hãy cài đặt Bot ở Sidebar để nhận tin nhắn.")
else: st.error("Không thể kết nối dữ liệu.")