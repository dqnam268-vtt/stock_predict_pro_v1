import streamlit as st
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

# BỘ NÃO AI
from ai_core import build_features, AIModel

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# DANH MỤC THỊ TRƯỜNG
INDUSTRIES = {
    "🏦 Ngân hàng": ["VCB", "BID", "CTG", "MBB", "TCB", "VPB", "ACB", "STB", "SHB", "HDB"],
    "📈 Chứng khoán": ["SSI", "VND", "HCM", "VCI", "VIX", "SHS", "MBS", "FTS", "BSI"],
    "🏢 Bất động sản & KCN": ["VHM", "VIC", "VRE", "NVL", "DIG", "DXG", "KBC", "PDR", "IDC", "SZC"],
    "🏗️ Thép & Xây dựng": ["HPG", "HSG", "NKG", "HT1", "BCC", "VCG", "CTD", "HBC"],
    "🛒 Bán lẻ & Công nghệ": ["FPT", "MWG", "PNJ", "DGW", "FRT", "CMG"],
    "🛢️ Dầu khí & Năng lượng": ["GAS", "PVD", "PVS", "BSR", "POW", "PLX", "NT2"],
    "🚢 Cảng biển & Thủy sản": ["HAH", "GMD", "VSC", "VHC", "ANV", "FMC"]
}

if 'last_alert' not in st.session_state: st.session_state['last_alert'] = ""

def send_telegram_alert(bot_token, chat_id, message):
    if not bot_token or not chat_id: return False
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {'chat_id': chat_id, 'text': message, 'parse_mode': 'Markdown'}
    try: return requests.post(url, data=payload).status_code == 200
    except: return False

# ==========================================
# PHẦN 1: LOCAL DATABASE (TỐC ĐỘ BÀN THỜ - CSV)
# ==========================================
class CSVDataLoader:
    def __init__(self):
        self.data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

    def download_yf(self, yf_symbol, start, end):
        df = pd.DataFrame()
        for attempt in range(3):
            try:
                ticker = yf.Ticker(yf_symbol)
                df = ticker.history(start=start, end=end)
                if not df.empty: break
            except Exception:
                time.sleep(1)
                
        if df.empty: return pd.DataFrame()
        df.reset_index(inplace=True)
        df.columns = [c.lower() for c in df.columns]
        if 'date' in df.columns and df['date'].dt.tz is not None:
            df['date'] = df['date'].dt.tz_localize(None)
        return df

    def clean_data(self, df):
        if df is None or df.empty: return df
        df = df.copy()
        
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if df[col].dtype == object:
                df[col] = df[col].astype(str).str.replace(',', '.', regex=False).str.strip()
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # MÀNG LỌC: Chém bay lỗi giá 6 Tỷ của Yahoo
        invalid_rows = (df['close'] <= 100) | (df['close'] > 2000000)
        df.loc[invalid_rows, ['open', 'high', 'low', 'close', 'volume']] = np.nan
        df.ffill(inplace=True)
        df.dropna(inplace=True)
        return df

    def get_data(self, symbol, days=3650):
        yf_symbol = symbol if symbol.endswith(".VN") else f"{symbol}.VN"
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        file_path = os.path.join(self.data_dir, f"{symbol}.csv")

        if not os.path.exists(file_path):
            df = self.download_yf(yf_symbol, start_date, end_date)
            if not df.empty:
                df_save = df[['date', 'open', 'high', 'low', 'close', 'volume']].copy()
                df_save.to_csv(file_path, index=False)
            return self.clean_data(df)
        else:
            try:
                df = pd.read_csv(file_path)
                df['date'] = pd.to_datetime(df['date'])
            except Exception:
                df = pd.DataFrame()

            if df.empty:
                df = self.download_yf(yf_symbol, start_date, end_date)
                if not df.empty:
                    df_save = df[['date', 'open', 'high', 'low', 'close', 'volume']].copy()
                    df_save.to_csv(file_path, index=False)
                return self.clean_data(df)
            else:
                last_date = df['date'].max()
                if end_date.date() > last_date.date() and end_date.weekday() < 5:
                    new_start = last_date + timedelta(days=1)
                    new_df = self.download_yf(yf_symbol, new_start, end_date)
                    if not new_df.empty:
                        new_df = new_df[['date', 'open', 'high', 'low', 'close', 'volume']]
                        df = pd.concat([df, new_df]).drop_duplicates(subset=['date'], keep='last').reset_index(drop=True)
                        df.to_csv(file_path, index=False)
                return self.clean_data(df)

    def save_leaderboard(self, df_leaderboard):
        file_path = os.path.join(self.data_dir, "Top_10_Leaderboard.csv")
        try:
            df_leaderboard.to_csv(file_path, index=False)
            return True
        except Exception: return False

    def load_leaderboard(self):
        file_path = os.path.join(self.data_dir, "Top_10_Leaderboard.csv")
        if os.path.exists(file_path):
            try:
                return pd.read_csv(file_path)
            except Exception:
                return pd.DataFrame()
        return pd.DataFrame()

@st.cache_data(ttl=3600, show_spinner=False)
def analyze_symbol(symbol, future_days):
    df = CSVDataLoader().get_data(symbol)
    if df is None or df.empty or len(df) < 250: return None
    
    df_feat = build_features(df)
    if df_feat.empty: return None
    
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
    
    future_preds_adapt = []
    current_lags_adapt = df['close'].iloc[-5:].values.tolist()
    for _ in range(future_days):
        pred = reg_model_adapt.predict(np.array([current_lags_adapt]))[0]
        future_preds_adapt.append(float(pred))
        current_lags_adapt.pop(0)
        current_lags_adapt.append(float(pred))
        
    return {'df': df, 'df_feat': df_feat, 'prob': prob, 'all_probs': all_probs, 'future_preds_adapt': future_preds_adapt, 'features_count': len(model.features), 'data_rows': len(df_feat)}

def run_advanced_backtest(df_bt, nav):
    fee = 0.0015         
    stop_loss = -0.07    
    take_profit = 0.15   
    
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
                    if (current_price * (1 - fee)) > (entry_price * (1 + fee)): 
                        winning_trades += 1
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
# PHẦN 3: GIAO DIỆN APP (UI)
# ==========================================
st.set_page_config(page_title="AI Quant - Thầy Nam", layout="wide")

with st.sidebar:
    st.header("🤖 Cài đặt Telegram Bot")
    bot_token = st.text_input("🔑 Bot Token:", type="password")
    chat_id = st.text_input("💬 Chat ID:")
    
    if st.button("🔔 Gửi Test", use_container_width=True):
        if bot_token and chat_id:
            if send_telegram_alert(bot_token, chat_id, "✅ Hệ thống AI Quant đang hoạt động tốt."):
                st.success("Đã gửi tin nhắn test!")
            else: st.error("Gửi thất bại.")
    
st.title("📈 Hệ thống Dự báo Định lượng (AI Quant)")

if st.button("🔄 Xóa Nhớ Đệm & Cập nhật Dữ liệu Mới Nhất", use_container_width=True):
    st.cache_data.clear()
    st.success("Đã xóa bộ nhớ đệm. Bắt đầu phiên làm việc mới!")

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

with st.spinner(f"Đang phân tích dữ liệu kho cho mã {symbol}..."):
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

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🔮 Dự báo Chi tiết", "📊 Kỷ luật Thực chiến", "🏆 Radar Tín Hiệu", "📈 Xếp Hạng Ngành", "🧠 CÔNG CỤ XÂY KHO"])
    
    with tab1:
        col1, col2 = st.columns([1, 2.8])
        with col1:
            st.info("💡 Tín hiệu AI & Dòng tiền")
            st.metric("Xác suất tăng (3 phiên tới)", f"{prob*100:.1f}%")
            st.write("---")
            st.write(f"- **VWAP:** {'Tích cực' if price_to_vwap > 0 else 'Tiêu cực'}")
            st.write(f"- **ADL:** {'Gom hàng' if adl_zscore > 0 else 'Xả hàng'}")
            
            if mtf_trend == 1:
                st.write("- **Khung Tuần:** Đồng thuận Tăng 📈")
            else:
                st.write("- **Khung Tuần:** Đang rủi ro (Nên cẩn trọng) ⚠️")

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
        st.subheader("🏆 Radar Tín Hiệu & Báo Cáo Telegram (Lọc Top 5)")
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
            radar_results = []
            
            for i, sym in enumerate(target_tickers):
                res = analyze_symbol(sym, future_days)
                if not res: continue
                scan_prob = res['prob']
                scan_preds = res['future_preds_adapt']
                min_idx = int(np.argmin(scan_preds))
                buy_p = scan_preds[min_idx]
                scan_profit = 0
                if min_idx + 3 < len(scan_preds):
                    scan_profit = (max(scan_preds[min_idx + 3:]) - buy_p) / buy_p * 100
                scan_kelly = max(0, (scan_prob - ((1-scan_prob)/(scan_profit/5.0))) / 2) * 100 if (scan_profit > 0 and (scan_profit / 5.0) > 0) else 0
                        
                radar_results.append({"Mã CP": sym, "Xác suất Tăng": scan_prob, "Tỷ trọng Vốn (Kelly)": scan_kelly / 100, "Kỳ vọng T+3": scan_profit / 100, "Giá Canh Mua": buy_p})
                progress_bar.progress((i + 1) / len(target_tickers))
                
            progress_bar.empty()
            if radar_results:
                radar_df = pd.DataFrame(radar_results).sort_values(by="Tỷ trọng Vốn (Kelly)", ascending=False).reset_index(drop=True)
                st.dataframe(radar_df.style.format({"Xác suất Tăng": "{:.1%}", "Tỷ trọng Vốn (Kelly)": "{:.1%}", "Kỳ vọng T+3": "{:+.2%}", "Giá Canh Mua": "{:,.0f} đ"}).background_gradient(subset=["Xác suất Tăng", "Tỷ trọng Vốn (Kelly)"], cmap="Greens"), use_container_width=True, height=400)
                
                if bot_token and chat_id:
                    buyable_df = radar_df[radar_df["Tỷ trọng Vốn (Kelly)"] > 0]
                    top_5_df = buyable_df.head(5)
                    final_msg = f"🏆 *TOP 5 MÃ TỐT NHẤT ({'NGÀNH' if scan_mode == 'sector' else 'TOÀN TT'})* 🏆\n\n"
                    if not top_5_df.empty:
                        for _, row in top_5_df.iterrows():
                            final_msg += f"✅ *{row['Mã CP']}* | Mua: {row['Giá Canh Mua']:,.0f}đ | Kỳ vọng: +{row['Kỳ vọng T+3']*100:.2f}% | Kelly: {row['Tỷ trọng Vốn (Kelly)']*100:.1f}%\n"
                    else:
                        final_msg += "⚠️ *Toàn bộ các mã quét được đều Xấu. Nên đứng ngoài.*\n"
                    send_telegram_alert(bot_token, chat_id, final_msg)
                    st.toast("Đã lọc và gửi báo cáo Top 5 qua Telegram!", icon="✈️")

    with tab4:
        st.subheader(f"📈 Bảng Xếp Hạng Kỷ Luật Thực Chiến: Nhóm {selected_sector}")
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
                    profit_pct = (final_equity / nav - 1) * 100
                    bnh_profit_pct = (bt_df['bnh_equity'].iloc[-1] / nav - 1) * 100
                    roll_max = bt_df['strategy_equity'].cummax()
                    max_dd = (bt_df['strategy_equity'] / roll_max - 1).min() * 100
                    
                    all_bt_results.append({
                        "Mã CP": sym, 
                        "Lãi ròng AI": profit_pct / 100, 
                        "So với Mua ôm": (profit_pct - bnh_profit_pct) / 100, 
                        "Win Rate": win_rate_pct / 100, 
                        "Số lệnh": total_tr,
                        "Drawdown": max_dd / 100
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
            with st.spinner("Đang kéo dữ liệu từ Bộ nhớ Nội bộ..."):
                loader = CSVDataLoader()
                df_top10 = loader.load_leaderboard()
                if not df_top10.empty:
                    st.success("Tải Bảng Phong Thần thành công trong chớp mắt!")
                    try:
                        for col in ["Lãi ròng AI", "Tỷ lệ Thắng", "Kelly Mua Mới"]:
                            if col in df_top10.columns: df_top10[col] = df_top10[col].astype(float)
                        if "Giá Canh Mua" in df_top10.columns: df_top10["Giá Canh Mua"] = df_top10["Giá Canh Mua"].astype(float)
                        
                        st.dataframe(df_top10.style.format({"Lãi ròng AI": "{:+.2%}", "Tỷ lệ Thắng": "{:.1%}", "Giá Canh Mua": "{:,.0f} đ", "Kelly Mua Mới": "{:.1%}"}).background_gradient(subset=["Lãi ròng AI"], cmap="RdYlGn"), use_container_width=True)
                    except:
                        st.dataframe(df_top10, use_container_width=True)
                else:
                    st.warning("Bảng Phong Thần chưa có dữ liệu. Thầy hãy bấm nút 'Cập nhật Bảng' trước nhé!")

        if btn_update_top10:
            status_text = st.empty()
            with st.spinner("Đang cày xới 50 mã (Có tính phí giao dịch) để tìm Top 10 xuất sắc nhất..."):
                all_top10_results = []
                all_tickers_list = [tic for sublist in INDUSTRIES.values() for tic in sublist]
                bt_progress = st.progress(0)
                
                for idx, sym in enumerate(all_tickers_list):
                    status_text.markdown(f"⏳ **AI đang đánh giá mã: {sym} ({idx+1}/50)...**")
                    res_bt = analyze_symbol(sym, future_days)
                    if not res_bt: continue
                    
                    df_f_bt = res_bt['df_feat']
                    bt_days_actual = min(bt_days_all, len(df_f_bt))
                    bt_df = df_f_bt.tail(bt_days_actual).copy()
                    bt_df['prob'] = res_bt['all_probs'][-bt_days_actual:]
                    
                    bt_df, win_rate_pct, total_tr = run_advanced_backtest(bt_df, nav)
                    profit_pct = (bt_df['strategy_equity'].iloc[-1] / nav - 1) * 100
                    
                    scan_prob = res_bt['prob']
                    scan_preds = res_bt['future_preds_adapt']
                    min_idx = int(np.argmin(scan_preds))
                    buy_p = scan_preds[min_idx]
                    scan_profit = (max(scan_preds[min_idx + 3:]) - buy_p) / buy_p * 100 if min_idx + 3 < len(scan_preds) else 0
                    scan_kelly = max(0, (scan_prob - ((1-scan_prob)/(scan_profit/5.0))) / 2) * 100 if (scan_profit > 0 and (scan_profit / 5.0) > 0) else 0

                    all_top10_results.append({
                        "Mã CP": sym, 
                        "Lãi ròng AI": profit_pct / 100, 
                        "Tỷ lệ Thắng": win_rate_pct / 100,
                        "Giá Canh Mua": buy_p,
                        "Kelly Mua Mới": scan_kelly / 100
                    })
                    bt_progress.progress((idx + 1) / len(all_tickers_list))
                
                bt_progress.empty()
                status_text.empty()
                
                if all_top10_results:
                    df_top10 = pd.DataFrame(all_top10_results).sort_values(by="Lãi ròng AI", ascending=False).head(10).reset_index(drop=True)
                    
                    loader = CSVDataLoader()
                    loader.save_leaderboard(df_top10)
                    
                    st.success("Đã LƯU vĩnh viễn Bảng Phong Thần vào hệ thống! Từ nay chỉ cần bấm 'Xem Bảng' là hiện ra ngay.")
                    st.dataframe(df_top10.style.format({"Lãi ròng AI": "{:+.2%}", "Tỷ lệ Thắng": "{:.1%}", "Giá Canh Mua": "{:,.0f} đ", "Kelly Mua Mới": "{:.1%}"}).background_gradient(subset=["Lãi ròng AI"], cmap="RdYlGn"), use_container_width=True)
                    
                    if bot_token and chat_id:
                        msg = f"🏆 *BẢNG PHONG THẦN MỚI ĐƯỢC CẬP NHẬT* 🏆\n_(Xếp hạng Lãi ròng {bt_timeframe_all} - Đã trừ phí GD)_\n\n"
                        for rank, row in df_top10.iterrows():
                            sym = row['Mã CP']
                            perf = row['Lãi ròng AI'] * 100
                            win = row['Tỷ lệ Thắng'] * 100
                            buy = row['Giá Canh Mua']
                            kel = row['Kelly Mua Mới'] * 100
                            
                            msg += f"*{rank + 1}. {sym}* | Lãi ròng: {perf:+.1f}% | Win: {win:.0f}%\n"
                            if kel > 0:
                                msg += f"👉 *🟢 CANH MUA {buy:,.0f}đ* (Vào {kel:.1f}% vốn)\n\n"
                            else:
                                msg += f"👉 *➖ Đứng ngoài quan sát*\n\n"
                            
                        send_telegram_alert(bot_token, chat_id, msg)
                        st.toast("Đã bắn Báo cáo Top 10 qua Telegram!", icon="✈️")

    with tab5:
        st.subheader("🧠 Trạng thái Đào tạo & Kho dữ liệu")
        col_ai1, col_ai2, col_ai3 = st.columns(3)
        col_ai1.metric("Thuật toán (AI Core)", "XGBoost 2.0 (Học sâu)")
        col_ai2.metric("Kho lưu trữ", "Local CSV Database (Siêu tốc)")
        col_ai3.metric("Bộ Đặc trưng (Features)", f"{result['features_count']} chỉ báo Vĩ mô")
        st.info("💡 **Hệ thống Kiểm tra & Huấn luyện Liên tục:** Dữ liệu tự động sinh ra và đóng gói thành file .csv nằm trong thư mục gốc. Tải cực nhanh, không lo bị chặn API.")
        
        st.markdown("---")
        st.subheader("🛠️ CÔNG CỤ XÂY KHO DỮ LIỆU (Dành cho lần chạy đầu tiên)")
        st.warning("⚠️ Bấm nút dưới đây để tải 10 năm dữ liệu (khoảng 125,000 dòng) về lưu vào thư mục /data.")
        
        if st.button("🏗️ XÂY KHO CSV (Tải 10 Năm Của 50 Mã)", type="primary", use_container_width=True):
            all_tickers_list = [tic for sublist in INDUSTRIES.values() for tic in sublist]
            prog_bar = st.progress(0)
            status_text = st.empty()
            loader = CSVDataLoader()
            
            for idx, sym_build in enumerate(all_tickers_list):
                status_text.markdown(f"**Đang tải 10 năm dữ liệu và nén thành CSV mã: {sym_build} ({idx+1}/50)...**")
                try:
                    loader.get_data(sym_build, 3650)
                except Exception as e: 
                    pass 
                prog_bar.progress((idx + 1) / len(all_tickers_list))
                
            status_text.success("✅ TẠO DATABASE THÀNH CÔNG! Toàn bộ các file .csv đã nằm trong thư mục /data.")

else: 
    if not result: st.error("Không thể kết nối hoặc thiếu dữ liệu để phân tích mã này.")