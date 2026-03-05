import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from src.data_loader import DataLoader
from src.features import build_features
from src.predictor import AIModel

st.set_page_config(page_title="AI Quant - Thầy Nam", layout="wide")
st.title("📈 Hệ thống Dự báo Định lượng (AI Quant)")

if st.button("🔄 Cập nhật dữ liệu & Huấn luyện lại thuật toán (Real-time)", use_container_width=True):
    st.cache_data.clear()

# 1. GIAO DIỆN TÙY CHỈNH
col_sel1, col_sel2, col_sel3 = st.columns(3)

with col_sel1:
    # ĐÃ THÊM MÃ VIX VÀO ĐÂY
    tickers = ["GAS", "HT1", "VCB", "MBB", "BID", "SSI", "VND", "HCM", "FPT", "VIX"]
    symbol = st.selectbox("🎯 Chọn mã cổ phiếu:", tickers)

with col_sel2:
    timeframe = st.selectbox(
        "🔙 Dò tìm Cực trị (Quá khứ):",
        ["Theo Tuần (5 phiên)", "Theo Tháng (21 phiên)", "Theo Quý (63 phiên)", "Theo Năm (252 phiên)"],
        index=1
    )

with col_sel3:
    future_horizon = st.selectbox(
        "🔮 AI Dự báo Xu hướng (Tương lai):",
        ["1 Tuần tới (5 phiên)", "1 Tháng tới (21 phiên)"],
        index=1
    )

window_dict = {"Theo Tuần (5 phiên)": 5, "Theo Tháng (21 phiên)": 21, "Theo Quý (63 phiên)": 63, "Theo Năm (252 phiên)": 252}
window = window_dict[timeframe]
future_days = 5 if "Tuần" in future_horizon else 21
offset = 5 # Điểm neo so sánh (Cách đây 1 tuần)

# 2. XỬ LÝ DỮ LIỆU & HUẤN LUYỆN AI
loader = DataLoader()
with st.spinner(f"Đang đồng bộ dữ liệu thị trường và phân tích sai số AI cho {symbol}..."):
    df = loader.get_data(symbol)

if not df.empty and len(df) > 50:
    df_feat = build_features(df)
    
    # AI Phân loại cơ bản
    model = AIModel()
    model.train(df_feat)
    prob = model.predict_prob(df_feat.tail(1))[0]
    
    df['local_max'] = df['close'] == df['close'].rolling(window=window, center=True).max()
    df['local_min'] = df['close'] == df['close'].rolling(window=window, center=True).min()
    df['ma50'] = df['close'].rolling(window=50).mean()
    
    # --- CHUẨN BỊ MA TRẬN CHO AUTOREGRESSIVE MODEL ---
    df_reg = df[['close']].copy()
    for i in range(1, 6):
        df_reg[f'lag_{i}'] = df_reg['close'].shift(i)
    df_reg = df_reg.dropna()
    features_reg = [f'lag_{5}', f'lag_{4}', f'lag_{3}', f'lag_{2}', f'lag_{1}']
    
    # ==========================================================
    # NÃO 1: MÔ HÌNH CƠ SỞ (BASELINE) - BỊ BỊT MẮT 5 NGÀY TRƯỚC
    # ==========================================================
    df_base = df_reg.iloc[:-offset] # Cắt bỏ 5 ngày gần nhất
    X_base = df_base[features_reg]
    y_base = df_base['close']
    
    reg_model_base = XGBRegressor(n_estimators=150, max_depth=4, learning_rate=0.05, random_state=42)
    reg_model_base.fit(X_base, y_base)
    
    # Ngoại suy từ T-5
    future_preds_base = []
    current_lags_base = df['close'].iloc[-offset-5 : -offset].values.tolist()
    
    for _ in range(offset + future_days):
        pred = reg_model_base.predict(np.array([current_lags_base]))[0]
        future_preds_base.append(float(pred))
        current_lags_base.pop(0)
        current_lags_base.append(float(pred))
        
    date_T_minus_5 = df['date'].iloc[-offset]
    base_dates = pd.bdate_range(start=date_T_minus_5, periods=offset + future_days)

    # ==========================================================
    # NÃO 2: MÔ HÌNH THÍCH NGHI (ADAPTIVE) - HỌC ĐẾN PHÚT HIỆN TẠI
    # ==========================================================
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
        
    last_date = df['date'].iloc[-1]
    future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=future_days)

    # --- ÁP DỤNG ĐIỀU KIỆN T+3 (TRÊN ĐƯỜNG THÍCH NGHI) ---
    future_min_idx = int(np.argmin(future_preds_adapt))
    buy_date = future_dates[future_min_idx]
    buy_price = future_preds_adapt[future_min_idx]
    
    if future_min_idx + 3 < len(future_preds_adapt):
        valid_sell_slice = future_preds_adapt[future_min_idx + 3:]
        valid_sell_dates = future_dates[future_min_idx + 3:]
        
        offset_idx = int(np.argmax(valid_sell_slice))
        future_max_idx = future_min_idx + 3 + offset_idx
        
        sell_date = future_dates[future_max_idx]
        sell_price = future_preds_adapt[future_max_idx]
        profit_pct = (sell_price - buy_price) / buy_price * 100
        can_sell_T3 = True
    else:
        can_sell_T3 = False

    # 3. HIỂN THỊ DASHBOARD
    col1, col2 = st.columns([1, 2.5])
    
    with col1:
        st.info("💡 So sánh & Tối ưu Thuật toán")
        
        # Tính toán sai số giữa AI cũ và thực tế (Trong 5 ngày qua)
        actual_5days = df['close'].iloc[-offset:].values
        predicted_5days = future_preds_base[:offset]
        rmse = np.sqrt(np.mean((actual_5days - predicted_5days)**2))
        
        st.metric(label="Sai số Dự báo 5 ngày qua (RMSE)", value=f"{rmse:,.0f} đ", delta="Độ lệch so với thực tế", delta_color="inverse")
        st.write("---")
        
        st.success(f"**Đề xuất T+3 (Dựa trên AI Thích nghi):**")
        st.write(f"🟢 **Canh Mua:** {buy_price:,.0f} đ")
        if can_sell_T3:
            st.write(f"🔴 **Canh Bán:** {sell_price:,.0f} đ")
            st.metric("Biên lợi nhuận:", f"{profit_pct:+.2f}%")
        else:
            st.warning("⏳ Khung thời gian hẹp, không đủ T+3.")
        
    with col2:
        st.subheader(f"Biểu đồ Đối chiếu Quỹ đạo AI - {symbol}")
        
        fig, ax = plt.subplots(figsize=(10, 5))
        
        # 1. Quá khứ thực tế
        ax.plot(df['date'].iloc[-100:], df['close'].iloc[-100:], label='Giá thực tế (Actual)', color='#1f77b4', linewidth=2)
        
        # 2. Đường Cơ sở (AI T-5 Cứng)
        ax.plot(base_dates, future_preds_base, label='AI Gốc (Dự báo từ 5 ngày trước)', color='gray', linestyle='-.', linewidth=1.5, alpha=0.8)
        
        # 3. Đường Thích nghi (AI T0 Cập nhật)
        ax.plot(future_dates, future_preds_adapt, label='AI Thích nghi (Cập nhật Real-time)', color='magenta', linestyle='--', linewidth=2.5)
        
        # Vùng highlight sai số (Shaded area)
        intersect_dates = pd.bdate_range(start=date_T_minus_5, periods=offset)
        ax.fill_between(intersect_dates, actual_5days, predicted_5days, color='red', alpha=0.15, label='Vùng sai số (Error)')
        
        # Vẽ điểm Mua/Bán có điều kiện T+3
        ax.scatter(buy_date, buy_price, color='lime', marker='^', s=180, label='MUA T+3', zorder=6, edgecolors='black')
        if can_sell_T3:
            ax.scatter(sell_date, sell_price, color='red', marker='v', s=180, label='BÁN T+3', zorder=6, edgecolors='black')
            ax.plot([buy_date, sell_date], [buy_price, sell_price], color='green', linestyle=':', linewidth=1.5)
        
        ax.set_ylabel("Giá (VND)")
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
        ax.grid(True, linestyle=':', alpha=0.6)
        
        st.pyplot(fig)
else:
    st.error("Không thể kết nối dữ liệu hoặc dữ liệu quá ngắn để huấn luyện.")