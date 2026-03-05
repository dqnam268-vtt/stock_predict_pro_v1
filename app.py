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

# 1. GIAO DIỆN TÙY CHỈNH MỚI (Chia làm 3 cột cho cân đối)
col_sel1, col_sel2, col_sel3 = st.columns(3)

with col_sel1:
    # Đã bổ sung đầy đủ các mã Ngân hàng và Chứng khoán theo yêu cầu
    tickers = ["GAS", "HT1", "VCB", "MBB", "BID", "SSI", "VND", "HCM", "FPT"]
    symbol = st.selectbox("🎯 Chọn mã cổ phiếu:", tickers)

with col_sel2:
    # Bổ sung chu kỳ "Theo Tuần (5 phiên)"
    timeframe = st.selectbox(
        "🔙 Dò tìm Cực trị (Quá khứ):",
        ["Theo Tuần (5 phiên)", "Theo Tháng (21 phiên)", "Theo Quý (63 phiên)", "Theo Năm (252 phiên)"],
        index=1 # Mặc định chọn Theo Tháng
    )

with col_sel3:
    # Combobox chọn khung thời gian dự báo tương lai
    future_horizon = st.selectbox(
        "🔮 AI Dự báo Xu hướng (Tương lai):",
        ["1 Tuần tới (5 phiên)", "1 Tháng tới (21 phiên)"]
    )

# Từ điển ánh xạ số phiên
window_dict = {
    "Theo Tuần (5 phiên)": 5,
    "Theo Tháng (21 phiên)": 21,
    "Theo Quý (63 phiên)": 63,
    "Theo Năm (252 phiên)": 252
}
window = window_dict[timeframe]
future_days = 5 if "Tuần" in future_horizon else 21

# 2. XỬ LÝ DỮ LIỆU & HUẤN LUYỆN AI
loader = DataLoader()
with st.spinner(f"Đang tính toán ma trận đặc trưng và huấn luyện AI cho {symbol}..."):
    df = loader.get_data(symbol)

if not df.empty:
    df_feat = build_features(df)
    
    # AI Phân loại (Xác suất tăng/giảm)
    model = AIModel()
    model.train(df_feat)
    latest_data = df_feat.tail(1)
    prob = model.predict_prob(latest_data)[0]
    
    # Toán học: Tìm Cực trị địa phương (Quá khứ)
    df['local_max'] = df['close'] == df['close'].rolling(window=window, center=True).max()
    df['local_min'] = df['close'] == df['close'].rolling(window=window, center=True).min()
    df['ma50'] = df['close'].rolling(window=50).mean()
    
    # --- TOÁN HỌC: AI DỰ BÁO TƯƠNG LAI (Mô hình Tự hồi quy - Auto-Regressive) ---
    # Dùng 5 phiên gần nhất làm Feature để đoán phiên tiếp theo
    df_reg = df[['close']].copy()
    for i in range(1, 6):
        df_reg[f'lag_{i}'] = df_reg['close'].shift(i)
    df_reg = df_reg.dropna()
    
    X_reg = df_reg[[f'lag_{i}' for i in range(1, 6)]]
    y_reg = df_reg['close']
    
    # Huấn luyện XGBoost Regressor cho việc vẽ đường giá
    reg_model = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42)
    reg_model.fit(X_reg, y_reg)
    
    # Vòng lặp dự báo N ngày tới
    future_preds = []
    current_lags = df['close'].iloc[-5:].values[::-1].tolist() # Lấy 5 giá gần nhất [t-1, t-2, t-3, t-4, t-5]
    
    for _ in range(future_days):
        pred = reg_model.predict(np.array([current_lags]))[0]
        future_preds.append(pred)
        # Cập nhật mảng lag: Xóa ngày xa nhất, chèn dự báo mới vào đầu
        current_lags.pop()
        current_lags.insert(0, pred)
        
    # Tạo trục thời gian cho tương lai (Chỉ lấy ngày làm việc, bỏ T7, CN)
    last_date = df['date'].iloc[-1]
    future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=future_days)
    
    # 3. HIỂN THỊ DASHBOARD NÂNG CẤP
    col1, col2 = st.columns([1, 2.5])
    
    with col1:
        st.info("💡 Tín hiệu AI Hiện tại")
        st.metric(label=f"Xác suất {symbol} tăng >1.5% (3 phiên tới)", value=f"{prob*100:.1f}%")
        st.write("---")
        
        # Phân tích đề xuất tương lai
        future_min = min(future_preds)
        future_max = max(future_preds)
        st.success(f"**AI Đề xuất {future_horizon}:**")
        st.write(f"- 🟢 **Canh Mua quanh:** {future_min:,.0f} đ")
        st.write(f"- 🔴 **Canh Bán quanh:** {future_max:,.0f} đ")
        
    with col2:
        st.subheader(f"Biểu đồ Hành vi Giá & Dự báo AI - {symbol}")
        
        fig, ax = plt.subplots(figsize=(10, 5))
        
        # 1. Vẽ Quá khứ
        ax.plot(df['date'], df['close'], label='Giá thực tế', color='#1f77b4', linewidth=1.5, alpha=0.8)
        ax.plot(df['date'], df['ma50'], label='MA 50 ngày', color='darkorange', linewidth=1.2, linestyle='--', alpha=0.6)
        
        sell_points = df[df['local_max']]
        ax.scatter(sell_points['date'], sell_points['close'], color='red', label='Đỉnh (Quá khứ)', marker='v', s=60, zorder=5)
        
        buy_points = df[df['local_min']]
        ax.scatter(buy_points['date'], buy_points['close'], color='green', label='Đáy (Quá khứ)', marker='^', s=60, zorder=5)
        
        # 2. Vẽ Tương lai (Đường nét đứt màu tím)
        ax.plot(future_dates, future_preds, label='Quỹ đạo AI dự báo', color='magenta', linestyle='--', linewidth=2)
        
        # Tìm index của điểm Mua/Bán trong mảng tương lai
        future_min_idx = np.argmin(future_preds)
        future_max_idx = np.argmax(future_preds)
        
        # Đánh dấu điểm Mua/Bán Tương lai bằng Ngôi sao
        ax.scatter(future_dates[future_max_idx], future_preds[future_max_idx], color='purple', marker='*', s=150, label='Đề xuất BÁN (Tương lai)', zorder=6)
        ax.scatter(future_dates[future_min_idx], future_preds[future_min_idx], color='lime', marker='*', s=150, label='Đề xuất MUA (Tương lai)', zorder=6)
        
        ax.set_ylabel("Giá (VND)")
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1)) # Đẩy chú thích ra ngoài cho dễ nhìn
        ax.grid(True, linestyle=':', alpha=0.6)
        
        st.pyplot(fig)
else:
    st.error("Không thể kết nối dữ liệu. Thầy vui lòng kiểm tra lại mạng hoặc mã cổ phiếu.")