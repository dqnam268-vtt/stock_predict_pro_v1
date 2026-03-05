import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from src.data_loader import DataLoader
from src.features import build_features
from src.predictor import AIModel

st.set_page_config(page_title="AI Quant - Thầy Nam", layout="wide")
st.title("📈 Hệ thống Dự báo Định lượng (AI Quant)")

# 1. GIAO DIỆN TÙY CHỈNH MỚI
col_sel1, col_sel2 = st.columns(2)

with col_sel1:
    # Bổ sung nhóm Ngân hàng và Chứng khoán
    tickers = ["GAS", "HT1", "VCB", "MBB", "BID", "SSI", "VND", "HCM", "FPT"]
    symbol = st.selectbox("Chọn mã cổ phiếu cần dự báo:", tickers)

with col_sel2:
    # Combobox chọn chu kỳ tối ưu (Tháng, Quý, Năm)
    timeframe = st.selectbox(
        "Dò tìm Điểm Mua/Bán tối ưu theo chu kỳ:",
        ["Theo Tháng (21 phiên)", "Theo Quý (63 phiên)", "Theo Năm (252 phiên)"]
    )

# Từ điển ánh xạ chu kỳ ra số phiên giao dịch thực tế
window_dict = {
    "Theo Tháng (21 phiên)": 21,
    "Theo Quý (63 phiên)": 63,
    "Theo Năm (252 phiên)": 252
}
window = window_dict[timeframe]

# 2. XỬ LÝ DỮ LIỆU VÀ AI
loader = DataLoader()
with st.spinner(f"Đang tải và tính toán dữ liệu {symbol}..."):
    df = loader.get_data(symbol)

if not df.empty:
    df_feat = build_features(df)
    model = AIModel()
    model.train(df_feat)
    
    latest_data = df_feat.tail(1)
    prob = model.predict_prob(latest_data)[0]
    
    # --- TOÁN HỌC: TÌM CỰC TRỊ ĐỊA PHƯƠNG ---
    # Sử dụng cửa sổ trượt (rolling window) có tâm (center=True) 
    # để tìm đỉnh/đáy cục bộ trong khoảng thời gian đã chọn.
    df['local_max'] = df['close'] == df['close'].rolling(window=window, center=True).max()
    df['local_min'] = df['close'] == df['close'].rolling(window=window, center=True).min()
    
    # 3. HIỂN THỊ DASHBOARD NÂNG CẤP
    col1, col2 = st.columns([1, 2.5])
    
    with col1:
        st.info("💡 Tín hiệu AI (XGBoost) Hiện tại")
        st.metric(label=f"Xác suất {symbol} tăng >1.5% (3 phiên tới)", value=f"{prob*100:.1f}%")
        st.write("---")
        st.write("**Thông số tham chiếu:**")
        st.write(f"- Khung dò tìm: **{window} ngày**")
        st.write("- Mô hình: **XGBoost Classifier**")
        st.write("- Chỉ số: Hurst, Z-Score, MACD")
        
    with col2:
        st.subheader(f"Biểu đồ Hành vi Giá & Cực trị - {symbol}")
        
        # Chuyển sang dùng Matplotlib để vẽ biểu đồ điểm (scatter) tối ưu
        fig, ax = plt.subplots(figsize=(10, 5))
        
        # Vẽ đường giá đóng cửa
        ax.plot(df['date'], df['close'], label='Giá đóng cửa', color='#1f77b4', linewidth=1.5, alpha=0.7)
        
        # Vẽ điểm Bán (Đỉnh - Tam giác ngược màu đỏ)
        sell_points = df[df['local_max']]
        ax.scatter(sell_points['date'], sell_points['close'], color='red', label='Điểm Bán Tối Ưu', marker='v', s=80, zorder=5)
        
        # Vẽ điểm Mua (Đáy - Tam giác màu xanh)
        buy_points = df[df['local_min']]
        ax.scatter(buy_points['date'], buy_points['close'], color='green', label='Điểm Mua Tối Ưu', marker='^', s=80, zorder=5)
        
        ax.set_ylabel("Giá (VND)")
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.5)
        
        # Hiển thị biểu đồ lên Streamlit
        st.pyplot(fig)
else:
    st.error("Không thể kết nối dữ liệu. Thầy vui lòng kiểm tra lại mạng hoặc mã cổ phiếu.")