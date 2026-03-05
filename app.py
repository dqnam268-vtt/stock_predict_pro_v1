import streamlit as st
from src.data_loader import DataLoader
from src.features import build_features
from src.predictor import AIModel

st.set_page_config(page_title="AI Quant - Thầy Nam", layout="wide")
st.title("📈 Hệ thống Dự báo Định lượng (AI Quant)")

# Cho phép chọn mã cổ phiếu
symbol = st.selectbox("Chọn mã cổ phiếu cần dự báo:", ["GAS", "HT1", "FPT", "VCB"])

# 1. Tải dữ liệu
loader = DataLoader()
with st.spinner(f"Đang tải dữ liệu {symbol} từ Yahoo Finance..."):
    df = loader.get_data(symbol)

if not df.empty:
    # 2. Xử lý toán học & Trích xuất đặc trưng
    df_feat = build_features(df)
    
    # 3. Huấn luyện AI (XGBoost)
    model = AIModel()
    model.train(df_feat)
    
    # 4. Dự báo xác suất cho phiên tiếp theo
    latest_data = df_feat.tail(1)
    prob = model.predict_prob(latest_data)[0]
    
    # 5. Hiển thị Dashboard
    col1, col2 = st.columns([1, 2])
    with col1:
        st.info("💡 Tín hiệu AI (XGBoost)")
        st.metric(label=f"Xác suất {symbol} tăng >1.5% (3 phiên tới)", value=f"{prob*100:.1f}%")
        st.write("---")
        st.write("**Các chỉ số toán học cấu thành:**")
        st.write("- Hệ số Hurst (Đo lường xu hướng)")
        st.write("- Z-Score (Chuẩn hóa phân phối)")
        st.write("- Realized Volatility (Độ biến động)")
        st.write("- MACD Histogram")
        
    with col2:
        st.subheader(f"Biểu đồ giá đóng cửa {symbol}")
        st.line_chart(df.set_index('date')['close'])
else:
    st.error("Không thể kết nối dữ liệu. Thầy vui lòng kiểm tra lại mã cổ phiếu hoặc mạng.")