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

# 1. GIAO DIỆN
col_sel1, col_sel2, col_sel3 = st.columns(3)
with col_sel1:
    tickers = ["GAS", "HT1", "VCB", "MBB", "BID", "SSI", "VND", "HCM", "FPT", "VIX"]
    symbol = st.selectbox("🎯 Chọn mã cổ phiếu:", tickers)
with col_sel2:
    timeframe = st.selectbox("🔙 Dò tìm Cực trị:", ["Theo Tuần (5 phiên)", "Theo Tháng (21 phiên)", "Theo Quý (63 phiên)", "Theo Năm (252 phiên)"], index=1)
with col_sel3:
    future_horizon = st.selectbox("🔮 AI Dự báo Tương lai:", ["1 Tuần tới (5 phiên)", "1 Tháng tới (21 phiên)"], index=1)

window_dict = {"Theo Tuần (5 phiên)": 5, "Theo Tháng (21 phiên)": 21, "Theo Quý (63 phiên)": 63, "Theo Năm (252 phiên)": 252}
window = window_dict[timeframe]
future_days = 5 if "Tuần" in future_horizon else 21
offset = 5 

# 2. XỬ LÝ DỮ LIỆU
loader = DataLoader()
with st.spinner(f"Đang đồng bộ dữ liệu thị trường và xuất báo cáo cho {symbol}..."):
    df = loader.get_data(symbol)

if not df.empty and len(df) > 50:
    df_feat = build_features(df)
    
    # Huấn luyện Classification
    model = AIModel()
    model.train(df_feat)
    latest_row = df_feat.tail(1)
    prob = model.predict_prob(latest_row)[0]
    
    # Trích xuất dữ liệu Vi mô (Dòng tiền)
    current_price = latest_row['close'].values[0]
    price_to_vwap = latest_row['price_to_vwap'].values[0]
    adl_zscore = latest_row['adl_zscore'].values[0]
    
    # Cực trị & MA
    df['local_max'] = df['close'] == df['close'].rolling(window=window, center=True).max()
    df['local_min'] = df['close'] == df['close'].rolling(window=window, center=True).min()
    df['ma50'] = df['close'].rolling(window=50).mean()
    
    # AR Model (Hồi quy)
    df_reg = df[['close']].copy()
    for i in range(1, 6):
        df_reg[f'lag_{i}'] = df_reg['close'].shift(i)
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
        
    last_date = df['date'].iloc[-1]
    future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=future_days)

    # Logic T+3
    future_min_idx = int(np.argmin(future_preds_adapt))
    buy_date = future_dates[future_min_idx]
    buy_price = future_preds_adapt[future_min_idx]
    
    can_sell_T3 = False
    if future_min_idx + 3 < len(future_preds_adapt):
        valid_sell_slice = future_preds_adapt[future_min_idx + 3:]
        valid_sell_dates = future_dates[future_min_idx + 3:]
        offset_idx = int(np.argmax(valid_sell_slice))
        future_max_idx = future_min_idx + 3 + offset_idx
        
        sell_date = future_dates[future_max_idx]
        sell_price = future_preds_adapt[future_max_idx]
        profit_pct = (sell_price - buy_price) / buy_price * 100
        can_sell_T3 = True

    # 3. HIỂN THỊ DASHBOARD
    col1, col2 = st.columns([1, 2.5])
    
    with col1:
        st.info("💡 Tín hiệu AI & Dòng tiền")
        st.metric("Xác suất tăng (3 phiên tới)", f"{prob*100:.1f}%")
        
        # Nhận định dòng tiền
        vwap_status = "Tích cực (Trên giá vốn cá mập)" if price_to_vwap > 0 else "Chiết khấu (Dưới giá vốn cá mập)"
        adl_status = "Dòng tiền đang Gom hàng" if adl_zscore > 0 else "Dòng tiền có dấu hiệu Xả"
        
        st.write("---")
        st.write(f"- **Khối lượng (VWAP):** {vwap_status}")
        st.write(f"- **Áp lực mua/bán (ADL):** {adl_status}")

    with col2:
        st.subheader(f"Biểu đồ Quỹ đạo AI - {symbol}")
        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.plot(df['date'].iloc[-100:], df['close'].iloc[-100:], label='Giá thực tế', color='#1f77b4', linewidth=2)
        ax.plot(future_dates, future_preds_adapt, label='AI Dự báo Tương lai', color='magenta', linestyle='--', linewidth=2.5)
        
        ax.scatter(buy_date, buy_price, color='lime', marker='^', s=180, label='MUA T+3', zorder=6, edgecolors='black')
        if can_sell_T3:
            ax.scatter(sell_date, sell_price, color='red', marker='v', s=180, label='BÁN T+3', zorder=6, edgecolors='black')
            ax.plot([buy_date, sell_date], [buy_price, sell_price], color='green', linestyle=':', linewidth=1.5)
        
        ax.set_ylabel("Giá (VND)")
        ax.legend(loc='upper left')
        ax.grid(True, linestyle=':', alpha=0.6)
        st.pyplot(fig)

    # ==========================================
    # 4. XUẤT BÁO CÁO DẠNG TEXT (MODULE MỚI)
    # ==========================================
    st.markdown("---")
    st.subheader("📝 BẢN GHI NHỚ GIAO DỊCH (TRADE PLAN)")
    
    # Xây dựng logic kết luận
    conclusion = ""
    if prob > 0.6 and can_sell_T3 and profit_pct > 1.5 and adl_zscore > 0:
        conclusion = "🌟 **RẤT TÍCH CỰC:** Hội tụ đủ yếu tố kỹ thuật, AI dự báo xu hướng tăng, dòng tiền đang gom hàng. Có thể xem xét giải ngân."
    elif prob > 0.5 and can_sell_T3 and profit_pct > 0:
        conclusion = "⚖️ **TRUNG LẬP / CÓ THỂ LƯỚT SÓNG:** Có biên lợi nhuận T+3 nhưng tín hiệu dòng tiền chưa quá mạnh. Nên đi vốn nhỏ."
    else:
        conclusion = "⛔ **RỦI RO / ĐỨNG NGOÀI:** Biên lợi nhuận mỏng hoặc AI báo rủi ro giảm giá cao. Khuyến nghị quan sát thêm."

    report_text = f"""
    **1. Tổng quan mã {symbol}:**
    - **Giá đóng cửa hiện tại:** {current_price:,.0f} đ
    - **Xác suất AI đánh giá tăng giá:** {prob*100:.1f}%
    - **Trạng thái dòng tiền lớn:** {adl_status}

    **2. Kế hoạch lướt sóng (Dựa trên ngoại suy T+3):**
    - 🟢 **Điểm chờ MUA:** Quanh vùng **{buy_price:,.0f} đ** (Thời gian dự kiến: {buy_date.strftime('%d/%m/%Y')})
    """
    
    if can_sell_T3:
        report_text += f"""- 🔴 **Điểm chờ BÁN:** Quanh vùng **{sell_price:,.0f} đ** (Thời gian dự kiến: {sell_date.strftime('%d/%m/%Y')})
    - 🎯 **Biên lợi nhuận kỳ vọng:** **{profit_pct:+.2f}%**
        """
    else:
        report_text += f"""- ⚠️ **Lưu ý:** Khung thời gian dự báo hiện tại không đủ để hoàn thành vòng quay T+3. Không có điểm bán tối ưu."""

    report_text += f"\n\n**3. Kết luận từ AI Quant:**\n{conclusion}"

    # Hiển thị báo cáo trong một khối màu dễ nhìn
    st.info(report_text)

else:
    st.error("Không thể kết nối dữ liệu hoặc dữ liệu quá ngắn để huấn luyện.")