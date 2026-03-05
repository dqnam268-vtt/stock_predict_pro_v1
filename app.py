import sys
import os
# Ép Python nhận diện thư mục hiện tại làm gốc để tìm thư mục 'src'
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from xgboost import XGBRegressor
from src.data_loader import DataLoader
from src.features import build_features
from src.predictor import AIModel

# Thiết lập trang
st.set_page_config(page_title="AI Quant - Thầy Nam", layout="wide")
st.title("📈 Hệ thống Dự báo Định lượng (AI Quant)")

# Nút cập nhật Real-time
if st.button("🔄 Cập nhật dữ liệu & Huấn luyện lại thuật toán (Real-time)", use_container_width=True):
    st.cache_data.clear()

# ==========================================
# 1. GIAO DIỆN LỰA CHỌN TÙY CHỈNH
# ==========================================
col_sel1, col_sel2, col_sel3 = st.columns(3)
with col_sel1:
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
        "🔮 AI Dự báo Tương lai:", 
        ["1 Tuần tới (5 phiên)", "1 Tháng tới (21 phiên)"], 
        index=1
    )

window_dict = {"Theo Tuần (5 phiên)": 5, "Theo Tháng (21 phiên)": 21, "Theo Quý (63 phiên)": 63, "Theo Năm (252 phiên)": 252}
window = window_dict[timeframe]
future_days = 5 if "Tuần" in future_horizon else 21

# ==========================================
# 2. XỬ LÝ DỮ LIỆU & HUẤN LUYỆN AI
# ==========================================
loader = DataLoader()
with st.spinner(f"Đang đồng bộ dữ liệu Vĩ mô, Vi mô và vẽ biểu đồ tương tác cho {symbol}..."):
    df = loader.get_data(symbol)

if not df.empty and len(df) > 50:
    df_feat = build_features(df)
    
    # Huấn luyện mô hình Phân loại (Xác suất)
    model = AIModel()
    model.train(df_feat)
    latest_row = df_feat.tail(1)
    prob = model.predict_prob(latest_row)[0]
    
    # Trích xuất dữ liệu Vi mô & Vĩ mô
    current_price = latest_row['close'].values[0]
    price_to_vwap = latest_row['price_to_vwap'].values[0]
    adl_zscore = latest_row['adl_zscore'].values[0]
    
    if 'market_corr' in latest_row.columns:
        market_corr = latest_row['market_corr'].values[0]
        if market_corr > 0.6:
            corr_status = f"Đồng pha mạnh với VN-Index ({market_corr:.2f})"
        elif market_corr < -0.3:
            corr_status = f"Đi ngược bão VN-Index ({market_corr:.2f})"
        else:
            corr_status = f"Ít phụ thuộc VN-Index ({market_corr:.2f})"
    else:
        corr_status = "Không có dữ liệu VN-Index"
    
    # Chuẩn bị dữ liệu cho Mô hình Hồi quy (Auto-Regressive)
    df_reg = df[['close']].copy()
    for i in range(1, 6):
        df_reg[f'lag_{i}'] = df_reg['close'].shift(i)
    df_reg = df_reg.dropna()
    features_reg = [f'lag_{5}', f'lag_{4}', f'lag_{3}', f'lag_{2}', f'lag_{1}']
    
    X_adapt = df_reg[features_reg]
    y_adapt = df_reg['close']
    
    # Huấn luyện mô hình Hồi quy
    reg_model_adapt = XGBRegressor(n_estimators=150, max_depth=4, learning_rate=0.05, random_state=99)
    reg_model_adapt.fit(X_adapt, y_adapt)
    
    # Ngoại suy tương lai
    future_preds_adapt = []
    current_lags_adapt = df['close'].iloc[-5:].values.tolist()
    
    for _ in range(future_days):
        pred = reg_model_adapt.predict(np.array([current_lags_adapt]))[0]
        future_preds_adapt.append(float(pred))
        current_lags_adapt.pop(0)
        current_lags_adapt.append(float(pred))
        
    last_date = df['date'].iloc[-1]
    future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=future_days)

    # Logic Tối ưu hóa điều kiện T+3
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

    # ==========================================
    # 3. HIỂN THỊ DASHBOARD & BIỂU ĐỒ PLOTLY
    # ==========================================
    col1, col2 = st.columns([1, 2.5])
    
    with col1:
        st.info("💡 Tín hiệu AI & Dòng tiền")
        st.metric("Xác suất tăng (3 phiên tới)", f"{prob*100:.1f}%")
        st.write("---")
        st.write(f"- **Khối lượng (VWAP):** {'Tích cực' if price_to_vwap > 0 else 'Tiêu cực'}")
        st.write(f"- **Áp lực gom/xả (ADL):** {'Gom hàng' if adl_zscore > 0 else 'Xả hàng'}")
        st.write(f"- **Tương quan:** {corr_status}")

    with col2:
        st.subheader(f"Biểu đồ Tương tác & Quỹ đạo AI - {symbol}")
        
        # 🚀 Khởi tạo biểu đồ Plotly
        fig = go.Figure()
        
        # Vẽ đường giá quá khứ (lấy 150 phiên gần nhất cho nhẹ và dễ nhìn)
        fig.add_trace(go.Scatter(
            x=df['date'].iloc[-150:], 
            y=df['close'].iloc[-150:],
            mode='lines',
            name='Giá thực tế',
            line=dict(color='#1f77b4', width=2)
        ))
        
        # Vẽ đường dự báo tương lai
        fig.add_trace(go.Scatter(
            x=future_dates, 
            y=future_preds_adapt,
            mode='lines',
            name='AI Dự báo Tương lai',
            line=dict(color='magenta', width=2.5, dash='dash')
        ))
        
        # Điểm MUA T+3
        fig.add_trace(go.Scatter(
            x=[buy_date], y=[buy_price],
            mode='markers',
            name='Điểm MUA T+3',
            marker=dict(color='lime', symbol='triangle-up', size=16, line=dict(color='black', width=1))
        ))
        
        # Điểm BÁN T+3
        if can_sell_T3:
            fig.add_trace(go.Scatter(
                x=[sell_date], y=[sell_price],
                mode='markers',
                name='Điểm BÁN T+3',
                marker=dict(color='red', symbol='triangle-down', size=16, line=dict(color='black', width=1))
            ))
            
            # Đường nối Mua - Bán
            fig.add_trace(go.Scatter(
                x=[buy_date, sell_date], y=[buy_price, sell_price],
                mode='lines',
                name='Biên lợi nhuận',
                line=dict(color='green', width=1.5, dash='dot')
            ))
        
        # Cấu hình giao diện tương tác (Hover, Zoom, Range Slider)
        fig.update_layout(
            xaxis_title="Thời gian (Kéo thanh trượt bên dưới để Zoom)",
            yaxis_title="Giá (VND)",
            hovermode="x unified", # Bảng chi tiết hiện ra khi di chuột
            margin=dict(l=0, r=0, t=30, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis=dict(
                rangeslider=dict(visible=True), # Thanh trượt thời gian
                type="date"
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)

    # ==========================================
    # 4. XUẤT BÁO CÁO DẠNG TEXT (TRADE PLAN)
    # ==========================================
    st.markdown("---")
    st.subheader("📝 BẢN GHI NHỚ GIAO DỊCH (TRADE PLAN)")
    
    conclusion = ""
    if prob > 0.6 and can_sell_T3 and profit_pct > 1.5 and adl_zscore > 0:
        conclusion = "🌟 **RẤT TÍCH CỰC:** Hội tụ đủ yếu tố kỹ thuật, AI dự báo xu hướng tăng, dòng tiền đang gom hàng. Có thể xem xét giải ngân."
    elif prob > 0.5 and can_sell_T3 and profit_pct > 0:
        conclusion = "⚖️ **TRUNG LẬP / CÓ THỂ LƯỚT SÓNG:** Có biên lợi nhuận T+3 nhưng tín hiệu dòng tiền chưa quá mạnh. Nên đi vốn nhỏ."
    else:
        conclusion = "⛔ **RỦI RO / ĐỨNG NGOÀI:** Biên lợi nhuận mỏng hoặc AI báo rủi ro giảm giá cao. Khuyến nghị quan sát thêm."

    report_text = f"""
    **1. Tổng quan Vi mô & Vĩ mô mã {symbol}:**
    - **Giá đóng cửa hiện tại:** {current_price:,.0f} đ
    - **Tương quan Vĩ mô:** {corr_status}
    - **Trạng thái dòng tiền lớn:** {'Đang Gom hàng' if adl_zscore > 0 else 'Có dấu hiệu Xả'}
    - **Xác suất XGBoost đánh giá tăng giá:** {prob*100:.1f}%

    **2. Kế hoạch lướt sóng (Ngoại suy T+3):**
    - 🟢 **Điểm chờ MUA:** Quanh vùng **{buy_price:,.0f} đ** (Dự kiến: {buy_date.strftime('%d/%m/%Y')})
    """
    if can_sell_T3:
        report_text += f"""- 🔴 **Điểm chờ BÁN:** Quanh vùng **{sell_price:,.0f} đ** (Dự kiến: {sell_date.strftime('%d/%m/%Y')})
    - 🎯 **Biên lợi nhuận kỳ vọng:** **{profit_pct:+.2f}%**"""
    else:
        report_text += f"""- ⚠️ **Lưu ý:** Khung thời gian hẹp không đủ để hoàn thành vòng quay T+3."""

    report_text += f"\n\n**3. Kết luận từ AI Quant:**\n{conclusion}"
    st.info(report_text)

else:
    st.error("Không thể kết nối dữ liệu hoặc dữ liệu quá ngắn để huấn luyện.")