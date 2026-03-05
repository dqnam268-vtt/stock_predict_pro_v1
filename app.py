import streamlit as st
from src.data_loader import DNSELoader
from src.features import build_features
from src.predictor import AIModel

st.set_page_config(page_title="Math-AI Stock Trader", layout="wide")
st.title("📈 AI Prediction for GAS (PV GAS)")

# Thực thi hệ thống
loader = DNSELoader()
df = loader.get_data("GAS")
df_feat = build_features(df)

model = AIModel()
prob = model.train_and_predict(df_feat)

# Hiển thị kết quả
st.metric("Xác suất tăng giá (3 phiên tới)", f"{prob*100:.1f}%")
st.line_chart(df.set_index('datetime')['close'])