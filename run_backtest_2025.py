from src.data_loader import DataLoader
from src.features import build_features
from src.predictor import AIModel
from src.backtester import QuantBacktester

def main():
    print("🚀 Đang xử lý mã GAS đến tháng 03/2026...")
    
    # 1. Tải và xử lý
    loader = DataLoader()
    df = loader.get_data("GAS", days=1200)
    df_feat = build_features(df)
    
    # 2. Chia dữ liệu: Học từ quá khứ, test trên 2025-2026
    train_data = df_feat[df_feat['date'] < '2025-01-01'].copy()
    test_data = df_feat[df_feat['date'] >= '2025-01-01'].copy()
    
    # 3. AI Học và Dự báo
    model = AIModel()
    model.train(train_data)
    test_data['predict_prob'] = model.predict_prob(test_data)
    
    # 4. Kiểm định (Backtest)
    backtester = QuantBacktester()
    result, metrics = backtester.run_backtest(test_data, prob_threshold=0.65)
    
    print("\n📊 KẾT QUẢ ĐÁNH GIÁ:")
    for k, v in metrics.items():
        print(f" > {k}: {v}")

if __name__ == "__main__":
    main()