# Cập nhật hàm _calculate_metrics trong src/backtester.py
def _calculate_metrics(self, strat_returns, market_returns):
    # Lợi nhuận tích lũy
    equity_curve = (1 + strat_returns).cumprod()
    total_strat_return = equity_curve.iloc[-1] - 1
    total_market_return = (1 + market_returns).cumprod().iloc[-1] - 1

    # Win Rate & Profit Factor
    wins = strat_returns[strat_returns > 0]
    losses = strat_returns[strat_returns < 0]
    win_rate = len(wins) / len(strat_returns[strat_returns != 0]) if len(strat_returns[strat_returns != 0]) > 0 else 0
    profit_factor = abs(wins.sum() / losses.sum()) if len(losses) > 0 and losses.sum() != 0 else 0

    # Max Drawdown
    roll_max = equity_curve.cummax()
    drawdown = equity_curve / roll_max - 1
    max_drawdown = drawdown.min()

    # Sharpe Ratio (Annualized)
    sharpe = np.sqrt(252) * strat_returns.mean() / (strat_returns.std() + 1e-9)

    return {
        "Lợi nhuận AI": f"{total_strat_return * 100:.2f}%",
        "Lợi nhuận Market": f"{total_market_return * 100:.2f}%",
        "Tỷ lệ thắng (Win Rate)": f"{win_rate * 100:.2f}%",
        "Hệ số lợi nhuận (Profit Factor)": f"{profit_factor:.2f}",
        "Sụt giảm tối đa (Max Drawdown)": f"{max_drawdown * 100:.2f}%",
        "Chỉ số Sharpe": f"{sharpe:.2f}",
        "Số lượng lệnh thực hiện": len(strat_returns[strat_returns != 0])
    }