import matplotlib
matplotlib.use("Agg")

from flask import Flask, send_file, render_template_string
import matplotlib.pyplot as plt
import threading
import time
from datetime import datetime

from live_virtual_rsi_backtest import build_cases, CASES, LiveVirtualBacktestEngine, INITIAL_CASH

app = Flask(__name__)

cases = build_cases(CASES)
engine = LiveVirtualBacktestEngine(cases)
engine_started = False
engine_lock = threading.Lock()

HTML = """
<!doctype html>
<html>
<head>
    <meta charset="utf-8">
    <title>Live Virtual RSI Backtest</title>
    <meta http-equiv="refresh" content="120">
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #111; color: #eee; }
        h1 { margin-bottom: 8px; }
        .meta { margin-bottom: 20px; color: #bbb; }
        img { max-width: 100%; border: 1px solid #444; background: white; }
        .cards { display: flex; flex-wrap: wrap; gap: 12px; margin-top: 20px; }
        .card {
            border: 1px solid #333;
            border-radius: 10px;
            padding: 12px;
            width: 280px;
            background: #1b1b1b;
        }
        .label { font-weight: bold; margin-bottom: 8px; }
        .small { color: #bbb; font-size: 14px; }
    </style>
</head>
<body>
    <h1>Live Virtual RSI Backtest Dashboard</h1>
    <div class="meta">5초마다 자동 새로고침</div>

    <img src="/plot.png?ts={{ ts }}" alt="dashboard">

    <div class="cards">
        {% for row in rows %}
        <div class="card">
            <div class="label">{{ row.label }}</div>
            <div>자산: {{ row.cash }}</div>
            <div>실현손익: {{ row.realized }}</div>
            <div>총 거래수: {{ row.trades }}</div>
            <div>승률: {{ row.winrate }}</div>
            <div>포지션: {{ row.position }}</div>
            <div class="small">업데이트: {{ row.updated }}</div>
        </div>
        {% endfor %}
    </div>
</body>
</html>
"""

def ensure_engine_started():
    global engine_started
    with engine_lock:
        if not engine_started:
            engine.start()
            engine_started = True

def make_plot():
    with engine.lock:
        fig, axes = plt.subplots(2, 2, figsize=(16, 9))
        ax_asset, ax_trades, ax_winrate, ax_equity = axes.ravel()

        labels = [case.case_id for case in engine.cases]
        assets = []
        trade_counts = []
        win_rates = []

        for case in engine.cases:
            current_equity = case.equity_history[-1] if case.equity_history else case.cash
            assets.append(current_equity)
            trade_counts.append(case.total_trades)
            win_rates.append(case.win_rate())

        ax_asset.bar(labels, assets)
        ax_asset.axhline(INITIAL_CASH, linestyle="--", linewidth=1)
        ax_asset.set_title("Current Asset (USDT)")

        ax_trades.bar(labels, trade_counts)
        ax_trades.set_title("Trade Count")

        ax_winrate.bar(labels, win_rates)
        ax_winrate.set_ylim(0, 100)
        ax_winrate.set_title("Win Rate (%)")

        for case in engine.cases:
            if case.time_history and case.equity_history:
                x = list(range(len(case.equity_history)))
                y = list(case.equity_history)
                ax_equity.plot(x, y, label=case.case_id)

        ax_equity.axhline(INITIAL_CASH, linestyle="--", linewidth=1)
        ax_equity.set_title("Equity Curves")
        ax_equity.legend(loc="upper left", fontsize=8)

        fig.tight_layout()
        fig.savefig("dashboard.png")
        plt.close(fig)

@app.route("/")
def home():
    ensure_engine_started()

    rows = []
    with engine.lock:
        for case in engine.cases:
            rows.append({
                "label": case.label(),
                "cash": f"{case.cash:.4f}",
                "realized": f"{case.realized_pnl:.4f}",
                "trades": case.total_trades,
                "winrate": f"{case.win_rate():.2f}%",
                "position": case.position.side if case.position else "None",
                "updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            })

    return render_template_string(HTML, ts=int(time.time()), rows=rows)

@app.route("/plot.png")
def plot_png():
    ensure_engine_started()
    make_plot()
    return send_file("dashboard.png", mimetype="image/png")

if __name__ == "__main__":
    ensure_engine_started()
    app.run(host="0.0.0.0", port=5000, debug=False)



