from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from collections import deque
import math
import threading
import time

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pybit.unified_trading import HTTP


# ==========================================================
# 사용자 설정
# ==========================================================
CASES = [
    {
        "symbol": "PUMPFUNUSDT",
        "interval": "5",
        "rsi_period":9,
        "long_switch_rsi": 28,
        "short_switch_rsi": 72,
        "close_long_rsi": 30,
        "close_short_rsi": 70,
        "mode": 1,
        "sl_roe": 10,
    },
    {
        "symbol": "PUMPFUNUSDT",
        "interval": "5",
        "rsi_period":14,
        "long_switch_rsi": 28,
        "short_switch_rsi": 72,
        "close_long_rsi": 30,
        "close_short_rsi": 70,
        "mode": 1,
        "sl_roe": 10,
    },
    {
        "symbol": "PUMPFUNUSDT",
        "interval": "15",
        "rsi_period":5,
        "long_switch_rsi": 28,
        "short_switch_rsi": 72,
        "close_long_rsi": 30,
        "close_short_rsi": 70,
        "mode": 1,
        "sl_roe": 10,
    },
    {
        "symbol": "PUMPFUNUSDT",
        "interval": "15",
        "rsi_period": 9,    
        "long_switch_rsi": 28,
        "short_switch_rsi": 72,
        "close_long_rsi": 30,
        "close_short_rsi": 70,
        "mode": 1,
        "sl_roe": 10,
    },
    {
        "symbol": "PUMPFUNUSDT",
        "interval": "15",
        "rsi_period": 14,    
        "long_switch_rsi": 28,
        "short_switch_rsi": 72,
        "close_long_rsi": 30,
        "close_short_rsi": 70,
        "mode": 1,
        "sl_roe": 10,
    },
    {
        "symbol": "PUMPFUNUSDT",
        "interval": "30",
        "rsi_period": 5,
        "long_switch_rsi": 28,
        "short_switch_rsi": 72,
        "close_long_rsi": 30,
        "close_short_rsi": 70,
        "mode": 1,
        "sl_roe": 10,
    },
    {
        "symbol": "PUMPFUNUSDT",
        "interval": "30",
        "rsi_period": 7,
        "long_switch_rsi": 28,
        "short_switch_rsi": 72,
        "close_long_rsi": 30,
        "close_short_rsi": 70,
        "mode": 1,
        "sl_roe": 10,
    },
    {
        "symbol": "PUMPFUNUSDT",
        "interval": "30",
        "rsi_period": 12,
        "long_switch_rsi": 28,
        "short_switch_rsi": 72,
        "close_long_rsi": 30,
        "close_short_rsi": 70,
        "mode": 1,
        "sl_roe": 10,
    },
    {
        "symbol": "PUMPFUNUSDT",
        "interval": "60",
        "rsi_period": 5,
        "long_switch_rsi": 28,
        "short_switch_rsi": 72,
        "close_long_rsi": 30,
        "close_short_rsi": 70,
        "mode": 1,
        "sl_roe": 10,
    },
    {
        "symbol": "PUMPFUNUSDT",
        "interval": "60",
        "rsi_period": 8,
        "long_switch_rsi": 28,
        "short_switch_rsi": 72,
        "close_long_rsi": 30,
        "close_short_rsi": 70,
        "mode": 1,
        "sl_roe": 10,
    },
    {
        "symbol": "PUMPFUNUSDT",
        "interval": "60",
        "rsi_period": 12,
        "long_switch_rsi": 28,
        "short_switch_rsi": 72,
        "close_long_rsi": 30,
        "close_short_rsi": 70,
        "mode": 1,
        "sl_roe": 10,
    },
]

INITIAL_CASH = 100.0          # 각 케이스 시작 자산
LEVERAGE = 5                  # RSI.py와 동일
PCT = 80                      # 진입 시 사용할 증거금 비율(자산의 40%)
ENTRY_BAND = 4                # RSI.py와 동일
COOLDOWN_BARS = 0             # RSI.py와 동일
POLL_SECONDS = 5              # 케이스별 폴링 간격
ROUND_DIGITS = 6              # 수량 반올림
HISTORY_LIMIT = 500           # 시각화용 히스토리 최대 길이
USE_TESTNET = False           # 공개 시세라 보통 False 그대로 두면 됨


# ==========================================================
# 내부 로직
# ==========================================================
@dataclass
class VirtualPosition:
    side: str                  # long / short
    entry_price: float
    qty: float
    margin_used: float
    notional: float
    entry_time: datetime


@dataclass
class StrategyCase:
    case_id: str
    symbol: str
    interval: str
    rsi_period: int
    long_switch_rsi: float
    short_switch_rsi: float
    close_long_rsi: float
    close_short_rsi: float
    mode: int
    sl_roe: float

    cash: float = INITIAL_CASH
    position: Optional[VirtualPosition] = None
    last_peak_level: Optional[float] = None
    last_trough_level: Optional[float] = None
    armed_short_switch: bool = False
    armed_long_switch: bool = False
    last_closed_price1: Optional[float] = None
    cooldown_bars: int = 0

    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    realized_pnl: float = 0.0

    time_history: deque = field(default_factory=lambda: deque(maxlen=HISTORY_LIMIT))
    equity_history: deque = field(default_factory=lambda: deque(maxlen=HISTORY_LIMIT))
    trade_count_history: deque = field(default_factory=lambda: deque(maxlen=HISTORY_LIMIT))
    win_rate_history: deque = field(default_factory=lambda: deque(maxlen=HISTORY_LIMIT))

    def label(self) -> str:
        return f"{self.symbol} {self.interval}m RSI{self.rsi_period}"

    def unrealized_pnl(self, current_price: float) -> float:
        if self.position is None:
            return 0.0
        if self.position.side == "long":
            return (current_price - self.position.entry_price) * self.position.qty
        return (self.position.entry_price - current_price) * self.position.qty

    def equity(self, current_price: float) -> float:
        return self.cash + self.unrealized_pnl(current_price)

    def roe(self, current_price: float) -> float:
        if self.position is None or self.position.margin_used <= 0:
            return 0.0
        return (self.unrealized_pnl(current_price) / self.position.margin_used) * 100.0

    def win_rate(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return (self.wins / self.total_trades) * 100.0

    def snapshot(self, now: datetime, current_price: float) -> None:
        self.time_history.append(now)
        self.equity_history.append(self.equity(current_price))
        self.trade_count_history.append(self.total_trades)
        self.win_rate_history.append(self.win_rate())


class BybitPublicMarket:
    def __init__(self, use_testnet: bool = False):
        self.session = HTTP(testnet=use_testnet)

    def fetch_market_state(self, symbol: str, interval: str, limit: int) -> Tuple[List[float], float, float, float]:
        kline = self.session.get_kline(
            category="linear",
            symbol=symbol,
            interval=interval,
            limit=limit,
        )
        raw = kline["result"]["list"]
        if not raw:
            raise RuntimeError(f"No kline data for {symbol} {interval}")

        # Bybit kline 응답은 최신 -> 과거 순서라 뒤집기
        rows = list(reversed(raw))
        closes = [float(row[4]) for row in rows]
        if len(closes) < 3:
            raise RuntimeError(f"Not enough kline data for {symbol} {interval}")

        c_prev2 = closes[-3]
        c_prev1 = closes[-2]
        current_from_kline = closes[-1]

        ticker = self.session.get_tickers(category="linear", symbol=symbol)
        ticker_list = ticker["result"]["list"]
        if not ticker_list:
            current_price = current_from_kline
        else:
            current_price = float(ticker_list[0]["lastPrice"])

        return closes, c_prev2, c_prev1, current_price


def compute_rsi(closes: List[float], period: int) -> float:
    if len(closes) < period + 1:
        raise ValueError(f"RSI 계산용 캔들이 부족합니다. need >= {period + 1}, got {len(closes)}")

    gains: List[float] = []
    losses: List[float] = []
    for i in range(-period, 0):
        change = closes[i] - closes[i - 1]
        gains.append(max(change, 0.0))
        losses.append(max(-change, 0.0))

    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period

    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def build_cases(configs: List[Dict]) -> List[StrategyCase]:
    built: List[StrategyCase] = []
    for i, cfg in enumerate(configs, start=1):
        built.append(
            StrategyCase(
                case_id=f"CASE{i}",
                symbol=cfg["symbol"].upper(),
                interval=str(cfg["interval"]),
                rsi_period=int(cfg["rsi_period"]),
                long_switch_rsi=float(cfg["long_switch_rsi"]),
                short_switch_rsi=float(cfg["short_switch_rsi"]),
                close_long_rsi=float(cfg["close_long_rsi"]),
                close_short_rsi=float(cfg["close_short_rsi"]),
                mode=int(cfg["mode"]),
                sl_roe=float(cfg["sl_roe"]),
            )
        )
    return built


def open_virtual_position(case: StrategyCase, side: str, current_price: float) -> None:
    margin = case.cash * (PCT / 100.0)
    notional = margin * LEVERAGE
    qty = round(notional / current_price, ROUND_DIGITS)
    if qty <= 0:
        return

    case.position = VirtualPosition(
        side=side,
        entry_price=current_price,
        qty=qty,
        margin_used=margin,
        notional=notional,
        entry_time=datetime.now(),
    )


def close_virtual_position(case: StrategyCase, current_price: float, reason: str) -> None:
    if case.position is None:
        return

    pnl = case.unrealized_pnl(current_price)
    case.cash += pnl
    case.realized_pnl += pnl
    case.total_trades += 1

    if pnl >= 0:
        case.wins += 1
    else:
        case.losses += 1

    closed_side = case.position.side
    entry_price = case.position.entry_price
    case.position = None

    print(
        f"{reason} | {case.case_id} {case.label()} | entry={entry_price:.6f} | exit={current_price:.6f} | pnl={pnl:.4f} | cash={case.cash:.4f}"
    )

    if closed_side == "short":
        case.armed_long_switch = False
        case.last_trough_level = None
    else:
        case.armed_short_switch = False
        case.last_peak_level = None
    case.cooldown_bars = COOLDOWN_BARS


def apply_peak_trough_logic(case: StrategyCase, rsi: float) -> None:
    if rsi <= case.long_switch_rsi:
        case.armed_long_switch = True
    if rsi >= case.short_switch_rsi:
        case.armed_short_switch = True

    if rsi >= 84:
        case.last_peak_level = 84
    elif rsi >= 80:
        if case.last_peak_level is None or case.last_peak_level < 80:
            case.last_peak_level = 80
    elif rsi >= 75:
        if case.last_peak_level is None or case.last_peak_level < 75:
            case.last_peak_level = 75
    elif rsi >= 70:
        if case.last_peak_level is None or case.last_peak_level < 70:
            case.last_peak_level = 70

    if rsi <= 20:
        case.last_trough_level = 20
    elif rsi <= 25:
        if case.last_trough_level is None or case.last_trough_level > 25:
            case.last_trough_level = 25
    elif rsi <= 30:
        if case.last_trough_level is None or case.last_trough_level > 30:
            case.last_trough_level = 30
    elif rsi <= 35:
        if case.last_trough_level is None or case.last_trough_level > 35:
            case.last_trough_level = 35


def process_case(case: StrategyCase, closes: List[float], c_prev1: float, current_price: float) -> None:
    rsi = compute_rsi(closes, case.rsi_period)

    new_bar = (case.last_closed_price1 is None) or (case.last_closed_price1 != c_prev1)
    if new_bar:
        case.last_closed_price1 = c_prev1
        if case.cooldown_bars > 0:
            case.cooldown_bars -= 1

    apply_peak_trough_logic(case, rsi)

    if case.position is None and case.cooldown_bars == 0:
        if case.last_peak_level is not None and case.armed_short_switch:
            short_trigger = case.last_peak_level - 3
            if (rsi <= short_trigger) and (rsi >= short_trigger - ENTRY_BAND):
                open_virtual_position(case, side="short", current_price=current_price)
                if case.position is not None:
                    case.cooldown_bars = COOLDOWN_BARS
                    case.last_peak_level = None
                    case.armed_short_switch = False
                    case.armed_long_switch = (rsi <= case.long_switch_rsi)
                    print(f"📉 {case.case_id} {case.label()} SHORT 진입 | entry={current_price:.6f} | mode={case.mode}")
                    case.snapshot(datetime.now(), current_price)
                    return

        if case.position is None and case.last_trough_level is not None and case.armed_long_switch and case.cooldown_bars == 0:
            long_trigger = case.last_trough_level + 3
            if (rsi >= long_trigger) and (rsi <= long_trigger + ENTRY_BAND):
                open_virtual_position(case, side="long", current_price=current_price)
                if case.position is not None:
                    case.cooldown_bars = COOLDOWN_BARS
                    case.last_trough_level = None
                    case.armed_long_switch = False
                    case.armed_short_switch = (rsi >= case.short_switch_rsi)
                    print(f"📈 {case.case_id} {case.label()} LONG 진입 | entry={current_price:.6f} | mode={case.mode}")
                    case.snapshot(datetime.now(), current_price)
                    return

    elif case.position is not None and case.position.side == "short":
        roe = case.roe(current_price)
        if case.mode == 1:
            if rsi <= case.close_long_rsi:
                close_virtual_position(case, current_price, f"✅ {case.case_id} {case.label()} SHORT 청산 | RSI 반대 시그널")
        elif case.mode == 2:
            if roe <= -case.sl_roe:
                close_virtual_position(case, current_price, f"🛑 {case.case_id} {case.label()} SHORT 손절 | ROE={roe:.2f}% | SL={case.sl_roe}%")
            elif rsi <= case.close_long_rsi:
                close_virtual_position(case, current_price, f"✅ {case.case_id} {case.label()} SHORT 익절 | RSI 반대 시그널")

    elif case.position is not None and case.position.side == "long":
        roe = case.roe(current_price)
        if case.mode == 1:
            if rsi >= case.close_short_rsi:
                close_virtual_position(case, current_price, f"✅ {case.case_id} {case.label()} LONG 청산 | RSI 반대 시그널")
        elif case.mode == 2:
            if roe <= -case.sl_roe:
                close_virtual_position(case, current_price, f"🛑 {case.case_id} {case.label()} LONG 손절 | ROE={roe:.2f}% | SL={case.sl_roe}%")
            elif rsi >= case.close_short_rsi:
                close_virtual_position(case, current_price, f"✅ {case.case_id} {case.label()} LONG 익절 | RSI 반대 시그널")

    pnl = case.equity(current_price) - INITIAL_CASH
    roe = case.roe(current_price)
    case.snapshot(datetime.now(), current_price)

    print(
        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
        f"🪙{case.case_id} {case.symbol} @{case.interval} "
        f"💲현재가: {current_price:.6f}$ "
        f"🚩포지션 {case.position.side if case.position else None} "
        f"| MODE={case.mode} "
        f"| RSI({case.rsi_period})={rsi:.2f} "
        f"| PnL: {pnl:.3f} "
        f"| ROE: {roe:.2f}"
    )


class LiveVirtualBacktestEngine:
    def __init__(self, cases: List[StrategyCase]):
        self.market = BybitPublicMarket(use_testnet=USE_TESTNET)
        self.cases = cases
        self.running = True
        self.lock = threading.Lock()

    def start(self) -> None:
        print("=" * 80)
        print("📡 라이브 가상 백테스트 시작")
        print(f"각 케이스 시작 자산: {INITIAL_CASH:.2f} USDT")
        print(f"케이스 수: {len(self.cases)}")
        for case in self.cases:
            print(
                f"- {case.case_id}: {case.label()} | long_switch={case.long_switch_rsi} | short_switch={case.short_switch_rsi} | "
                f"close_long={case.close_long_rsi} | close_short={case.close_short_rsi} | mode={case.mode} | sl_roe={case.sl_roe}"
            )
        print("=" * 80)

        thread = threading.Thread(target=self.run_loop, daemon=True)
        thread.start()

    def run_loop(self) -> None:
        while self.running:
            grouped: Dict[Tuple[str, str], List[StrategyCase]] = {}
            for case in self.cases:
                grouped.setdefault((case.symbol, case.interval), []).append(case)

            for (symbol, interval), group_cases in grouped.items():
                try:
                    max_period = max(case.rsi_period for case in group_cases)
                    closes, _c_prev2, c_prev1, current_price = self.market.fetch_market_state(
                        symbol=symbol,
                        interval=interval,
                        limit=max(120, max_period + 20),
                    )
                    with self.lock:
                        for case in group_cases:
                            try:
                                process_case(case, closes, c_prev1, current_price)
                            except Exception as e:
                                print(f"[ERR] {case.case_id} {case.label()}: {type(e).__name__} {e}")
                except Exception as e:
                    print(f"[ERR] {symbol} {interval}: {type(e).__name__} {e}")

                time.sleep(POLL_SECONDS)

    def stop(self) -> None:
        self.running = False


def run_dashboard(engine: LiveVirtualBacktestEngine) -> None:
    plt.ion()
    fig, axes = plt.subplots(2, 2, figsize=(16, 9))
    ax_asset, ax_trades, ax_winrate, ax_equity = axes.ravel()
    fig.suptitle("Live Virtual RSI Backtest Dashboard", fontsize=16)

    def refresh(_frame: int):
        with engine.lock:
            labels = [case.case_id for case in engine.cases]
            full_labels = [f"{case.case_id}\n{case.symbol}-{case.interval}m-RSI{case.rsi_period}" for case in engine.cases]
            assets = []
            trade_counts = []
            win_rates = []

            ax_asset.clear()
            ax_trades.clear()
            ax_winrate.clear()
            ax_equity.clear()

            for case in engine.cases:
                current_equity = case.equity_history[-1] if case.equity_history else case.cash
                assets.append(current_equity)
                trade_counts.append(case.total_trades)
                win_rates.append(case.win_rate())

            ax_asset.bar(labels, assets)
            ax_asset.axhline(INITIAL_CASH, linestyle="--", linewidth=1)
            ax_asset.set_title("Current Asset (USDT)")
            ax_asset.set_xticks(range(len(labels)), labels)

            ax_trades.bar(labels, trade_counts)
            ax_trades.set_title("Trade Count")
            ax_trades.set_xticks(range(len(labels)), labels)

            ax_winrate.bar(labels, win_rates)
            ax_winrate.set_ylim(0, 100)
            ax_winrate.set_title("Win Rate (%)")
            ax_winrate.set_xticks(range(len(labels)), labels)

            for case in engine.cases:
                if case.time_history and case.equity_history:
                    x = list(range(len(case.equity_history)))
                    ax_equity.plot(x, list(case.equity_history), label=case.case_id)

            ax_equity.axhline(INITIAL_CASH, linestyle="--", linewidth=1)
            ax_equity.set_title("Equity Curves")
            ax_equity.legend(loc="upper left", fontsize=8)

            for ax in (ax_asset, ax_trades, ax_winrate):
                ax.tick_params(axis="x", rotation=0, labelsize=8)
            ax_equity.tick_params(axis="x", labelsize=8)
            fig.tight_layout()

    _ani = FuncAnimation(fig, refresh, interval=2000, cache_frame_data=False)
    plt.show(block=True)


def main() -> None:
    cases = build_cases(CASES)
    engine = LiveVirtualBacktestEngine(cases)
    engine.start()
    try:
        run_dashboard(engine)
    except KeyboardInterrupt:
        print("\n🛑 사용자 중단")
    finally:
        engine.stop()


if __name__ == "__main__":
    main()
