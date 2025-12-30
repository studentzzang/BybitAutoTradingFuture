import os
import time
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
import numpy as np
from pybit.unified_trading import HTTP

# ====== 사용자 설정 변수 ======
SYMBOL = ["PUMPFUNUSDT"]
LEVERAGE = 5
TIMEFRAME = [3,5,15,30]
RSI_PERIOD = [5,7,9,12,15]
EQUITY = 100.0
START = "2025-09-01"
END = "2025-12-28"
OUT_DIR = "test"

MAX_CANDLES = 20000

# RSI 트리거 값 (원본 설정값)
OPEN_SHORT_RSI  = 72.0   # 숏 진입 기준
OPEN_LONG_RSI   = 28.0   # 롱 진입 기준
CLOSE_SHORT_RSI = 70.0
CLOSE_LONG_RSI  = 30.0

# DOORSTEP 밴드 (원본 로직 전용)
DOORSTEP = 3.0
 
# ====== TP / SL 배열 ======
TP_ROE_ARR   = [1.5,2,3]
SL_ROE_ARR   = [7,9]
TP_MODE_ARR  = [1,2]   # 1 = DOORSTEP + TP/SL, 2 = TP/SL만 의존
# ==========================

session = HTTP()

# ---------- 유틸리티 ----------
def _as_list(x):
    return x if isinstance(x, (list, tuple)) else [x]

def parse_date(s: Optional[str]) -> Optional[int]:
    if not s: return None
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        dt = datetime.strptime(s, "%Y-%m-%d")
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)

def bybit_interval(tf: str) -> str:
    return str(tf)

def fetch_ohlcv_10000(symbol: str, tf: str, start_ms=None, end_ms=None, max_candles: int = MAX_CANDLES) -> pd.DataFrame:
    interval = bybit_interval(tf)
    if end_ms is None:
        end_ms = int(datetime.now(tz=timezone.utc).timestamp() * 1000)

    rows = []
    while len(rows) < max_candles:
        resp = None
        last_err = None
        for attempt in range(3):
            try:
                r = session.get_kline(
                    category="linear", symbol=symbol, interval=interval,
                    end=end_ms, limit=1000
                )
                if r.get("retCode") == 0:
                    resp = r
                    break
                else:
                    last_err = RuntimeError(f"retCode {r.get('retCode')} {r.get('retMsg')}")
            except Exception as e:
                last_err = e
            time.sleep(0.4)

        if resp is None: break

        lst = resp["result"]["list"]
        if not lst: break
        for it in lst:
            ts = int(it[0]); o, h, l, c, v = map(float, it[1:6])
            rows.append((ts, o, h, l, c, v))
        end_ms = min(int(x[0]) for x in lst) - 1
        if len(lst) < 1000: break
        time.sleep(0.12)

    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"]).drop_duplicates("ts")
    df.sort_values("ts", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df.head(max_candles)

def compute_rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0)
    down = (-delta).clip(lower=0)
    roll_up = up.ewm(alpha=1/period, adjust=False).mean()
    roll_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-12)
    return 100 - (100/(1+rs))

# ---------- 시뮬레이션 ----------
def run(symbol: str, tf: str, rsi_period: int, leverage: float, equity: float,
        start: Optional[str], end: Optional[str], out_dir: str,
        tp_roe: float, sl_roe: float, tp_mode: int) -> str:
    
    start_ms = parse_date(start)
    end_ms = parse_date(end)

    ohlc = fetch_ohlcv_10000(symbol, tf, start_ms, end_ms)
    if ohlc.empty: return ""

    ohlc["rsi"] = compute_rsi(ohlc["close"], rsi_period)

    cols = ["datetime", "symbol", "timeframe", "close", "rsi", "포지션", "비고", "entry_price", "미실현PnL", "ROE"]
    log = []

    position = None
    entry_px = None
    qty = None
    init_margin = None

    for i in range(len(ohlc)):
        ts = int(ohlc.loc[i, "ts"]) // 1000
        dt = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        px = float(ohlc.loc[i, "close"])
        rv = float(ohlc.loc[i, "rsi"]) if not np.isnan(ohlc.loc[i, "rsi"]) else None

        remark = ""
        unreal = 0.0
        roe = 0.0

        # === 진입 ===
        if position is None and rv is not None:
            if rv >= OPEN_SHORT_RSI:
                position = "SHORT"; entry_px = px
                notional = equity * leverage
                qty = notional / entry_px
                init_margin = equity
                remark = "SHORT 진입"
            elif rv <= OPEN_LONG_RSI:
                position = "LONG"; entry_px = px
                notional = equity * leverage
                qty = notional / entry_px
                init_margin = equity
                remark = "LONG 진입"

        # === 보유 중 ===
        elif position is not None and rv is not None:
            if position == "LONG":
                unreal = (px - entry_px) * qty
                roe = (unreal / init_margin) * 100

                if tp_mode == 1:
                    if roe <= -sl_roe:
                        remark = f"close LONG (SL {roe:.1f}%)"; position = None
                    elif roe >= tp_roe:
                        if rv >= OPEN_SHORT_RSI:
                            if (OPEN_SHORT_RSI - DOORSTEP) <= rv <= (OPEN_SHORT_RSI + DOORSTEP):
                                remark = f"close LONG (DOORSTEP TP, ROE {roe:.1f}%)"; position = None
                        else:
                            remark = f"close LONG (TP {roe:.1f}%)"; position = None
                elif tp_mode == 2:
                    if roe >= tp_roe:
                        remark = f"close LONG (TP {roe:.1f}%)"; position = None
                    elif roe <= -sl_roe:
                        remark = f"close LONG (SL {roe:.1f}%)"; position = None

            elif position == "SHORT":
                unreal = (entry_px - px) * qty
                roe = (unreal / init_margin) * 100

                if tp_mode == 1:
                    if roe <= -sl_roe:
                        remark = f"close SHORT (SL {roe:.1f}%)"; position = None
                    elif roe >= tp_roe:
                        if rv <= OPEN_LONG_RSI:
                            if (OPEN_LONG_RSI - DOORSTEP) <= rv <= (OPEN_LONG_RSI + DOORSTEP):
                                remark = f"close SHORT (DOORSTEP TP, ROE {roe:.1f}%)"; position = None
                        else:
                            remark = f"close SHORT (TP {roe:.1f}%)"; position = None
                elif tp_mode == 2:
                    if roe >= tp_roe:
                        remark = f"close SHORT (TP {roe:.1f}%)"; position = None
                    elif roe <= -sl_roe:
                        remark = f"close SHORT (SL {roe:.1f}%)"; position = None

            # === 청산 로그 기록 ===
            if remark and "close" in remark:
                log.append([dt, symbol, tf, px, rv, "CLOSE", remark, entry_px, unreal, roe])
                position = None; entry_px = None; qty = None

    if not log: return ""
    df = pd.DataFrame(log, columns=cols)
    os.makedirs(out_dir, exist_ok=True)
    fname = f"{symbol}_{tf}_R{rsi_period}_TP{tp_roe}_SL{sl_roe}_M{tp_mode}.csv"
    path = os.path.join(out_dir, fname)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path

# ---------- 실행 ----------
if __name__ == "__main__":
    print("--- RSI 원본 로직 백테스팅 시작 ---")
    for s in _as_list(SYMBOL):
        for tf in _as_list(TIMEFRAME):
            for rp in _as_list(RSI_PERIOD):
                for tp in TP_ROE_ARR:
                    for sl in SL_ROE_ARR:
                        for mode in TP_MODE_ARR:
                            csv_path = run(s, str(tf), rp, LEVERAGE, EQUITY, START, END, OUT_DIR, tp, sl, mode)
                            if csv_path:
                                print(f"✅ 완료: {os.path.basename(csv_path)}")