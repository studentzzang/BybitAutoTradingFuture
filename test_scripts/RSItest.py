import os
import time
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
import numpy as np
from pybit.unified_trading import HTTP

SYMBOL = ["PUMPFUNUSDT", "FARTCOINUSDT"]
LEVERAGE = 5
TIMEFRAME = [5,15,30,60]
RSI_PERIOD = [5, 9, 14]
EQUITY = 100.0
START = "2026-02-01"
END = "2026-03-07"
OUT_DIR = "test"

MAX_CANDLES = 20000

OPEN_SHORT_RSI = 72.0
OPEN_LONG_RSI = 28.0
CLOSE_SHORT_RSI = 70.0
CLOSE_LONG_RSI = 30.0

SL_ROE_ARR = [5, 10, 15]
MODE_ARR = [1, 2]

session = HTTP()

def _as_list(x):
    return x if isinstance(x, (list, tuple)) else [x]

def parse_date(s: Optional[str]) -> Optional[int]:
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        dt = datetime.strptime(s, "%Y-%m-%d")
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)

def bybit_interval(tf: str) -> str:
    tf = str(tf).upper()
    mapping = {
        "1": "1", "3": "3", "5": "5", "15": "15", "30": "30",
        "60": "60", "120": "120", "240": "240", "360": "360", "720": "720",
        "D": "D", "W": "W", "M": "M"
    }
    if tf not in mapping:
        raise ValueError(f"지원하지 않는 분봉/주기: {tf}")
    return mapping[tf]

def fetch_ohlcv_10000(symbol: str, tf: str, start_ms=None, end_ms=None, max_candles: int = MAX_CANDLES) -> pd.DataFrame:
    interval = bybit_interval(tf)
    if end_ms is None:
        end_ms = int(datetime.now(tz=timezone.utc).timestamp() * 1000)

    rows = []
    while len(rows) < max_candles + 2000:
        resp = None
        last_err = None

        for _ in range(3):
            try:
                r = session.get_kline(
                    category="linear",
                    symbol=symbol,
                    interval=interval,
                    end=end_ms,
                    limit=1000
                )
                if r.get("retCode") == 0:
                    resp = r
                    break
                last_err = RuntimeError(f"retCode {r.get('retCode')} {r.get('retMsg')}")
            except Exception as e:
                last_err = e
            time.sleep(0.4)

        if resp is None:
            raise last_err if last_err else RuntimeError("Unknown API error")

        lst = resp["result"]["list"]
        if not lst:
            break

        for it in lst:
            ts = int(it[0])
            o, h, l, c, v = map(float, it[1:6])
            rows.append((ts, o, h, l, c, v))

        oldest_ts = min(int(x[0]) for x in lst)
        end_ms = oldest_ts - 1

        if start_ms is not None and oldest_ts < start_ms:
            break

        if len(lst) < 1000:
            break

        time.sleep(0.12)

    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"]).drop_duplicates("ts")
    df.sort_values("ts", inplace=True)
    df.reset_index(drop=True, inplace=True)

    if start_ms is not None:
        df = df[df["ts"] >= start_ms].copy()
    if end_ms is not None:
        pass

    if len(df) > max_candles:
        df = df.tail(max_candles).copy()

    df.reset_index(drop=True, inplace=True)
    return df

def compute_rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0)
    down = (-delta).clip(lower=0)
    roll_up = up.ewm(alpha=1 / period, adjust=False).mean()
    roll_down = down.ewm(alpha=1 / period, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-12)
    return 100 - (100 / (1 + rs))

def run(symbol: str, tf: str, rsi_period: int, leverage: float, equity: float,
        start: Optional[str], end: Optional[str], out_dir: str,
        sl_roe: float, mode: int) -> str:

    start_ms = parse_date(start)
    end_ms = parse_date(end)

    ohlc = fetch_ohlcv_10000(symbol, tf, start_ms, end_ms, MAX_CANDLES)
    if ohlc.empty:
        raise SystemExit("❌ 시세 데이터가 비었습니다. 심볼/기간/분봉을 확인하세요.")

    ohlc["rsi"] = compute_rsi(ohlc["close"], rsi_period)

    cols = [
        "datetime", "symbol", "timeframe", "close", "rsi",
        "포지션", "비고", "entry_price", "exit_price",
        "수량", "미실현PnL", "ROE",
        "원금_before", "원금_after", "mode", "sl_roe"
    ]
    log = []

    cash = float(equity)

    position = None
    entry_px = None
    qty = None
    margin = None

    for i in range(len(ohlc)):
        ts = int(ohlc.loc[i, "ts"]) // 1000
        dt = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        px = float(ohlc.loc[i, "close"])
        rv = float(ohlc.loc[i, "rsi"]) if not np.isnan(ohlc.loc[i, "rsi"]) else None

        if rv is None:
            continue

        if position is None:
            if rv >= OPEN_SHORT_RSI:
                position = "SHORT"
                entry_px = px
                margin = cash
                qty = (margin * leverage) / entry_px

            elif rv <= OPEN_LONG_RSI:
                position = "LONG"
                entry_px = px
                margin = cash
                qty = (margin * leverage) / entry_px

            continue

        remark = ""
        pnl = 0.0
        roe = 0.0

        if position == "LONG":
            pnl = (px - entry_px) * qty
            roe = (pnl / margin) * 100 if margin else 0.0

            if mode == 1:
                if rv >= CLOSE_SHORT_RSI:
                    remark = "close LONG (RSI 반대 시그널)"
            elif mode == 2:
                if roe <= -sl_roe:
                    remark = f"close LONG (SL {roe:.2f}%)"
                elif rv >= CLOSE_SHORT_RSI:
                    remark = "close LONG (RSI 반대 시그널)"

        elif position == "SHORT":
            pnl = (entry_px - px) * qty
            roe = (pnl / margin) * 100 if margin else 0.0

            if mode == 1:
                if rv <= CLOSE_LONG_RSI:
                    remark = "close SHORT (RSI 반대 시그널)"
            elif mode == 2:
                if roe <= -sl_roe:
                    remark = f"close SHORT (SL {roe:.2f}%)"
                elif rv <= CLOSE_LONG_RSI:
                    remark = "close SHORT (RSI 반대 시그널)"

        if remark:
            before_cash = cash
            cash = cash + pnl
            log.append([
                dt, symbol, tf, px, rv,
                "CLOSE", remark, entry_px, px,
                qty, pnl, roe,
                before_cash, cash, mode, sl_roe
            ])
            position = None
            entry_px = None
            qty = None
            margin = None

    df = pd.DataFrame(log, columns=cols)

    os.makedirs(out_dir, exist_ok=True)
    fname = f"{symbol}_{tf}_{rsi_period}_MODE{mode}_SL{sl_roe}.csv"
    path = os.path.join(out_dir, fname)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path

if __name__ == "__main__":
    for s in _as_list(SYMBOL):
        for tf in _as_list(TIMEFRAME):
            for rp in _as_list(RSI_PERIOD):
                for mode in MODE_ARR:
                    if mode == 1:
                        csv_path = run(s, tf, rp, LEVERAGE, EQUITY, START, END, OUT_DIR, 0, mode)
                        print(f"✅ 저장 완료: {csv_path}")
                    else:
                        for sl in SL_ROE_ARR:
                            csv_path = run(s, tf, rp, LEVERAGE, EQUITY, START, END, OUT_DIR, sl, mode)
                            print(f"✅ 저장 완료: {csv_path}")