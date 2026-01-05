from dotenv import load_dotenv, find_dotenv
import os, sys, time
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
from pybit.unified_trading import HTTP
import bybit

load_dotenv(find_dotenv(), override=True)
_api_key = os.getenv("API_KEY")
_api_secret = os.getenv("API_KEY_SECRET")
if not _api_key or not _api_secret:
    print("❌ API_KEY 또는 API_KEY_SECRET을 .env에서 못 찾았습니다.")
    sys.exit(1)

session = HTTP(api_key=_api_key, api_secret=_api_secret, recv_window=60000, max_retries=0)

SYMBOLS = ["PUMPFUNUSDT"]
INTERVALS = ["1"]

LEVERAGE = "5"
PCT = 40

MODE = [1]
TP_ROE = [2.0]
SL_ROE = [2.0]

STOCH_PERIOD = [14]
K_SMOOTH = [3]
D_SMOOTH = [3]

OVERSOLD = [20.0]
OVERBOUGHT = [80.0]
STRICT_ZONE = [False]

COOLDOWN_BARS = 0
SLEEP_SEC = 3

bybit.PCT = PCT
for s in SYMBOLS:
    if s not in bybit.SYMBOLS:
        bybit.SYMBOLS.append(s)

position = {s: None for s in SYMBOLS}   # "long"/"short"/None
entry_px = {s: None for s in SYMBOLS}
qty = {s: None for s in SYMBOLS}
cooldown = {s: 0 for s in SYMBOLS}
last_closed_ts = {s: None for s in SYMBOLS}

def get_kline_http(symbol: str, interval: str, limit: int = 200):
    r = session.get_kline(category="linear", symbol=str(symbol).upper(), interval=str(interval), limit=int(limit))
    return r["result"]["list"][::-1]

def compute_stoch_from_klines(kl, period: int, k_smooth: int, d_smooth: int):
    df = pd.DataFrame(kl, columns=["ts","open","high","low","close","volume","turnover"])
    df["ts"] = df["ts"].astype(np.int64)
    for c in ["open","high","low","close"]:
        df[c] = df[c].astype(float)

    low_min = df["low"].rolling(window=period).min()
    high_max = df["high"].rolling(window=period).max()
    denom = (high_max - low_min).replace(0, np.nan)

    k_raw = 100.0 * (df["close"] - low_min) / denom
    k = k_raw.rolling(window=k_smooth).mean()
    d = k.rolling(window=d_smooth).mean()

    df["k"] = k
    df["d"] = d
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def bull_cross(k_prev: float, d_prev: float, k_now: float, d_now: float) -> bool:
    return (k_prev < d_prev) and (k_now > d_now)

def bear_cross(k_prev: float, d_prev: float, k_now: float, d_now: float) -> bool:
    return (k_prev > d_prev) and (k_now < d_now)

def in_zone_long(k: float, d: float, os_: float, strict: bool) -> bool:
    return (k <= os_ and d <= os_) if strict else (k <= os_)

def in_zone_short(k: float, d: float, ob_: float, strict: bool) -> bool:
    return (k >= ob_ and d >= ob_) if strict else (k >= ob_)

def close_long(symbol: str):
    bybit.close_position(symbol, "Sell")

def close_short(symbol: str):
    bybit.close_position(symbol, "Buy")

def enter_long(symbol: str, leverage: str):
    px, q = bybit.entry_position(symbol, "Buy", leverage)
    if px is not None and q > 0:
        position[symbol] = "long"
        entry_px[symbol] = px
        qty[symbol] = q
        return True
    return False

def enter_short(symbol: str, leverage: str):
    px, q = bybit.entry_position(symbol, "Sell", leverage)
    if px is not None and q > 0:
        position[symbol] = "short"
        entry_px[symbol] = px
        qty[symbol] = q
        return True
    return False

def start():
    base = bybit.get_usdt()
    print(f"🔧 보유금액: {base:.2f} USDT")
    for s in SYMBOLS:
        bybit.set_leverage(symbol=s, leverage=LEVERAGE)

def update():
    while True:
        for idx, symbol in enumerate(SYMBOLS):
            try:
                interval = INTERVALS[idx]
                mode = MODE[idx]

                tp = TP_ROE[idx] if mode == 1 else None
                sl = SL_ROE[idx] if mode == 1 else None

                period = STOCH_PERIOD[idx]
                ks = K_SMOOTH[idx]
                ds = D_SMOOTH[idx]
                os_ = OVERSOLD[idx]
                ob_ = OVERBOUGHT[idx]
                strict = STRICT_ZONE[idx]

                kl = get_kline_http(symbol, interval, limit=300)
                if len(kl) < (period + ks + ds + 10):
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] {symbol} 데이터 부족")
                    continue

                st = compute_stoch_from_klines(kl, period, ks, ds)
                if len(st) < 5:
                    continue

                ts_prev = int(st.iloc[-3]["ts"])
                ts_now = int(st.iloc[-2]["ts"])  # 마지막 '닫힌' 봉으로 보려고 -2 사용
                if last_closed_ts[symbol] is None:
                    last_closed_ts[symbol] = ts_now

                new_bar = (ts_now != last_closed_ts[symbol])
                if new_bar:
                    last_closed_ts[symbol] = ts_now
                    if cooldown[symbol] > 0:
                        cooldown[symbol] -= 1

                k_prev = float(st.iloc[-3]["k"])
                d_prev = float(st.iloc[-3]["d"])
                k_now = float(st.iloc[-2]["k"])
                d_now = float(st.iloc[-2]["d"])

                bc = bull_cross(k_prev, d_prev, k_now, d_now)
                sc = bear_cross(k_prev, d_prev, k_now, d_now)

                ROE = bybit.get_ROE(symbol)
                PnL = bybit.get_PnL(symbol)

                closed = False

                if position[symbol] == "long":
                    if mode == 1:
                        if (tp is not None and ROE >= tp) or (sl is not None and ROE <= -sl):
                            close_long(symbol)
                            closed = True
                    else:
                        if new_bar and sc:
                            close_long(symbol)
                            closed = True

                    if closed:
                        position[symbol] = None
                        entry_px[symbol] = None
                        qty[symbol] = None
                        cooldown[symbol] = COOLDOWN_BARS

                elif position[symbol] == "short":
                    if mode == 1:
                        if (tp is not None and ROE >= tp) or (sl is not None and ROE <= -sl):
                            close_short(symbol)
                            closed = True
                    else:
                        if new_bar and bc:
                            close_short(symbol)
                            closed = True

                    if closed:
                        position[symbol] = None
                        entry_px[symbol] = None
                        qty[symbol] = None
                        cooldown[symbol] = COOLDOWN_BARS

                if position[symbol] is None and cooldown[symbol] == 0 and new_bar:
                    if bc and in_zone_long(k_now, d_now, os_, strict):
                        enter_long(symbol, LEVERAGE)

                    elif sc and in_zone_short(k_now, d_now, ob_, strict):
                        enter_short(symbol, LEVERAGE)

                print(
                    f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                    f"🪙 {symbol} 🕧 {interval} | 🚩포지션:{position[symbol]} "
                    f"| K:{k_now:.2f} D:{d_now:.2f} "
                    f"|💸 PnL:{PnL:.3f} |💎 ROE:{ROE:.2f} "
                    f"| cooldown:{cooldown[symbol]}"
                )

            except Exception as e:
                print(f"[ERROR] {symbol}: {type(e).__name__} {e}")
                continue

            time.sleep(SLEEP_SEC)

        time.sleep(1)

start()
update()
