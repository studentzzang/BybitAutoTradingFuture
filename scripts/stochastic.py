
from dotenv import load_dotenv, find_dotenv
import os, sys, time
from datetime import datetime
import requests
import numpy as np
import pandas as pd

import bybit  

# =========================
# ENV
# =========================
load_dotenv(find_dotenv(), override=True)
_api_key = os.getenv("API_KEY")
_api_secret = os.getenv("API_KEY_SECRET")
if not _api_key or not _api_secret:
    print("❌ API_KEY 또는 API_KEY_SECRET을 .env에서 못 찾았습니다.")
    sys.exit(1)

# =========================
# CONFIG (배열 유지)
# =========================
SYMBOLS = ["FARTCOINUSDT"]
INTERVALS = ["5"]
CATEGORY = ["linear"]          # 중요: v5 kline category (linear/inverse/spot/option)

LEVERAGE = "10"
PCT = 50
bybit.PCT = PCT

MODE = [1]                     # 1: TP/SL by ROE, 2: opposite cross exit
TP_ROE = [2.5]                 # mode1 only
SL_ROE = [15]                 # mode1 only

STOCH_PERIOD = [5]
K_SMOOTH = [5]
D_SMOOTH = [3]

OVERSOLD = [25.0]
OVERBOUGHT = [75.0]
STRICT_ZONE = [False]

COOLDOWN_BARS = 0
SLEEP_SEC = 2              

# =========================
# STATE
# =========================
position = {s: None for s in SYMBOLS}   
cooldown = {s: 0 for s in SYMBOLS}
last_closed_ts = {s: None for s in SYMBOLS} 

# =========================
# BYBIT KLINE (requests)
# =========================
def get_kline_http(symbol: str, interval: str, limit: int = 300, category: str = "linear"):
    symbol = str(symbol).upper().strip()
    interval = str(interval).strip()
    category = str(category).strip()

    params = {
        "category": category,
        "symbol": symbol,
        "interval": interval,
        "limit": int(limit),
    }

    r = requests.get("https://api.bybit.com/v5/market/kline", params=params, timeout=10)
    j = r.json()

    if j.get("retCode", 0) != 0:
        # 여기서 params 찍어줘야 "category 틀림/심볼 틀림/interval 틀림" 바로 보임
        raise RuntimeError(f"get_kline failed: retCode={j.get('retCode')} retMsg={j.get('retMsg')} params={params}")

    lst = j.get("result", {}).get("list", [])
    if not lst:
        return []

    # Bybit은 보통 최신→과거로 줌. ts 기준 오름차순으로 정렬해서 안전하게 처리
    lst_sorted = sorted(lst, key=lambda row: int(row[0]))
    return lst_sorted

# =========================
# INDICATOR
# =========================
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

# =========================
# ORDER WRAPPERS (bybit.py 그대로 사용)
# =========================
def close_long(symbol: str):
    bybit.close_position(symbol, "Sell")

def close_short(symbol: str):
    bybit.close_position(symbol, "Buy")

def enter_long(symbol: str, leverage: str):
    px, q = bybit.entry_position(symbol, "Buy", leverage)
    if px is not None and q and q > 0:
        position[symbol] = "long"
        return True
    return False

def enter_short(symbol: str, leverage: str):
    px, q = bybit.entry_position(symbol, "Sell", leverage)
    if px is not None and q and q > 0:
        position[symbol] = "short"
        return True
    return False

# =========================
# MAIN
# =========================
def start():
    base = bybit.get_usdt()
    print(f"🔧 보유금액: {base:.2f} USDT")

    for s in SYMBOLS:
        if hasattr(bybit, "SYMBOLS") and s not in bybit.SYMBOLS:
            bybit.SYMBOLS.append(s)

    for i, s in enumerate(SYMBOLS):
        bybit.set_leverage(symbol=s, leverage=LEVERAGE)

def update():
    while True:
        for idx, symbol in enumerate(SYMBOLS):
            try:
                interval = INTERVALS[idx]
                category = CATEGORY[idx]
                mode = MODE[idx]

                tp = TP_ROE[idx] if mode == 1 else None
                sl = SL_ROE[idx] if mode == 1 else None

                period = STOCH_PERIOD[idx]
                ks = K_SMOOTH[idx]
                ds = D_SMOOTH[idx]
                os_ = OVERSOLD[idx]
                ob_ = OVERBOUGHT[idx]
                strict = STRICT_ZONE[idx]

                kl = get_kline_http(symbol, interval, limit=400, category=category)
                if len(kl) < (period + ks + ds + 20):
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] {symbol} 데이터 부족")
                    continue

                st = compute_stoch_from_klines(kl, period, ks, ds)
                if len(st) < 5:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] {symbol} stoch 데이터 부족")
                    continue

                # live(진행중) / close(닫힌 봉) 분리
                ts_live = int(st.iloc[-1]["ts"])
                ts_close = int(st.iloc[-2]["ts"])

                if last_closed_ts[symbol] is None:
                    last_closed_ts[symbol] = ts_close

                new_bar = (ts_close != last_closed_ts[symbol])
                if new_bar:
                    last_closed_ts[symbol] = ts_close
                    if cooldown[symbol] > 0:
                        cooldown[symbol] -= 1

                k_prev = float(st.iloc[-3]["k"])
                d_prev = float(st.iloc[-3]["d"])

                k_close = float(st.iloc[-2]["k"])
                d_close = float(st.iloc[-2]["d"])

                k_live = float(st.iloc[-1]["k"])
                d_live = float(st.iloc[-1]["d"])

                # 교차는 "닫힌 봉 기준"으로만 판단
                bc = bull_cross(k_prev, d_prev, k_close, d_close)
                sc = bear_cross(k_prev, d_prev, k_close, d_close)

                ROE = float(bybit.get_ROE(symbol))
                PnL = float(bybit.get_PnL(symbol))

                closed = False

                # ====== EXIT ======
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
                        cooldown[symbol] = COOLDOWN_BARS

                # ====== ENTRY (새로 닫힌 봉에서만) ======
                if position[symbol] is None and cooldown[symbol] == 0 and new_bar:
                    if bc and in_zone_long(k_close, d_close, os_, strict):
                        enter_long(symbol, LEVERAGE)
                    elif sc and in_zone_short(k_close, d_close, ob_, strict):
                        enter_short(symbol, LEVERAGE)

                print(
                    f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                    f"🪙 {symbol} 🕧 {interval} | 🚩포지션:{position[symbol]} "
                    f"| K(close):{k_close:.2f} D(close):{d_close:.2f} "
                    f"| K(live):{k_live:.2f} D(live):{d_live:.2f} "
                    f"|💸 PnL:{PnL:.3f} |💎 ROE:{ROE:.2f} "
                    f"| new_bar:{1 if new_bar else 0} cooldown:{cooldown[symbol]}"
                )

            except Exception as e:
                print(f"[ERROR] {symbol}: {type(e).__name__} {e}")

            time.sleep(SLEEP_SEC)

        time.sleep(0.2)

if __name__ == "__main__":
    start()
    update()
