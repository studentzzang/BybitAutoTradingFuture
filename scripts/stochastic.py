from dotenv import load_dotenv, find_dotenv
import os, sys, time
from datetime import datetime

import pandas as pd
import numpy as np

import bybit

load_dotenv(find_dotenv(), override=True)
_api_key = os.getenv("API_KEY")
_api_secret = os.getenv("API_KEY_SECRET")
if not _api_key or not _api_secret:
    print("❌ API_KEY 또는 API_KEY_SECRET을 .env에서 못 찾았습니다.")
    sys.exit(1)

# ===================== 사용자 설정 =====================

SYMBOLS   = ["PUMPFUNUSDT"]

# 봉 간격 (Bybit interval string: "1","3","5","15","30","60"...)
INTERVALS = ["30"]

# Stochastic 파라미터 (심볼별로 1:1 매칭)
STOCH_PERIODS = [14]   # %K period
K_SMOOTHS     = [3]    # %K smoothing
D_SMOOTHS     = [3]    # %D smoothing

# 존(심볼별)
OVERSOLD   = [20.0]
OVERBOUGHT = [80.0]

# strict_zone=True면
#   LONG: K,D 둘다 OVERSOLD 이하에서 발생한 골든크로스만 진입
#   SHORT: K,D 둘다 OVERBOUGHT 이상에서 발생한 데드크로스만 진입
STRICT_ZONE = [True]

# 거래 설정
LEVERAGE      = "5"
PCT           = 40
COOLDOWN_BARS = 0

# 모드1 트레일링에 쓰는 doorstep(ROE 기준)
DOORSTEP      = 2.0

# TP/SL (ROE %), 심볼별
TP_ROE  = [15]
SL_ROE  = [10]

# TP_MODE
# 1: 모드1 (TP 돌파 후 조건 만족 시 ROE 피크 트레일링으로 익절)
# 2: 기본 (TP/SL 단순)
TP_MODE = [2]

# ===================== 상태 변수 =====================

position    = {s: None for s in SYMBOLS}   # "long" / "short" / None
entry_px    = {s: None for s in SYMBOLS}
init_margin = {s: None for s in SYMBOLS}
qty         = {s: None for s in SYMBOLS}

# 쿨다운(봉 기준)
last_closed_price1 = {s: None for s in SYMBOLS}
cooldown_bars      = {s: 0   for s in SYMBOLS}

# 모드1 TP 유지용
tp_hold  = {s: False for s in SYMBOLS}
roe_peak = {s: None  for s in SYMBOLS}

BASE_CASH = None

# bybit 모듈 설정
bybit.PCT = PCT
for s in SYMBOLS:
    if s not in bybit.SYMBOLS:
        bybit.SYMBOLS.append(s)


# ===================== Stochastic 계산 =====================

def _compute_stoch_series(high, low, close, period, k_smooth, d_smooth):
    """pandas Series/ndarray 받아서 stoch K,D Series 반환"""
    high_s = pd.Series(high, dtype="float64")
    low_s  = pd.Series(low, dtype="float64")
    close_s= pd.Series(close, dtype="float64")

    low_min  = low_s.rolling(window=period).min()
    high_max = high_s.rolling(window=period).max()
    denom = (high_max - low_min).replace(0, np.nan)

    k_raw = 100.0 * (close_s - low_min) / denom
    k = k_raw.rolling(window=k_smooth).mean()
    d = k.rolling(window=d_smooth).mean()
    return k, d


def get_stoch_last2(symbol, interval, period=14, k_smooth=3, d_smooth=3, limit=200):
    """
    최근 캔들로 Stoch(K,D) 계산해서
    (k_prev, d_prev, k_now, d_now) 반환
    """
    kl = bybit.get_kline_http(symbol, interval, limit=limit)  # oldest -> newest (bybit.py가 reverse 해줌)
    if not kl or len(kl) < (period + k_smooth + d_smooth + 5):
        return None

    highs  = [float(k[2]) for k in kl]
    lows   = [float(k[3]) for k in kl]
    closes = [float(k[4]) for k in kl]

    k_s, d_s = _compute_stoch_series(highs, lows, closes, period, k_smooth, d_smooth)
    k_s = k_s.dropna()
    d_s = d_s.dropna()
    if len(k_s) < 2 or len(d_s) < 2:
        return None

    # 같은 인덱스 기준으로 마지막 2개 맞추기 위해 tail(2)
    k2 = k_s.tail(2).to_numpy()
    d2 = d_s.tail(2).to_numpy()
    return float(k2[0]), float(d2[0]), float(k2[1]), float(d2[1])


def in_zone(k, d, threshold, strict_zone, is_oversold):
    if not strict_zone:
        return True
    if is_oversold:
        return (k <= threshold) and (d <= threshold)
    else:
        return (k >= threshold) and (d >= threshold)


# ===================== 유틸 =====================

def start():
    """시작 시 USDT 잔고 및 레버리지 설정"""
    global BASE_CASH
    BASE_CASH = bybit.get_usdt()
    print(f"🔧 보유금액: {BASE_CASH:.2f} USDT")
    for s in SYMBOLS:
        bybit.set_leverage(symbol=s, leverage=LEVERAGE)


def close_long(symbol):
    bybit.close_position(symbol, "Sell")


def close_short(symbol):
    bybit.close_position(symbol, "Buy")


def enter_long(symbol, px, q, leverage):
    position[symbol]    = "long"
    entry_px[symbol]    = px
    qty[symbol]         = q
    init_margin[symbol] = (px * q) / float(leverage)
    tp_hold[symbol]  = False
    roe_peak[symbol] = None


def enter_short(symbol, px, q, leverage):
    position[symbol]    = "short"
    entry_px[symbol]    = px
    qty[symbol]         = q
    init_margin[symbol] = (px * q) / float(leverage)
    tp_hold[symbol]  = False
    roe_peak[symbol] = None


def reset_after_close(symbol):
    tp_hold[symbol]  = False
    roe_peak[symbol] = None


# ===================== 메인 루프 =====================

def update():
    while True:
        for idx, symbol in enumerate(SYMBOLS):
            try:
                interval = INTERVALS[idx]

                stoch_p  = STOCH_PERIODS[idx]
                k_sm     = K_SMOOTHS[idx]
                d_sm     = D_SMOOTHS[idx]
                osd      = float(OVERSOLD[idx])
                obd      = float(OVERBOUGHT[idx])
                strict   = bool(STRICT_ZONE[idx])

                tp_roe   = float(TP_ROE[idx])
                sl_roe   = float(SL_ROE[idx])
                tp_mode  = int(TP_MODE[idx])

                # ROE/PnL (실거래 기준은 bybit에서 가져옴)
                Pnl = bybit.get_PnL(symbol)
                ROE = bybit.get_ROE(symbol)

                # 새 봉 체크(쿨다운 감소용)
                c_prev2, c_prev1, _cur3 = bybit.get_close_price(symbol, interval=interval)
                new_bar = (last_closed_price1[symbol] is None) or (last_closed_price1[symbol] != c_prev1)
                if new_bar:
                    last_closed_price1[symbol] = c_prev1
                    if cooldown_bars[symbol] > 0:
                        cooldown_bars[symbol] -= 1

                # Stochastic 계산
                st = get_stoch_last2(symbol, interval, period=stoch_p, k_smooth=k_sm, d_smooth=d_sm, limit=200)
                if st is None:
                    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 🪙 {symbol} 🕧 {interval} | 📉 Stoch 계산 데이터 부족")
                    time.sleep(2)
                    continue

                k_prev, d_prev, k_now, d_now = st
                bull_cross = (k_prev < d_prev) and (k_now > d_now)
                bear_cross = (k_prev > d_prev) and (k_now < d_now)

                # ===== 1) 포지션 없음 & 쿨다운 끝 → 진입 =====
                if position[symbol] is None and cooldown_bars[symbol] == 0:
                    # LONG: 골든크로스 + (strict_zone면 oversold에서만)
                    if bull_cross and in_zone(k_now, d_now, osd, strict, is_oversold=True):
                        px, q = bybit.entry_position(symbol, "Buy", LEVERAGE)
                        if q > 0 and px is not None:
                            enter_long(symbol, px, q, LEVERAGE)
                            cooldown_bars[symbol] = COOLDOWN_BARS

                    # SHORT: 데드크로스 + (strict_zone면 overbought에서만)
                    if position[symbol] is None and cooldown_bars[symbol] == 0:
                        if bear_cross and in_zone(k_now, d_now, obd, strict, is_oversold=False):
                            px, q = bybit.entry_position(symbol, "Sell", LEVERAGE)
                            if q > 0 and px is not None:
                                enter_short(symbol, px, q, LEVERAGE)
                                cooldown_bars[symbol] = COOLDOWN_BARS

                # ===== 2) 포지션 보유 시 청산 로직 =====
                closed = False

                if position[symbol] == "short":
                    roe = ROE

                    # (a) SL 먼저
                    if roe <= -sl_roe:
                        close_short(symbol)
                        closed = True

                    # (b) TP 처리
                    if not closed:
                        if tp_mode == 1:
                            # TP 첫 돌파
                            if not tp_hold[symbol] and roe >= tp_roe:
                                tp_hold[symbol] = True
                                roe_peak[symbol] = roe

                            if tp_hold[symbol]:
                                # 숏의 "유지 조건": (더 내려가서) oversold에 진입해있으면 트레일링, 아니면 그냥 익절
                                if (k_now <= osd) and (d_now <= osd):
                                    if roe > (roe_peak[symbol] or roe):
                                        roe_peak[symbol] = roe
                                    if (roe_peak[symbol] - roe) >= DOORSTEP:
                                        close_short(symbol)
                                        closed = True
                                else:
                                    close_short(symbol)
                                    closed = True
                        else:
                            if roe >= tp_roe or roe <= -sl_roe:
                                close_short(symbol)
                                closed = True

                elif position[symbol] == "long":
                    roe = ROE

                    # (a) SL 먼저
                    if roe <= -sl_roe:
                        close_long(symbol)
                        closed = True

                    # (b) TP 처리
                    if not closed:
                        if tp_mode == 1:
                            if not tp_hold[symbol] and roe >= tp_roe:
                                tp_hold[symbol] = True
                                roe_peak[symbol] = roe

                            if tp_hold[symbol]:
                                # 롱의 "유지 조건": overbought에 진입해있으면 트레일링, 아니면 그냥 익절
                                if (k_now >= obd) and (d_now >= obd):
                                    if roe > (roe_peak[symbol] or roe):
                                        roe_peak[symbol] = roe
                                    if (roe_peak[symbol] - roe) >= DOORSTEP:
                                        close_long(symbol)
                                        closed = True
                                else:
                                    close_long(symbol)
                                    closed = True
                        else:
                            if roe >= tp_roe or roe <= -sl_roe:
                                close_long(symbol)
                                closed = True

                if closed:
                    position[symbol]    = None
                    entry_px[symbol]    = None
                    qty[symbol]         = None
                    init_margin[symbol] = None
                    cooldown_bars[symbol] = COOLDOWN_BARS
                    reset_after_close(symbol)

                # ===== 상태 출력 =====
                print(
                    f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                    f"🪙 {symbol} 🕧 {interval} | 🚩포지션:{position[symbol]} "
                    f"| STOCH K:{k_now:.2f} D:{d_now:.2f} "
                    f"|💸 PnL:{Pnl:.3f} |💎 ROE:{ROE:.2f} "
                )

            except Exception as e:
                print(f"[ERROR] {symbol}: {type(e).__name__} {e}")

            time.sleep(5)

        time.sleep(3)


start()
update()
