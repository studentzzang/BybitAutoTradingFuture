import os
import time
from datetime import datetime, timezone
from typing import Optional, Literal

import numpy as np
import pandas as pd
from pybit.unified_trading import HTTP

# =========================
# "유튜버가 실제로 하는 방식"에 가깝게(재량을 규칙화) 만든 버전
# - EMA9/EMA21: 방향(레짐) 필터
# - RSI(9), 레벨 21: "되돌림 후 재가속" 진입
# - 같은 방향으로 여러 번 재진입 가능(쿨타임)
# - SL: 직전 캔들 뒤
# - 관리: +1R에서 본절(BE), 2R 목표/또는 역교차/RSI 약화 시 청산
# =========================

# ====== 사용자 설정 ======
SYMBOLS = ["ETHUSDT"]
LEVERAGE = 5
TIMEFRAMES = [1]

EQUITY = 100.0
START = "2025-11-15"
END   = "2025-01-05"
OUT_DIR = "test"
MAX_CANDLES = 50000

# ====== 지표 설정 (유튜브) ======
EMA_FAST = 9
EMA_SLOW = 21
RSI_PERIOD = 9
RSI_LEVEL = 21.0

# ====== "영상처럼 트레이드 수 나오는" 핵심 완화 파라미터 ======
# 레짐(EMA9>EMA21이면 롱만, EMA9<EMA21이면 숏만)
USE_REGIME_FILTER = True

# RSI가 레벨을 '방금' 통과한 경우 뿐 아니라,
# 최근 LOOKBACK_BARS 안에 레벨 반대편에 있었고(되돌림),
# 지금은 레벨을 다시 회복/이탈하면서(재가속) 진입
LOOKBACK_BARS = 10

# RSI가 레벨을 넘었다고 바로 들어가면 노이즈 많아서,
# "상승/하락 모멘텀" 확인용: 최근 2~3봉 RSI가 같은 방향인지
MOMENTUM_BARS = 2  # 2면 rsi[i] > rsi[i-1] > rsi[i-2] (롱)
# RSI 50 조건은 영상에 "롱은 50 이상으로 이동"이 있었지만,
# 실제 영상 매매는 종종 50 전에 들어가서, 옵션으로 둠
REQUIRE_RSI_50_FOR_LONG = False

# 재진입 허용을 위한 쿨타임 (같은 방향 연속 스캘핑)
COOLDOWN_BARS = 3

# ====== 청산/관리 ======
# 1) 기본 SL: 진입 직전 캔들 뒤
# 2) +1R 도달 시 SL을 진입가(본절)로 이동
MOVE_SL_TO_BE_AT_R = 1.0

# 3) TP: 2R (영상의 1:2 예시)
RR_MULT = 2.0

# 4) 역교차(EMA9/21) 나오면 청산
CLOSE_ON_REVERSE_CROSS = True

# 5) RSI 약화(롱인데 RSI가 꺾임 / 숏인데 RSI가 꺾임)면 청산 (영상의 "구매력 감소")
CLOSE_ON_RSI_WEAKNESS = True

# RSI 약화 판정: 롱은 rsi가 최근 WEAK_BARS 동안 하락이면 약화로 보고 종료
WEAK_BARS = 2

# ====== 비용(영상엔 없어서 0) ======
FEE_RATE = 0.0
SLIPPAGE_RATE = 0.0

session = HTTP(testnet=False)

# ---------- 유틸 ----------
def parse_date_ms(s: str) -> int:
    dt = datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)

def ms_to_dt(ms: int) -> str:
    return datetime.fromtimestamp(ms // 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

def calc_ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def calc_rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = (-delta).clip(lower=0)
    roll_up = up.ewm(alpha=1 / period, adjust=False).mean()
    roll_down = down.ewm(alpha=1 / period, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-12)
    return 100 - (100 / (1 + rs))

def fetch_ohlcv_10000(symbol: str, tf_min: int, end_ms: Optional[int] = None) -> pd.DataFrame:
    interval = str(tf_min)
    if end_ms is None:
        end_ms = int(datetime.now(tz=timezone.utc).timestamp() * 1000)

    rows = []
    while len(rows) < MAX_CANDLES:
        r = session.get_kline(category="linear", symbol=symbol, interval=interval, end=end_ms, limit=1000)
        if r.get("retCode") != 0:
            break
        lst = r.get("result", {}).get("list", [])
        if not lst:
            break

        for it in lst:
            rows.append((int(it[0]), float(it[1]), float(it[2]), float(it[3]), float(it[4]), float(it[5])))

        end_ms = min(int(x[0]) for x in lst) - 1
        if len(lst) < 1000:
            break

        time.sleep(0.02)

    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"]).drop_duplicates("ts")
    df.sort_values("ts", inplace=True)
    df.reset_index(drop=True, inplace=True)

    start_ms = parse_date_ms(START)
    end_ms2 = parse_date_ms(END) + 24 * 60 * 60 * 1000 - 1
    #df = df[(df["ts"] >= start_ms) & (df["ts"] <= end_ms2)].copy()
    df.reset_index(drop=True, inplace=True)
    return df

def apply_costs(entry_px: float, exit_px: float, qty: float) -> float:
    notional_entry = entry_px * qty
    notional_exit = exit_px * qty
    fee = (notional_entry + notional_exit) * FEE_RATE
    slip = (notional_entry + notional_exit) * SLIPPAGE_RATE
    return fee + slip

def calc_unreal(pos: str, entry_px: float, px: float, qty: float) -> float:
    return (px - entry_px) * qty if pos == "LONG" else (entry_px - px) * qty

# ---------- 진입 보조 함수 ----------
def is_momentum_up(rsi_series: np.ndarray, idx: int, bars: int) -> bool:
    # rsi[idx] > rsi[idx-1] > ... > rsi[idx-bars]
    for k in range(bars):
        if not (rsi_series[idx - k] > rsi_series[idx - k - 1]):
            return False
    return True

def is_momentum_down(rsi_series: np.ndarray, idx: int, bars: int) -> bool:
    for k in range(bars):
        if not (rsi_series[idx - k] < rsi_series[idx - k - 1]):
            return False
    return True

def was_below_level_recently(rsi_series: np.ndarray, idx: int, level: float, lookback: int) -> bool:
    start = max(0, idx - lookback)
    return np.nanmin(rsi_series[start:idx+1]) < level

def was_above_level_recently(rsi_series: np.ndarray, idx: int, level: float, lookback: int) -> bool:
    start = max(0, idx - lookback)
    return np.nanmax(rsi_series[start:idx+1]) > level

# ---------- 시뮬레이션 ----------
def run_simulation(symbol: str, tf: int):
    ohlc = fetch_ohlcv_10000(symbol, tf)
    if ohlc.empty or len(ohlc) < 200:
        print(f"⚠️ 데이터 부족: {symbol} tf={tf}")
        return

    ohlc["ema9"] = calc_ema(ohlc["close"], EMA_FAST)
    ohlc["ema21"] = calc_ema(ohlc["close"], EMA_SLOW)
    ohlc["rsi"] = calc_rsi(ohlc["close"], RSI_PERIOD)

    ohlc.dropna(inplace=True)
    ohlc.reset_index(drop=True, inplace=True)

    ema9 = ohlc["ema9"].to_numpy()
    ema21 = ohlc["ema21"].to_numpy()
    rsi = ohlc["rsi"].to_numpy()
    close = ohlc["close"].to_numpy()
    low = ohlc["low"].to_numpy()
    high = ohlc["high"].to_numpy()
    ts = ohlc["ts"].to_numpy()

    log = []

    position: Optional[str] = None
    entry_px = 0.0
    qty = 0.0
    init_margin = 0.0

    stop_px = 0.0
    target_px = 0.0
    risk_per_unit = 0.0

    last_exit_i = -10**9  # 쿨타임용

    for i in range(2, len(ohlc)):
        px = float(close[i])

        # 레짐(방향) 판단
        long_regime = (ema9[i] > ema21[i])
        short_regime = (ema9[i] < ema21[i])

        # 역교차 감지(청산용)
        bull_cross = (ema9[i - 1] <= ema21[i - 1]) and (ema9[i] > ema21[i])
        bear_cross = (ema9[i - 1] >= ema21[i - 1]) and (ema9[i] < ema21[i])

        # ===== 포지션 관리/청산 =====
        if position is not None:
            unreal = calc_unreal(position, entry_px, px, qty)
            roe = (unreal / init_margin) * 100.0 if init_margin > 0 else 0.0

            # SL 체크
            stop_hit = (px <= stop_px) if position == "LONG" else (px >= stop_px)
            if stop_hit:
                cost = apply_costs(entry_px, px, qty)
                pnl = unreal - cost
                roe_net = (pnl / init_margin) * 100.0 if init_margin > 0 else 0.0
                log.append([ms_to_dt(int(ts[i])), symbol, tf, "CLOSE_SL", entry_px, px, float(rsi[i]), float(ema9[i]), float(ema21[i]), stop_px, target_px, pnl, roe_net])
                position = None
                last_exit_i = i
                continue

            # +1R 도달 시 본절로 SL 이동 (영상 예시)
            if risk_per_unit > 0:
                if position == "LONG" and px >= entry_px + MOVE_SL_TO_BE_AT_R * risk_per_unit:
                    stop_px = max(stop_px, entry_px)
                if position == "SHORT" and px <= entry_px - MOVE_SL_TO_BE_AT_R * risk_per_unit:
                    stop_px = min(stop_px, entry_px)

            # TP(2R) 도달 시 익절
            if target_px > 0:
                tp_hit = (px >= target_px) if position == "LONG" else (px <= target_px)
                if tp_hit:
                    cost = apply_costs(entry_px, px, qty)
                    pnl = unreal - cost
                    roe_net = (pnl / init_margin) * 100.0 if init_margin > 0 else 0.0
                    log.append([ms_to_dt(int(ts[i])), symbol, tf, "CLOSE_TP", entry_px, px, float(rsi[i]), float(ema9[i]), float(ema21[i]), stop_px, target_px, pnl, roe_net])
                    position = None
                    last_exit_i = i
                    continue

            # 역교차 청산
            if CLOSE_ON_REVERSE_CROSS:
                if position == "LONG" and bear_cross:
                    cost = apply_costs(entry_px, px, qty)
                    pnl = unreal - cost
                    roe_net = (pnl / init_margin) * 100.0 if init_margin > 0 else 0.0
                    log.append([ms_to_dt(int(ts[i])), symbol, tf, "CLOSE_REV", entry_px, px, float(rsi[i]), float(ema9[i]), float(ema21[i]), stop_px, target_px, pnl, roe_net])
                    position = None
                    last_exit_i = i
                    continue
                if position == "SHORT" and bull_cross:
                    cost = apply_costs(entry_px, px, qty)
                    pnl = unreal - cost
                    roe_net = (pnl / init_margin) * 100.0 if init_margin > 0 else 0.0
                    log.append([ms_to_dt(int(ts[i])), symbol, tf, "CLOSE_REV", entry_px, px, float(rsi[i]), float(ema9[i]), float(ema21[i]), stop_px, target_px, pnl, roe_net])
                    position = None
                    last_exit_i = i
                    continue

            # RSI 약화 청산 (구매력 감소)
            if CLOSE_ON_RSI_WEAKNESS and i >= WEAK_BARS + 1:
                if position == "LONG":
                    # RSI가 연속 하락이면 약화
                    weak = True
                    for k in range(WEAK_BARS):
                        if not (rsi[i - k] < rsi[i - k - 1]):
                            weak = False
                            break
                    if weak:
                        cost = apply_costs(entry_px, px, qty)
                        pnl = unreal - cost
                        roe_net = (pnl / init_margin) * 100.0 if init_margin > 0 else 0.0
                        log.append([ms_to_dt(int(ts[i])), symbol, tf, "CLOSE_RSI_WEAK", entry_px, px, float(rsi[i]), float(ema9[i]), float(ema21[i]), stop_px, target_px, pnl, roe_net])
                        position = None
                        last_exit_i = i
                        continue

                if position == "SHORT":
                    weak = True
                    for k in range(WEAK_BARS):
                        if not (rsi[i - k] > rsi[i - k - 1]):
                            weak = False
                            break
                    if weak:
                        cost = apply_costs(entry_px, px, qty)
                        pnl = unreal - cost
                        roe_net = (pnl / init_margin) * 100.0 if init_margin > 0 else 0.0
                        log.append([ms_to_dt(int(ts[i])), symbol, tf, "CLOSE_RSI_WEAK", entry_px, px, float(rsi[i]), float(ema9[i]), float(ema21[i]), stop_px, target_px, pnl, roe_net])
                        position = None
                        last_exit_i = i
                        continue

            # 포지션 유지
            continue

        # ===== 포지션 없으면 진입(여기서 트레이드 수가 갈림) =====
        if i - last_exit_i <= COOLDOWN_BARS:
            continue

        # (유튜브 개념) EMA 교차는 "추세가 바뀌었다" 신호지만,
        # 실제 매매는 교차 직후뿐 아니라 "레짐 유지 중 RSI 되돌림->재가속"에서 여러 번 들어감.

        # 레짐 필터
        if USE_REGIME_FILTER:
            allow_long = long_regime
            allow_short = short_regime
        else:
            allow_long = True
            allow_short = True

        # RSI 되돌림->회복(롱)
        # - 최근 LOOKBACK 안에 RSI가 21 아래로 내려간 적이 있고(되돌림)
        # - 지금 RSI가 21 위로 회복했거나(회복)
        # - RSI 모멘텀이 상승중
        long_reclaim = (rsi[i] >= RSI_LEVEL) and was_below_level_recently(rsi, i, RSI_LEVEL, LOOKBACK_BARS)
        long_momo = is_momentum_up(rsi, i, MOMENTUM_BARS) if i >= MOMENTUM_BARS + 1 else False

        # RSI 되돌림->이탈(숏)
        short_reject = (rsi[i] <= RSI_LEVEL) and was_above_level_recently(rsi, i, RSI_LEVEL, LOOKBACK_BARS)
        short_momo = is_momentum_down(rsi, i, MOMENTUM_BARS) if i >= MOMENTUM_BARS + 1 else False

        # 롱 진입
        if allow_long and long_reclaim and long_momo and (not REQUIRE_RSI_50_FOR_LONG or rsi[i] >= 50.0):
            position = "LONG"
            entry_px = px
            qty = (EQUITY * LEVERAGE) / entry_px
            init_margin = EQUITY

            # SL: "이전 캔들 뒤"
            stop_px = float(low[i - 1])
            risk_per_unit = max(entry_px - stop_px, 0.0)
            target_px = entry_px + RR_MULT * risk_per_unit if risk_per_unit > 0 else 0.0

            log.append([ms_to_dt(int(ts[i])), symbol, tf, "OPEN_LONG", entry_px, px, float(rsi[i]), float(ema9[i]), float(ema21[i]), stop_px, target_px, 0.0, 0.0])
            continue

        # 숏 진입
        if allow_short and short_reject and short_momo:
            position = "SHORT"
            entry_px = px
            qty = (EQUITY * LEVERAGE) / entry_px
            init_margin = EQUITY

            stop_px = float(high[i - 1])
            risk_per_unit = max(stop_px - entry_px, 0.0)
            target_px = entry_px - RR_MULT * risk_per_unit if risk_per_unit > 0 else 0.0

            log.append([ms_to_dt(int(ts[i])), symbol, tf, "OPEN_SHORT", entry_px, px, float(rsi[i]), float(ema9[i]), float(ema21[i]), stop_px, target_px, 0.0, 0.0])
            continue

    # 저장
    if log:
        os.makedirs(OUT_DIR, exist_ok=True)
        df = pd.DataFrame(
            log,
            columns=[
                "datetime", "symbol", "timeframe", "action",
                "entry_px", "close_px", "rsi", "ema9", "ema21",
                "stop_px", "target_px", "pnl", "roe"
            ],
        )
        fname = (
            f"YT_LIKE_EMA9_21_RSI9_L{int(RSI_LEVEL)}_{symbol}"
            f"_tf{tf}_LEV{LEVERAGE}_LB{LOOKBACK_BARS}_CD{COOLDOWN_BARS}.csv"
        )
        path = os.path.join(OUT_DIR, fname)
        df.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"✅ 저장: {fname} (기록: {len(df)}줄)")
    else:
        print(f"⚠️ 거래 없음: {symbol} tf={tf}")

if __name__ == "__main__":
    print("--- (영상 실매매 근사) EMA9/21 + RSI(9, level 21) 스캘핑 ---")
    for s in SYMBOLS:
        for tf in TIMEFRAMES:
            run_simulation(s, tf)
