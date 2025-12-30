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
TIMEFRAME = [5, 15, 30]
RSI_PERIOD = [7, 9, 12]

# ✅ ADX 파라미터 (기간, 추세판단 임계값)
ADX_PERIOD = [7, 10, 14]
ADX_MIN_ARR = [10, 15, 20]

# ✅ 추세 모드 RSI 눌림/되돌림 트리거 (배열로 경우의 수)
TREND_LONG_PULLBACK_RSI_ARR  = [35, 42.0, 45.0, 48.0]
TREND_SHORT_PULLBACK_RSI_ARR = [65, 58.0, 55.0, 52.0]

EQUITY = 100.0
START = "2025-09-01"
END = "2025-12-28"
OUT_DIR = "test"

MAX_CANDLES = 15000

# ====== RSI 트리거 값 (횡보/비추세 모드 평균회귀) ======
OPEN_SHORT_RSI = 72.0
OPEN_LONG_RSI = 28.0

# ====== TP / SL 배열 ======
TP_ROE_ARR = [7, 10]
SL_ROE_ARR = [10, 12.5]
TP_MODE_ARR = [2]

# ✅ 연쇄 손실 방지(핵심)
MAX_CONSEC_SL_SAME_DIR = 2     # 같은 방향 SL 연속 N회면 차단
COOLDOWN_BARS = 6              # 차단 후 N봉 동안 진입 금지
UNBLOCK_ON_DI_FLIP = True      # DI 방향 바뀌면 즉시 차단 해제

session = HTTP()

# ---------- 유틸리티 ----------
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
    return str(tf)

def fetch_ohlcv_10000(symbol: str, tf: str, start_ms=None, end_ms=None, max_candles: int = MAX_CANDLES) -> pd.DataFrame:
    interval = bybit_interval(tf)
    if end_ms is None:
        end_ms = int(datetime.now(tz=timezone.utc).timestamp() * 1000)

    rows = []
    last_min_ts = None
    max_calls = 80
    calls = 0

    while len(rows) < max_candles and calls < max_calls:
        calls += 1

        resp = None
        for attempt in range(3):
            try:
                r = session.get_kline(
                    category="linear",
                    symbol=symbol,
                    interval=str(interval),
                    end=end_ms,
                    limit=1000
                )
                if r.get("retCode") == 0:
                    resp = r
                    break
            except Exception:
                pass
            time.sleep(0.35)

        if resp is None:
            break

        lst = resp.get("result", {}).get("list", [])
        if not lst:
            break

        min_ts_this_page = min(int(x[0]) for x in lst)

        # 같은 구간 반복 방지
        if last_min_ts is not None and min_ts_this_page >= last_min_ts:
            break
        last_min_ts = min_ts_this_page

        for it in lst:
            ts = int(it[0])
            o, h, l, c, v = map(float, it[1:6])
            if start_ms is not None and ts < start_ms:
                continue
            rows.append((ts, o, h, l, c, v))

        end_ms = min_ts_this_page - 1
        if start_ms is not None and end_ms < start_ms:
            break

        time.sleep(0.12)

    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"]).drop_duplicates("ts")
    df.sort_values("ts", inplace=True)
    df.reset_index(drop=True, inplace=True)

    if start_ms is not None:
        df = df[df["ts"] >= start_ms].copy()
    if end_ms is not None:
        df = df[df["ts"] <= parse_date(END)].copy()

    df.reset_index(drop=True, inplace=True)
    return df.head(max_candles)

def compute_rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0)
    down = (-delta).clip(lower=0)
    roll_up = up.ewm(alpha=1/period, adjust=False).mean()
    roll_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-12)
    return 100 - (100 / (1 + rs))

def compute_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int):
    high = high.astype(float)
    low = low.astype(float)
    close = close.astype(float)

    prev_high = high.shift(1)
    prev_low = low.shift(1)
    prev_close = close.shift(1)

    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)

    up_move = high - prev_high
    down_move = prev_low - low

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    plus_dm = pd.Series(plus_dm, index=high.index)
    minus_dm = pd.Series(minus_dm, index=high.index)

    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    plus_dm_sm = plus_dm.ewm(alpha=1/period, adjust=False).mean()
    minus_dm_sm = minus_dm.ewm(alpha=1/period, adjust=False).mean()

    plus_di = 100 * (plus_dm_sm / (atr + 1e-12))
    minus_di = 100 * (minus_dm_sm / (atr + 1e-12))

    dx = 100 * ((plus_di - minus_di).abs() / ((plus_di + minus_di) + 1e-12))
    adx = dx.ewm(alpha=1/period, adjust=False).mean()

    return adx, plus_di, minus_di

# ---------- 시뮬레이션 ----------
def run(symbol: str, tf: str,
        rsi_period: int,
        adx_period: int,
        adx_min: float,
        trend_long_pb: float,
        trend_short_pb: float,
        leverage: float, equity: float,
        start: Optional[str], end: Optional[str], out_dir: str,
        tp_roe: float, sl_roe: float, tp_mode: int) -> str:

    start_ms = parse_date(start)
    end_ms = parse_date(end)

    ohlc = fetch_ohlcv_10000(symbol, tf, start_ms, end_ms)
    if ohlc.empty:
        return ""

    ohlc["rsi"] = compute_rsi(ohlc["close"], rsi_period)
    ohlc["adx"], ohlc["pdi"], ohlc["mdi"] = compute_adx(ohlc["high"], ohlc["low"], ohlc["close"], adx_period)

    cols = [
        "datetime", "symbol", "timeframe",
        "close", "rsi", "adx", "pdi", "mdi",
        "adx_period", "adx_min",
        "trend_long_pb", "trend_short_pb",
        "block_dir", "cooldown_left",
        "포지션", "비고", "entry_price", "미실현PnL", "ROE"
    ]
    log = []

    position = None
    entry_px = None
    qty = None
    init_margin = None

    # ✅ 연쇄 손실 차단 상태
    block_dir = None            # "UP" or "DOWN" (DI 기반 추세 방향)
    consec_sl_up = 0
    consec_sl_down = 0
    cooldown_left = 0

    prev_trend_dir = None

    for i in range(len(ohlc)):
        ts = int(ohlc.loc[i, "ts"]) // 1000
        dt = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        px = float(ohlc.loc[i, "close"])

        rv = ohlc.loc[i, "rsi"]
        ax = ohlc.loc[i, "adx"]
        pdi = ohlc.loc[i, "pdi"]
        mdi = ohlc.loc[i, "mdi"]

        rv = float(rv) if not np.isnan(rv) else None
        ax = float(ax) if not np.isnan(ax) else None
        pdi = float(pdi) if not np.isnan(pdi) else None
        mdi = float(mdi) if not np.isnan(mdi) else None

        remark = ""
        unreal = 0.0
        roe = 0.0

        # cooldown tick
        if cooldown_left > 0:
            cooldown_left -= 1

        trend_mode = (ax is not None) and (ax >= adx_min) and (pdi is not None) and (mdi is not None)
        trend_dir = None
        if trend_mode:
            trend_dir = "UP" if (pdi > mdi) else "DOWN"

        # ✅ DI 방향이 뒤집히면(추세 전환) 차단 해제 옵션
        if UNBLOCK_ON_DI_FLIP and trend_dir is not None and prev_trend_dir is not None:
            if trend_dir != prev_trend_dir:
                block_dir = None
                cooldown_left = 0
                consec_sl_up = 0
                consec_sl_down = 0
        if trend_dir is not None:
            prev_trend_dir = trend_dir

        # ✅ 진입 차단 판단
        blocked_now = (cooldown_left > 0) or (block_dir is not None and trend_dir == block_dir)

        # === 진입 ===
        if position is None and rv is not None and not blocked_now:
            if trend_mode:
                if trend_dir == "UP":
                    if rv <= trend_long_pb:
                        position = "LONG"
                        entry_px = px
                        notional = equity * leverage
                        qty = notional / entry_px
                        init_margin = equity
                        remark = f"LONG 진입 (TREND ADX{ax:.1f} UP, RSI<= {trend_long_pb})"
                else:
                    if rv >= trend_short_pb:
                        position = "SHORT"
                        entry_px = px
                        notional = equity * leverage
                        qty = notional / entry_px
                        init_margin = equity
                        remark = f"SHORT 진입 (TREND ADX{ax:.1f} DOWN, RSI>= {trend_short_pb})"
            else:
                if rv >= OPEN_SHORT_RSI:
                    position = "SHORT"
                    entry_px = px
                    notional = equity * leverage
                    qty = notional / entry_px
                    init_margin = equity
                    remark = f"SHORT 진입 (MR RSI>= {OPEN_SHORT_RSI})"
                elif rv <= OPEN_LONG_RSI:
                    position = "LONG"
                    entry_px = px
                    notional = equity * leverage
                    qty = notional / entry_px
                    init_margin = equity
                    remark = f"LONG 진입 (MR RSI<= {OPEN_LONG_RSI})"

        # === 보유 중 ===
        elif position is not None and rv is not None:
            if position == "LONG":
                unreal = (px - entry_px) * qty
                roe = (unreal / init_margin) * 100
                if tp_mode == 2:
                    if roe >= tp_roe:
                        remark = f"close LONG (TP {roe:.1f}%)"
                    elif roe <= -sl_roe:
                        remark = f"close LONG (SL {roe:.1f}%)"

            elif position == "SHORT":
                unreal = (entry_px - px) * qty
                roe = (unreal / init_margin) * 100
                if tp_mode == 2:
                    if roe >= tp_roe:
                        remark = f"close SHORT (TP {roe:.1f}%)"
                    elif roe <= -sl_roe:
                        remark = f"close SHORT (SL {roe:.1f}%)"

            # 청산 처리
            if remark.startswith("close"):
                # ✅ SL이면 연속손실 카운트 누적 + 차단
                if "(SL" in remark:
                    loss_dir = "UP" if position == "LONG" else "DOWN"
                    if loss_dir == "UP":
                        consec_sl_up += 1
                        consec_sl_down = 0
                        if consec_sl_up >= MAX_CONSEC_SL_SAME_DIR:
                            block_dir = "UP"
                            cooldown_left = COOLDOWN_BARS
                    else:
                        consec_sl_down += 1
                        consec_sl_up = 0
                        if consec_sl_down >= MAX_CONSEC_SL_SAME_DIR:
                            block_dir = "DOWN"
                            cooldown_left = COOLDOWN_BARS
                else:
                    # TP면 해당 방향 연속 SL 카운트 리셋
                    if position == "LONG":
                        consec_sl_up = 0
                    else:
                        consec_sl_down = 0

                # 포지션 정리
                position = None
                entry_px = None
                qty = None
                init_margin = None

        # === 로그 기록(진입/청산/차단상태 포함) ===
        if remark:
            log.append([
                dt, symbol, tf,
                px, rv, ax, pdi, mdi,
                adx_period, adx_min,
                trend_long_pb, trend_short_pb,
                block_dir, cooldown_left,
                ("CLOSE" if "close" in remark else "OPEN"),
                remark, entry_px if entry_px is not None else np.nan, unreal, roe
            ])

    if not log:
        return ""

    df = pd.DataFrame(log, columns=cols)
    os.makedirs(out_dir, exist_ok=True)

    fname = (
        f"{symbol}_{tf}_R{rsi_period}"
        f"_ADX{adx_period}_MIN{adx_min}"
        f"_LPB{trend_long_pb}_SPB{trend_short_pb}"
        f"_TP{tp_roe}_SL{sl_roe}_M{tp_mode}"
        f"_CSL{MAX_CONSEC_SL_SAME_DIR}_CD{COOLDOWN_BARS}.csv"
    )
    path = os.path.join(out_dir, fname)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path

# ---------- 실행 ----------
if __name__ == "__main__":
    print("--- RSI + ADX(DI) + Trend Pullback + 연속SL차단 백테스팅 시작 ---")
    for s in _as_list(SYMBOL):
        for tf in _as_list(TIMEFRAME):
            for rp in _as_list(RSI_PERIOD):
                for ap in _as_list(ADX_PERIOD):
                    for amin in _as_list(ADX_MIN_ARR):
                        for lpb in _as_list(TREND_LONG_PULLBACK_RSI_ARR):
                            for spb in _as_list(TREND_SHORT_PULLBACK_RSI_ARR):
                                for tp in TP_ROE_ARR:
                                    for sl in SL_ROE_ARR:
                                        for mode in TP_MODE_ARR:
                                            csv_path = run(
                                                s, str(tf),
                                                rp, ap, amin,
                                                float(lpb), float(spb),
                                                LEVERAGE, EQUITY,
                                                START, END, OUT_DIR,
                                                tp, sl, mode
                                            )
                                            if csv_path:
                                                print(f"✅ 완료: {os.path.basename(csv_path)}")
