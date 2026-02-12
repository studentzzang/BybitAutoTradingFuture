import os
import time
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
import numpy as np
from pybit.unified_trading import HTTP

# ====== 사용자 설정 변수 ======
SYMBOL = ["PUMPFUNUSDT", "FARTCOINUSDT"]
LEVERAGE = 3
TIMEFRAME = [3,5, 15, 30]
RSI_PERIOD = [5, 7, 9, 12]
EQUITY = 100.0
START = "2025-12-30"
END = "2026-02-08"
OUT_DIR = "test"

MAX_CANDLES = 30000

# RSI 트리거 값 (원본 설정값)
OPEN_SHORT_RSI = 72.0   # 숏 진입 기준
OPEN_LONG_RSI = 28.0    # 롱 진입 기준
CLOSE_SHORT_RSI = 70.0
CLOSE_LONG_RSI = 30.0

# DOORSTEP 밴드 (원본 로직 전용)
DOORSTEP = 3.0

# ====== TP / SL 배열 ======
TP_ROE_ARR = [3, 5, 10]
SL_ROE_ARR = [3, 5, 10]
TP_MODE_ARR = [1, 2]   # 1 = DOORSTEP(청산) + 진입=즉시, 2 = 진입=재돌파(진입방지) + 청산=TP/SL
# ==========================

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

def fetch_ohlcv_10000(
    symbol: str,
    tf: str,
    start_ms=None,
    end_ms=None,
    max_candles: int = MAX_CANDLES
) -> pd.DataFrame:
    interval = bybit_interval(tf)
    if end_ms is None:
        end_ms = int(datetime.now(tz=timezone.utc).timestamp() * 1000)

    rows = []
    while len(rows) < max_candles:
        resp = None
        for attempt in range(3):
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
            except Exception:
                pass
            time.sleep(0.4)

        if resp is None:
            break

        lst = resp["result"]["list"]
        if not lst:
            break
        
        batch_min_ts = None

        for it in lst:
            ts = int(it[0])
            o, h, l, c, v = map(float, it[1:6])
            rows.append((ts, o, h, l, c, v))
            if batch_min_ts is None or ts < batch_min_ts:
                batch_min_ts = ts

        end_ms = min(int(x[0]) for x in lst) - 1

        if start_ms is not None and batch_min_ts is not None and batch_min_ts <= start_ms:
            break

        if len(lst) < 1000:
            break
        time.sleep(0.12)

    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"]).drop_duplicates("ts")
    df.sort_values("ts", inplace=True)
    df.reset_index(drop=True, inplace=True)

    if start_ms is not None:
        df = df[df["ts"] >= start_ms]
        df.reset_index(drop=True, inplace=True)
    if end_ms is not None:

        pass

    return df.head(max_candles)


def compute_rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0)
    down = (-delta).clip(lower=0)
    roll_up = up.ewm(alpha=1/period, adjust=False).mean()
    roll_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-12)
    return 100 - (100 / (1 + rs))

# ---------- 시뮬레이션 ----------
def run(
    symbol: str,
    tf: str,
    rsi_period: int,
    leverage: float,
    equity: float,
    start: Optional[str],
    end: Optional[str],
    out_dir: str,
    tp_roe: float,
    sl_roe: float,
    tp_mode: int
) -> str:

    start_ms = parse_date(start)
    end_ms = parse_date(end)

    ohlc = fetch_ohlcv_10000(symbol, tf, start_ms, end_ms)
    if ohlc.empty:
        return ""

    ohlc["rsi"] = compute_rsi(ohlc["close"], rsi_period)

    # ✅ 원금 누적 반영을 위해 current_equity 사용
    current_equity = float(equity)

    cols = [
        "datetime", "symbol", "timeframe",
        "close", "rsi",
        "포지션", "비고",
        "entry_price",
        "미실현PnL", "ROE",
        "원금_before", "원금_after"
    ]
    log = []

    position = None
    entry_px = None
    qty = None
    init_margin = None  # 진입 당시 원금(마진) 기록용

    prev_rsi = None  # 직전 RSI (MODE2 재돌파 진입용)

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
            # MODE 1: 기존과 동일 (과매수/과매도 값에 닿으면 즉시 진입)
            if tp_mode == 1:
                if rv >= OPEN_SHORT_RSI:
                    position = "SHORT"
                    entry_px = px
                    notional = current_equity * leverage
                    qty = notional / entry_px
                    init_margin = current_equity
                    remark = "SHORT 진입"
                elif rv <= OPEN_LONG_RSI:
                    position = "LONG"
                    entry_px = px
                    notional = current_equity * leverage
                    qty = notional / entry_px
                    init_margin = current_equity
                    remark = "LONG 진입"

            # MODE 2: DOORSTEP 미사용. "재돌파(되돌림)"로만 진입
            # - 너무 과매수/과매도일 때 바로 진입하는 것을 방지
            # - SHORT: (이전 RSI > OPEN_SHORT_RSI)였다가 (현재 RSI <= OPEN_SHORT_RSI)로 내려오면 진입
            # - LONG : (이전 RSI < OPEN_LONG_RSI)였다가 (현재 RSI >= OPEN_LONG_RSI)로 올라오면 진입
            elif tp_mode == 2 and prev_rsi is not None:
                if prev_rsi > OPEN_SHORT_RSI and rv <= OPEN_SHORT_RSI:
                    position = "SHORT"
                    entry_px = px
                    notional = current_equity * leverage
                    qty = notional / entry_px
                    init_margin = current_equity
                    remark = "SHORT 진입 (RE-CROSS)"
                elif prev_rsi < OPEN_LONG_RSI and rv >= OPEN_LONG_RSI:
                    position = "LONG"
                    entry_px = px
                    notional = current_equity * leverage
                    qty = notional / entry_px
                    init_margin = current_equity
                    remark = "LONG 진입 (RE-CROSS)"

        # === 보유 중 ===
        elif position is not None and rv is not None:
            closed = False

            if position == "LONG":
                unreal = (px - entry_px) * qty
                roe = (unreal / init_margin) * 100

                if tp_mode == 1:
                    if roe <= -sl_roe:
                        remark = f"close LONG (SL {roe:.1f}%)"
                        closed = True
                    elif roe >= tp_roe:
                        if rv >= OPEN_SHORT_RSI:
                            if (OPEN_SHORT_RSI - DOORSTEP) <= rv <= (OPEN_SHORT_RSI + DOORSTEP):
                                remark = f"close LONG (DOORSTEP TP, ROE {roe:.1f}%)"
                                closed = True
                        else:
                            remark = f"close LONG (TP {roe:.1f}%)"
                            closed = True
                elif tp_mode == 2:
                    if roe >= tp_roe:
                        remark = f"close LONG (TP {roe:.1f}%)"
                        closed = True
                    elif roe <= -sl_roe:
                        remark = f"close LONG (SL {roe:.1f}%)"
                        closed = True

            elif position == "SHORT":
                unreal = (entry_px - px) * qty
                roe = (unreal / init_margin) * 100

                if tp_mode == 1:
                    if roe <= -sl_roe:
                        remark = f"close SHORT (SL {roe:.1f}%)"
                        closed = True
                    elif roe >= tp_roe:
                        if rv <= OPEN_LONG_RSI:
                            if (OPEN_LONG_RSI - DOORSTEP) <= rv <= (OPEN_LONG_RSI + DOORSTEP):
                                remark = f"close SHORT (DOORSTEP TP, ROE {roe:.1f}%)"
                                closed = True
                        else:
                            remark = f"close SHORT (TP {roe:.1f}%)"
                            closed = True
                elif tp_mode == 2:
                    if roe >= tp_roe:
                        remark = f"close SHORT (TP {roe:.1f}%)"
                        closed = True
                    elif roe <= -sl_roe:
                        remark = f"close SHORT (SL {roe:.1f}%)"
                        closed = True

            # === 청산 처리: pnl/roe 크기 무시하고 "부호만" 보고 TP/SL로 강제 반영 ===
            if closed:
                equity_before = current_equity

                # ✅ 승/패는 기존 unreal(또는 roe) 부호로만 판단
                is_win = (unreal >= 0)

                # ✅ 실제 pnl이 얼마든, TP/SL 퍼센트로 강제 치환
                if is_win:
                    fixed_roe = float(tp_roe)          # 예: +15%
                else:
                    fixed_roe = -float(sl_roe)         # 예: -10%

                fixed_pnl = equity_before * (fixed_roe / 100.0)
                current_equity = equity_before + fixed_pnl

                # CSV에는 "고정 반영된 결과"만 기록 (열 이름/구조 유지)
                log.append([
                    dt, symbol, tf,
                    px, rv,
                    "CLOSE", f"{remark} | FIXED {'TP' if is_win else 'SL'}",
                    entry_px,
                    fixed_pnl, fixed_roe,
                    equity_before, current_equity
                ])

                # 포지션 정리
                position = None
                entry_px = None
                qty = None
                init_margin = None

        # 다음 캔들을 위한 prev_rsi 갱신 (NaN이면 갱신 안 함)
        if rv is not None:
            prev_rsi = rv

    if not log:
        return ""

    df = pd.DataFrame(log, columns=cols)
    os.makedirs(out_dir, exist_ok=True)

    # ✅ 파일명/경로 형식 절대 변경 X
    fname = f"{symbol}_{tf}_R{rsi_period}_TP{tp_roe}_SL{sl_roe}_M{tp_mode}.csv"
    path = os.path.join(out_dir, fname)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path

# ---------- 실행 ----------
if __name__ == "__main__":
    print("--- RSI 원본 로직 백테스팅 시작 (청산 시 TP/SL 퍼센트로 강제 치환, 원금 누적 반영) ---")
    for s in _as_list(SYMBOL):
        for tf in _as_list(TIMEFRAME):
            for rp in _as_list(RSI_PERIOD):
                for tp in TP_ROE_ARR:
                    for sl in SL_ROE_ARR:
                        for mode in TP_MODE_ARR:
                            csv_path = run(s, str(tf), rp, LEVERAGE, EQUITY, START, END, OUT_DIR, tp, sl, mode)
                            if csv_path:
                                print(f"✅ 완료: {os.path.basename(csv_path)}")
