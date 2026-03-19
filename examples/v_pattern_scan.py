# -*- coding: utf-8 -*-
"""
V字形三日形态选股 / 全市场扫描（改进版）

形态定义（三阳V底）：
  三日窗口 Day1-Day2-Day3，三天都是涨的（阳线），但中间低、两头高：
  1. 三天都是阳线：Day1/Day2/Day3 的 close > open
  2. Day2 收盘是三日最低 —— V 形底部
  3. Day1 和 Day3 收盘较高 —— V 形两翼
  4. 最小振幅：三日收盘极差/均价 ≥ a_min（默认 2%），排除横盘噪音
  5. 可选：Day3 收盘 ≥ Day1 收盘（右腿不低于左腿）
  6. 可选：量能配合（底部放量加分）
  7. 可选：利用技术指标（KDJ_J / RSI_6 / CCI 超卖）加分

评分机制：
  不再用固定模板的欧氏距离，改为多因子加权评分：
  - V 形深度（越深越标准）
  - V 形对称性（左右两腿越对称越好）
  - K 线形态质量（Day2 长下影 / Day3 阳包阴等）
  - 量能配合（底部放量加分）
  - 技术指标超卖信号（可选，数据中有指标时自动启用）

用法（在 Kronos 项目根目录）:
  python examples/v_pattern_scan.py
  python examples/v_pattern_scan.py --workers 8 --top_k_global 2000
  python examples/v_pattern_scan.py --backtest
  python examples/v_pattern_scan.py --max_stocks 200
  python examples/v_pattern_scan.py --a_min 0.05 --depth_min 0.03
  python examples/v_pattern_scan.py --no_recovery      # 不要求 Day3 >= Day1
  python examples/v_pattern_scan.py --no_uptrend_3d     # Day1/Day3 不要求阳线
"""
from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

COLUMN_MAP = {
    "交易日期": "timestamps",
    "开盘价": "open",
    "收盘价": "close",
    "最高价": "high",
    "最低价": "low",
    "成交量(手)": "volume",
    "成交额(千元)": "amount",
}

INDICATOR_COLS = [
    "KDJ_K", "KDJ_D", "KDJ_J",
    "RSI_6", "RSI_12", "RSI_24",
    "CCI",
    "MACD_DIF", "MACD_DEA", "MACD",
    "BOLL_UPPER", "BOLL_MID", "BOLL_LOWER",
]


def load_stock_basic(stock_basic_path: str) -> pd.DataFrame:
    basic = pd.read_csv(stock_basic_path, dtype=str)
    if "股票代码" not in basic.columns:
        raise ValueError(f"stock_basic 缺少列「股票代码」: {stock_basic_path}")
    basic = basic.rename(columns={"股票代码": "stock_code"})
    wanted = [
        "stock_code", "股票名称", "地区", "行业", "市场类型",
        "上市状态", "上市日期", "是否沪深港通标的", "企业性质",
    ]
    keep = [c for c in wanted if c in basic.columns]
    return basic[keep]


@dataclass
class ScanParams:
    """V字形扫描参数"""
    # --- 硬性过滤 ---
    # 最小振幅：三日收盘极差 / 均价 ≥ a_min
    a_min: float = 0.04
    # V 形凹陷深度：Day2 收盘比两侧均值低的最小比例（越大越像标准 V）
    depth_min: float = 0.03
    # Day3 收盘 ≥ Day1 收盘（反弹到位）
    require_recovery: bool = True

    # --- 评分权重 ---
    w_depth: float = 30.0
    w_symmetry: float = 20.0
    w_kline_quality: float = 20.0
    w_volume: float = 15.0
    w_indicator: float = 15.0

    # --- 其他 ---
    top_k_per_stock: int = 3
    top_k_global: int = 2000
    distance_metric: str = "score"
    backtest_holding_days: List[int] = field(default_factory=lambda: [5, 10, 20])
    scan_start_date: str = "2025-12-25"
    scan_end_date: str = "2026-02-25"


# --------------- 数据读取 ---------------
def load_stk_factor_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    rename = {k: v for k, v in COLUMN_MAP.items() if k in df.columns}
    if "交易日期" not in df.columns:
        raise ValueError(f"缺少列「交易日期」: {csv_path}")
    df = df.rename(columns=rename)
    df["timestamps"] = pd.to_datetime(df["timestamps"])
    df = df.sort_values("timestamps").reset_index(drop=True)
    for col in ["open", "high", "low", "close", "volume", "amount"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in INDICATOR_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "volume" not in df.columns:
        df["volume"] = 0.0
    if "amount" not in df.columns:
        df["amount"] = 0.0
    return df


# --------------- 向量化规则过滤 ---------------
def _vectorized_mask(df: pd.DataFrame, p: ScanParams) -> np.ndarray:
    n = len(df)
    if n < 3:
        return np.array([], dtype=bool)

    c = df["close"].values.astype(np.float64)
    o = df["open"].values.astype(np.float64)
    h = df["high"].values.astype(np.float64)
    lo = df["low"].values.astype(np.float64)

    c1, c2, c3 = c[:-2], c[1:-1], c[2:]
    o1, o2, o3 = o[:-2], o[1:-1], o[2:]
    h1, h2, h3 = h[:-2], h[1:-1], h[2:]
    l1, l2, l3 = lo[:-2], lo[1:-1], lo[2:]

    c_bar = np.maximum((c1 + c2 + c3) / 3.0, 1e-12)

    # 核心条件 1: 三天都是阳线（收盘 > 开盘）
    mask = (c1 > o1) & (c2 > o2) & (c3 > o3)

    # 核心条件 2: 收盘价 V 形 — Day2 收盘 < Day1 且 < Day3
    mask &= (c2 < c1) & (c2 < c3)

    # 核心条件 3: 开盘价 V 形 — Day2 开盘 < Day1 且 < Day3
    mask &= (o2 < o1) & (o2 < o3)

    # 核心条件 4: 收盘价凹陷深度 ≥ depth_min
    c_side = (c1 + c3) / 2.0
    c_depth = (c_side - c2) / np.maximum(c_side, 1e-12)
    mask &= c_depth >= p.depth_min

    # 核心条件 5: 开盘价凹陷深度 ≥ depth_min
    o_bar = np.maximum((o1 + o2 + o3) / 3.0, 1e-12)
    o_side = (o1 + o3) / 2.0
    o_depth = (o_side - o2) / np.maximum(o_side, 1e-12)
    mask &= o_depth >= p.depth_min

    # 核心条件 6: 最小振幅 — 收盘极差 / 均价，排除横盘噪音
    amp = (np.maximum(c1, c3) - c2) / c_bar
    mask &= amp >= p.a_min

    # 可选：反弹到位 — Day3 收盘 ≥ Day1 收盘
    if p.require_recovery:
        mask &= c3 >= c1

    # 日期范围过滤
    ts = pd.to_datetime(df["timestamps"].values)
    ts1, ts3 = ts[:-2], ts[2:]
    start_ts = pd.Timestamp(p.scan_start_date)
    end_ts = pd.Timestamp(p.scan_end_date)
    mask &= (ts1 >= start_ts) & (ts3 <= end_ts)

    return mask


# --------------- 多因子评分 ---------------
def _compute_score(w: pd.DataFrame, p: ScanParams) -> Tuple[float, dict]:
    """
    对通过硬性过滤的三日窗口计算综合评分。
    返回 (总分, 各因子明细 dict)。
    """
    o1, o2, o3 = w["open"].iloc[0], w["open"].iloc[1], w["open"].iloc[2]
    h1, h2, h3 = w["high"].iloc[0], w["high"].iloc[1], w["high"].iloc[2]
    l1, l2, l3 = w["low"].iloc[0], w["low"].iloc[1], w["low"].iloc[2]
    c1, c2, c3 = w["close"].iloc[0], w["close"].iloc[1], w["close"].iloc[2]

    c_bar = (c1 + c2 + c3) / 3.0
    if c_bar <= 0:
        return 0.0, {}

    # --- 因子 1: V 形深度 — 中间收盘比两头低多少 (0~100) ---
    side_avg = (c1 + c3) / 2.0
    depth_pct = (side_avg - c2) / side_avg if side_avg > 0 else 0.0
    s_depth = min(max(depth_pct / 0.10, 0.0), 1.0) * 100.0

    # --- 因子 2: V 形对称性 — Day1和Day3收盘越接近越好 (0~100) ---
    if max(c1, c3) > 0:
        sym_ratio = min(c1, c3) / max(c1, c3)
    else:
        sym_ratio = 0.0
    recovery_ratio = min(c3 / c1, 1.2) / 1.2 if c1 > 0 else 0.0
    s_symmetry = (sym_ratio * 0.4 + recovery_ratio * 0.6) * 100.0

    # --- 因子 3: K 线阳线质量 — 三天阳线实体占比 (0~100) ---
    kline_score = 0.0
    for i, (oi, hi, li, ci) in enumerate([(o1,h1,l1,c1),(o2,h2,l2,c2),(o3,h3,l3,c3)]):
        r = hi - li
        if r > 0:
            body = (ci - oi) / r
            kline_score += body * 25.0
    # Day3 反弹力度加分 — Day3 涨幅 > Day1 涨幅
    gain1 = (c1 - o1) / o1 if o1 > 0 else 0.0
    gain3 = (c3 - o3) / o3 if o3 > 0 else 0.0
    if gain3 > gain1 and gain1 > 0:
        kline_score += min((gain3 / gain1 - 1.0), 1.0) * 25.0
    s_kline = min(kline_score, 100.0)

    # --- 因子 4: 量能配合 (0~100) ---
    v1 = w["volume"].iloc[0]
    v2 = w["volume"].iloc[1]
    v3 = w["volume"].iloc[2]
    vol_score = 0.0
    if v1 > 0 and v2 > 0:
        # Day3 放量反弹加分
        if v3 > v2:
            vol_score += min((v3 / v2 - 1.0) / 1.0, 1.0) * 50.0
        if v3 > v1:
            vol_score += min((v3 / v1 - 1.0) / 1.0, 1.0) * 50.0
    s_volume = min(vol_score, 100.0)

    # --- 因子 5: 技术指标信号 (0~100) ---
    ind_score = 0.0
    has_indicator = False
    if "KDJ_J" in w.columns:
        j2 = w["KDJ_J"].iloc[1]
        if pd.notna(j2):
            has_indicator = True
            if j2 < 0:
                ind_score += 35.0
            elif j2 < 20:
                ind_score += 25.0 * (1.0 - j2 / 20.0)
    if "RSI_6" in w.columns:
        rsi2 = w["RSI_6"].iloc[1]
        if pd.notna(rsi2):
            has_indicator = True
            if rsi2 < 20:
                ind_score += 35.0
            elif rsi2 < 30:
                ind_score += 25.0 * (1.0 - (rsi2 - 20.0) / 10.0)
    if "CCI" in w.columns:
        cci2 = w["CCI"].iloc[1]
        if pd.notna(cci2):
            has_indicator = True
            if cci2 < -200:
                ind_score += 30.0
            elif cci2 < -100:
                ind_score += 20.0 * (1.0 + (cci2 + 100.0) / 100.0)
    s_indicator = min(ind_score, 100.0) if has_indicator else 50.0

    # --- 加权汇总 ---
    total = (
        p.w_depth * s_depth
        + p.w_symmetry * s_symmetry
        + p.w_kline_quality * s_kline
        + p.w_volume * s_volume
        + p.w_indicator * s_indicator
    ) / (p.w_depth + p.w_symmetry + p.w_kline_quality + p.w_volume + p.w_indicator)

    detail = {
        "s_depth": round(s_depth, 2),
        "s_symmetry": round(s_symmetry, 2),
        "s_kline": round(s_kline, 2),
        "s_volume": round(s_volume, 2),
        "s_indicator": round(s_indicator, 2),
        "depth_pct": round(depth_pct * 100, 2),
    }
    return round(total, 4), detail


# --------------- 单股扫描 ---------------
def _scan_one_stock(code: str, df: pd.DataFrame, p: ScanParams) -> List[dict]:
    if len(df) < 3:
        return []
    mask = _vectorized_mask(df, p)
    indices = np.where(mask)[0]
    if len(indices) == 0:
        return []
    timestamps = df["timestamps"].values
    rows = []
    for i in indices:
        w = df.iloc[i: i + 3]
        score, detail = _compute_score(w, p)
        rows.append({
            "stock_code": code,
            "date_start": timestamps[i],
            "date_end": timestamps[i + 2],
            "score": score,
            "c1": w["close"].iloc[0],
            "c2": w["close"].iloc[1],
            "c3": w["close"].iloc[2],
            "low2": w["low"].iloc[1],
            "depth_pct": detail.get("depth_pct", 0.0),
            "s_depth": detail.get("s_depth", 0.0),
            "s_symmetry": detail.get("s_symmetry", 0.0),
            "s_kline": detail.get("s_kline", 0.0),
            "s_volume": detail.get("s_volume", 0.0),
            "s_indicator": detail.get("s_indicator", 0.0),
        })
    return rows


def _worker(args: Tuple[str, str, ScanParams]) -> List[dict]:
    data_dir, fname, p = args
    try:
        df = load_stk_factor_csv(os.path.join(data_dir, fname))
        return _scan_one_stock(os.path.splitext(fname)[0], df, p)
    except Exception:
        return []


# --------------- 全市场扫描 ---------------
def run_scan(
    data_dir: str,
    params: Optional[ScanParams] = None,
    max_stocks: Optional[int] = None,
    workers: int = 4,
) -> pd.DataFrame:
    p = params or ScanParams()
    csv_files = [f for f in sorted(os.listdir(data_dir)) if f.endswith(".csv")]
    if max_stocks is not None:
        csv_files = csv_files[:max_stocks]
    if not csv_files:
        return pd.DataFrame()

    if workers <= 1:
        rows: List[dict] = []
        for f in tqdm(csv_files, desc="扫描股票", unit="只"):
            try:
                df = load_stk_factor_csv(os.path.join(data_dir, f))
                rows.extend(_scan_one_stock(os.path.splitext(f)[0], df, p))
            except Exception:
                continue
    else:
        jobs = [(data_dir, f, p) for f in csv_files]
        rows = []
        with ProcessPoolExecutor(max_workers=workers) as ex:
            it = ex.map(_worker, jobs, chunksize=max(1, len(jobs) // (workers * 4)))
            for res in tqdm(it, total=len(jobs), desc="扫描股票", unit="只"):
                rows.extend(res)

    if not rows:
        return pd.DataFrame()

    full = pd.DataFrame(rows)
    full_sorted = full.sort_values("score", ascending=False)
    per_stock = full_sorted.groupby("stock_code", group_keys=False).head(p.top_k_per_stock).reset_index(drop=True)
    return per_stock.nlargest(p.top_k_global, "score").reset_index(drop=True)


# --------------- 回测 ---------------
def add_backtest_returns(df: pd.DataFrame, data_dir: str, holding_days: List[int]) -> pd.DataFrame:
    if df.empty:
        return df
    code_to_df: dict = {}
    for code in tqdm(df["stock_code"].unique(), desc="加载行情", unit="只"):
        try:
            code_to_df[code] = load_stk_factor_csv(os.path.join(data_dir, code + ".csv"))
        except Exception:
            code_to_df[code] = None

    out = df.copy()
    for H in holding_days:
        rets = []
        for i in tqdm(range(len(df)), desc="回测 +%dd" % H, unit="条", leave=False):
            r = df.iloc[i]
            tdf = code_to_df.get(r["stock_code"])
            if tdf is None:
                rets.append(np.nan)
                continue
            tdf = tdf.sort_values("timestamps").reset_index(drop=True)
            idx = tdf["timestamps"].searchsorted(pd.Timestamp(r["date_end"]))
            if idx >= len(tdf):
                rets.append(np.nan)
                continue
            close_end = tdf["close"].iloc[idx]
            idx_f = idx + H
            if idx_f >= len(tdf) or close_end <= 0:
                rets.append(np.nan)
                continue
            rets.append((tdf["close"].iloc[idx_f] - close_end) / close_end)
        out[f"ret_{H}d"] = rets
    return out


# --------------- main ---------------
def main():
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    default_data_dir = os.path.join(_script_dir, "stk_factor")
    default_stock_basic = os.path.join(_script_dir, "stock_basic.csv")

    parser = argparse.ArgumentParser(
        description="V字形三日形态全市场扫描（改进版）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python examples/v_pattern_scan.py
  python examples/v_pattern_scan.py --a_min 0.05 --depth_min 0.03
  python examples/v_pattern_scan.py --backtest --workers 8
  python examples/v_pattern_scan.py --no_recovery --no_uptrend_3d
  python examples/v_pattern_scan.py --scan_start 2026-01-01 --scan_end 2026-03-15
        """,
    )
    parser.add_argument("--data_dir", type=str, default=default_data_dir)
    parser.add_argument("--a_min", type=float, default=0.04,
                        help="最小振幅（收盘极差/均价），默认 0.04 (4%%)")
    parser.add_argument("--depth_min", type=float, default=0.03,
                        help="V形凹陷深度下限（Day2比两侧均值低的比例），默认 0.03 (3%%)")
    parser.add_argument("--no_recovery", action="store_true",
                        help="不要求 Day3 收盘 >= Day1 收盘")
    parser.add_argument("--top_k_per_stock", type=int, default=3)
    parser.add_argument("--top_k_global", type=int, default=2000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--backtest", action="store_true", help="计算形态后 5/10/20 日收益")
    parser.add_argument("--out", type=str, default="")
    parser.add_argument("--max_stocks", type=int, default=None)
    parser.add_argument("--scan_start", type=str, default="2025-12-12")
    parser.add_argument("--scan_end", type=str, default="2026-03-12")
    parser.add_argument("--stock_basic", type=str, default=default_stock_basic)
    args = parser.parse_args()

    params = ScanParams(
        a_min=args.a_min,
        depth_min=args.depth_min,
        require_recovery=not args.no_recovery,
        top_k_per_stock=args.top_k_per_stock,
        top_k_global=args.top_k_global,
        scan_start_date=args.scan_start,
        scan_end_date=args.scan_end,
    )

    print("=" * 60)
    print("V字形扫描（三阳V底）")
    print("  数据目录: %s" % args.data_dir)
    print("  日期范围: [%s, %s]" % (args.scan_start, args.scan_end))
    print("  最小振幅: %.1f%%" % (args.a_min * 100))
    print("  V形深度: %.1f%%" % (args.depth_min * 100))
    print("  三天阳线: 是  Day3>=Day1: %s" % ("是" if params.require_recovery else "否"))
    print("  workers=%d" % args.workers)
    print("=" * 60)

    result = run_scan(args.data_dir, params, max_stocks=args.max_stocks, workers=args.workers)

    if result.empty:
        print("\n未找到符合条件的 V 形窗口。可尝试降低 --a_min。")
        return

    if args.backtest:
        result = add_backtest_returns(result, args.data_dir, params.backtest_holding_days)
        print("\n回测结果:")
        for H in params.backtest_holding_days:
            col = f"ret_{H}d"
            if col in result.columns:
                valid = result[col].dropna()
                if len(valid):
                    win_rate = (valid > 0).mean() * 100
                    print("  +%dd: mean=%.2f%%  std=%.2f%%  win_rate=%.1f%%  count=%d" % (
                        H, valid.mean() * 100, valid.std() * 100, win_rate, len(valid)))

    if args.stock_basic:
        try:
            basic = load_stock_basic(args.stock_basic)
            result = result.merge(basic, on="stock_code", how="left")
        except Exception as e:
            print(f"Warning: failed to merge stock_basic ({args.stock_basic}): {e}")

    result["同花顺链接"] = result["stock_code"].apply(
        lambda x: "https://stockpage.10jqka.com.cn/" + str(x).split(".")[0] + "/"
    )

    out_path = args.out or os.path.join(_script_dir, "outputs", "v_pattern_scan_result.csv")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    result.to_csv(out_path, index=False, encoding="utf-8-sig")
    print("\n保存 %d 条结果到 %s" % (len(result), out_path))

    print("\nTop 10:")
    show_cols = ["stock_code", "date_start", "date_end", "score", "depth_pct", "c1", "c2", "c3"]
    if "股票名称" in result.columns:
        show_cols.insert(1, "股票名称")
    print(result[show_cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
