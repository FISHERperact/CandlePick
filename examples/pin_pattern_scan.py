# -*- coding: utf-8 -*-
"""
品字形三日形态选股 / 全市场扫描

形态定义：中间高、两头低（∧），即 Day2 收盘最高，Day1 和 Day3 收盘接近且较低。
前置条件（默认开启）：这三天股票都是在涨，即 Day1/Day2/Day3 均为阳线（收盘 > 开盘）。
流程：数据读取 → 滑窗生成 → 规则过滤 → 特征计算 → 相似度排序 → 结果导出（CSV）。
可选：形态出现后第 5/10/20 日收益分布回测。

用法（在 Kronos 项目根目录，先激活环境如 conda activate kronos）:
  python examples/pin_pattern_scan.py
  python examples/pin_pattern_scan.py --data_dir examples/stk_factor --top_k_global 2000 --workers 8
  python examples/pin_pattern_scan.py --backtest
  python examples/pin_pattern_scan.py --max_stocks 200   # 快速测试
  python examples/pin_pattern_scan.py --no_uptrend_3d    # 不要求三天都是涨（仅品字形）

默认参数与调参建议:
  epsilon: 0.02  → 两边收盘接近度，|C1-C3|/C_bar < ε；可试 0.01~0.05，越小越严。
  a_min:   0.01  → 最小形态振幅，避免横盘；(max-min)/C_bar > a_min；可试 0.005~0.03。
  --require_yang / --require_close_near_high → 可选 K 线约束，默认关闭。
"""
from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

# 数据列名映射（与 stk_factor CSV 一致）
COLUMN_MAP = {
    "交易日期": "timestamps",
    "开盘价": "open",
    "收盘价": "close",
    "最高价": "high",
    "最低价": "low",
    "成交量(手)": "volume",
    "成交额(千元)": "amount",
}

def load_stock_basic(stock_basic_path: str) -> pd.DataFrame:
    """
    读取股票基本信息表（examples/stock_basic.csv），用于在结果中补充股票名称/行业等字段。
    返回的 DataFrame 至少包含 stock_code 以及若干基础列。
    """
    basic = pd.read_csv(stock_basic_path, dtype=str)
    if "股票代码" not in basic.columns:
        raise ValueError(f"stock_basic 缺少列「股票代码」: {stock_basic_path}")
    basic = basic.rename(columns={"股票代码": "stock_code"})
    wanted = [
        "stock_code",
        "股票名称",
        "地区",
        "行业",
        "市场类型",
        "上市状态",
        "上市日期",
        "是否沪深港通标的",
        "企业性质",
    ]
    keep = [c for c in wanted if c in basic.columns]
    return basic[keep]


# --------------- 默认参数与调参建议 ---------------
@dataclass
class ScanParams:
    """品字形扫描参数（可调）"""
    # 两头接近：|C1-C3|/C_bar < epsilon（两侧收盘接近）
    epsilon: float = 0.02
    # 形态最小振幅：(max-min)/C_bar > a_min，避免太平
    a_min: float = 0.01
    # 前置条件：三天都是在涨（Day1/Day2/Day3 均为阳线，close > open）
    require_uptrend_3d: bool = True
    # 是否要求 Day1、Day3 为阳线（close > open）；若 require_uptrend_3d 已为 True 则三日都已为阳线
    require_yang_d1_d3: bool = False
    # 是否要求 Day1、Day3 收盘靠近当日高点 (close >= high * ratio)
    require_close_near_high: bool = False
    close_near_high_ratio: float = 0.98
    # 相似度：距离度量 "euclidean" | "cosine"
    distance_metric: str = "euclidean"
    # 每只股票最多保留 Top-K 窗口
    top_k_per_stock: int = 5
    # 全市场最终保留条数，直接保存到单文件（默认 2000）
    top_k_global: int = 2000
    # 回测：形态结束后持有 N 天的收益
    backtest_holding_days: List[int] = field(default_factory=lambda: [5, 10, 20])
    # 搜索范围：只保留三日窗口完全落在此区间内（Day1 起、Day3 止）
    scan_start_date: str = "2025-12-25"
    scan_end_date: str = "2026-02-25"

    # 调参建议（见文档字符串）
    # epsilon: 0.01~0.05，越小两边越接近；a_min: 0.005~0.03，太小会筛出横盘


# --------------- 1. 数据读取 ---------------
def load_stk_factor_csv(csv_path: str) -> pd.DataFrame:
    """读取单只股票 stk_factor 格式 CSV，统一为 open/high/low/close/volume/amount/timestamps。"""
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
    if "volume" not in df.columns:
        df["volume"] = 0.0
    if "amount" not in df.columns:
        df["amount"] = 0.0
    return df


def load_all_stocks(data_dir: str, max_stocks: Optional[int] = None) -> List[Tuple[str, pd.DataFrame]]:
    """加载 data_dir 下所有 *\.csv，返回 [(stock_code, df), ...]。max_stocks 用于快速测试。"""
    results = []
    for f in sorted(os.listdir(data_dir)):
        if not f.endswith(".csv"):
            continue
        if max_stocks is not None and len(results) >= max_stocks:
            break
        path = os.path.join(data_dir, f)
        try:
            df = load_stk_factor_csv(path)
            if len(df) < 3:
                continue
            code = os.path.splitext(f)[0]
            results.append((code, df))
        except Exception:
            continue
    return results


# --------------- 2. 向量化滑窗 + 3. 规则过滤 ---------------
def _vectorized_mask(df: pd.DataFrame, p: ScanParams) -> np.ndarray:
    """向量化硬性判定，返回 (n-2,) 布尔数组，True 表示该 3 日窗口通过。"""
    n = len(df)
    if n < 3:
        return np.array([], dtype=bool)
    c = df["close"].values
    o = df["open"].values
    h = df["high"].values
    l = df["low"].values
    c1, c2, c3 = c[:-2], c[1:-1], c[2:]
    o1, o2, o3 = o[:-2], o[1:-1], o[2:]
    h1, h3 = h[:-2], h[2:]
    c_bar = (c1 + c2 + c3) / 3.0
    c_bar = np.maximum(c_bar, 1e-12)

    # 品字形：中间高、两头低 —— Day2 收盘 > max(Day1, Day3)，且 Day1 ≈ Day3
    mask = (c2 > np.maximum(c1, c3)) & (np.abs(c1 - c3) / c_bar <= p.epsilon)
    amp = (np.maximum(np.maximum(c1, c2), c3) - np.minimum(np.minimum(c1, c2), c3)) / c_bar
    mask &= amp >= p.a_min

    if p.require_uptrend_3d:
        mask &= (c1 > o1) & (c2 > o2) & (c3 > o3)
    if p.require_yang_d1_d3 and not p.require_uptrend_3d:
        mask &= (c1 > o1) & (c3 > o3)
    if p.require_close_near_high:
        mask &= (c1 >= h1 * p.close_near_high_ratio) & (c3 >= h3 * p.close_near_high_ratio)

    # 仅保留三日窗口完全落在 [scan_start_date, scan_end_date] 内
    ts = pd.to_datetime(df["timestamps"].values)
    ts1, ts3 = ts[:-2], ts[2:]
    start_ts = pd.Timestamp(p.scan_start_date)
    end_ts = pd.Timestamp(p.scan_end_date)
    mask &= (ts1 >= start_ts) & (ts3 <= end_ts)
    return mask


def normalize_closes(c1: float, c2: float, c3: float) -> Tuple[float, float, float]:
    """3 日内 min-max 归一化到 [0,1]，避免除零。"""
    mn = min(c1, c2, c3)
    mx = max(c1, c2, c3)
    if mx <= mn:
        return (1.0, 0.0, 1.0)
    return (
        (c1 - mn) / (mx - mn),
        (c2 - mn) / (mx - mn),
        (c3 - mn) / (mx - mn),
    )


def _scan_one_stock(code: str, df: pd.DataFrame, p: ScanParams) -> List[dict]:
    """单只股票：向量化过滤后仅对通过窗口计算特征与相似度，返回行列表。"""
    n = len(df)
    if n < 3:
        return []
    mask = _vectorized_mask(df, p)
    indices = np.where(mask)[0]
    if len(indices) == 0:
        return []
    timestamps = df["timestamps"].values
    rows = []
    for i in indices:
        w = df.iloc[i : i + 3]
        feat = extract_features(w)
        score = similarity_to_pin(feat, p.distance_metric)
        rows.append({
            "stock_code": code,
            "date_start": timestamps[i],
            "date_end": timestamps[i + 2],
            "score": score,
            "c1": w["close"].iloc[0],
            "c2": w["close"].iloc[1],
            "c3": w["close"].iloc[2],
        })
    return rows


# --------------- 4. 特征计算 ---------------
def extract_features(w: pd.DataFrame) -> np.ndarray:
    """
    特征向量：标准化收盘 [c1,c2,c3]，实体比例(3日平均)，上影线比例，下影线比例，振幅，成交量相对值。
    实体 = |close-open|/range, 上影 = (high-max(o,c))/range, 下影 = (min(o,c)-low)/range，range=high-low。
    """
    c1, c2, c3 = w["close"].iloc[0], w["close"].iloc[1], w["close"].iloc[2]
    nc1, nc2, nc3 = normalize_closes(c1, c2, c3)
    c_bar = (c1 + c2 + c3) / 3.0

    body_ratios = []
    upper_ratios = []
    lower_ratios = []
    for i in range(3):
        o, h, l, c = w["open"].iloc[i], w["high"].iloc[i], w["low"].iloc[i], w["close"].iloc[i]
        r = h - l
        if r <= 0:
            body_ratios.append(0.0)
            upper_ratios.append(0.0)
            lower_ratios.append(0.0)
        else:
            body_ratios.append(abs(c - o) / r)
            upper_ratios.append((h - max(o, c)) / r)
            lower_ratios.append((min(o, c) - l) / r)
    body = np.mean(body_ratios)
    upper = np.mean(upper_ratios)
    lower = np.mean(lower_ratios)

    amp = (max(c1, c2, c3) - min(c1, c2, c3)) / c_bar if c_bar > 0 else 0.0
    vol = w["volume"].values
    vol_rel = float(np.mean(vol) / (np.mean(vol) + 1e-8))  # 3日平均相对自身，可改为相对全市场
    # 简单用 3 日成交量均值归一化（与自身比）
    vol_mean = np.mean(vol)
    vol_rel = 1.0 if vol_mean <= 0 else float(vol_mean / (vol_mean + 1e-8))

    return np.array([nc1, nc2, nc3, body, upper, lower, amp, vol_rel], dtype=np.float64)


# 品字形理想模板：归一化收盘为 (低, 高, 低) -> (0, 1, 0)，其余特征用典型值
PIN_TEMPLATE = np.array([0.0, 1.0, 0.0, 0.4, 0.1, 0.1, 0.03, 0.5], dtype=np.float64)


def similarity_to_pin(feat: np.ndarray, metric: str = "euclidean") -> float:
    """与品字形模板的相似度：欧氏距离取负（越大越相似），或余弦相似度（越大越相似）。"""
    if metric == "euclidean":
        d = np.sqrt(np.sum((feat - PIN_TEMPLATE) ** 2))
        return -d
    if metric == "cosine":
        a = feat.ravel()
        b = PIN_TEMPLATE.ravel()
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na < 1e-10 or nb < 1e-10:
            return 0.0
        return float(np.dot(a, b) / (na * nb))
    raise ValueError(f"Unknown metric: {metric}")


# --------------- 5. 相似度排序 + 6. 结果导出 ---------------
def _worker(args: Tuple[str, str, ScanParams]) -> List[dict]:
    """多进程 worker：读取单文件并扫描，返回该股票的所有通过窗口行。"""
    data_dir, fname, p = args
    path = os.path.join(data_dir, fname)
    try:
        df = load_stk_factor_csv(path)
        code = os.path.splitext(fname)[0]
        return _scan_one_stock(code, df, p)
    except Exception:
        return []


def run_scan(
    data_dir: str,
    params: Optional[ScanParams] = None,
    max_stocks: Optional[int] = None,
    workers: int = 1,
) -> pd.DataFrame:
    """
    全流程：加载 → 向量化规则过滤 → 特征 → 相似度 → 每股 Top-K → 取前 top_k_global 条保存。
    返回 DataFrame：stock_code, date_start, date_end, score, ...
    workers>1 时多进程并行扫股票。
    """
    p = params or ScanParams()
    csv_files = [f for f in sorted(os.listdir(data_dir)) if f.endswith(".csv")]
    if max_stocks is not None:
        csv_files = csv_files[: max_stocks]
    if not csv_files:
        return pd.DataFrame()

    if workers <= 1:
        rows = []
        for f in tqdm(csv_files, desc="扫描股票", unit="只"):
            path = os.path.join(data_dir, f)
            try:
                df = load_stk_factor_csv(path)
                code = os.path.splitext(f)[0]
                rows.extend(_scan_one_stock(code, df, p))
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
    per_stock = (
        full.sort_values("score", ascending=False)
        .groupby("stock_code", group_keys=False)
        .apply(lambda g: g.head(p.top_k_per_stock))
        .reset_index(drop=True)
    )
    top = per_stock.nlargest(p.top_k_global, "score").reset_index(drop=True)
    return top


def add_backtest_returns(
    df: pd.DataFrame,
    data_dir: str,
    holding_days: List[int],
) -> pd.DataFrame:
    """为每条记录添加形态结束后持有 N 天的收益率（需要能取到后续日线）。"""
    if df.empty:
        return df
    codes = df["stock_code"].unique()
    code_to_df = {}
    for code in tqdm(codes, desc="加载行情", unit="只"):
        path = os.path.join(data_dir, code + ".csv")
        try:
            code_to_df[code] = load_stk_factor_csv(path)
        except Exception:
            code_to_df[code] = None

    out = df.copy()
    for H in holding_days:
        rets = []
        for i in tqdm(range(len(df)), desc="回测 +%dd" % H, unit="条", leave=False):
            r = df.iloc[i]
            code, end_date = r["stock_code"], r["date_end"]
            tdf = code_to_df.get(code)
            if tdf is None:
                rets.append(np.nan)
                continue
            tdf = tdf.sort_values("timestamps").reset_index(drop=True)
            idx = tdf["timestamps"].searchsorted(pd.Timestamp(end_date))
            if idx >= len(tdf):
                rets.append(np.nan)
                continue
            # 形态结束日的收盘价（即 Day3 收盘）
            close_end = tdf["close"].iloc[idx]
            idx_future = idx + H
            if idx_future >= len(tdf):
                rets.append(np.nan)
                continue
            close_future = tdf["close"].iloc[idx_future]
            if close_end <= 0:
                rets.append(np.nan)
            else:
                rets.append((close_future - close_end) / close_end)
        out[f"ret_{H}d"] = rets

    return out


def main():
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    default_data_dir = os.path.join(_script_dir, "stk_factor")
    default_stock_basic = os.path.join(_script_dir, "stock_basic.csv")

    parser = argparse.ArgumentParser(description="品字形三日形态全市场扫描")
    parser.add_argument("--data_dir", type=str, default=default_data_dir, help="stk_factor CSV 目录")
    parser.add_argument("--epsilon", type=float, default=0.02, help="|C1-C3|/C_bar 上界")
    parser.add_argument("--a_min", type=float, default=0.01, help="最小振幅 (max-min)/C_bar")
    parser.add_argument("--no_uptrend_3d", action="store_true", help="关闭「三天都是在涨」前置条件（默认开启：三日均为阳线）")
    parser.add_argument("--require_yang", action="store_true", help="Day1/Day3 要求阳线")
    parser.add_argument("--require_close_near_high", action="store_true", help="Day1/Day3 收盘靠近当日高点")
    parser.add_argument("--top_k_per_stock", type=int, default=5)
    parser.add_argument("--top_k_global", type=int, default=2000, help="保存前 N 条到单文件，默认 2000")
    parser.add_argument("--metric", type=str, default="euclidean", choices=["euclidean", "cosine"])
    parser.add_argument("--workers", type=int, default=4, help="并行进程数，默认 4，设为 1 则单进程")
    parser.add_argument("--backtest", action="store_true", help="计算形态后 5/10/20 日收益")
    parser.add_argument("--out", type=str, default="", help="输出 CSV 路径，默认 outputs/pin_pattern_scan_result.csv")
    parser.add_argument("--max_stocks", type=int, default=None, help="仅加载前 N 只股票（快速测试）")
    parser.add_argument("--scan_start", type=str, default="2025-12-25", help="搜索区间起始日，三日窗口须完全在此区间内")
    parser.add_argument("--scan_end", type=str, default="2026-02-25", help="搜索区间结束日")
    parser.add_argument("--stock_basic", type=str, default=default_stock_basic, help="股票基本信息表路径；设为空字符串则不合并")
    args = parser.parse_args()

    params = ScanParams(
        epsilon=args.epsilon,
        a_min=args.a_min,
        require_uptrend_3d=not args.no_uptrend_3d,
        require_yang_d1_d3=args.require_yang,
        require_close_near_high=args.require_close_near_high,
        distance_metric=args.metric,
        top_k_per_stock=args.top_k_per_stock,
        top_k_global=args.top_k_global,
        scan_start_date=args.scan_start,
        scan_end_date=args.scan_end,
    )

    print("Scanning", args.data_dir, "workers=%d  date_range=[%s, %s]" % (args.workers, args.scan_start, args.scan_end))
    result = run_scan(
        args.data_dir, params, max_stocks=args.max_stocks, workers=args.workers
    )
    if result.empty:
        print("No windows passed the rules. Try loosening epsilon / a_min.")
        return

    if args.backtest:
        result = add_backtest_returns(result, args.data_dir, params.backtest_holding_days)
        for H in params.backtest_holding_days:
            col = f"ret_{H}d"
            if col in result.columns:
                valid = result[col].dropna()
                if len(valid) > 0:
                    print(f"  Ret +{H}d: mean={valid.mean():.4f}, std={valid.std():.4f}, count={len(valid)}")

    if args.stock_basic:
        try:
            basic = load_stock_basic(args.stock_basic)
            result = result.merge(basic, on="stock_code", how="left")
        except Exception as e:
            print(f"Warning: failed to merge stock_basic ({args.stock_basic}): {e}")

    # 同花顺个股页链接（用 stock_code 的数字部分，如 300197.SZ -> 300197）
    result["同花顺链接"] = result["stock_code"].apply(
        lambda x: "https://stockpage.10jqka.com.cn/" + str(x).split(".")[0] + "/"
    )

    out_path = args.out or os.path.join(_script_dir, "outputs", "pin_pattern_scan_result.csv")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    result.to_csv(out_path, index=False, encoding="utf-8-sig")
    n = len(result)
    print("Saved %d records to %s" % (n, out_path))


if __name__ == "__main__":
    main()
