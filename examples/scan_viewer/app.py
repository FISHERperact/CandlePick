# -*- coding: utf-8 -*-
"""
品字形 / V字形 扫描结果展示网站。
读取 outputs 下的 CSV，在页面上展示两个结果表，每行股票带同花顺链接。

运行（在 Kronos 项目根或 examples 目录）:
  cd Kronos/examples/scan_viewer && python app.py
  然后访问 http://127.0.0.1:5000
"""
import csv
import io
import os
import sys

import pandas as pd
import numpy as np

# 确保能找到项目根
_VIEWER_DIR = os.path.dirname(os.path.abspath(__file__))
_EXAMPLES_DIR = os.path.dirname(_VIEWER_DIR)
if _EXAMPLES_DIR not in sys.path:
    sys.path.insert(0, _EXAMPLES_DIR)

OUTPUTS_DIR = os.path.join(_EXAMPLES_DIR, "outputs")
SAMPLE_DIR = os.path.join(_EXAMPLES_DIR, "sample_results")
STK_FACTOR_DIR = os.path.join(_EXAMPLES_DIR, "stk_factor")
PIN_CSV = os.path.join(OUTPUTS_DIR, "pin_pattern_scan_result.csv")
V_CSV = os.path.join(OUTPUTS_DIR, "v_pattern_scan_result.csv")
PIN_SAMPLE_CSV = os.path.join(SAMPLE_DIR, "品字形_前200条.csv")
V_SAMPLE_CSV = os.path.join(SAMPLE_DIR, "V字形_前200条.csv")


def _pin_csv_path():
    """优先用 outputs 全量结果，否则用 sample_results 的 200 条展示样本。"""
    return PIN_CSV if os.path.isfile(PIN_CSV) else PIN_SAMPLE_CSV


def _v_csv_path():
    return V_CSV if os.path.isfile(V_CSV) else V_SAMPLE_CSV

COLUMN_MAP = {
    "交易日期": "timestamps",
    "开盘价": "open",
    "收盘价": "close",
    "最高价": "high",
    "最低价": "low",
}


def _stock_link(code: str) -> str:
    """stock_code 如 300197.SZ -> 同花顺个股页"""
    return "https://stockpage.10jqka.com.cn/" + str(code).split(".")[0] + "/"


def _load_csv(path: str):
    if not os.path.isfile(path):
        return None
    df = pd.read_csv(path, encoding="utf-8-sig")
    if "同花顺链接" not in df.columns and "stock_code" in df.columns:
        df["同花顺链接"] = df["stock_code"].apply(_stock_link)
    for col in ["date_start", "date_end"]:
        if col in df.columns:
            df[col] = df[col].astype(str)
    return df


_stk_cache = {}


def _get_kline(stock_code: str, date_start: str, date_end: str):
    """从 stk_factor 目录读取该股票三日 OHLC，返回 [[o,h,l,c], [o,h,l,c], [o,h,l,c]]。"""
    if not stock_code or not date_start or not date_end:
        return []
    path = os.path.join(STK_FACTOR_DIR, str(stock_code).strip() + ".csv")
    if path not in _stk_cache:
        if not os.path.isfile(path):
            return []
        try:
            d = pd.read_csv(path, encoding="utf-8-sig")
            rename = {k: v for k, v in COLUMN_MAP.items() if k in d.columns}
            if "交易日期" not in d.columns:
                return []
            d = d.rename(columns=rename)
            d["timestamps"] = pd.to_datetime(d["timestamps"])
            d = d.sort_values("timestamps").reset_index(drop=True)
            for c in ["open", "high", "low", "close"]:
                if c in d.columns:
                    d[c] = pd.to_numeric(d[c], errors="coerce")
            _stk_cache[path] = d
        except Exception:
            return []
    df = _stk_cache[path]
    try:
        start_ts = pd.to_datetime(date_start)
        end_ts = pd.to_datetime(date_end)
    except Exception:
        return []
    mask = (df["timestamps"] >= start_ts) & (df["timestamps"] <= end_ts)
    block = df.loc[mask, ["open", "high", "low", "close"]].head(3)
    if len(block) != 3:
        return []
    out = []
    for _, r in block.iterrows():
        out.append([float(r["open"]), float(r["high"]), float(r["low"]), float(r["close"])])
    return out


def _get_forward_return(stock_code: str, date_end: str, holding_days: int = 60):
    """
    形态结束日（date_end）收盘价 -> 后续最多 holding_days 个交易日的收盘价涨跌幅（%）。
    约 40 个交易日 ≈ 两月。若后续不足 40 日则用实际可用天数计算。
    返回 (pct, days_used)：pct 为涨跌幅，days_used 为实际使用的交易日数（不足 40 时才有，用于前端标注）。
    """
    if not stock_code or not date_end:
        return None, None
    path = os.path.join(STK_FACTOR_DIR, str(stock_code).strip() + ".csv")
    if path not in _stk_cache:
        if not os.path.isfile(path):
            return None, None
        try:
            d = pd.read_csv(path, encoding="utf-8-sig")
            rename = {k: v for k, v in COLUMN_MAP.items() if k in d.columns}
            if "交易日期" not in d.columns:
                return None, None
            d = d.rename(columns=rename)
            d["timestamps"] = pd.to_datetime(d["timestamps"])
            d = d.sort_values("timestamps").reset_index(drop=True)
            d["close"] = pd.to_numeric(d["close"], errors="coerce")
            _stk_cache[path] = d
        except Exception:
            return None, None
    df = _stk_cache[path]
    try:
        end_ts = pd.to_datetime(date_end)
    except Exception:
        return None, None
    idx = df["timestamps"].searchsorted(end_ts)
    if idx >= len(df):
        return None, None
    # 形态结束日之后还有多少交易日
    available = len(df) - idx - 1
    if available < 1:
        return None, None
    holding = min(holding_days, available)
    close_end = float(df["close"].iloc[idx])
    close_future = float(df["close"].iloc[idx + holding])
    if close_end <= 0:
        return None, None
    pct = round((close_future - close_end) / close_end * 100, 2)
    days_used = holding if holding < holding_days else None
    return pct, days_used


def _to_json_serializable(obj):
    """将 numpy/pandas 标量转为原生 Python 类型，便于 JSON 序列化。"""
    if isinstance(obj, dict):
        return {k: _to_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_json_serializable(v) for v in obj]
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if pd.isna(obj):
        return None
    return obj


def _enrich_with_kline(rows: list, columns: list):
    """为每行添加 kline、两月涨幅，并在列中增加 K线、两月涨幅。"""
    if "K线" not in columns:
        columns.append("K线")
    if "两月涨幅" not in columns:
        columns.append("两月涨幅")
    for row in rows:
        code = row.get("stock_code")
        start = row.get("date_start")
        end = row.get("date_end")
        row["kline"] = _get_kline(code, start, end)
        pct, days_used = _get_forward_return(code, end, 40)
        row["两月涨幅"] = pct
        row["两月涨幅_天数"] = days_used  # 不足 40 日时才有，供前端显示「N日涨幅」
    return columns


def _build_export_data():
    """加载品字形 / V字形 数据并做 K线、两月涨幅  enrichment，返回 (rows, columns) 用于 CSV 导出。"""
    pin_df = _load_csv(_pin_csv_path())
    v_df = _load_csv(_v_csv_path())
    pin_data = pin_df.to_dict("records") if pin_df is not None and len(pin_df) else []
    v_data = v_df.to_dict("records") if v_df is not None and len(v_df) else []
    columns_pin = list(pin_df.columns) if pin_df is not None else []
    columns_v = list(v_df.columns) if v_df is not None else []
    columns_pin = _enrich_with_kline(pin_data, columns_pin)
    columns_v = _enrich_with_kline(v_data, columns_v)
    export_cols = ["形态"] + [c for c in columns_pin if c != "K线"]
    rows = []
    for row in pin_data:
        r = dict(row)
        r["形态"] = "品字形"
        rows.append(r)
    for row in v_data:
        r = dict(row)
        r["形态"] = "V字形"
        rows.append(r)
    for r in rows:
        if "K线" in r:
            del r["K线"]
    return rows, export_cols


def create_app():
    from flask import Flask, make_response, render_template

    app = Flask(__name__, template_folder="templates")

    @app.route("/export.csv")
    def export_csv():
        rows, columns = _build_export_data()
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=columns, extrasaction="ignore", encoding="utf-8")
        writer.writeheader()
        for row in rows:
            out = {k: row.get(k) for k in columns}
            for k, v in out.items():
                if v is None or (isinstance(v, float) and pd.isna(v)):
                    out[k] = ""
                elif isinstance(v, (np.integer, np.int64, np.int32)):
                    out[k] = int(v)
                elif isinstance(v, (np.floating, np.float64, np.float32)):
                    out[k] = float(v)
            writer.writerow(out)
        resp = make_response(buf.getvalue().encode("utf-8-sig"))
        resp.headers["Content-Type"] = "text/csv; charset=utf-8"
        resp.headers["Content-Disposition"] = "attachment; filename=形态选股结果.csv"
        return resp

    @app.route("/")
    def index():
        DISPLAY_LIMIT = 200
        pin_df = _load_csv(_pin_csv_path())
        v_df = _load_csv(_v_csv_path())
        pin_data = pin_df.head(DISPLAY_LIMIT).to_dict("records") if pin_df is not None and len(pin_df) else []
        v_data = v_df.head(DISPLAY_LIMIT).to_dict("records") if v_df is not None and len(v_df) else []
        columns_pin = list(pin_df.columns) if pin_df is not None else []
        columns_v = list(v_df.columns) if v_df is not None else []
        columns_pin = _enrich_with_kline(pin_data, columns_pin)
        columns_v = _enrich_with_kline(v_data, columns_v)
        pin_data = _to_json_serializable(pin_data)
        v_data = _to_json_serializable(v_data)
        return render_template(
            "index.html",
            pin_data=pin_data,
            v_data=v_data,
            columns_pin=columns_pin,
            columns_v=columns_v,
        )

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=5000, debug=True)
