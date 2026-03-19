# -*- coding: utf-8 -*-
"""直接导出形态选股结果为 CSV（不启动网页）。"""
import csv
import os
import sys

_EXAMPLES_DIR = os.path.dirname(os.path.abspath(__file__))
if _EXAMPLES_DIR not in sys.path:
    sys.path.insert(0, _EXAMPLES_DIR)

from scan_viewer.app import _build_export_data
import numpy as np
import pandas as pd

OUTPUTS_DIR = os.path.join(_EXAMPLES_DIR, "outputs")
OUT_CSV = os.path.join(OUTPUTS_DIR, "形态选股结果.csv")


def main():
    rows, columns = _build_export_data()
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    with open(OUT_CSV, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
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
    print("已导出:", OUT_CSV)
    print("共", len(rows), "条")


if __name__ == "__main__":
    main()
