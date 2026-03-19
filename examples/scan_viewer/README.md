# 形态选股结果展示

在浏览器中查看品字形、V字形扫描结果，并可通过「同花顺链接」列跳转到 [同花顺个股页](https://stockpage.10jqka.com.cn/)。

## 依赖

- Python 3.8+
- Flask、pandas（若项目已安装 `requirements.txt` 一般已满足）

```bash
pip install flask pandas
```

## 运行

在项目根目录或 `examples` 下执行：

```bash
cd Kronos/examples/scan_viewer
python app.py
```

浏览器访问：**http://127.0.0.1:5000**

## 数据来源

- 品字形表：`examples/outputs/pin_pattern_scan_result.csv`
- V字形表：`examples/outputs/v_pattern_scan_result.csv`

先运行 `pin_pattern_scan.py` 和 `v_pattern_scan.py` 生成 CSV 后，本站在刷新页面时会自动读取最新结果。每行股票的「同花顺链接」格式为：`https://stockpage.10jqka.com.cn/{股票代码数字部分}/`（如 300197.SZ → 300197）。
