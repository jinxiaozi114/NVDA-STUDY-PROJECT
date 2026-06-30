import json
from pathlib import Path


ROOT = Path(r"C:\Users\JINYIHANG\Documents\Codex\2026-06-26\w")
OUT = ROOT / "NVDA_logistic_regression_model_report.ipynb"

cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.splitlines(True)})


def code(text):
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(True),
    })


md("""# NVIDIA Stock Return Prediction Report

## 基于 Logistic Regression、市场因子与财报事件的 NVDA 上涨概率预测

本报告基于已经完成的模型输出文件，对 NVIDIA（NVDA）的短期收益预测模型进行总结和分析。

研究目标不是预测具体股价，而是预测：

> NVDA 未来 5 个交易日是否上涨。

如果模型预测上涨概率高于设定阈值，则策略选择持有 NVDA；否则选择空仓。报告重点分析 Logistic Regression 作为基准模型的表现，并与 Random Forest、Gradient Boosting、Hist Gradient Boosting 等模型进行对比。
""")

md("""## 1. 使用的数据文件

本报告使用以下模型输出文件：

| 文件 | 作用 |
|---|---|
| `nvda_model_dataset.csv` | 清洗后并构造好特征的建模数据集 |
| `nvda_multi_model_comparison.csv` | 多模型表现对比结果 |
| `nvda_best_model_test_result.csv` | 最优模型在测试集上的每日预测与策略表现 |
| `nvda_best_model_feature_importance.csv` | 最优模型特征重要性 |
| `nvda_earnings_events_used.csv` | 财报事件变量 |

数据中包含三类核心信息：

1. **NVDA 自身交易特征**：收益率、波动率、均线偏离、成交量变化。
2. **市场因子**：SPY、QQQ、SOXX，用于刻画大盘、科技股和半导体行业环境。
3. **财报事件因子**：营收增长、数据中心收入增长、毛利率变化、EPS 变化、AI 情绪、出口风险等。
""")

code(r'''from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# If this notebook is placed inside the GitHub project root, use ./output.
# Otherwise, it falls back to the local output folder used during analysis.
output_dir = Path("output")

if not output_dir.exists():
    output_dir = Path(r"D:\工作\七月份股票学习，以及相关作业完成\output")

model_df = pd.read_csv(output_dir / "nvda_model_dataset.csv")
comparison = pd.read_csv(output_dir / "nvda_multi_model_comparison.csv")
best_result = pd.read_csv(output_dir / "nvda_best_model_test_result.csv")
feature_importance = pd.read_csv(output_dir / "nvda_best_model_feature_importance.csv")
earnings_events = pd.read_csv(output_dir / "nvda_earnings_events_used.csv")

print("Model dataset shape:", model_df.shape)
print("Model dataset date range:", model_df["Date"].iloc[0], "to", model_df["Date"].iloc[-1])
print("Test result date range:", best_result["Date"].iloc[0], "to", best_result["Date"].iloc[-1])''')

md("""## 2. 预测目标与建模逻辑

模型目标是预测未来 5 个交易日 NVDA 是否上涨：

```text
Future_Return_5D > 0  →  Target = 1
Future_Return_5D <= 0 →  Target = 0
```

模型输出的是上涨概率 `Pred_Prob_Up`，不是具体股价。

交易规则为：

```text
如果 Pred_Prob_Up >= Threshold，则持有 NVDA
如果 Pred_Prob_Up < Threshold，则空仓
```

这个阈值决定了模型从“概率判断”转成“交易信号”的标准。
""")

code('''target_counts = model_df["Target"].value_counts().rename(index={0: "Down / Non-positive", 1: "Up"})
target_counts''')

md("""## 3. Logistic Regression 基准模型表现

Logistic Regression 是本项目中的基准模型。它的优点是简单、可解释、能直接输出上涨概率。

从结果看，Logistic Regression 在不同阈值下的表现如下。重点观察：

- `Accuracy`：方向预测准确率。
- `Final_Strategy_Return`：模型策略最终收益。
- `Final_BuyHold_Return`：买入持有收益。
- `Strategy_Max_Drawdown`：模型策略最大回撤。
- `Strategy_Sharpe`：模型策略夏普比率。
""")

code('''logistic_results = comparison[comparison["Model"] == "Logistic Regression"].copy()
logistic_results''')

md("""### Logistic Regression 结果解读

在阈值 `0.48` 下，Logistic Regression 的表现相对最好：

| 指标 | 结果 |
|---|---:|
| Accuracy | 50.42% |
| Strategy Return | 29.54% |
| Buy and Hold Return | 29.63% |
| Excess Return | -0.09% |
| Strategy Max Drawdown | -9.39% |
| Buy and Hold Max Drawdown | -20.22% |
| Strategy Sharpe | 1.31 |
| Buy and Hold Sharpe | 0.95 |

这说明 Logistic Regression 并没有明显跑赢 Buy and Hold 的收益，但它显著降低了最大回撤，并提高了风险调整后收益。

换句话说：

> Logistic Regression 的主要价值不是提高收益，而是改善风险控制。
""")

md("""## 4. 多模型对比

除了 Logistic Regression，本项目也比较了 Random Forest、Gradient Boosting 和 Hist Gradient Boosting。

多模型对比可以帮助判断：线性模型是否足够，还是非线性模型更适合捕捉 NVDA 与市场因子之间的关系。
""")

code('''comparison.head(10)''')

md("""### 最优模型结果

根据 `Excess_Return` 排序，最优模型为：

```text
Random Forest
Threshold = 0.48
```

表现如下：

| 指标 | Random Forest 策略 | Buy and Hold |
|---|---:|---:|
| Final Return | 32.79% | 29.63% |
| Excess Return | +3.16% | - |
| Max Drawdown | -10.83% | -20.22% |
| Sharpe Ratio | 1.21 | 0.95 |
| Accuracy | 54.20% | - |

Random Forest 相比 Logistic Regression，收益略高，并且仍然明显降低了回撤。

这说明 NVDA 的短期上涨概率可能存在一定非线性结构，仅靠线性模型不一定能完全捕捉。
""")

md("""## 5. 策略净值曲线

下图比较最优模型策略与 Buy and Hold 的净值曲线。
""")

code('''best_result["Date"] = pd.to_datetime(best_result["Date"])

plt.figure(figsize=(12, 6))
plt.plot(best_result["Date"], best_result["Strategy_Equity"], label="Model Strategy", linewidth=2)
plt.plot(best_result["Date"], best_result["BuyHold_Equity"], label="Buy and Hold", linewidth=2)

plt.title("NVDA Model Strategy vs Buy and Hold")
plt.xlabel("Date")
plt.ylabel("Equity Curve")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()''')

md("""## 6. 特征重要性分析

最优模型的特征重要性显示，模型最关注以下几类变量：

1. NVDA 与半导体板块 SOXX 的相关性。
2. NVDA 自身中期收益率。
3. NVDA 与 SPY、QQQ 的滚动相关性。
4. SOXX、QQQ、SPY 等市场因子。
5. NVDA 波动率和均线偏离。

这说明模型不是只看 NVDA 自己，而是在判断：

> NVDA 的走势是否与大盘、科技股和半导体行业环境配合。
""")

code('''feature_importance.head(20)''')

code('''top_features = feature_importance.head(15).sort_values("Importance")

plt.figure(figsize=(10, 6))
plt.barh(top_features["Feature"], top_features["Importance"])
plt.title("Top 15 Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.show()''')

md("""## 7. 财报事件变量

本项目加入了 NVIDIA 财报事件变量，包括：

- Revenue_QoQ：营收环比增长
- DataCenter_QoQ：数据中心收入环比增长
- Gross_Margin_Change：毛利率变化
- EPS_QoQ：EPS 环比变化
- Guidance_Sentiment：管理层指引情绪
- AI_Sentiment：AI 需求情绪
- Export_Risk：出口限制风险

这些变量不是每天都有影响，而是作为财报窗口的事件变量，帮助模型识别重要信息披露期。
""")

code('''earnings_events''')

md("""## 8. 研究结论

本项目构建了一个结合技术指标、市场因子和财报事件的 NVDA 上涨概率预测框架。

主要结论如下：

1. **Logistic Regression 是一个清晰的基准模型。**  
   它没有显著跑赢 Buy and Hold，但有效降低了最大回撤，并提高了夏普比率。

2. **Random Forest 表现最佳。**  
   在测试期内，Random Forest 策略收益为 32.79%，高于 Buy and Hold 的 29.63%，同时最大回撤从 -20.22% 降至 -10.83%。

3. **市场因子非常重要。**  
   特征重要性显示，NVDA 与 SOXX、SPY、QQQ 的相关性和市场状态对预测结果有明显影响。

4. **模型的核心价值是风险控制。**  
   本项目的模型并不是“精准预测股价”，而是通过概率判断帮助策略减少部分不利阶段的暴露。
""")

md("""## 9. 局限性与后续改进

本项目仍有一些局限：

1. 测试期较短，需要更长时间的 out-of-sample 检验。
2. 财报和新闻情绪变量仍带有人工主观判断。
3. 当前策略只考虑持有或空仓，没有考虑仓位大小。
4. 没有加入更真实的滑点、税费和流动性约束。
5. 还没有做 walk-forward 回测。

后续可以继续改进：

- 加入 walk-forward backtest。
- 用更严格的时间序列交叉验证。
- 加入更多半导体股票作为横截面因子。
- 加入真实新闻文本情绪分析。
- 增加仓位管理规则，而不是简单 0/1 持仓。
""")

md("""## 10. GitHub 项目说明

建议 GitHub 项目结构如下：

```text
NVDA-STUDY-PROJECT/
├── data/
│   ├── NVDA.csv
│   ├── SPY.csv
│   ├── QQQ.csv
│   └── SOXX.csv
├── output/
│   ├── nvda_model_dataset.csv
│   ├── nvda_multi_model_comparison.csv
│   ├── nvda_best_model_test_result.csv
│   ├── nvda_best_model_feature_importance.csv
│   └── nvda_earnings_events_used.csv
├── reports/
│   └── NVDA_logistic_regression_model_report.ipynb
└── README.md
```

项目描述可以写：

> A quantitative research project predicting NVIDIA short-term return direction using Logistic Regression, market factors, earnings events, and machine learning model comparison.
""")

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.x"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(OUT)
