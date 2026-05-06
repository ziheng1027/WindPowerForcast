# WindPowerForecast

基于深度学习的风电功率超短期预测系统。输入历史 96 步 (24h) 多源数据，预测未来 16 步 (4h) 功率曲线。

**核心成果**：提出 DualMeteo-Net 双分支气象融合网络，在国网 GBT+40607-2021 考核指标（第 4 小时月均准确率）上达到 **89.08%**，突破 87% 合格线，优于 7 个主流时序基线模型。

---

## 背景

风电超短期功率预测是电网调度与电力交易的关键环节。国网标准 GBT+40607-2021 规定：

- **考核步长**：第 16 步（4h ahead）
- **准确率**：`(1 - RMSE / Cap) × 100%`，月均 ≥ 87% 方为合格
- **合格率**：误差 ≤ 25%Cap 的样本占比
- **装机容量**：70,000 kW（14 台 × 5000kW）

---

## 基线实验

在相同数据集（2025 年全年 GFS 气象预报 + 测风塔实测 + 功率数据，12 月测试）上复现了 7 个主流时序预测模型：

| 模型 | 类型 | 第 16 步准确率 | 合格率 |
|:-----|:-----|:--------------:|:------:|
| LSTM | 循环网络 | 85.18% | 88.15% |
| BiLSTM | 双向循环网络 | 85.61% | 88.36% |
| TCN | 时序卷积网络 | 85.79% | 87.69% |
| DLinear | 线性基线 | 84.85% | 85.34% |
| iTransformer | 倒置 Transformer | 84.66% | 86.62% |
| PatchTST | 补丁 Transformer | 84.63% | 85.86% |
| TimeAlign | 时序对齐 Transformer | 85.03% | 86.46% |
| **DualMeteo-Net** | **气象-功率双分支融合** | **89.08%** | **92.64%** |

> 基线模型均采用 23 维统一拼接输入、标准 Encoder-Decoder 框架。

**关键发现**：基线模型在第 1 步 (15min) 均达到 94%+ 准确率，但第 16 步 (4h) 集体降至 84-86%，距离 87% 合格线存在系统性差距。

---

## 数据瓶颈分析

基线模型集体未能达标的核心原因：**缺失未来气象信息注入**。

4 小时后的功率变化主要由未来的风况决定。GFS 数值天气预报提供了未来 4h 的风速、风向、温度、湿度、气压等 6 维气象预报，但基线模型仅依赖历史 96 步数据外推，无法获取未来气象变化信号，导致在功率急变日（如寒潮大风、静风期突起）出现大幅偏差。

此外还有两个次要因素：

- **单流拼接输入**：23 维特征直接拼接送入模型，功率 (1 维) 与气象 (12 维) 的模态差异被忽略，模型难以区分哪些特征在驱动功率变化
- **无跨域交互机制**：功率与气象之间的非线性关系 (P ∝ v³) 需要显式建模，而非隐式期待模型自行发现

---

## DualMeteo-Net

### 架构

```
历史功率 [B, 96, 1]  ──→ BiLSTM ──→ Power 编码 [B, 128] ─────────┐
                                                                    │
历史 GFS [B, 96, 6]  ──→┐                                          │
                         ├──→ 共享 TCN ──→ Meteo 编码 ──→ KV     ├──→ Cross-Attn ──→ 融合向量 [B, 192]
历史 Tower [B, 96, 6] ──→┘                         (拼接 GFS+Tower)│
                                                                    │
未来 GFS [B, 16, 6]   ──→ 独立 TCN ──→ 未来气象编码 [B, 32]  ─────┤
                                                                    │
                                                            ┌───────┘
                                                            ↓
                                                    MLP Decoder
                                                            ↓
                                                 输出 [B, 16, 17]
```

### 设计要点

| 组件 | 作用 |
|------|------|
| **BiLSTM 功率编码器** | 捕捉功率序列的双向时序依赖 |
| **共享 TCN 气象编码器** | GFS 与 Tower 共享权重，提取多源气象特征，减少参数量 |
| **Cross-Attention 融合** | Power 做 Query，Meteo 做 Key/Value，让模型聚焦与功率最相关的气象信号 |
| **未来 GFS 注入** | 独立 TCN 编码未来 4h 气象预报，为长步预测提供关键前瞻信息 |
| **MLP 解码器** | 将融合特征映射为 16 步 × 17 维输出 |

### 超参数（经 Optuna 贝叶斯搜索）

| 参数 | 值 |
|------|----|
| BiLSTM hidden_size | 64 |
| TCN channels (历史) | [32, 64] |
| TCN channels (未来) | [16, 32] |
| TCN kernel_size | 3 |
| Cross-Attention dim | 64 |
| Dropout | 0.2 |
| Optimizer | AdamW |
| Learning Rate | 3e-4 |
| Scheduler | ReduceLROnPlateau |
| Batch Size | 64 |
| Early Stopping Patience | 10 |

---

## 项目结构

```
WindPowerForecast/
├── config/                    # 模型配置 (YAML)
│   ├── meteo_power.yml        #   DualMeteo-Net
│   ├── lstm.yml / bilstm.yml  #   循环网络基线
│   ├── tcn.yml                #   TCN 基线
│   ├── dlinear.yml            #   DLinear 基线
│   ├── itransformer.yml       #   iTransformer 基线
│   ├── patchtst.yml           #   PatchTST 基线
│   └── timealign.yml          #   TimeAlign 基线
├── data/                      # 原始数据 (不入库)
│   └── wind_power/
│       ├── raw/               #   power.csv / gfs.csv / tower.csv
│       └── processed/         #   预处理产物 (features.pkl, norm_params.json)
├── dataset/                   # 数据集构建
│   ├── wind_power.py          #   通用 23 维拼接数据集
│   └── meteo_power.py         #   DualMeteo-Net 三路拆分数据集
├── model/                     # 模型定义
│   ├── meteo_power.py         #   DualMeteo-Net (MeteoPowerNet)
│   ├── lstm.py / tcn.py       #   基线模型
│   └── ...
├── trainer/                   # 训练器 (基类 + 派生)
│   ├── trainer_base.py        #   通用训练循环、优化器、早停
│   ├── trainer_meteo.py       #   DualMeteo-Net 专用前向逻辑
│   └── trainer_timealign.py   #   TimeAlign 专用前向逻辑
├── tool/                      # 工具集
│   ├── metrics.py             #   国标评估指标 (GBT+40607-2021)
│   ├── logger.py              #   日志记录
│   ├── early_stopping.py      #   早停策略
│   ├── visualize.py           #   可视化 (预测曲线、准确率对比)
│   └── utils.py               #   配置加载、设备管理、随机种子
├── preprocess.py              # 数据预处理入口
├── train.py                   # 训练入口
├── test.py                    # 测试入口 (评估 + 可视化 + 导出)
├── tune.py                    # Optuna 超参搜索
└── output/                    # 运行产物 (不入库)
    ├── checkpoint/            #   模型权重
    ├── figure/                #   可视化图表
    ├── log/                   #   训练日志
    └── result/                #   逐日/逐点评估 CSV
```

---

## 快速开始

### 环境要求

- Python ≥ 3.10
- PyTorch ≥ 2.0 (推荐 CUDA)
- 依赖：`numpy`, `pandas`, `scikit-learn`, `optuna`, `matplotlib`, `pyyaml`

### 数据准备

将原始数据放入 `data/wind_power/raw/`：

```
raw/
├── power.csv    # 时间戳 + 实际功率
├── gfs.csv      # 时间戳 + GFS 6维气象预报
└── tower.csv    # 时间戳 + 测风塔 6维实测
```

运行预处理（只需执行一次）：

```bash
python preprocess.py
```

产物保存在 `data/wind_power/processed/`，包含归一化特征矩阵、归一化参数、划分信息。

### 训练

```bash
# DualMeteo-Net
python train.py --config config/meteo_power.yml

# 基线模型
python train.py --config config/lstm.yml
python train.py --config config/itransformer.yml
```

### 测试

```bash
python test.py --config config/meteo_power.yml
```

输出包括：
- `output/result/{model}/step16_daily.csv` — 逐日准确率/合格率
- `output/result/{model}/step16_detail.csv` — 逐点预测明细
- `output/figure/{model}/` — 预测曲线、准确率对比图

### 超参搜索

```bash
python tune.py --n_trials 30
```

基于 Optuna TPE 采样器搜索最优配置，结果保存在 `output/tune/`。

---

## 评估指标说明

| 指标 | 定义 | 说明 |
|------|------|------|
| 准确率 | `(1 - RMSE / Cap) × 100%` | Cap = 70,000 kW |
| 合格率 | `\|误差\| ≤ 25%Cap` 的样本占比 | 单点判定 |
| 相关系数 R | Pearson 相关系数 | 衡量趋势一致性 |
| 计算流程 | 日均值 → 月均值 | 按国标逐日计算后取月均 |

---

## 许可证

[Apache License 2.0](LICENSE)
