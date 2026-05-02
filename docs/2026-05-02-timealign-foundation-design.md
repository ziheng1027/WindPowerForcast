# TimeAlign 基础设施 + 模型实现设计

> 日期: 2026-05-02
> 状态: 待审核
> 范围: TimeAlign 训练链路的最小基础设施（数据处理、工具集、训练器、模型、入口脚本）

---

## 1. 背景与目标

### 1.1 项目背景

风电功率超短期预测系统（WPF），基于深度学习的多模型时序预测。
- 预测任务：15min 间隔，输入 96 步（24h）→ 预测 16 步（4h）
- 数据：2025 年全年，35,007 条，gfs(6维) + tower(6维) + power(1维)
- 评估：国网 GBT+40607-2021 标准（准确率、合格率）
- 本期目标模型：TimeAlign（ICLR 2026）

### 1.2 本次目标

搭建 TimeAlign 训练所需的最小基础设施，后续模型（iTransformer、DLinear、PatchTST）复用此基础。

### 1.3 核心约束

- 数据处理执行一次，结果持久化，所有模型复用
- 数据划分按时间顺序，不 shuffle
- 归一化仅用 train 集统计量（min-max）
- 不做数据增强（已验证有害）
- 代码风格遵循 Agent.md 规范

---

## 2. 整体架构

```
                         ┌─────────────────────┐
                         │  preprocess.py      │  ← 一次性执行
                         │  raw → processed    │
                         └────────┬────────────┘
                                  │
              data/wind_power/processed/*.pkl
                                  │
         ┌────────────────────────┼────────────────────────┐
         │                        │                        │
    dataset/wind_power.py         │                        │
    (加载processed + 滑动窗口)     │                        │
         │                        │                        │
    DataLoader                    │                        │
         │                        │                        │
    model/timealign.py ◄── module/timealign.py             │
         │                                                │
    trainer/trainer_timealign.py ◄── trainer/trainer_base.py
         │                                                │
    train.py ──→ best_model                    test.py ──→ 评估结果
         │                                                │
    tool/logger.py    tool/metrics.py                     │
    tool/early_stop   tool/utils.py                      │
    tool/visualize.py                                     │
```

---

## 3. 两阶段数据管线

### 3.1 阶段一：数据预处理（一次性）

**脚本**: `preprocess.py`（项目根目录）

**流程**:
1. 加载 `data/wind_power/raw/` 下的 power.csv、gfs.csv、tower.csv
2. 按时间列对齐合并（外连接，缺失值前向填充 → 后向填充）
3. 特征工程：拼接 12 维原始特征 + 4 维时间编码（hour_sin, hour_cos, month_sin, month_cos）
4. 按 8:1:1 时间划分 train/val/test
5. 用 train 集计算各特征 min/max
6. min-max 归一化（仅特征，目标列单独统计）
7. 持久化到 `data/wind_power/processed/`

**输出文件**（`data/wind_power/processed/`）:

| 文件 | 内容 |
|------|------|
| `features.pkl` | 合并后的完整 DataFrame（含时间索引、16维特征） |
| `norm_params.json` | 各特征的 min/max（来自 train 集），目标列单独标记 |
| `split_info.json` | train/val/test 的起止日期、样本数 |

**日志**: 单独写入 `data/wind_power/processed/preprocess.log`，记录：
- 原始数据加载信息（文件、行数、时间范围）
- 缺失值统计与处理方式
- 特征列表与维度
- train/val/test 划分点与样本数
- 归一化参数摘要

**调用方式**:
```bash
python preprocess.py
```

幂等设计：检查 processed/ 下三个文件均存在，且 raw 文件修改时间早于 processed 文件，则跳过并打印提示。

### 3.2 阶段二：数据集构建（每个模型复用）

**文件**: `dataset/wind_power.py`

**职责**: 加载 processed 文件 → 滑动窗口切分 → DataLoader

**不做**: 归一化、特征构建、缺失值处理（这些在阶段一完成）

**接口**:

```python
def get_dataloader(config, flag="train"):
    """
    根据配置返回对应数据集的 DataLoader。

    Parameters
    ----------
    config : dict
        包含 seq_len, pred_len, batch_size 等参数
    flag : str
        "train" / "val" / "test"

    Returns
    -------
    DataLoader
    """
```

**Dataset 类输出**: `(x_enc, y_enc)`
- `x_enc`: [seq_len, 16]（含目标列，作为第0列）
- `y_enc`: [pred_len, 16]（含目标列，作为第0列）

---

## 4. 模型定义

### 4.1 `module/timealign.py` — 子模块

从 TimeAlign 官方仓库提取，适配项目结构。包含：

| 类 | 来源 | 职责 |
|----|------|------|
| `PositionalEmbedding` | layers/Embed.py | 正弦位置编码 |
| `Normalize` | layers/StandardNorm.py | RevIN 风格实例归一化（norm/denorm） |
| `GlocalAlignAblation` | layers/Alignment.py | local + global 对齐损失（glocal_align_ablation） |

### 4.2 `model/timealign.py` — 完整模型

从 TimeAlign 官方仓库 `models/TimeAlign.py` 适配。包含：

| 类 | 职责 |
|----|------|
| `PatchEmbed` | 将时间序列分 patch 投影到 d_model 维 |
| `TimeAlign(Model)` | 主模型：双分支编码 + alignment + 解码投影 |

**forward 签名**:
```python
def forward(self, x, y, is_training=True):
    """
    Parameters
    ----------
    x : [B, seq_len, C] 过去序列（含目标列）
    y : [B, pred_len, C] 未来序列（训练时传入真值）
    is_training : bool

    Returns
    -------
    output : [B, pred_len, C] 预测结果
    y_recon : [B, pred_len, C] 重建结果（训练时）
    align_loss : float 对齐损失
    """
```

**关键设计**:
- enc_in = 16（12 原始 + 4 时间编码）
- patch_num = 4（96/4 = 24 步/patch，16/4 = 4 步/patch）
- 使用 `glocal_align_ablation`（local + global 对齐）

---

## 5. 工具集

### 5.1 `tool/logger.py` — 全流程日志

**设计**: 单例模式，贯穿数据处理→特征构建→模型训练→测试评估。

**四阶段标记**:

| 标记 | 阶段 | 记录内容 |
|------|------|----------|
| `[DATA]` | 数据处理 | 原始数据信息、缺失值处理、归一化参数、划分信息 |
| `[FEAT]` | 特征构建 | 加载的特征列表、维度、矩阵形状 |
| `[TRAIN]` | 模型训练 | 配置摘要、参数量、每epoch的loss分解（pred/recon/align）、lr、耗时、最佳epoch、早停 |
| `[EVAL]` | 测试评估 | MAE/MSE/RMSE/MAPE、国网准确率、合格率、第16步指标、误差统计 |

**输出**:
- 文件: `checkpoints/{model_name}/train.log`
- 控制台: 带颜色同步输出

**接口**:
```python
class Logger:
    def __init__(self, log_dir, model_name):
        """初始化日志器，创建日志文件。"""

    def log_data(self, msg):
        """记录数据处理阶段信息。"""

    def log_feat(self, msg):
        """记录特征构建阶段信息。"""

    def log_train(self, msg):
        """记录训练阶段信息。"""

    def log_eval(self, msg):
        """记录评估阶段信息。"""

    def log_config(self, config):
        """记录完整配置信息。"""
```

### 5.2 `tool/metrics.py` — 评估指标

**指标清单**:

| 指标 | 公式 | 用途 |
|------|------|------|
| MAE | mean(\|pred - true\|) | 基础 |
| MSE | mean((pred - true)²) | 基础 |
| RMSE | √MSE | 基础 |
| 准确率 | 1 - RMSE / Cap × 100% | 国网标准 |
| 合格率 | (\|error\| ≤ 0.25 × Cap) 占比 | 国网标准 |

**装机容量**: Cap = 70,000 kW

**接口**:
```python
def calculate_metrics(preds, trues, capacity=70000):
    """
    计算全部评估指标。

    Returns
    -------
    dict: 包含 mae, mse, rmse, mape, accuracy, qualification_rate
    """

def calculate_step_metrics(preds, trues, step, capacity=70000):
    """
    计算指定步长的评估指标（如第16步）。
    """
```

### 5.3 `tool/early_stopping.py` — 早停

```python
class EarlyStopping:
    def __init__(self, patience=7, verbose=True, delta=0):
        """
        Parameters
        ----------
        patience : int
            容忍的 epoch 数
        verbose : bool
            是否打印信息
        delta : float
            最小改善量
        """

    def __call__(self, val_loss, model, path):
        """
        检查是否需要早停并保存最佳模型。
        """
```

### 5.4 `tool/utils.py` — 辅助工具

```python
def set_seed(seed=42):
    """设置全局随机种子（random/numpy/torch）。"""

def load_config(config_path):
    """加载 YAML 配置文件，返回字典。"""

def get_device(config):
    """根据配置返回 torch.device。"""

def count_parameters(model):
    """统计模型可训练参数量。"""
```

### 5.5 `tool/visualize.py` — 可视化

```python
def plot_loss_curve(train_losses, val_losses, save_path):
    """绘制训练/验证 loss 曲线。"""

def plot_prediction(preds, trues, save_path, n_samples=5):
    """绘制预测 vs 真值对比图。"""

def plot_step_accuracy(preds, trues, capacity, save_path):
    """绘制各步长准确率/合格率。"""
```

保存路径: `figures/{model_name}/`

---

## 6. 训练器

### 6.1 `trainer/trainer_base.py` — 训练基类

**职责**: 定义通用训练流程框架，子类通过覆写钩子方法实现模型特有逻辑。

**接口**:
```python
class TrainerBase:
    def __init__(self, model, config, train_loader, val_loader,
                 test_loader, logger):
        """初始化训练器。"""

    def _select_optimizer(self):
        """根据配置选择优化器（Adam/AdamW/SGD）。"""

    def _select_scheduler(self, optimizer):
        """根据配置选择学习率调度器。"""

    def forward_batch(self, batch, is_training=True):
        """
        前向传播（子类必须覆写）。

        Returns
        -------
        loss : torch.Tensor
        pred_loss : float
        """
        raise NotImplementedError

    def train_epoch(self, epoch):
        """执行一个 epoch 的训练。"""

    def valid(self):
        """在验证集上评估。"""

    def train(self):
        """完整训练流程（epoch循环 + 早停 + 调度）。"""

    def test(self):
        """在测试集上评估。"""
```

**支持的学习率调度器**: CosineAnnealingLR, ReduceLROnPlateau, OneCycleLR, StepLR

**梯度裁剪**: 可配置 `grad_clip` 值

### 6.2 `trainer/trainer_timealign.py` — TimeAlign 训练器

```python
class TrainerTimeAlign(TrainerBase):
    def forward_batch(self, batch, is_training=True):
        """
        TimeAlign 特有的前向逻辑。

        Parameters
        ----------
        batch : tuple (x_enc, y_enc)
            x_enc: [B, seq_len, 16]
            y_enc: [B, pred_len, 16]

        Returns
        -------
        loss : 总损失 = pred_loss + w_recon × recon_loss + w_align × align_loss
        metrics : dict(pred_loss, recon_loss, align_loss)
        """
```

**损失函数**: L1Loss (MAE) 作为基础损失
**损失组合**: `1.0 × pred_loss + w_recon × recon_loss + w_align × align_loss`
**验证/测试**: `is_training=False`，仅返回 x 分支预测

---

## 7. 入口脚本

### 7.1 `train.py` — 训练入口

```bash
python train.py --config config/timealign.yml
```

**流程**:
1. 加载配置
2. 设置随机种子 + 设备
3. 初始化 Logger
4. 构建 DataLoader（调用 dataset/wind_power.py）
5. 日志记录 `[FEAT]` 阶段信息
6. 实例化模型
7. 日志记录模型信息（参数量等）
8. 初始化 Trainer
9. 执行训练（`trainer.train()`）
10. 保存最佳模型到 `checkpoints/{model_name}/best_model.pth`
11. 绘制 loss 曲线

### 7.2 `test.py` — 测试入口

```bash
python test.py --config config/timealign.yml
```

**流程**:
1. 加载配置
2. 构建 DataLoader
3. 加载最佳模型权重
4. 执行推理
5. 计算全部指标（`tool/metrics.py`）
6. 日志记录 `[EVAL]` 阶段信息
7. 生成可视化图表

### 7.3 `preprocess.py` — 数据预处理入口

```bash
python preprocess.py
```

**流程**: 见 §3.1

---

## 8. 配置文件

### `config/timealign.yml`

```yaml
# 模型
model: timealign
seq_len: 96
pred_len: 16
enc_in: 16            # 12原始特征 + 4时间编码
patch_num: 4          # seq_len/patch_num=24步/patch
d_model: 64
d_ff: 256
e_layers: 2
dropout: 0.1
layer_norm: true
pos: true
local_margin: 0.0
global_margin: 0.0
loc: true
glo: true

# 损失权重
w_recon: 1.0
w_align: 0.1

# 训练
seed: 42
device: cuda
lr: 0.001
batch_size: 32
epochs: 100
patience: 10
optimizer: adamw
scheduler: cosine
grad_clip: 1.0

# 数据
data_dir: data/wind_power/processed
target_col: actual_power    # 目标列在特征中的位置（第0列）
```

---

## 9. 文件清单

### 新增文件

| 文件 | 行数估计 | 职责 |
|------|----------|------|
| `preprocess.py` | ~150 | 数据预处理入口 |
| `dataset/wind_power.py` | ~120 | 数据集构建 |
| `module/timealign.py` | ~160 | 子模块（Embed + Normalize + Alignment） |
| `model/timealign.py` | ~100 | 完整模型 |
| `config/timealign.yml` | ~35 | 配置文件 |
| `trainer/trainer_base.py` | ~200 | 训练基类 |
| `trainer/trainer_timealign.py` | ~80 | TimeAlign 训练器 |
| `tool/logger.py` | ~100 | 全流程日志 |
| `tool/metrics.py` | ~80 | 评估指标 |
| `tool/early_stopping.py` | ~60 | 早停 |
| `tool/utils.py` | ~60 | 辅助工具 |
| `tool/visualize.py` | ~80 | 可视化 |
| `train.py` | ~60 | 训练入口 |
| `test.py` | ~60 | 测试入口 |

### 总计：14 个文件，约 ~1,350 行代码

---

## 10. 不在本次范围内

- 其他三个模型（iTransformer、DLinear、PatchTST）的实现
- 模型对比和消融实验
- 超参数搜索
- 部署/推理服务
