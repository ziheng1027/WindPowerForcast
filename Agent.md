# WPF - 风电功率预测

基于深度学习的多模型风电功率时序预测项目，支持 iTransformer、DLinear、PatchTST、TimeAlign 等模型。

---

## Goal

构建超短期风电功率预测系统（15min 间隔，输入 96 步 → 预测 16 步），从数据管线到模型训练 + 国网标准评估。

---

## Constraints & Preferences

### 代码风格

- **命名**: PEP 8（类 PascalCase，方法/变量 snake_case）
- **注释**: 中文，文档注释 `"""简洁中文描述，限定三行以内"""`
- **缩进**: 4 空格，行宽 ≤ 80 字符
- **布局**: 类间 2 空行，方法间 1 空行，同方法内用空行划分代码块
- **导入顺序**: 标准库 → 第三方库 → 本地模块
- **设计**: 模块化，单一职责，简洁可扩展

### 运行环境

- **conda env**: `TSF`（PyTorch 2.11.0+cu128）
- **工作目录**: `D:\YZH\Item\WPF`

### 数据规则

- 滚动统计必须后向窗口，避免数据泄露
- 数据划分按时间顺序，不 shuffle
- 归一化仅用 train 集统计量(min-max)
- 数据增强有害（已验证）

### 国网评估标准 (GBT+40607-2021)

- **装机容量**: Cap = 70,000 kW（14 台 × 5000kW, EN-171/5.0）
- **准确率**: 1 - nRMSE = 1 - RMSE/Cap × 100%
- **合格率**: 误差 ≤ 25%Cap 的占比
- **考核指标**: 月平均 & 第 16 步（4h ahead）准确率 + 合格率

---

## 项目架构

```
WPF/
├── Agent.md                  # 项目架构与职责定义
├── config/                   # 模型配置文件
│   ├── dliner.yml
│   ├── itransformer.yml
│   ├── patchtst.yml
│   └── timealign.yml
├── data/                     # 数据目录
│   └── wind_power/           # 风电数据（2025年GFS/Tower/Power数据）
├── dataset/                  # 数据集构建
│   └── wind_power.py
├── model/                    # 完整模型定义
│   ├── dliner.py
│   ├── itransformer.py
│   ├── patchtst.py
│   └── timealign.py
├── module/                   # 模型子模块
│   ├── dliner.py
│   ├── itransformer.py
│   ├── patchtst.py
│   └── timealign.py
├── trainer/                  # 训练器
│   ├── trainer_base.py
│   ├── trainer_dliner.py
│   ├── trainer_itransformer.py
│   ├── trainer_patchtst.py
│   └── trainer_timealign.py
├── tool/                     # 工具集
│   ├── early_stopping.py
│   ├── logger.py
│   ├── metrics.py
│   ├── utils.py
│   └── visualize.py
├── train.py                  # 训练入口
└── test.py                   # 测试入口
```

---

## 目录职责

### `data/` — 数据存储

存放原始数据和清洗后的数据，按数据集名称组织子目录。

- `wind_power/`: 风电功率原始数据及处理后的数据文件
- 仅负责存储，不包含代码逻辑

### `dataset/` — 数据集构建

负责特征工程、滑动窗口构建以及 DataLoader 生成。

- `wind_power.py`:
  - 数据加载与预处理（缺失值处理、归一化等）
  - 滑动窗口切分（`seq_len` → `pred_len`）
  - `Dataset` 类实现
  - `get_dataloader()` 工厂函数，返回 train/val/test 的 DataLoader

**输入**: `data/` 中的原始数据文件
**输出**: 可直接送入模型的 DataLoader

### `model/` — 完整模型

每个文件定义一个完整的可训练模型，负责将子模块组装成端到端的前向传播流程。

- 每个模型类应实现 `__init__()` 和 `forward()` 方法
- `forward()` 接收原始输入（如 `x_enc`），返回预测结果
- 可直接实例化为 `nn.Module` 送入训练器

**依赖**: `module/` 中的子模块

### `module/` — 模型子模块

存放各模型的核心组件和基础构建块，如注意力机制、嵌入层、MLP 块等。

- 与 `model/` 一一对应，每个文件为同名模型提供子模块
- 每个子模块应职责单一、可独立测试
- 避免跨模型的耦合依赖

### `trainer/` — 训练器

训练逻辑的核心，采用基类 + 派生类的设计模式。

- `trainer_base.py`:
  - 训练基类，定义标准训练流程（epoch 循环、前向传播、损失计算、反向传播、梯度裁剪等）
  - 提供 `train_batch()`、`train_epoch()`、`evaluate_batch(mode="valid/test")`、`valid()`、`train()`、`test()` 等通用接口
  - 预定义多种优化器（Adam、AdamW、SGD 等）和学习率调度器（CosineAnnealingLR、StepLR、ReduceLROnPlateau、OneCycleLR 等），通过配置文件参数选择
- `trainer_{model}.py`:
  - 继承基类，实现特定模型的训练细节
  - 可覆写损失函数、前向逻辑、特殊调度策略等

**依赖**: `model/`、`tool/`（logger、early_stopping、metrics）

### `tool/` — 工具集

提供训练和评估过程中所需的通用工具。

| 文件 | 职责 |
|------|------|
| `logger.py` | 日志记录（训练过程、配置信息、系统输出） |
| `metrics.py` | 评估指标计算（MAE、MSE、RMSE、MAPE 等） |
| `early_stopping.py` | 早停机制（监控验证集指标，防止过拟合） |
| `visualize.py` | 可视化工具（预测曲线、损失曲线、对比图） |
| `utils.py` | 其他辅助函数（随机种子设置、配置加载等） |

### `config/` — 配置文件

每个模型对应一个 YAML 配置文件，包含：

- 模型超参数（`seq_len`、`pred_len`、`d_model`、`n_heads` 等）
- 训练超参数（`lr`、`batch_size`、`epochs` 等）
- 数据路径与预处理参数

### `train.py` — 训练入口

训练脚本的唯一入口，负责：

1. 解析命令行参数或加载配置文件
2. 构建数据集和 DataLoader
3. 实例化模型和训练器
4. 执行训练流程
5. 保存最佳模型权重
6. 保存并绘制训练/验证 loss 曲线

### `test.py` — 测试入口

测试脚本的唯一入口，负责：

1. 加载训练好的最佳模型权重
2. 在测试集上执行推理
3. 计算并输出各项评估指标
4. 生成可视化结果（预测 vs 真值对比图等）

---

## 数据流

```
data/ → dataset/ → DataLoader → model/ → trainer/ → train.py → best_model
                                                  ↘ tool/metrics → test.py → 评估结果
                                                  ↘ tool/visualize → 可视化图表
```

---

## 编码规范

- **命名**: 遵循 PEP 8（类 PascalCase、方法/变量 snake_case）
- **注释**: 使用中文注释，docstring 控制在三行以内
- **缩进**: 4 空格，行宽不超过 80 字符
- **导入顺序**: 标准库 → 第三方库 → 本地模块
- **模型注册**: 新增模型时需同步创建 `config/{name}.yml`、`module/{name}.py`、`model/{name}.py`、`trainer/trainer_{name}.py`
