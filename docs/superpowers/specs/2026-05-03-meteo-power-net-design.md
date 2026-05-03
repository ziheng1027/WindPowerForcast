# MeteoPower-Net：气象-功率融合网络设计

> 日期: 2026-05-03
> 目标: 突破国网 87% 准确率要求（当前最优 BiLSTM 86.29%）

## 1. 问题分析

### 1.1 当前瓶颈

当前所有模型将 23 维特征（Power + GFS + Tower + 衍生 + 时间）
拼接后统一输入，存在两个根本问题：

1. **数据类型平等化**：GFS 预报、Tower 实测、Power 历史物理意义完全不同，
   拼接后模型无法区分它们的语义
2. **未利用未来气象信息**：实际场景中 GFS 能提供未来 4h 预报，
   但当前框架只用历史数据预测未来

### 1.2 突破方向

严格模拟真实预报场景：

- 12 月测试时，只用 GFS 数据作为"未来预报"
- Tower 和 Power 仅作为历史输入
- Power 只作为评估标签

## 2. 模型架构

### 2.1 整体数据流

```
历史编码 (t-96 ~ t-1)                      未来注入 (t ~ t+15)
                                           
Power [B,96,1] ─→ BiLSTM ─→ h_pow [B,128]─┐
                                            │
GFS [B,96,6] ─→ TCN_enc ─→ h_gfs [B,32] ──┼─→ CrossAttn ─→ fused [B,192]
                                            │     (Q=h_pow)
Tower [B,96,6] ─→ TCN_enc ─→ h_tow [B,32]─┘
                                                              │
GFS_future [B,16,6] ─→ TCN_dec ─→ h_fut [B,32] ─────────────┤
                                                              │
                                              concat ─→ [B,224]
                                                              │
                                           MLP Decoder ─→ [B,16,17]
                                                              │
                                        loss = L1(output[:,:,0], target)
```

### 2.2 模块详细设计

#### Power 编码器 (BiLSTM)

- 输入: `[B, 96, 1]` (actual_power 历史)
- 双向 LSTM, hidden_size=64, 1 层, dropout=0.2
- 取最后一层双向隐状态拼接 → `[B, 128]`
- 选择理由: 功率序列时序依赖最强，双向编码捕捉升降趋势前后的模式

#### GFS 编码器 (TCN)

- 输入: `[B, 96, 6]` (6 个 GFS 特征)
- TCN, channels=[32, 32, 32], kernel_size=3, dropout=0.2
- 取最后时间步 → `[B, 32]`
- 选择理由: 气象变量连续渐变，卷积擅长捕捉局部模式

#### Tower 编码器 (TCN, 权重共享)

- 输入: `[B, 96, 6]` (6 个 Tower 特征)
- **与 GFS 编码器共享 TCN 权重**（Tower 和 GFS 特征结构相同:
  风速、风向sin/cos、温度、湿度、气压）
- 取最后时间步 → `[B, 32]`
- 权重共享减少参数量，强制网络学到气象特征的通用表征

#### 跨模态融合 (Cross-Attention)

- Query: `h_pow [B, 128]` → Linear(128, 64) → `[B, 1, 64]`
- Key/Value: `cat(h_gfs, h_tow) [B, 64]`
  → Linear(64, 64) → `[B, 1, 64]`
- 单头注意力, 输出 `[B, 64]`
- 与 `h_pow` 拼接 → `[B, 192]`
- 设计意图: Power 编码器主动"询问"气象编码器关键信息

#### 未来 GFS 注入 (TCN + Concat)

- 输入: `[B, 16, 6]` (未来 16 步 GFS 预报)
- 独立 TCN, channels=[32, 32], kernel_size=3
- 取最后时间步 → `[B, 32]`
- 与融合结果 concat → `[B, 224]`

#### 解码器 (MLP)

- `[B, 224]` → Linear(224, 512) → ReLU → Dropout(0.2)
  → Linear(512, 16 × 17) → reshape `[B, 16, 17]`
- Loss 只取第 0 维 (actual_power), L1Loss
- 输出 17 维与现有模型一致，复用评估流程

### 2.3 参数量估算

| 模块 | 参数量 |
|------|--------|
| BiLSTM (input=1, hidden=64, bidirectional) | ~50K |
| TCN-气象 (input=6, channels=[32,32,32]) | ~25K |
| Cross-Attention (64d) | ~12K |
| TCN-未来 (input=6, channels=[32,32]) | ~10K |
| MLP Decoder (224→512→272) | ~130K |
| **总计** | **~230K** |

## 3. 数据集设计

### 3.1 新增 MeteoPowerDataset

新建 `dataset/meteo_power.py`，不影响现有 `WindPowerDataset`。

数据列索引映射（来自 feature_cols 顺序）：

| 索引 | 含义 | 用途 |
|------|------|------|
| [0] | actual_power | Power 编码器 |
| [1:7] | GFS 6 维 | GFS 编码器 |
| [7:13] | Tower 6 维 | Tower 编码器 |
| [13:23] | 衍生+时间 10 维 | 暂不使用 |

### 3.2 返回值

```python
def __getitem__(self, idx) -> tuple:
    return (
        power_hist,    # [seq_len, 1]     历史 Power
        gfs_hist,      # [seq_len, 6]     历史 GFS
        tower_hist,    # [seq_len, 6]     历史 Tower
        gfs_future,    # [pred_len, 6]    未来 GFS（关键！）
        power_target,  # [pred_len, 17]   未来 Power（标签）
        months,        # [pred_len]       月份
        dates,         # [pred_len]       日期编码
        minutes,       # [pred_len]       分钟数
    )
```

### 3.3 未来 GFS 的截取

当前 `y_enc` 只包含 17 维（无 GFS）。
需要从原始数据中额外截取未来 `pred_len` 步的 GFS 列：

```
gfs_future = data[i+seq_len : i+total_len, 1:7]  # GFS 列索引
```

这从同一个 data 数组中直接切片，无需额外数据文件。

### 3.4 数据加载函数

```python
def get_meteo_dataloader(config, flag="train"):
    """返回 MeteoPowerDataset 的 DataLoader。"""
```

与现有 `get_dataloader` 并列，模型通过 config 中的
`dataset_type: "meteo"` 字段选择使用哪个数据集。

## 4. 训练/测试数据流

### 4.1 训练和验证 (1-10 月训练, 11 月验证)

与现有流程相同，额外提供 gfs_future：

| 输入 | 来源 | 维度 |
|------|------|------|
| power_hist | 历史窗口内 Power | [96, 1] |
| gfs_hist | 历史窗口内 GFS | [96, 6] |
| tower_hist | 历史窗口内 Tower | [96, 6] |
| gfs_future | 预测窗口内 GFS | [16, 6] |
| power_target | 预测窗口内 Power | [16, 17] |

### 4.2 测试 (12 月，严格模拟预报)

测试集的 gfs_future 来自 12 月的 GFS 数据（模拟未来预报）。
由于滑动窗口内的 data 数组已经包含了 12 月的 GFS 数据，
gfs_future 的截取方式与训练时完全相同，无需特殊处理。

Tower 和 Power 历史数据来自 12 月窗口的前 96 步，
这在真实场景中对应"已知的历史实测值"，是合理的。

## 5. Trainer 设计

### 5.1 新增 TrainerMeteoPower

继承 TrainerBase，重写 `forward_batch`：

```python
class TrainerMeteoPower(TrainerBase):
    def forward_batch(self, batch, is_training=True):
        (power_hist, gfs_hist, tower_hist,
         gfs_future, y_enc, _, _, _) = batch

        output = self.model(
            power_hist, gfs_hist, tower_hist,
            gfs_future, is_training=is_training,
        )

        pred_target = output[:, :, 0]
        true_target = y_enc[:, :, 0]
        loss = self.criterion(pred_target, true_target)
        return loss, {"pred_loss": loss.item()}
```

### 5.2 模型 forward 签名

```python
class MeteoPowerNet(nn.Module):
    def forward(self, power_hist, gfs_hist, tower_hist,
                gfs_future, is_training=True):
        ...
        return output, None, 0.0
        # output: [B, pred_len, 17]
```

## 6. 文件清单

| 操作 | 文件 | 说明 |
|------|------|------|
| 新建 | `model/meteo_power.py` | MeteoPowerNet 模型 |
| 新建 | `dataset/meteo_power.py` | MeteoPowerDataset 数据集 |
| 新建 | `trainer/trainer_meteo.py` | TrainerMeteoPower 训练器 |
| 新建 | `config/meteo_power.yml` | 配置文件 |
| 修改 | `train.py` | 注册新模型和数据集 |
| 修改 | `test.py` | 注册新模型和适配测试流程 |

现有模型和数据集完全不受影响。

## 7. 评估指标

复用现有 test.py 的评估流程：

- 全局 RMSE / MAE
- 国标第 16 步月均准确率 / 合格率 / 相关系数
- 国标第 1 步月均准确率 / 合格率 / 相关系数
- 目标: 第 16 步月均准确率 ≥ 87%

## 8. 风险与缓解

| 风险 | 缓解措施 |
|------|----------|
| 参数量 ~230K, 数据 3.5 万可能过拟合 | EarlyStopping (patience=10), Dropout=0.2, Weight Decay |
| 权重共享可能限制 Tower 表达力 | 先验证共享效果，不行再解耦 |
| 未来 GFS 注入可能主导预测，忽略历史 | 观察 attention 权重分布，必要时加辅助损失 |
