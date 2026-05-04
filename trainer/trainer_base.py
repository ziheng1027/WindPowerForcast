"""
训练基类：定义通用训练流程框架。

子类通过覆写 forward_batch() 实现模型特有逻辑。
"""
import torch
import torch.nn as nn
from torch import optim

from tool.early_stopping import EarlyStopping


class TrainerBase:
    """
    训练基类，提供标准训练流程。

    支持 Adam/AdamW/SGD 优化器和
    CosineAnnealing/ReduceLROnPlateau/OneCycleLR 调度器。
    """

    def __init__(self, model, config, train_loader, valid_loader,
                 test_loader, logger, device):
        """
        Parameters
        ----------
        model : nn.Module
            待训练的模型
        config : dict
            训练配置
        train_loader : DataLoader
            训练数据加载器
        valid_loader : DataLoader
            验证数据加载器
        test_loader : DataLoader
            测试数据加载器
        logger : Logger
            日志记录器
        device : torch.device
            计算设备
        """
        self.model = model.to(device)
        self.config = config
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.logger = logger
        self.device = device

        self.epochs = config.get("epochs", 100)
        self.patience = config.get("patience", 10)
        self.grad_clip = config.get("grad_clip", 1.0)

        self.optimizer = self._select_optimizer()
        self.scheduler = self._select_scheduler(self.optimizer)
        self.criterion = nn.L1Loss()

        self.train_losses = []
        self.valid_losses = []

    def _log(self, method, msg):
        """安全日志记录（兼容 logger=None 的静默模式）。"""
        if self.logger is not None:
            getattr(self.logger, method)(msg)

    def _select_optimizer(self):
        """
        根据配置选择优化器。

        Returns
        -------
        optimizer
        """
        name = self.config.get("optimizer", "adamw").lower()
        lr = self.config.get("lr", 0.001)
        params = self.model.parameters()

        if name == "adam":
            return optim.Adam(params, lr=lr)
        elif name == "adamw":
            return optim.AdamW(params, lr=lr)
        elif name == "sgd":
            return optim.SGD(params, lr=lr, momentum=0.9)
        else:
            return optim.AdamW(params, lr=lr)

    def _select_scheduler(self, optimizer):
        """
        根据配置选择学习率调度器。

        Parameters
        ----------
        optimizer
            优化器实例

        Returns
        -------
        scheduler or None
        """
        name = self.config.get("scheduler", "cosine").lower()
        epochs = self.config.get("epochs", 100)

        if name == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs
            )
        elif name == "plateau":
            return optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=5
            )
        elif name == "onecycle":
            steps = len(self.train_loader) * epochs
            return optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=self.config.get("lr", 0.001),
                total_steps=steps
            )
        elif name == "step":
            return optim.lr_scheduler.StepLR(
                optimizer, step_size=20, gamma=0.5
            )
        return None

    def forward_batch(self, batch, is_training=True):
        """
        前向传播（默认实现: 仅计算预测损失）。

        标准接口: model.forward(x, y, is_training) 返回
        (output, y_recon, align_loss)。大多数模型只需
        预测损失，无需覆写此方法。

        仅在需要特殊损失组合时覆写（如 TimeAlign
        的重建损失 + 对齐损失）。

        Parameters
        ----------
        batch : tuple
            数据批次 (x_enc, y_enc, months)
        is_training : bool
            是否训练模式

        Returns
        -------
        loss : torch.Tensor
            总损失
        metrics : dict
            详细指标
        """
        x_enc, y_enc, _, _, _ = batch
        x_enc = x_enc.to(self.device)
        y_enc = y_enc.to(self.device)

        output, _, _ = self.model(
            x_enc, y_enc, is_training=is_training
        )

        # 模型输出 [B, pred_len, C], 只对目标变量(第0维)计算loss
        pred_target = output[:, :, 0]
        true_target = y_enc[:, :, 0]

        loss = self.criterion(pred_target, true_target)
        metrics = {"pred_loss": loss.item()}

        return loss, metrics

    def train_epoch(self, epoch):
        """
        执行一个 epoch 的训练。

        Parameters
        ----------
        epoch : int
            当前 epoch 编号

        Returns
        -------
        float
            平均训练损失
        """
        self.model.train()
        total_loss = []
        total_metrics = {}

        for batch in self.train_loader:
            self.optimizer.zero_grad()

            loss, metrics = self.forward_batch(batch, is_training=True)

            loss.backward()
            if self.grad_clip > 0:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_clip
                )
            self.optimizer.step()

            total_loss.append(loss.item())
            for k, v in metrics.items():
                total_metrics.setdefault(k, [])
                total_metrics[k].append(
                    v if isinstance(v, float) else v
                )

        avg_loss = sum(total_loss) / len(total_loss)
        avg_metrics = {
            k: sum(v) / len(v) for k, v in total_metrics.items()
        }
        return avg_loss, avg_metrics

    def valid(self):
        """
        在验证集上评估。

        Returns
        -------
        float
            平均验证损失
        """
        self.model.eval()
        total_loss = []

        with torch.no_grad():
            for batch in self.valid_loader:
                loss, _ = self.forward_batch(
                    batch, is_training=False
                )
                total_loss.append(loss.item())

        return sum(total_loss) / len(total_loss)

    def train(self):
        """
        完整训练流程。

        epoch 循环 + 早停 + 调度器 + 日志记录。
        """
        model_name = self.config.get("model", "unknown")
        save_path = f"output/checkpoint/{model_name}"
        silent = self.config.get("_silent", False)
        early_stopping = EarlyStopping(
            patience=self.patience, verbose=not silent
        )

        self._log("log_train", "开始训练...")

        for epoch in range(1, self.epochs + 1):
            # 训练
            train_loss, train_metrics = self.train_epoch(epoch)

            # 验证
            valid_loss = self.valid()

            # 记录
            self.train_losses.append(train_loss)
            self.valid_losses.append(valid_loss)

            # 调度器步进
            if self.scheduler is not None:
                if isinstance(
                    self.scheduler,
                    optim.lr_scheduler.ReduceLROnPlateau,
                ):
                    self.scheduler.step(valid_loss)
                else:
                    self.scheduler.step()

            lr = self.optimizer.param_groups[0]["lr"]

            # 日志
            metrics_str = " | ".join(
                f"{k}: {v:.6f}"
                for k, v in train_metrics.items()
            )
            self._log(
                "log_train",
                f"Epoch {epoch}/{self.epochs} | "
                f"Train: {train_loss:.6f} | "
                f"Valid: {valid_loss:.6f} | "
                f"LR: {lr:.2e} | {metrics_str}"
            )

            # 早停检查
            early_stopping(valid_loss, self.model, save_path)
            if early_stopping.early_stop:
                self._log(
                    "log_train",
                    f"早停触发于 epoch {epoch}"
                )
                break

        # 加载最佳模型
        best_path = f"{save_path}/best_model.pth"
        self.model.load_state_dict(
            torch.load(best_path, weights_only=True)
        )
        self._log("log_train", "已加载最佳模型权重")

    def test(self):
        """
        在测试集上评估。

        Returns
        -------
        tuple(np.ndarray, np.ndarray)
            (predictions, ground_truths)
        """
        self.model.eval()
        preds = []
        trues = []

        with torch.no_grad():
            for batch in self.test_loader:
                x_enc, y_enc, _, _, _ = batch
                x_enc = x_enc.to(self.device)
                y_enc = y_enc.to(self.device)

                output, _, _ = self.model(
                    x_enc, y_enc, is_training=False
                )

                # 模型输出 [B, pred_len, C], 取第0维(目标变量)
                preds.append(
                    output[:, :, 0].detach().cpu().numpy()
                )
                trues.append(
                    y_enc[:, :, 0].detach().cpu().numpy()
                )

        import numpy as np
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        return preds, trues
