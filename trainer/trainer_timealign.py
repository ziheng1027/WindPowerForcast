"""
TimeAlign 训练器：双输入 + 三部分损失。
"""

import torch
import torch.nn as nn

from trainer.trainer_base import TrainerBase


class TrainerTimeAlign(TrainerBase):
    """
    TimeAlign 专用训练器。

    损失 = pred_loss + w_recon × recon_loss + w_align × align_loss
    """

    def __init__(self, model, config, train_loader, valid_loader,
                 test_loader, logger, device):
        """
        Parameters
        ----------
        model : nn.Module
            TimeAlign 模型实例
        config : dict
            训练配置，额外需要 w_recon 和 w_align
        """
        super().__init__(
            model, config, train_loader, valid_loader,
            test_loader, logger, device
        )
        self.w_recon = config.get("w_recon", 1.0)
        self.w_align = config.get("w_align", 0.1)
        self.pred_len = config["pred_len"]

    def forward_batch(self, batch, is_training=True):
        """
        TimeAlign 前向传播。

        Parameters
        ----------
        batch : tuple(x_enc, y_enc)
            x_enc: [B, seq_len, C]
            y_enc: [B, pred_len, C]
        is_training : bool
            是否训练模式

        Returns
        -------
        loss : torch.Tensor
            总损失
        metrics : dict
            分项损失详情
        """
        x_enc, y_enc, _, _, _ = batch
        x_enc = x_enc.to(self.device)
        y_enc = y_enc.to(self.device)

        # 模型前向
        output, y_recon, align_loss = self.model(
            x_enc, y_enc, is_training=is_training
        )

        # 模型输出 [B, pred_len, C], 只对目标变量(第0维)计算loss
        pred_target = output[:, -self.pred_len :, 0]
        true_target = y_enc[:, -self.pred_len :, 0]

        # 预测损失
        pred_loss = self.criterion(pred_target, true_target)

        if is_training:
            # 重建损失
            recon_target = y_recon[:, -self.pred_len :, 0]
            recon_loss = self.criterion(recon_target, true_target)

            # 总损失
            loss = (
                pred_loss
                + self.w_recon * recon_loss
                + self.w_align * align_loss
            )

            metrics = {
                "pred_loss": pred_loss.item(),
                "recon_loss": recon_loss.item(),
                "align_loss": (
                    align_loss.item()
                    if torch.is_tensor(align_loss)
                    else align_loss
                ),
            }
        else:
            # 验证/测试时只计算预测损失
            loss = pred_loss
            metrics = {"valid_loss": pred_loss.item()}

        return loss, metrics
