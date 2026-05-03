"""
MeteoPowerNet 专用训练器。

重写 forward_batch 以处理 8-tuple batch。
"""

import torch

from trainer.trainer_base import TrainerBase


class TrainerMeteoPower(TrainerBase):
    """
    MeteoPowerNet 训练器。

    重写 forward_batch 处理拆分的三路输入
    和未来 GFS 注入。
    """

    def forward_batch(self, batch, is_training=True):
        """
        处理 MeteoPowerDataset 的 8-tuple batch。

        Parameters
        ----------
        batch : tuple
            (power_hist, gfs_hist, tower_hist,
             gfs_future, y_enc, months, dates, minutes)
        is_training : bool

        Returns
        -------
        loss : torch.Tensor
        metrics : dict
        """
        (power_hist, gfs_hist, tower_hist,
         gfs_future, y_enc, _, _, _) = batch

        power_hist = power_hist.to(self.device)
        gfs_hist = gfs_hist.to(self.device)
        tower_hist = tower_hist.to(self.device)
        gfs_future = gfs_future.to(self.device)
        y_enc = y_enc.to(self.device)

        output, _, _ = self.model(
            power_hist, gfs_hist, tower_hist,
            gfs_future, is_training=is_training,
        )

        # 只对目标变量(第0维)计算 loss
        pred_target = output[:, :, 0]
        true_target = y_enc[:, :, 0]

        loss = self.criterion(pred_target, true_target)
        metrics = {"pred_loss": loss.item()}

        return loss, metrics
