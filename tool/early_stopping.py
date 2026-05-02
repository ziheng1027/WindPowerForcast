"""
早停机制：监控验证集指标，自动保存最佳模型。
"""

import os

import torch


class EarlyStopping:
    """
    早停控制器。

    当验证损失连续 patience 个 epoch 未改善时触发早停，
    并在每次改善时自动保存最佳模型权重。
    """

    def __init__(self, patience=7, verbose=True, delta=0):
        """
        Parameters
        ----------
        patience : int
            容忍的 epoch 数
        verbose : bool
            是否打印改善信息
        delta : float
            最小改善量阈值
        """
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.valid_loss_min = float("inf")

    def __call__(self, valid_loss, model, path):
        """
        检查是否需要早停并保存最佳模型。

        Parameters
        ----------
        valid_loss : float
            当前验证损失
        model : nn.Module
            待保存的模型
        path : str
            模型保存目录
        """
        score = -valid_loss

        if self.best_score is None:
            self.best_score = score
            self._save_checkpoint(valid_loss, model, path)
            return

        if score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(
                    f"EarlyStopping counter: "
                    f"{self.counter}/{self.patience}"
                )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self._save_checkpoint(valid_loss, model, path)
            self.counter = 0

    def _save_checkpoint(self, valid_loss, model, path):
        """
        保存最佳模型权重。

        Parameters
        ----------
        valid_loss : float
            当前验证损失
        model : nn.Module
            待保存的模型
        path : str
            模型保存目录
        """
        os.makedirs(path, exist_ok=True)
        save_path = os.path.join(path, "best_model.pth")
        if self.verbose:
            print(
                f"验证损失改善: "
                f"{self.valid_loss_min:.6f} -> {valid_loss:.6f}, "
                f"保存模型 -> {save_path}"
            )
        torch.save(model.state_dict(), save_path)
        self.valid_loss_min = valid_loss
