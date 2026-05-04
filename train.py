"""
训练入口：加载数据 → 构建模型 → 训练 → 保存。
"""

import argparse

from tool.logger import Logger
from tool.utils import (
    count_parameters,
    get_device,
    load_config,
    set_seed,
)
from tool.visualize import plot_loss_curve


def get_trainer(model_name):
    """
    根据模型名返回对应的训练器类。

    Parameters
    ----------
    model_name : str
        模型名称

    Returns
    -------
    type
        训练器类
    """
    from trainer.trainer_timealign import TrainerTimeAlign
    from trainer.trainer_meteo import TrainerMeteoPower
    from trainer.trainer_base import TrainerBase

    trainers = {
        "timealign": TrainerTimeAlign,
        "meteo_power": TrainerMeteoPower,
        "itransformer": TrainerBase,
        "patchtst": TrainerBase,
        "dlinear": TrainerBase,
        "lstm": TrainerBase,
        "bilstm": TrainerBase,
        "tcn": TrainerBase,
    }
    if model_name not in trainers:
        raise ValueError(f"未知模型: {model_name}")
    return trainers[model_name]


def get_model(model_name, config):
    """
    根据模型名实例化模型。

    Parameters
    ----------
    model_name : str
        模型名称
    config : dict
        模型配置

    Returns
    -------
    nn.Module
    """
    from model.timealign import TimeAlign
    from model.itransformer import ITransformer
    from model.patchtst import PatchTST
    from model.dlinear import DLinear
    from model.lstm import LSTMForecaster
    from model.tcn import TCNForecaster
    from model.meteo_power import MeteoPowerNet

    models = {
        "timealign": TimeAlign,
        "meteo_power": MeteoPowerNet,
        "itransformer": ITransformer,
        "patchtst": PatchTST,
        "dlinear": DLinear,
        "lstm": LSTMForecaster,
        "bilstm": LSTMForecaster,
        "tcn": TCNForecaster,
    }
    if model_name not in models:
        raise ValueError(f"未知模型: {model_name}")
    return models[model_name](config)


def train_and_evaluate(config, silent=False):
    """
    可编程调用的训练接口。

    完成数据加载 → 构建模型 → 训练，
    返回最佳验证 loss 和训练器实例。

    Parameters
    ----------
    config : dict
        模型和训练配置
    silent : bool
        静默模式（超参调优时关闭文件日志和绘图）

    Returns
    -------
    float
        最佳验证 loss
    TrainerBase
        训练器实例（含模型权重和 loss 历史）
    """
    model_name = config["model"]

    # 1. 设置种子和设备
    set_seed(config.get("seed", 42))
    device = get_device(config)

    # 2. 日志
    if silent:
        logger = None
    else:
        log_dir = f"output/log/{model_name}"
        logger = Logger(log_dir, model_name)
        logger.log_config(config)

    # 3. 构建数据
    dataset_type = config.get("dataset_type", "default")
    if dataset_type == "meteo":
        from dataset.meteo_power import get_meteo_dataloader
        get_dl = get_meteo_dataloader
    else:
        from dataset.wind_power import get_dataloader
        get_dl = get_dataloader

    if not silent:
        logger.log_feat("加载数据集...")

    train_loader = get_dl(config, "train")
    valid_loader = get_dl(config, "valid")
    test_loader = get_dl(config, "test")

    if not silent:
        logger.log_feat(
            f"训练集: {len(train_loader.dataset)} 样本"
        )
        logger.log_feat(
            f"验证集: {len(valid_loader.dataset)} 样本"
        )
        logger.log_feat(
            f"测试集: {len(test_loader.dataset)} 样本"
        )

    # 4. 构建模型
    model = get_model(model_name, config)
    n_params = count_parameters(model)

    if not silent:
        logger.log_train(
            f"模型: {model_name}, 参数量: {n_params:,}"
        )
    else:
        print(f"  参数量: {n_params:,}")

    # 5. 构建训练器
    TrainerClass = get_trainer(model_name)
    if silent:
        config = {**config, "_silent": True}
    trainer = TrainerClass(
        model, config, train_loader, valid_loader,
        test_loader, logger, device
    )

    # 6. 训练
    trainer.train()

    # 7. 后处理
    best_valid_loss = min(trainer.valid_losses)

    if not silent:
        fig_path = f"output/figure/{model_name}/loss_curve.png"
        plot_loss_curve(
            trainer.train_losses, trainer.valid_losses,
            fig_path
        )
        logger.log_train(
            f"Loss 曲线保存至: {fig_path}\n\n"
        )
        logger.close()
        print("训练完成。")
    else:
        print(f"  Best valid loss: {best_valid_loss:.6f}")

    return best_valid_loss, trainer


def main():
    """CLI 训练入口。"""
    parser = argparse.ArgumentParser(description="WPF 训练入口")
    parser.add_argument(
        "--config", type=str, required=True,
        help="配置文件路径"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    train_and_evaluate(config, silent=False)


if __name__ == "__main__":
    main()
