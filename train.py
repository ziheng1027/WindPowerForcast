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
    from trainer.trainer_base import TrainerBase

    trainers = {
        "timealign": TrainerTimeAlign,
        "itransformer": TrainerBase,
        "patchtst": TrainerBase,
        "dlinear": TrainerBase,
        "lstm": TrainerBase,
        "bilstm": TrainerBase,
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

    models = {
        "timealign": TimeAlign,
        "itransformer": ITransformer,
        "patchtst": PatchTST,
        "dlinear": DLinear,
        "lstm": LSTMForecaster,
        "bilstm": LSTMForecaster,
    }
    if model_name not in models:
        raise ValueError(f"未知模型: {model_name}")
    return models[model_name](config)


def main():
    """主训练流程。"""
    parser = argparse.ArgumentParser(description="WPF 训练入口")
    parser.add_argument(
        "--config", type=str, required=True, help="配置文件路径"
    )
    args = parser.parse_args()

    # 1. 加载配置
    config = load_config(args.config)
    model_name = config["model"]

    # 2. 设置种子和设备
    set_seed(config.get("seed", 42))
    device = get_device(config)

    # 3. 初始化日志
    log_dir = f"output/log/{model_name}"
    logger = Logger(log_dir, model_name)
    logger.log_config(config)

    # 4. 构建数据
    from dataset.wind_power import get_dataloader

    logger.log_feat("加载数据集...")
    train_loader = get_dataloader(config, "train")
    valid_loader = get_dataloader(config, "valid")
    test_loader = get_dataloader(config, "test")

    logger.log_feat(f"训练集: {len(train_loader.dataset)} 样本")
    logger.log_feat(f"验证集: {len(valid_loader.dataset)} 样本")
    logger.log_feat(f"测试集: {len(test_loader.dataset)} 样本")

    # 5. 构建模型
    model = get_model(model_name, config)
    n_params = count_parameters(model)
    logger.log_train(f"模型: {model_name}, 参数量: {n_params:,}")

    # 6. 构建训练器
    TrainerClass = get_trainer(model_name)
    trainer = TrainerClass(
        model, config, train_loader, valid_loader,
        test_loader, logger, device
    )

    # 7. 训练
    trainer.train()

    # 8. 绘制 loss 曲线
    fig_path = f"output/figure/{model_name}/loss_curve.png"
    plot_loss_curve(trainer.train_losses, trainer.valid_losses,
                    fig_path)
    logger.log_train(f"Loss 曲线保存至: {fig_path}")

    # 9. 关闭日志
    logger.close()
    print("训练完成。")


if __name__ == "__main__":
    main()
