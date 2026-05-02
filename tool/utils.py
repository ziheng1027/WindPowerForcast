"""
通用工具函数：随机种子、配置加载、设备选择、参数统计。
"""

import random

import numpy as np
import torch
import yaml


def set_seed(seed=42):
    """设置全局随机种子，确保可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path):
    """
    加载 YAML 配置文件。

    Parameters
    ----------
    config_path : str
        配置文件路径

    Returns
    -------
    dict
        配置字典
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def get_device(config):
    """
    根据配置返回计算设备。

    Parameters
    ----------
    config : dict
        包含 device 字段的配置

    Returns
    -------
    torch.device
    """
    device_str = config.get("device", "cpu")
    if device_str == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def count_parameters(model):
    """
    统计模型可训练参数量。

    Parameters
    ----------
    model : nn.Module
        PyTorch 模型

    Returns
    -------
    int
        可训练参数总数
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
