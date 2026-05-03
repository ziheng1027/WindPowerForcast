"""
全流程训练日志，支持四阶段标记：[DATA]/[FEAT]/[TRAIN]/[EVAL]。
"""

import os
import datetime


# ANSI 颜色码
_COLORS = {
    "DATA": "\033[94m",    # 蓝色
    "FEAT": "\033[93m",    # 黄色
    "TRAIN": "\033[92m",   # 绿色
    "EVAL": "\033[96m",    # 青色
    "RESET": "\033[0m",
}


class Logger:
    """
    全流程日志记录器。

    四阶段标记: [DATA] 数据处理 / [FEAT] 特征构建
                [TRAIN] 模型训练 / [EVAL] 测试评估
    """

    def __init__(self, log_dir, model_name):
        """
        初始化日志器，创建日志目录和文件。

        Parameters
        ----------
        log_dir : str
            日志保存目录
        model_name : str
            模型名称，用于日志文件命名
        """
        self.model_name = model_name
        os.makedirs(log_dir, exist_ok=True)
        date_str = datetime.datetime.now().strftime("%Y-%m-%d")
        log_path = os.path.join(
            log_dir, f"train_{date_str}.log"
        )
        self._file = open(log_path, "a", encoding="utf-8")

        # 写入分隔线标记新会话
        self._write_raw("=" * 60)
        self._write_raw(
            f"新会话: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        self._write_raw("=" * 60)

    def _write_raw(self, msg):
        """写入原始消息到文件和控制台。"""
        self._file.write(msg + "\n")
        self._file.flush()
        print(msg)

    def _log(self, tag, msg):
        """
        带标签的日志记录。

        Parameters
        ----------
        tag : str
            阶段标记 (DATA/FEAT/TRAIN/EVAL)
        msg : str
            日志内容
        """
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        line = f"[{tag}] {timestamp} | {msg}"

        # 文件写入（无颜色）
        self._file.write(line + "\n")
        self._file.flush()

        # 控制台（带颜色）
        color = _COLORS.get(tag, "")
        reset = _COLORS["RESET"]
        print(f"{color}{line}{reset}")

    def log_data(self, msg):
        """记录数据处理阶段信息。"""
        self._log("DATA", msg)

    def log_feat(self, msg):
        """记录特征构建阶段信息。"""
        self._log("FEAT", msg)

    def log_train(self, msg):
        """记录训练阶段信息。"""
        self._log("TRAIN", msg)

    def log_eval(self, msg):
        """记录评估阶段信息。"""
        self._log("EVAL", msg)

    def log_config(self, config):
        """
        记录完整配置信息。

        Parameters
        ----------
        config : dict
            配置字典
        """
        self._write_raw("\n--- 配置信息 ---")
        for key, value in sorted(config.items()):
            self._write_raw(f"  {key}: {value}")
        self._write_raw("--- 配置结束 ---\n")

    def close(self):
        """关闭日志文件。"""
        if self._file and not self._file.closed:
            self._file.close()
