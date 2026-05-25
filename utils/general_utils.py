import torch
import random
import numpy as np
import os,logging,sys

def setup_logging(log_dir):
    """
    设置日志记录，将日志输出到控制台和文件。
    """
    # 1. 获取根记录器
    logger = logging.getLogger()
    logger.setLevel(logging.INFO) # 设置全局级别为 INFO

    # 2. 清除任何现有的处理器，防止重复记录
    if logger.hasHandlers():
        logger.handlers.clear()

    # 3. 创建格式化器
    # 格式：2025-10-27 17:30:01 - 您的日志消息
    formatter = logging.Formatter(
        '%(asctime)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 4. 创建文件处理器 (写入 run.log)
    log_file = os.path.join(log_dir, 'run.log')
    file_handler = logging.FileHandler(log_file, mode='w') # 'w' 模式会覆盖旧日志
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    # 5. 创建控制台处理器 (输出到 stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # 6. 将处理器添加到根记录器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

def set_seed(seed):
    """
    Sets the random seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Ensure that computations are deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(state, is_best, checkpoint_dir='checkpoints', filename='checkpoint.pth.tar'):
    """
    Saves a model checkpoint.
    Args:
        state (dict): Contains model's state_dict, optimizer's state_dict, epoch, etc.
        is_best (bool): If true, saves a copy as 'model_best.pth.tar'.
        checkpoint_dir (str): Directory to save checkpoints.
        filename (str): The name of the checkpoint file.
    """
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)

    if is_best:
        best_filepath = os.path.join(checkpoint_dir, 'model_best.pth.tar')
        torch.save(state, best_filepath)
        print(f"==> Saved new best model to {best_filepath}")