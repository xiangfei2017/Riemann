import os
from .data import *

def get_project_root():
    """获取项目根目录"""
    # 获取当前文件所在目录（utils目录）
    utils_dir = os.path.dirname(os.path.abspath(__file__))
    # 向上导航到项目根目录
    # utils -> riemann -> src -> project_root
    return os.path.abspath(os.path.join(utils_dir, '..', '..', '..'))