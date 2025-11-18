from pathlib import Path

# 获取load_env.py所在的目录（utils目录）
# 这个不要加载任何其他模块，PROJECT_DIR是非常基础的信息
utils_dir = Path(__file__).parent
# src目录是utils的父目录
src_dir = utils_dir.parent
CURRENT_DIR = src_dir
PROJECT_DIR = src_dir.parent


def get_base_scan_path():
    """获取基础扫描路径"""
    return CURRENT_DIR
