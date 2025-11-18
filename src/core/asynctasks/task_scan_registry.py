from typing import List


class TaskScanDirectoriesRegistry:
    """扫描目录注册器"""

    def __init__(self):
        """初始化扫描目录注册器"""
        self.scan_directories: List[str] = []

    def add_scan_path(self, path: str) -> 'TaskScanDirectoriesRegistry':
        """添加扫描目录"""
        self.scan_directories.append(path)
        return self

    def get_scan_directories(self) -> List[str]:
        """获取扫描目录"""
        return self.scan_directories

    def clear(self) -> 'TaskScanDirectoriesRegistry':
        """清空扫描目录"""
        self.scan_directories = []
        return self
