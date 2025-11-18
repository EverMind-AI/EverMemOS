from typing import List


class ScannerPathsRegistry:
    """扫描路径注册器"""

    def __init__(self):
        """初始化扫描路径注册器"""
        self.scan_paths: List[str] = []

    def add_scan_path(self, path: str) -> 'ScannerPathsRegistry':
        """添加扫描路径"""
        self.scan_paths.append(path)
        return self

    def get_scan_paths(self) -> List[str]:
        """获取扫描路径"""
        return self.scan_paths

    def clear(self) -> 'ScannerPathsRegistry':
        """清空扫描路径"""
        self.scan_paths = []
        return self
