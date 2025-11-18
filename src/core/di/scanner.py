# -*- coding: utf-8 -*-
"""
组件扫描器
"""

import os
import sys
import importlib
from pathlib import Path
from typing import List, Set, Optional, Dict, Any
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

from core.observation.logger import get_logger
from core.di.scan_context import ScanContextRegistry, scan_context


class ComponentScanner:
    """组件扫描器"""

    def __init__(self):
        self.scan_paths: List[str] = []
        self.scan_packages: List[str] = []
        # 使用'di'会导致audit类似的目录会被过滤，因此需要字段全量匹配
        self.exclude_paths: Set[str] = {
            '/di/',
            '/config/',
            '__pycache__',
            '.git',
            '.pytest_cache',
        }
        self.exclude_patterns: Set[str] = {'test_', '_test', 'tests'}
        self.include_patterns: Set[str] = set()
        self.recursive = True
        # self.parallel = True if os.getenv("ENV") == 'dev' else False
        self.parallel = False
        self.max_workers = 8

        # 创建专门的日志记录器
        self.logger = get_logger(__name__)

        # 扫描上下文注册器
        self.context_registry = ScanContextRegistry()

        # 需要预加载的关键模块，避免并行导入时的循环依赖
        self.preload_modules = [
            # SQLAlchemy 核心模块
            'sqlalchemy.engine',
            'sqlalchemy.engine.base',
            'sqlalchemy.engine.default',
            'sqlalchemy.pool',
            'sqlalchemy.sql',
            'sqlalchemy.sql.schema',
            'sqlalchemy.sql.sqltypes',
            'sqlalchemy.orm',
            'sqlalchemy.orm.session',
            'sqlalchemy.orm.query',
            # Pydantic 核心模块
            'pydantic',
            'pydantic.fields',
            'pydantic.main',
            'pydantic.validators',
            'pydantic.v1',
            'pydantic.v1.fields',
            'pydantic.v1.main',
            # 其他可能引起循环依赖的模块
            'typing_extensions',
            'dataclasses',
        ]

    def add_scan_path(self, path: str) -> 'ComponentScanner':
        """添加扫描路径"""
        self.scan_paths.append(path)
        return self

    def add_scan_package(self, package: str) -> 'ComponentScanner':
        """添加扫描包"""
        self.scan_packages.append(package)
        return self

    def exclude_path(self, path: str) -> 'ComponentScanner':
        """排除路径"""
        self.exclude_paths.add(path)
        return self

    def exclude_pattern(self, pattern: str) -> 'ComponentScanner':
        """排除模式"""
        self.exclude_patterns.add(pattern)
        return self

    def include_pattern(self, pattern: str) -> 'ComponentScanner':
        """包含模式"""
        self.include_patterns.add(pattern)
        return self

    def set_recursive(self, recursive: bool) -> 'ComponentScanner':
        """设置是否递归扫描"""
        self.recursive = recursive
        return self

    def set_parallel(self, parallel: bool) -> 'ComponentScanner':
        """设置是否并行扫描"""
        self.parallel = parallel
        return self

    def set_max_workers(self, max_workers: int) -> 'ComponentScanner':
        """设置最大工作线程数"""
        self.max_workers = max_workers
        return self

    def _preload_critical_modules(self):
        """
        预加载关键模块，避免并行导入时的循环依赖问题。
        在并行扫描之前调用此方法。
        """
        self.logger.info("🔄 预加载关键模块以避免循环依赖...")

        loaded_count = 0
        failed_count = 0

        for module_name in self.preload_modules:
            try:
                importlib.import_module(module_name)
                loaded_count += 1
            except ImportError:
                # 某些模块可能不存在，这是正常的
                failed_count += 1
            except Exception:
                # 其他异常需要记录但不阻止流程
                failed_count += 1

        self.logger.info(
            "📦 预加载完成: %d/%d 个模块成功加载",
            loaded_count,
            len(self.preload_modules),
        )
        if failed_count > 0:
            self.logger.debug("跳过了 %d 个不可用的模块", failed_count)

    def add_preload_module(self, module_name: str) -> 'ComponentScanner':
        """添加需要预加载的模块"""
        if module_name not in self.preload_modules:
            self.preload_modules.append(module_name)
        return self

    def register_scan_context(
        self, path: str, metadata: Dict[str, Any]
    ) -> 'ComponentScanner':
        """
        注册扫描路径的上下文元数据

        Args:
            path: 扫描路径
            metadata: 上下文元数据，可以包含任意自定义信息

        Returns:
            self，支持链式调用

        Example:
            ```python
            scanner = ComponentScanner()
            scanner.register_scan_context(
                "src/plugins",
                {"plugin_type": "core", "load_priority": 1}
            )
            scanner.add_scan_path("src/plugins").scan()
            ```
        """
        self.context_registry.register(path, metadata)
        return self

    def get_context_registry(self) -> ScanContextRegistry:
        """
        获取上下文注册器

        Returns:
            上下文注册器实例
        """
        return self.context_registry

    def scan(self) -> 'ComponentScanner':
        """执行扫描"""
        self.logger.info("🔍 开始组件扫描...")

        # 收集所有Python文件
        python_files = self._collect_python_files()
        self.logger.info("📄 发现 %d 个Python文件", len(python_files))

        if not python_files:
            self.logger.warning("⚠️  未发现任何Python文件")
            return self

        # 扫描组件
        if self.parallel and len(python_files) > 1:
            self.logger.info(
                "⚡ 使用并行扫描模式 (最大 %d 个工作线程)", self.max_workers
            )
            # 并行扫描前预加载关键模块
            self._preload_critical_modules()
            self._parallel_scan(python_files)
        else:
            self.logger.info("📝 使用顺序扫描模式")
            self._sequential_scan(python_files)

        self.logger.info("✅ 组件扫描完成")
        return self

    def _collect_python_files(self) -> List[Path]:
        """收集所有Python文件"""
        python_files = []

        # 扫描路径
        if self.scan_paths:
            self.logger.debug("扫描路径: %s", ', '.join(self.scan_paths))
        for scan_path in self.scan_paths:
            files_from_path = self._collect_files_from_path(scan_path)
            python_files.extend(files_from_path)

        # 扫描包
        if self.scan_packages:
            self.logger.debug("扫描包: %s", ', '.join(self.scan_packages))
        for package in self.scan_packages:
            files_from_package = self._collect_files_from_package(package)
            python_files.extend(files_from_package)

        # 去重
        unique_files = list(set(python_files))
        if len(python_files) != len(unique_files):
            self.logger.debug(
                "去重后文件数量: %d -> %d", len(python_files), len(unique_files)
            )

        return unique_files

    def _collect_files_from_path(self, path: str) -> List[Path]:
        """从路径收集Python文件"""
        files = []
        path_obj = Path(path)

        if not path_obj.exists():
            self.logger.warning("扫描路径不存在: %s", path)
            return files

        if path_obj.is_file() and path_obj.suffix == '.py':
            if self._should_include_file(path_obj):
                files.append(path_obj)
        elif path_obj.is_dir():
            pattern = "**/*.py" if self.recursive else "*.py"
            for file_path in path_obj.glob(pattern):
                if self._should_include_file(file_path):
                    files.append(file_path)

        return files

    def _collect_files_from_package(self, package_name: str) -> List[Path]:
        """从包收集Python文件"""
        try:
            package = importlib.import_module(package_name)
            if hasattr(package, '__file__') and package.__file__:
                package_path = Path(package.__file__).parent
                return self._collect_files_from_path(str(package_path))
        except ImportError as e:
            self.logger.warning("无法导入包 %s: %s", package_name, e)

        return []

    def _should_include_file(self, file_path: Path) -> bool:
        """检查是否应该包含文件"""
        # 排除特殊文件
        if file_path.name.startswith('__') and file_path.name.endswith('__.py'):
            return False

        # 检查排除路径
        for exclude_path in self.exclude_paths:
            if exclude_path in str(file_path):
                return False

        # 检查排除模式
        for pattern in self.exclude_patterns:
            if pattern in file_path.name:
                return False

        # 检查包含模式
        if self.include_patterns:
            return any(pattern in file_path.name for pattern in self.include_patterns)

        return True

    def _parallel_scan(self, python_files: List[Path]):
        """并行扫描"""
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._scan_file, file_path): file_path
                for file_path in python_files
            }

            for future in as_completed(futures):
                file_path = futures[future]
                try:
                    future.result()
                except Exception as e:
                    self.logger.error("并行扫描文件失败 %s: %s", file_path, e)

    def _sequential_scan(self, python_files: List[Path]):
        """顺序扫描"""
        for file_path in python_files:
            try:
                self._scan_file(file_path)
            except Exception as e:
                self.logger.error("顺序扫描文件失败 %s: %s", file_path, e)

    def _scan_file(self, file_path: Path):
        """
        扫描单个文件。
        通过导入模块，可以触发模块中定义的组件装饰器，从而完成自动注册。
        """
        module_name = self._file_to_module_name(file_path)
        if not module_name:
            return

        try:
            # 获取该文件路径对应的上下文元数据
            metadata = self.context_registry.get_metadata_for_path(file_path)

            # 在扫描上下文中导入模块
            # 被导入的模块可以通过 get_current_scan_context() 获取上下文信息
            with scan_context(file_path, module_name, metadata):
                # 导入模块以触发装饰器
                importlib.import_module(module_name)

        except ImportError as e:
            self.logger.error("导入模块失败 %s: %s", module_name, e)
            traceback.print_exc()
            sys.exit(1)
        except Exception as e:
            self.logger.error("扫描文件时出现未知错误 %s: %s", file_path, e)
            traceback.print_exc()
            sys.exit(1)

    def _file_to_module_name(self, file_path: Path) -> Optional[str]:
        """将文件路径转换为模块名"""
        try:
            # 获取相对于sys.path的路径，并按照路径深度倒排
            # 解决src.a.b.c和a.b.c的导入问题
            sorted_sys_paths = sorted(
                [(path, len(Path(path).resolve().parts)) for path in sys.path],
                key=lambda x: x[1],
                reverse=True,  # 路径越深，排在越前面
            )

            # 遍历排序后的路径
            for sys_path, _ in sorted_sys_paths:
                sys_path_obj = Path(sys_path).resolve()
                try:
                    relative_path = file_path.resolve().relative_to(sys_path_obj)
                    module_parts = list(relative_path.with_suffix("").parts)
                    return ".".join(module_parts)
                except ValueError:
                    continue
        except Exception:
            pass

        return None
