"""
Dual Storage Mixin - Model 层拦截方案

最小侵入方案：通过拦截 self.model 的 MongoDB 调用实现双存储
- Repository 代码零改动（连 append/get_by_id 等方法都不需要改）
- 主分支更新 CRUD 时，无需同步更新
- 双存储完全透明

工作原理：
1. 在 __init__ 中替换 self.model 为 DualStorageModelProxy
2. Proxy 拦截所有 MongoDB 调用（find, get 等）
3. Monkey patch document 类的实例方法（insert, save, delete）
4. 自动处理双存储同步

使用示例：
    class EpisodicMemoryRawRepository(
        DualStorageMixin,  # 只需添加 Mixin
        BaseRepository[EpisodicMemory]
    ):
        # 所有代码完全不变
        pass
"""

from typing import TypeVar, Generic, Type, Optional

from core.observation.logger import get_logger
from infra_layer.adapters.out.persistence.kv_storage.kv_storage_interface import (
    KVStorageInterface,
)
from infra_layer.adapters.out.persistence.repository.dual_storage_model_proxy import (
    DualStorageModelProxy,
    DocumentInstanceWrapper,
)

logger = get_logger(__name__)

TDocument = TypeVar("TDocument")


class DualStorageMixin(Generic[TDocument]):
    """
    Dual Storage Mixin - Model 层拦截实现

    通过拦截 self.model 来自动实现双存储，Repository 代码零改动。

    工作流程：
    1. __init__ 时替换 self.model 为 ModelProxy
    2. Proxy 拦截 find(), get() 等方法
    3. Monkey patch Document 类的 insert(), save(), delete()
    4. 所有 MongoDB 操作自动同步 KV-Storage

    优势：
    - Repository 所有代码完全不需要改动
    - 主分支更新时无需同步更新
    - 双存储逻辑完全透明
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize mixin and setup dual storage interception

        自动：
        1. 获取 KV-Storage 实例
        2. 替换 self.model 为 ModelProxy
        3. Monkey patch Document 实例方法
        """
        super().__init__(*args, **kwargs)

        # 立即初始化双存储（不延迟）
        self._kv_storage: Optional[KVStorageInterface] = None
        self._setup_dual_storage()

    def _setup_dual_storage(self):
        """
        Setup dual storage interception immediately

        在 __init__ 时立即设置拦截
        """
        try:
            # 1. 获取 KV-Storage 实例
            self._kv_storage = self._get_kv_storage()

            # 2. 替换 self.model 为 ModelProxy
            original_model = self.model
            self.model = DualStorageModelProxy(
                original_model=original_model,
                kv_storage=self._kv_storage,
                full_model_class=original_model,  # 当前设计：Full = Lite
            )

            # 3. Monkey patch Document 类的实例方法
            self._patch_document_methods(original_model)

            logger.debug(
                f"✅ Dual storage initialized for {original_model.__name__}"
            )

        except Exception as e:
            logger.error(f"❌ Failed to initialize dual storage: {e}")
            raise

    def _get_kv_storage(self) -> KVStorageInterface:
        """Lazy load KV-Storage instance from DI container"""
        if self._kv_storage is None:
            from core.di import get_bean_by_type

            self._kv_storage = get_bean_by_type(KVStorageInterface)
        return self._kv_storage

    def _patch_document_methods(self, document_class):
        """
        Monkey patch Document 类的实例方法

        Wrap insert(), save(), delete() 以自动同步 KV-Storage

        Args:
            document_class: Document model class (e.g., EpisodicMemory)
        """
        kv_storage = self._kv_storage

        # 保存原始方法
        if not hasattr(document_class, "_original_insert"):
            document_class._original_insert = document_class.insert
            document_class._original_save = document_class.save
            document_class._original_delete = document_class.delete

            # Wrap 实例方法
            document_class.insert = DocumentInstanceWrapper.wrap_insert(
                document_class._original_insert, kv_storage
            )
            document_class.save = DocumentInstanceWrapper.wrap_save(
                document_class._original_save, kv_storage
            )
            document_class.delete = DocumentInstanceWrapper.wrap_delete(
                document_class._original_delete, kv_storage
            )

            logger.debug(
                f"✅ Patched instance methods for {document_class.__name__}"
            )


__all__ = ["DualStorageMixin"]
