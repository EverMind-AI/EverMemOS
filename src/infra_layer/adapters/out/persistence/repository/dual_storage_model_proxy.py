"""
Dual Storage Model Proxy - 拦截 MongoDB 调用层（Lite 版本方案）

通过拦截 self.model 的所有 MongoDB 调用来实现双存储，Repository 代码零改动。

工作原理：
1. 运行时自动提取 Document 的索引字段（无需手动维护 Lite 类）
2. 写入时：
   - MongoDB 只存储 Lite 版本（索引字段）- 用于查询
   - KV-Storage 存储完整数据（加密存储）- 用于数据读取
3. 查询时：
   - MongoDB 查询返回 Lite 数据（包含 ID）
   - 根据 ID 从 KV-Storage 批量加载完整数据
4. 安全性：敏感字段只存在 KV-Storage，不存在 MongoDB

优势：
- Repository 代码完全不需要改动（零改动）
- 索引字段自动提取，第三方修改索引后无需改代码
- 敏感数据只存 KV-Storage（加密），安全性更高
"""

from typing import TYPE_CHECKING, Optional, Any, List, Set
from pymongo.asynchronous.client_session import AsyncClientSession
from pydantic import BaseModel, ConfigDict, Field
from beanie import PydanticObjectId

from core.observation.logger import get_logger
from infra_layer.adapters.out.persistence.repository.lite_model_extractor import (
    LiteModelExtractor,
)

if TYPE_CHECKING:
    from infra_layer.adapters.out.persistence.kv_storage.kv_storage_interface import (
        KVStorageInterface,
    )

logger = get_logger(__name__)


class LiteStorageQueryError(Exception):
    """Exception raised when query uses fields not available in Lite storage"""
    pass


# Minimal projection model for queries - only returns _id
class IdOnlyProjection(BaseModel):
    """Minimal projection to only retrieve document IDs from MongoDB"""
    # MongoDB uses _id, Beanie Documents map it to id
    # For projection models, we need to handle _id directly
    id: Optional[PydanticObjectId] = Field(None, alias="_id")

    model_config = ConfigDict(extra="ignore", populate_by_name=True)


class DualStorageQueryProxy:
    """
    Query Cursor Proxy - 拦截 MongoDB 查询游标操作

    拦截 find() 返回的 Cursor 对象，自动从 KV-Storage 加载完整数据
    MongoDB 只返回 Lite 数据（ID + 索引字段），完整数据从 KV 加载
    """

    def __init__(
        self,
        mongo_cursor,
        kv_storage: "KVStorageInterface",
        full_model_class,
    ):
        """
        Initialize query cursor proxy

        Args:
            mongo_cursor: MongoDB query cursor (from model.find())
            kv_storage: KV-Storage instance
            full_model_class: Full model class (e.g., EpisodicMemory)
        """
        self._mongo_cursor = mongo_cursor
        self._kv_storage = kv_storage
        self._full_model_class = full_model_class

    def sort(self, *args, **kwargs):
        """Proxy sort method"""
        self._mongo_cursor = self._mongo_cursor.sort(*args, **kwargs)
        return self

    def skip(self, *args, **kwargs):
        """Proxy skip method"""
        self._mongo_cursor = self._mongo_cursor.skip(*args, **kwargs)
        return self

    def limit(self, *args, **kwargs):
        """Proxy limit method"""
        self._mongo_cursor = self._mongo_cursor.limit(*args, **kwargs)
        return self

    async def to_list(self, *args, **kwargs) -> List[Any]:
        """
        Execute query and load full data from KV-Storage（Lite 存储模式）

        Lite 存储模式：
        1. 使用 PyMongo 直接查询 MongoDB 获取 Lite 数据（原始 dict，避免 Beanie 验证）
        2. 提取所有 IDs
        3. 从 KV-Storage 批量加载完整数据

        Returns:
            List of full model instances (from KV-Storage)
        """
        try:
            # 1. 使用 Beanie 的 project() 方法只返回 _id 字段
            # 使用 IdOnlyProjection 模型避免完整 Document 验证

            # 添加投影：只返回 _id 字段（使用 Pydantic 模型）
            projected_cursor = self._mongo_cursor.project(IdOnlyProjection)

            # 执行查询获取 IdOnlyProjection 对象列表
            length = kwargs.get("length", None) or (args[0] if args else None)
            id_projections = await projected_cursor.to_list(length=length)

            if not id_projections:
                return []

            # 2. 提取所有 document IDs from projection objects
            try:
                doc_ids = [str(proj.id) for proj in id_projections if proj.id]
                logger.debug(f"📋 Query returned {len(doc_ids)} IDs from MongoDB")
            except Exception as e:
                logger.error(f"❌ Failed to extract IDs: {e}, projections type={type(id_projections)}, first item={id_projections[0] if id_projections else 'empty'}")
                return []

            # 3. 从 KV-Storage 批量加载完整数据
            full_docs = []
            for doc_id in doc_ids:
                try:
                    kv_value = await self._kv_storage.get(key=doc_id)
                    if kv_value:
                        # 从 KV 反序列化完整数据
                        full_doc = self._full_model_class.model_validate_json(kv_value)
                        full_docs.append(full_doc)
                    else:
                        # KV miss - Lite 模式下无法恢复完整数据
                        logger.warning(f"⚠️  KV miss for {doc_id} - cannot return full document")
                        # 跳过此文档（因为无法从 MongoDB Lite 数据构建完整文档）
                except Exception as e:
                    logger.error(f"❌ Failed to load from KV for {doc_id}: {e}")

            logger.debug(f"✅ Loaded {len(full_docs)}/{len(doc_ids)} full documents from KV-Storage")
            return full_docs

        except Exception as e:
            import traceback
            logger.error(f"❌ Failed in to_list: {e}\n{traceback.format_exc()}")
            return []

    async def delete(self, *args, **kwargs):
        """
        Delete documents matching query（Lite 存储模式）

        Lite 模式：使用 project() 获取 IDs，避免 Beanie 验证

        Also deletes from KV-Storage
        """
        try:
            # 1. 使用 project() 获取要删除的文档 IDs（避免 Beanie 验证）
            projected_cursor = self._mongo_cursor.project(IdOnlyProjection)
            id_projections = await projected_cursor.to_list(length=None)
            doc_ids = [str(proj.id) for proj in id_projections if proj.id]

            # 2. 删除 MongoDB
            result = await self._mongo_cursor.delete(*args, **kwargs)

            # 3. 批量删除 KV-Storage
            if doc_ids:
                try:
                    await self._kv_storage.batch_delete(keys=doc_ids)
                    logger.debug(f"✅ Deleted {len(doc_ids)} documents from KV-Storage")
                except Exception as e:
                    logger.warning(f"⚠️  Failed to delete from KV-Storage: {e}")

            return result

        except Exception as e:
            logger.error(f"❌ Failed to delete with dual storage: {e}")
            raise

    async def count(self, *args, **kwargs):
        """Proxy count method to original cursor"""
        return await self._mongo_cursor.count(*args, **kwargs)

    def __getattr__(self, name):
        """Proxy all other methods to original cursor"""
        return getattr(self._mongo_cursor, name)


class DualStorageModelProxy:
    """
    Model Proxy - 拦截 MongoDB Model 层调用（Lite 版本方案）

    替换 Repository 的 self.model，拦截所有 MongoDB 操作：
    - 运行时提取索引字段（自动适配第三方修改）
    - find() -> 返回 QueryProxy（从 KV 加载完整数据）
    - get() -> 优先从 KV 读取完整数据
    - 写入 -> MongoDB 只存 Lite，KV 存完整
    """

    def __init__(
        self,
        original_model,
        kv_storage: "KVStorageInterface",
        full_model_class,
    ):
        """
        Initialize model proxy and extract indexed fields

        Args:
            original_model: Original Beanie Document model class
            kv_storage: KV-Storage instance
            full_model_class: Full model class (same as original_model)
        """
        self._original_model = original_model
        self._kv_storage = kv_storage
        self._full_model_class = full_model_class

        # 运行时自动提取索引字段（无需手动维护 Lite 类）
        self._indexed_fields = LiteModelExtractor.extract_indexed_fields(full_model_class)
        logger.info(
            f"🔍 Auto-extracted {len(self._indexed_fields)} indexed fields for {full_model_class.__name__}"
        )

    def _extract_query_fields(self, filter_dict: Any) -> Set[str]:
        """
        递归提取查询条件中使用的所有字段名

        支持：
        - 简单查询：{"user_id": "123"}
        - 操作符查询：{"timestamp": {"$gt": date}}
        - 逻辑操作符：{"$and": [...], "$or": [...]}
        - 数组操作符：{"keywords": {"$in": [...]}}

        Args:
            filter_dict: MongoDB filter query

        Returns:
            Set[str]: 查询中使用的所有字段名
        """
        fields = set()

        if not isinstance(filter_dict, dict):
            return fields

        for key, value in filter_dict.items():
            # 跳过 MongoDB 操作符（以 $ 开头）
            if key.startswith("$"):
                # 对于 $and, $or 等逻辑操作符，递归提取子条件
                if isinstance(value, list):
                    for sub_condition in value:
                        fields.update(self._extract_query_fields(sub_condition))
                elif isinstance(value, dict):
                    fields.update(self._extract_query_fields(value))
            else:
                # 这是一个实际的字段名
                fields.add(key)

        return fields

    def _validate_query_fields(self, filter_dict: Any) -> None:
        """
        验证查询字段是否在 Lite 数据中

        如果查询使用了非 Lite 字段，抛出清晰的错误提示

        Args:
            filter_dict: MongoDB filter query

        Raises:
            LiteStorageQueryError: 如果查询字段不在 Lite 存储中
        """
        if not filter_dict:
            return

        # 提取所有查询字段
        queried_fields = self._extract_query_fields(filter_dict)

        if not queried_fields:
            return

        # MongoDB 字段别名映射：_id -> id
        # MongoDB 内部使用 _id，但 Beanie 映射为 id
        normalized_queried_fields = set()
        for field in queried_fields:
            if field == "_id":
                # _id 是 id 的别名，总是可用
                normalized_queried_fields.add("id")
            else:
                normalized_queried_fields.add(field)

        # 检查是否有字段不在 indexed_fields 中
        missing_fields = normalized_queried_fields - self._indexed_fields

        if missing_fields:
            # 构建清晰的错误消息
            error_msg = (
                f"❌ Query uses fields not available in Lite storage: {sorted(missing_fields)}\n\n"
                f"These fields are not indexed and not in query_fields.\n"
                f"In Lite storage mode, MongoDB only stores indexed fields and query_fields.\n\n"
                f"To fix this issue, add these fields to Settings.query_fields in {self._full_model_class.__name__}:\n\n"
                f"  class Settings:\n"
                f"      query_fields = {sorted(list(missing_fields))}\n\n"
                f"Current indexed fields: {sorted(self._indexed_fields)}\n"
                f"Queried fields: {sorted(normalized_queried_fields)}\n"
            )
            raise LiteStorageQueryError(error_msg)

    def find(self, *args, **kwargs):
        """
        Intercept find() - 返回 QueryProxy 自动处理双存储

        Returns:
            DualStorageQueryProxy
        """
        # 验证查询字段
        filter_query = args[0] if args else {}
        self._validate_query_fields(filter_query)

        # 调用原始 model 的 find 方法
        mongo_cursor = self._original_model.find(*args, **kwargs)

        # 包装成 QueryProxy
        return DualStorageQueryProxy(
            mongo_cursor=mongo_cursor,
            kv_storage=self._kv_storage,
            full_model_class=self._full_model_class,
        )

    async def get(
        self, doc_id, session: Optional[AsyncClientSession] = None, **kwargs
    ):
        """
        Intercept get() - 优先从 KV-Storage 读取（Lite 存储模式）

        Lite 存储模式下：
        - MongoDB 只存 Lite 数据（索引字段）
        - KV-Storage 存完整数据
        - 必须从 KV 读取，MongoDB 无法提供完整文档

        Args:
            doc_id: Document ID (ObjectId or str)
            session: Optional MongoDB session

        Returns:
            Full document or None
        """
        try:
            # 必须从 KV-Storage 读取完整数据
            doc_id_str = str(doc_id)
            kv_value = await self._kv_storage.get(key=doc_id_str)

            if kv_value:
                # KV hit - 返回完整数据
                document = self._full_model_class.model_validate_json(kv_value)
                logger.debug(f"✅ KV hit: {doc_id_str}")
                return document

            # KV miss - Lite 模式下无法从 MongoDB 恢复完整数据
            # MongoDB 只有索引字段，不满足 required fields
            logger.warning(f"⚠️  KV miss for {doc_id_str} - cannot recover full document from MongoDB Lite data")
            return None

        except Exception as e:
            logger.error(f"❌ Failed to get document: {e}")
            return None

    async def find_one(self, *args, **kwargs):
        """
        Intercept find_one() - Lite 存储模式下使用 PyMongo 直接查询

        Lite 存储模式：
        1. 使用 PyMongo 查询 MongoDB 获取 Lite 数据（避免 Beanie 验证）
        2. 从 KV-Storage 加载完整数据

        Args:
            *args: filter query
            **kwargs: additional options

        Returns:
            Document or None

        Raises:
            LiteStorageQueryError: 如果查询字段不在 Lite 存储中
        """
        # 提取查询条件
        filter_query = args[0] if args else {}

        # 验证查询字段（如果失败会抛出 LiteStorageQueryError）
        self._validate_query_fields(filter_query)

        try:
            # 使用 PyMongo 直接查询（避免 Beanie 验证 Lite 数据）
            mongo_collection = self._original_model.get_pymongo_collection()
            session = kwargs.get("session", None)

            lite_doc = await mongo_collection.find_one(filter_query, session=session)

            if not lite_doc:
                return None

            # 从 KV-Storage 加载完整数据
            doc_id = str(lite_doc["_id"])
            kv_value = await self._kv_storage.get(key=doc_id)

            if kv_value:
                full_doc = self._full_model_class.model_validate_json(kv_value)
                logger.debug(f"✅ find_one loaded from KV: {doc_id}")
                return full_doc
            else:
                # KV miss - 无法恢复完整数据
                logger.warning(f"⚠️  KV miss in find_one for {doc_id}")
                return None

        except LiteStorageQueryError:
            # 重新抛出查询字段验证错误
            raise
        except Exception as e:
            logger.error(f"❌ Failed in find_one: {e}")
            return None

    async def delete_many(self, *args, **kwargs):
        """
        Intercept delete_many() - Lite 存储模式下的批量软删除

        Lite 存储模式下的批量软删除行为：
        - MongoDB：标记deleted_at（批量更新Lite数据）
        - KV-Storage：保留完整数据（不删除）

        原因：MongoDB只有索引字段，如果删除KV，恢复时无法重建完整数据

        Args:
            *args: filter query
            **kwargs: additional options

        Returns:
            Delete result
        """
        try:
            # 验证查询字段
            filter_query = args[0] if args else {}
            self._validate_query_fields(filter_query)

            # 执行批量软删除（只在MongoDB标记deleted_at）
            result = await self._original_model.delete_many(*args, **kwargs)

            # Lite模式：不从KV删除，保留完整数据以便恢复
            logger.debug(f"✅ Batch soft deleted in MongoDB (KV data preserved)")

            return result

        except Exception as e:
            logger.error(f"❌ Failed to delete_many with dual storage: {e}")
            raise

    def hard_find_one(self, *args, **kwargs):
        """
        Intercept hard_find_one() - 查询包括已删除的文档，并回填 KV

        Args:
            *args: filter query
            **kwargs: additional options

        Returns:
            FindOne query object
        """
        # hard_find_one returns a query object, we need to wrap it
        # But since it's a class method returning a query object, we'll just pass through
        # and handle backfill in the wrapper if needed
        return self._original_model.hard_find_one(*args, **kwargs)

    async def hard_delete_many(self, *args, **kwargs):
        """
        Intercept hard_delete_many() - 物理删除并同步 KV-Storage（Lite 存储模式）

        Lite 模式：使用 PyMongo 直接查询获取 IDs，避免 Beanie 验证

        Args:
            *args: filter query
            **kwargs: additional options

        Returns:
            Delete result
        """
        try:
            # 1. 验证查询字段
            filter_query = args[0] if args else {}
            self._validate_query_fields(filter_query)

            # 2. 使用 PyMongo 直接查询要删除的文档 IDs（避免 Beanie 验证）
            mongo_collection = self._original_model.get_pymongo_collection()
            session = kwargs.get("session", None)

            # 只查询 _id 字段（包括软删除的文档）
            lite_docs = await mongo_collection.find(filter_query, {"_id": 1}, session=session).to_list(length=None)
            doc_ids = [str(doc["_id"]) for doc in lite_docs]

            # 2. 执行物理删除
            result = await self._original_model.hard_delete_many(*args, **kwargs)

            # 3. 批量删除 KV-Storage
            if doc_ids:
                try:
                    await self._kv_storage.batch_delete(keys=doc_ids)
                    logger.debug(f"✅ Hard deleted {len(doc_ids)} documents from KV-Storage")
                except Exception as e:
                    logger.warning(f"⚠️  Failed to delete from KV-Storage: {e}")

            return result

        except Exception as e:
            logger.error(f"❌ Failed to hard_delete_many with dual storage: {e}")
            raise

    async def restore_many(self, *args, **kwargs):
        """
        Intercept restore_many() - 恢复已删除文档并同步 KV-Storage（Lite 存储模式）

        注意：restore 不需要更新 KV，因为 KV 中已经有完整数据
        只需要更新 MongoDB 的 deleted_at 字段（Lite 数据）

        Args:
            *args: filter query
            **kwargs: additional options

        Returns:
            Update result
        """
        try:
            # 验证查询字段
            filter_query = args[0] if args else {}
            self._validate_query_fields(filter_query)

            # 执行恢复操作（只更新 MongoDB 的 deleted_at 字段）
            result = await self._original_model.restore_many(*args, **kwargs)

            # Lite 模式下，KV 中已经有完整数据，无需额外同步
            # restore 只修改 MongoDB 的 deleted_at 字段（索引字段）

            logger.debug(f"✅ Restored documents in MongoDB (Lite data)")
            return result

        except Exception as e:
            logger.error(f"❌ Failed to restore_many with dual storage: {e}")
            raise

    def __getattr__(self, name):
        """Proxy all other methods to original model"""
        return getattr(self._original_model, name)


class DocumentInstanceWrapper:
    """
    Document Instance Wrapper - 拦截 Document 实例方法（Lite 版本方案）

    拦截 insert(), save(), delete() 等实例方法：
    - MongoDB 只存 Lite 版本（索引字段）
    - KV-Storage 存完整数据（加密存储）
    """

    @staticmethod
    def wrap_insert(original_insert, kv_storage: "KVStorageInterface", indexed_fields: Set[str]):
        """
        Wrap document.insert() to implement Lite storage

        使用底层 pymongo API 来确保 MongoDB 只存 Lite 数据

        MongoDB: Lite data (indexed fields only)
        KV-Storage: Full data (all fields, encrypted)
        """
        async def wrapped_insert(self, **kwargs):
            # 1. 提取 Lite 数据（只包含索引字段）
            lite_data = LiteModelExtractor.extract_lite_data(self, indexed_fields)

            # 2. 保存完整数据到 KV-Storage（在 insert 之前，避免 ID 问题）
            full_data_for_kv = self.model_dump(mode="python")

            # 3. 使用底层 pymongo API 直接插入 Lite 数据到 MongoDB
            mongo_collection = self.get_pymongo_collection()

            # 获取 session 参数（如果有）
            session = kwargs.get("session", None)

            # 直接插入 Lite 数据
            insert_result = await mongo_collection.insert_one(lite_data, session=session)

            # 4. 将生成的 ID 赋值给 document 对象
            self.id = insert_result.inserted_id

            # 5. 将完整数据存入 KV-Storage
            try:
                kv_key = str(self.id)

                # 更新 full_data 的 ID
                full_data_for_kv["id"] = self.id

                # 序列化完整数据
                full_document = self.__class__.model_validate(full_data_for_kv)
                kv_value = full_document.model_dump_json()

                await kv_storage.put(key=kv_key, value=kv_value)
                logger.debug(f"💾 MongoDB: Lite ({len(lite_data)} fields), KV: Full ({len(full_data_for_kv)} fields) - {kv_key}")
            except Exception as e:
                logger.warning(f"⚠️  Failed to sync full data to KV-Storage: {e}")

            # 6. 返回 document 对象（Beanie 的 insert 返回 self）
            return self

        return wrapped_insert

    @staticmethod
    def wrap_save(original_save, kv_storage: "KVStorageInterface", indexed_fields: Set[str]):
        """
        Wrap document.save() to implement Lite storage

        使用底层 pymongo API 来确保 MongoDB 只存 Lite 数据

        MongoDB: Lite data (indexed fields only)
        KV-Storage: Full data (all fields, encrypted)
        """
        async def wrapped_save(self, **kwargs):
            if not self.id:
                # 如果没有 ID，应该使用 insert 而不是 save
                logger.warning("save() called on document without ID, should use insert()")
                return await self.insert(**kwargs)

            # 1. 提取 Lite 数据
            lite_data = LiteModelExtractor.extract_lite_data(self, indexed_fields)

            # 2. 保存完整数据
            full_data = self.model_dump(mode="python")

            # 3. 使用底层 pymongo API 更新 MongoDB（只更新 Lite 字段）
            mongo_collection = self.get_pymongo_collection()
            session = kwargs.get("session", None)

            # 使用 replace_one 替换整个文档为 Lite 数据
            from bson import ObjectId
            await mongo_collection.replace_one(
                {"_id": ObjectId(self.id)},
                lite_data,
                session=session
            )

            # 4. 将完整数据存入 KV-Storage
            try:
                kv_key = str(self.id)
                kv_value = self.model_dump_json()
                await kv_storage.put(key=kv_key, value=kv_value)
                logger.debug(f"💾 MongoDB: Lite ({len(lite_data)} fields), KV: Full ({len(full_data)} fields) - {kv_key}")
            except Exception as e:
                logger.warning(f"⚠️  Failed to sync full data to KV-Storage: {e}")

            # 5. 返回 document 对象
            return self

        return wrapped_save

    @staticmethod
    def wrap_delete(original_delete, kv_storage: "KVStorageInterface"):
        """
        Wrap document.delete() - Lite 存储模式下的软删除

        Lite 模式下的软删除行为：
        - MongoDB：标记 deleted_at（只更新 Lite 数据）
        - KV-Storage：保留完整数据（不删除）

        原因：MongoDB 只有索引字段，如果删除 KV，恢复时无法重建完整数据
        """
        async def wrapped_delete(self, **kwargs):
            # 调用原始 delete（只在 MongoDB 标记 deleted_at）
            result = await original_delete(self, **kwargs)

            # Lite 模式下不从 KV 删除，保留完整数据以便恢复
            # KV中的数据仍然存在，只是MongoDB标记为已删除
            logger.debug(f"✅ Soft deleted in MongoDB (KV data preserved): {self.id}")

            return result

        return wrapped_delete

    @staticmethod
    def wrap_restore(original_restore, kv_storage: "KVStorageInterface"):
        """
        Wrap document.restore() - Lite 存储模式下的恢复

        Lite 模式下的恢复行为：
        - MongoDB：清除 deleted_at（只更新 Lite 数据）
        - KV-Storage：无需操作（数据一直都在）

        原因：软删除时KV数据未被删除，所以恢复时无需同步
        """
        async def wrapped_restore(self, **kwargs):
            # 调用原始 restore（只在 MongoDB 清除 deleted_at）
            result = await original_restore(self, **kwargs)

            # Lite 模式下 KV 数据未被删除，无需同步
            # KV中的完整数据一直存在，可以直接使用
            logger.debug(f"✅ Restored in MongoDB (KV data was preserved): {self.id}")

            return result

        return wrapped_restore

    @staticmethod
    def __original_wrap_restore_not_used(original_restore, kv_storage: "KVStorageInterface"):
        """DEPRECATED: 原始restore实现（已弃用）"""
        async def wrapped_restore(self, **kwargs):
            # 调用原始 restore (传递 self)
            result = await original_restore(self, **kwargs)

            # 恢复后同步回 KV-Storage
            if self.id:
                try:
                    kv_key = str(self.id)
                    kv_value = self.model_dump_json()
                    await kv_storage.put(key=kv_key, value=kv_value)
                    logger.debug(f"✅ Synced to KV-Storage after restore: {kv_key}")
                except Exception as e:
                    logger.warning(f"⚠️  Failed to sync to KV-Storage after restore: {e}")

            return result

        return wrapped_restore

    @staticmethod
    def wrap_hard_delete(original_hard_delete, kv_storage: "KVStorageInterface"):
        """Wrap document.hard_delete() to remove from KV-Storage"""
        async def wrapped_hard_delete(self, **kwargs):
            doc_id = str(self.id) if self.id else None

            # 调用原始 hard_delete (传递 self)
            result = await original_hard_delete(self, **kwargs)

            # 从 KV-Storage 删除
            if doc_id:
                try:
                    await kv_storage.delete(key=doc_id)
                    logger.debug(f"✅ Deleted from KV-Storage after hard_delete: {doc_id}")
                except Exception as e:
                    logger.warning(f"⚠️  Failed to delete from KV-Storage after hard_delete: {e}")

            return result

        return wrapped_hard_delete


__all__ = [
    "DualStorageModelProxy",
    "DualStorageQueryProxy",
    "DocumentInstanceWrapper",
    "LiteStorageQueryError",
]
