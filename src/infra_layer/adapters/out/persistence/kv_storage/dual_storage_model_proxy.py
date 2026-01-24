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
from infra_layer.adapters.out.persistence.kv_storage.lite_model_extractor import (
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


class FindOneQueryProxy:
    """
    FindOne Query Proxy - 支持 find_one().delete() 链式调用和直接 await

    包装 DualStorageModelProxy 的 find_one 逻辑，支持：
    1. 直接 await：await find_one(...) -> Document
    2. 链式 delete：await find_one(...).delete() -> DeleteResult

    确保删除操作能触发 DualStorageMixin 的 KV 同步
    """

    def __init__(
        self,
        original_model,
        kv_storage: "KVStorageInterface",
        full_model_class,
        indexed_fields: Set[str],
        filter_args,
        filter_kwargs,
    ):
        """
        Initialize find_one query proxy

        Args:
            original_model: Original Beanie model class
            kv_storage: KV-Storage instance
            full_model_class: Full model class
            indexed_fields: Set of indexed field names
            filter_args: Positional arguments from find_one(*args)
            filter_kwargs: Keyword arguments from find_one(**kwargs)
        """
        self._original_model = original_model
        self._kv_storage = kv_storage
        self._full_model_class = full_model_class
        self._indexed_fields = indexed_fields
        self._filter_args = filter_args
        self._filter_kwargs = filter_kwargs

    def __await__(self):
        """
        Support direct await: doc = await find_one(...)

        Returns full document from KV-Storage
        """
        return self._execute_find_one().__await__()

    async def _execute_find_one(self):
        """
        Execute find_one query and return document from KV-Storage

        This is the core logic that both __await__ and delete() use
        """
        try:
            # 检测是否使用字典语法
            is_dict_syntax = self._filter_args and isinstance(self._filter_args[0], dict)

            if is_dict_syntax:
                # 字典语法：验证查询字段并使用 PyMongo
                filter_query = self._filter_args[0]
                self._validate_query_fields(filter_query)

                mongo_collection = self._original_model.get_pymongo_collection()
                session = self._filter_kwargs.get("session", None)
                lite_doc = await mongo_collection.find_one(filter_query, session=session)
            else:
                # Beanie 操作符语法：使用 Beanie 的原生 find_one
                lite_doc = await self._original_model.find_one(
                    *self._filter_args,
                    projection_model=IdOnlyProjection,
                    **self._filter_kwargs
                )

                # 转换 IdOnlyProjection 对象为字典格式
                if lite_doc:
                    lite_doc = {"_id": lite_doc.id}

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

    def _extract_query_fields(self, filter_dict: Any) -> Set[str]:
        """递归提取查询条件中使用的所有字段名（与 DualStorageModelProxy 相同的逻辑）"""
        fields = set()
        if not isinstance(filter_dict, dict):
            return fields

        for key, value in filter_dict.items():
            if key.startswith("$"):
                if isinstance(value, list):
                    for sub_condition in value:
                        fields.update(self._extract_query_fields(sub_condition))
                elif isinstance(value, dict):
                    fields.update(self._extract_query_fields(value))
            else:
                fields.add(key)

        return fields

    def _validate_query_fields(self, filter_dict: Any) -> None:
        """验证查询字段是否在 Lite 数据中（与 DualStorageModelProxy 相同的逻辑）"""
        if not filter_dict:
            return

        queried_fields = self._extract_query_fields(filter_dict)
        if not queried_fields:
            return

        # MongoDB 字段别名映射：_id -> id
        normalized_queried_fields = set()
        for field in queried_fields:
            if field == "_id":
                normalized_queried_fields.add("id")
            else:
                normalized_queried_fields.add(field)

        # 检查是否有字段不在 indexed_fields 中
        missing_fields = normalized_queried_fields - self._indexed_fields

        if missing_fields:
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

    async def delete(self, *args, **kwargs):
        """
        Execute find_one and delete the result

        Supports chaining: await find_one(...).delete()

        Returns:
            Delete result with deleted_count
        """
        try:
            # 1. Execute find_one to get document ID only
            is_dict_syntax = self._filter_args and isinstance(self._filter_args[0], dict)

            if is_dict_syntax:
                # 字典语法
                filter_query = self._filter_args[0]
                self._validate_query_fields(filter_query)

                mongo_collection = self._original_model.get_pymongo_collection()
                session = self._filter_kwargs.get("session", None)
                lite_doc = await mongo_collection.find_one(filter_query, {"_id": 1}, session=session)
            else:
                # Beanie 操作符语法
                lite_doc = await self._original_model.find_one(
                    *self._filter_args,
                    projection_model=IdOnlyProjection,
                    **self._filter_kwargs
                )
                if lite_doc:
                    lite_doc = {"_id": lite_doc.id}

            if not lite_doc:
                # No document found
                class DeleteResult:
                    deleted_count = 0
                return DeleteResult()

            doc_id = str(lite_doc["_id"])

            # 2. Delete from MongoDB
            if is_dict_syntax:
                from bson import ObjectId
                delete_result = await mongo_collection.delete_one(
                    {"_id": ObjectId(doc_id)},
                    session=self._filter_kwargs.get("session", None)
                )
            else:
                # Use Beanie's find_one().delete()
                delete_query = self._original_model.find_one(*self._filter_args, **self._filter_kwargs)
                delete_result = await delete_query.delete(*args, **kwargs)

            # 3. Delete from KV-Storage
            if delete_result and hasattr(delete_result, 'deleted_count') and delete_result.deleted_count > 0:
                try:
                    await self._kv_storage.delete(key=doc_id)
                    logger.debug(f"✅ Deleted document {doc_id} from KV-Storage via find_one().delete()")
                except Exception as e:
                    logger.warning(f"⚠️  Failed to delete from KV-Storage: {e}")

            return delete_result

        except Exception as e:
            logger.error(f"❌ Failed to delete via find_one: {e}")
            raise


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

    async def update_many(self, update_data: dict, **kwargs):
        """
        Update documents matching query - Cursor上的批量更新并同步 KV-Storage

        这是在 find() 返回的Cursor上调用的 update_many()，不同于 model.update_many()
        例如：await model.find({"user_id": "123"}).update_many({"$set": {"field": "value"}})

        Args:
            update_data: Update operations (e.g., {"$set": {"field": value}})
            **kwargs: Additional options (e.g., session)

        Returns:
            Update result with modified_count
        """
        try:
            # 1. 获取要更新的文档IDs（使用project避免Beanie验证）
            projected_cursor = self._mongo_cursor.project(IdOnlyProjection)
            id_projections = await projected_cursor.to_list(length=None)
            doc_ids = [str(proj.id) for proj in id_projections if proj.id]

            if not doc_ids:
                # No documents to update
                class UpdateResult:
                    modified_count = 0
                return UpdateResult()

            # 1.5. Auto-set updated_at if document has this field (AuditBase)
            from common_utils.datetime_utils import get_now_with_timezone
            if hasattr(self._full_model_class, 'model_fields') and 'updated_at' in self._full_model_class.model_fields:
                # Add updated_at to $set operator
                if "$set" not in update_data:
                    update_data = {"$set": {}}
                update_data["$set"]["updated_at"] = get_now_with_timezone()

            # 2. 执行MongoDB批量更新
            result = await self._mongo_cursor.update_many(update_data, **kwargs)

            # 3. 同步到KV-Storage
            if result and result.modified_count > 0:
                import json
                from bson import ObjectId
                from datetime import datetime

                def json_serializer(obj):
                    """Custom JSON serializer for ObjectId and datetime"""
                    if isinstance(obj, ObjectId):
                        return str(obj)
                    elif isinstance(obj, datetime):
                        return obj.isoformat()
                    raise TypeError(f"Type {type(obj)} not serializable")

                # Extract update fields from $set operator
                update_fields = {}
                if "$set" in update_data:
                    update_fields = update_data["$set"]
                else:
                    logger.warning(f"⚠️  Cursor.update_many() only supports $set operator, got: {update_data.keys()}")

                # Update each document in KV-Storage
                for doc_id in doc_ids:
                    try:
                        # Load existing full data from KV
                        kv_value = await self._kv_storage.get(key=doc_id)
                        if kv_value:
                            # Parse existing data
                            full_data = json.loads(kv_value)
                            # Apply update fields
                            full_data.update(update_fields)
                            # Write back to KV
                            kv_value = json.dumps(full_data, default=json_serializer)
                            await self._kv_storage.put(key=doc_id, value=kv_value)
                        else:
                            logger.warning(f"⚠️  KV miss for {doc_id}, cannot update")
                    except Exception as e:
                        logger.warning(f"⚠️  Failed to sync to KV-Storage for {doc_id}: {e}")

                logger.debug(f"✅ Cursor.update_many() updated {result.modified_count} documents in MongoDB and KV-Storage")

            return result

        except Exception as e:
            logger.error(f"❌ Failed to update_many on cursor with dual storage: {e}")
            raise

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

        Supports both:
        - Dict syntax: find({"user_id": "123"})
        - Beanie operator syntax: find(Model.user_id == "123")

        Returns:
            DualStorageQueryProxy
        """
        # 只在使用字典语法时验证查询字段
        # Beanie 操作符语法会直接传递给底层 MongoDB
        if args and isinstance(args[0], dict):
            filter_query = args[0]
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

    def find_one(self, *args, **kwargs):
        """
        Intercept find_one() - 返回 FindOneQueryProxy 支持链式调用

        Supports both:
        - Dict syntax: find_one({"user_id": "123", "group_id": "456"})
        - Beanie operator syntax: find_one(Model.user_id == "123", Model.group_id == "456")

        Returns FindOneQueryProxy that supports:
        1. Direct await: doc = await find_one(...)
        2. Chained delete: await find_one(...).delete()

        Args:
            *args: filter query (dict or Beanie operators)
            **kwargs: additional options

        Returns:
            FindOneQueryProxy (can be awaited or chained with .delete())

        Raises:
            LiteStorageQueryError: 如果查询字段不在 Lite 存储中（仅字典语法）
        """
        return FindOneQueryProxy(
            original_model=self._original_model,
            kv_storage=self._kv_storage,
            full_model_class=self._full_model_class,
            indexed_fields=self._indexed_fields,
            filter_args=args,
            filter_kwargs=kwargs,
        )

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

    async def update_many(self, filter_query: dict, update_data: dict, **kwargs):
        """
        Intercept update_many() - 批量更新并同步 KV-Storage

        为了确保 KV-Storage 同步，需要：
        1. 查询所有匹配的文档（获取 ID）
        2. 执行 MongoDB 批量更新
        3. 遍历文档，更新 KV-Storage 中的对应字段

        Args:
            filter_query: MongoDB filter query (dict)
            update_data: Update operations (e.g., {"$set": {"field": value}})
            **kwargs: Additional options (e.g., session)

        Returns:
            Update result with modified_count

        Example:
            await self.model.update_many(
                {"group_id": "123", "sync_status": -1},
                {"$set": {"sync_status": 0}}
            )
        """
        try:
            # 1. Validate query fields
            self._validate_query_fields(filter_query)

            # 2. Find all documents to update (get IDs before update)
            # Use self.find() which returns DualStorageQueryProxy
            session = kwargs.get("session", None)
            docs_to_update = await self.find(filter_query, session=session).to_list()

            if not docs_to_update:
                # No documents to update
                class UpdateResult:
                    modified_count = 0
                return UpdateResult()

            # 2.5. Auto-set updated_at if document has this field (AuditBase)
            from common_utils.datetime_utils import get_now_with_timezone
            if hasattr(self._full_model_class, 'model_fields') and 'updated_at' in self._full_model_class.model_fields:
                # Add updated_at to $set operator
                if "$set" not in update_data:
                    update_data = {"$set": {}}
                update_data["$set"]["updated_at"] = get_now_with_timezone()

            # 3. Execute MongoDB batch update using PyMongo
            collection = self._original_model.get_pymongo_collection()
            result = await collection.update_many(filter_query, update_data, **kwargs)

            # 4. Sync to KV-Storage
            if result and result.modified_count > 0:
                import json
                from bson import ObjectId
                from datetime import datetime

                def json_serializer(obj):
                    """Custom JSON serializer for ObjectId and datetime"""
                    if isinstance(obj, ObjectId):
                        return str(obj)
                    elif isinstance(obj, datetime):
                        return obj.isoformat()
                    raise TypeError(f"Type {type(obj)} not serializable")

                # Extract update fields from $set operator
                update_fields = {}
                if "$set" in update_data:
                    update_fields = update_data["$set"]
                else:
                    logger.warning(f"⚠️  update_many only supports $set operator, got: {update_data.keys()}")

                # Update each document in KV-Storage
                for doc in docs_to_update:
                    try:
                        kv_key = str(doc.id)
                        # Load existing full data from KV
                        kv_value = await self._kv_storage.get(key=kv_key)
                        if kv_value:
                            # Parse existing data
                            full_data = json.loads(kv_value)
                            # Apply update fields
                            full_data.update(update_fields)
                            # Write back to KV
                            kv_value = json.dumps(full_data, default=json_serializer)
                            await self._kv_storage.put(key=kv_key, value=kv_value)
                        else:
                            logger.warning(f"⚠️  KV miss for {doc.id}, cannot update")
                    except Exception as e:
                        logger.warning(f"⚠️  Failed to sync to KV-Storage for {doc.id}: {e}")

                logger.debug(f"✅ update_many() updated {result.modified_count} documents in MongoDB and KV-Storage")

            return result

        except Exception as e:
            logger.error(f"❌ Failed to update_many with dual storage: {e}")
            raise

    async def delete_all(self, **kwargs):
        """
        Intercept delete_all() - 删除所有文档并同步 KV-Storage

        为了确保 KV-Storage 同步，需要：
        1. 获取所有文档
        2. 逐个调用 delete() 触发 DualStorageMixin 的 wrap_delete
        3. 返回删除计数

        Returns:
            DeleteResult with deleted_count
        """
        try:
            # Get all documents first to ensure KV-Storage deletion via DualStorageMixin
            all_docs = await self.find({}).to_list()
            count = 0

            for doc in all_docs:
                try:
                    await doc.delete()
                    count += 1
                except Exception as e:
                    logger.error(f"❌ Failed to delete document {doc.id}: {e}")

            # Return a result object compatible with Beanie's DeleteResult
            class DeleteAllResult:
                def __init__(self, deleted_count):
                    self.deleted_count = deleted_count

            logger.debug(f"✅ delete_all() removed {count} documents from MongoDB and KV-Storage")
            return DeleteAllResult(deleted_count=count)

        except Exception as e:
            logger.error(f"❌ Failed to delete_all with dual storage: {e}")
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

    async def insert_many(
        self,
        documents: List[Any],
        session: Optional[AsyncClientSession] = None,
        **kwargs
    ):
        """
        Intercept insert_many() - 批量插入并同步 KV-Storage（Lite 存储模式）

        Lite 模式下的批量插入：
        - MongoDB：只存储 Lite 数据（索引字段）
        - KV-Storage：存储完整数据（全部字段）

        Args:
            documents: 要插入的文档列表
            session: Optional MongoDB session
            **kwargs: 其他参数

        Returns:
            InsertManyResult
        """
        try:
            from bson import ObjectId
            from datetime import datetime
            import json

            def json_serializer(obj):
                """Custom JSON serializer for ObjectId and datetime"""
                if isinstance(obj, ObjectId):
                    return str(obj)
                elif isinstance(obj, datetime):
                    return obj.isoformat()
                raise TypeError(f"Type {type(obj)} not serializable")

            # 1. 触发 before_event 钩子（批量设置审计字段）
            if hasattr(self._full_model_class, 'prepare_for_insert_many'):
                self._full_model_class.prepare_for_insert_many(documents)
            else:
                # 手动设置审计字段
                from common_utils.datetime_utils import get_now_with_timezone
                now = get_now_with_timezone()
                for doc in documents:
                    if hasattr(doc, 'created_at') and doc.created_at is None:
                        doc.created_at = now
                    if hasattr(doc, 'updated_at') and doc.updated_at is None:
                        doc.updated_at = now

            # 2. 提取所有文档的 Lite 数据
            lite_data_list = []
            full_data_list = []

            for doc in documents:
                # 提取 Lite 数据
                lite_data = LiteModelExtractor.extract_lite_data(doc, self._indexed_fields)
                lite_data_list.append(lite_data)

                # 保存完整数据（用于 KV-Storage）
                full_data = doc.model_dump(mode="python", exclude={'_id', 'id', 'revision_id'})
                full_data_list.append(full_data)

            # 3. 使用 PyMongo 批量插入 Lite 数据到 MongoDB
            mongo_collection = self._original_model.get_pymongo_collection()
            insert_result = await mongo_collection.insert_many(lite_data_list, session=session)

            # 4. 将生成的 IDs 赋值给 document 对象
            for doc, inserted_id in zip(documents, insert_result.inserted_ids):
                doc.id = inserted_id

            # 5. 批量存储完整数据到 KV-Storage
            for doc, full_data in zip(documents, full_data_list):
                try:
                    kv_key = str(doc.id)
                    full_data["id"] = doc.id  # 添加生成的 ID
                    kv_value = json.dumps(full_data, default=json_serializer)
                    await self._kv_storage.put(key=kv_key, value=kv_value)
                except Exception as e:
                    logger.warning(f"⚠️  Failed to sync to KV-Storage for {doc.id}: {e}")

            # 6. CRITICAL: 确保返回的 documents 对象包含完整数据
            # 因为 PyMongo 直接插入 lite_data 后，documents 对象可能被 Beanie 修改
            # 需要将 full_data 的所有字段重新设置回 doc 对象
            for doc, full_data in zip(documents, full_data_list):
                for field_name, field_value in full_data.items():
                    # 只设置非特殊字段（排除 _id 等）
                    if not field_name.startswith('_') and field_name != 'id':
                        try:
                            setattr(doc, field_name, field_value)
                        except Exception:
                            pass  # 忽略只读字段

            logger.debug(
                f"💾 insert_many: MongoDB Lite ({len(lite_data_list)} docs), "
                f"KV Full ({len(full_data_list)} docs), restored full data to documents"
            )

            # IMPORTANT: Return InsertManyResult, NOT documents
            # BaseRepository.create_batch expects InsertManyResult and will handle assigning IDs
            return insert_result

        except Exception as e:
            logger.error(f"❌ Failed to insert_many with dual storage: {e}")
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
            # Debug: Check self's fields
            logger.debug(f"🔍 Inserting {self.__class__.__name__}, fields: {self.model_fields.keys()}")

            # 0. Trigger Beanie's before_event hooks (e.g., AuditBase.set_created_at)
            # Since we're using PyMongo directly, we need to manually trigger these hooks
            if hasattr(self, 'set_created_at'):
                try:
                    await self.set_created_at()
                except Exception as e:
                    logger.warning(f"⚠️  Failed to call set_created_at hook: {e}")

            try:
                # 1. 提取 Lite 数据（只包含索引字段）
                lite_data = LiteModelExtractor.extract_lite_data(self, indexed_fields)
            except Exception as e:
                logger.error(f"❌ Failed to extract lite data: {e}")
                logger.error(f"Document type: {type(self)}")
                logger.error(f"Document __dict__: {self.__dict__.keys()}")

                # Check for ExpressionField in instance
                for key, value in self.__dict__.items():
                    logger.error(f"  {key}: {type(value)}")

                import traceback
                traceback.print_exc()
                raise

            try:
                # 2. 保存完整数据到 KV-Storage（在 insert 之前，避免 ID 问题）
                # Exclude Beanie internal fields
                full_data_for_kv = self.model_dump(mode="python", exclude={'_id', 'id', 'revision_id'})
            except Exception as e:
                logger.error(f"❌ Failed to dump full data: {e}")
                import traceback
                traceback.print_exc()
                raise

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

                # 直接序列化字典为 JSON（避免重新创建 Document 导致 ExpressionField 问题）
                import json
                from bson import ObjectId
                from datetime import datetime

                def json_serializer(obj):
                    """Custom JSON serializer for ObjectId and datetime"""
                    if isinstance(obj, ObjectId):
                        return str(obj)
                    elif isinstance(obj, datetime):
                        return obj.isoformat()
                    raise TypeError(f"Type {type(obj)} not serializable")

                kv_value = json.dumps(full_data_for_kv, default=json_serializer)

                await kv_storage.put(key=kv_key, value=kv_value)
                logger.debug(f"💾 MongoDB: Lite ({len(lite_data)} fields), KV: Full ({len(full_data_for_kv)} fields) - {kv_key}")
            except Exception as e:
                logger.warning(f"⚠️  Failed to sync full data to KV-Storage: {e}")
                import traceback
                traceback.print_exc()

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

            # 0. Trigger Beanie's before_event hooks (e.g., AuditBase.set_updated_at)
            # Since we're using PyMongo directly, we need to manually trigger these hooks
            if hasattr(self, 'set_updated_at'):
                try:
                    await self.set_updated_at()
                except Exception as e:
                    logger.warning(f"⚠️  Failed to call set_updated_at hook: {e}")

            try:
                # 1. 提取 Lite 数据
                lite_data = LiteModelExtractor.extract_lite_data(self, indexed_fields)

                # 2. 使用底层 pymongo API 更新 MongoDB（只更新 Lite 字段）
                mongo_collection = self.get_pymongo_collection()
                session = kwargs.get("session", None)

                # 使用 replace_one 替换整个文档为 Lite 数据
                from bson import ObjectId
                await mongo_collection.replace_one(
                    {"_id": ObjectId(self.id)},
                    lite_data,
                    session=session
                )

                # 3. 将完整数据存入 KV-Storage
                try:
                    kv_key = str(self.id)

                    # 使用 model_dump + json.dumps 避免 ExpressionField 问题
                    # model_dump_json() 可能失败，因为从 KV 恢复的对象可能有 lazy_model 的 ExpressionField
                    import json
                    from bson import ObjectId
                    from datetime import datetime

                    def json_serializer(obj):
                        """Custom JSON serializer for ObjectId and datetime"""
                        if isinstance(obj, ObjectId):
                            return str(obj)
                        elif isinstance(obj, datetime):
                            return obj.isoformat()
                        raise TypeError(f"Type {type(obj)} not serializable")

                    full_data = self.model_dump(mode="python", exclude={'_id', 'revision_id'})
                    kv_value = json.dumps(full_data, default=json_serializer)

                    await kv_storage.put(key=kv_key, value=kv_value)
                    logger.debug(f"💾 MongoDB: Lite ({len(lite_data)} fields), KV: Full - {kv_key}")
                except Exception as e:
                    logger.warning(f"⚠️  Failed to sync full data to KV-Storage: {e}")
                    import traceback
                    logger.error(traceback.format_exc())

                # 4. 返回 document 对象
                return self

            except Exception as e:
                logger.error(f"❌ Failed in wrapped_save: {e}")
                import traceback
                logger.error(traceback.format_exc())
                raise

        return wrapped_save

    @staticmethod
    def wrap_delete(original_delete, kv_storage: "KVStorageInterface"):
        """
        Wrap document.delete() - 支持软删除和硬删除

        行为取决于文档是否有 hard_delete 方法：
        - 有 hard_delete（软删除文档）：
          - MongoDB：标记 deleted_at（只更新 Lite 数据）
          - KV-Storage：保留完整数据（不删除）
        - 无 hard_delete（普通文档）：
          - MongoDB：物理删除
          - KV-Storage：物理删除
        """
        async def wrapped_delete(self, **kwargs):
            doc_id = str(self.id) if self.id else None

            # 调用原始 delete
            result = await original_delete(self, **kwargs)

            # 判断是软删除还是硬删除
            has_hard_delete = hasattr(self.__class__, "hard_delete")

            if has_hard_delete:
                # 软删除文档：保留 KV 数据
                logger.debug(f"✅ Soft deleted in MongoDB (KV data preserved): {self.id}")
            else:
                # 硬删除文档：删除 KV 数据
                if doc_id:
                    try:
                        await kv_storage.delete(key=doc_id)
                        logger.debug(f"✅ Hard deleted from KV-Storage: {doc_id}")
                    except Exception as e:
                        logger.warning(f"⚠️  Failed to delete from KV-Storage: {e}")

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
    "FindOneQueryProxy",
    "DocumentInstanceWrapper",
    "LiteStorageQueryError",
]
