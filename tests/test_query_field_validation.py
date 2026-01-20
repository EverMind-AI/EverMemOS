"""
测试查询字段自动验证功能

验证当查询使用了不在 Lite 存储中的字段时，系统能否正确检测并提示。
"""

import pytest
import pytest_asyncio
import uuid
from datetime import datetime

from infra_layer.adapters.out.persistence.repository.dual_storage_model_proxy import (
    LiteStorageQueryError,
)

# Mark all test functions in this module as asyncio tests
pytestmark = pytest.mark.asyncio


@pytest_asyncio.fixture
async def memcell_repository():
    """Get MemCell repository instance"""
    from core.di import get_bean_by_type
    from infra_layer.adapters.out.persistence.repository.memcell_raw_repository import (
        MemCellRawRepository,
    )
    return get_bean_by_type(MemCellRawRepository)


@pytest_asyncio.fixture
async def kv_storage():
    """Get KV-Storage instance"""
    from core.di import get_bean_by_type
    from infra_layer.adapters.out.persistence.kv_storage.kv_storage_interface import (
        KVStorageInterface,
    )
    return get_bean_by_type(KVStorageInterface)


@pytest.fixture
def test_user_id():
    """Generate unique test user ID"""
    return f"test_user_{uuid.uuid4().hex[:8]}"


@pytest.mark.asyncio
class TestQueryFieldValidation:
    """测试查询字段验证功能"""

    async def test_query_with_valid_fields(self, memcell_repository, test_user_id):
        """
        测试：使用有效字段查询 - 应该成功

        user_id 是索引字段，应该能正常查询
        """
        # 应该不抛出异常
        result = await memcell_repository.model.find_one({"user_id": test_user_id})
        # 结果可能为 None（没有数据），但不应该抛异常
        assert True  # 如果到这里没抛异常就成功了

    async def test_query_with_invalid_field_raises_error(
        self, memcell_repository, test_user_id
    ):
        """
        测试：使用无效字段查询 - 应该抛出 LiteStorageQueryError

        假设有一个字段 'invalid_field' 不在 indexed_fields 中
        """
        with pytest.raises(LiteStorageQueryError) as exc_info:
            await memcell_repository.model.find_one({"invalid_field": "some_value"})

        # 验证错误消息包含关键信息
        error_msg = str(exc_info.value)
        assert "invalid_field" in error_msg
        assert "query_fields" in error_msg
        assert "Settings" in error_msg

    async def test_query_with_keywords_succeeds(self, memcell_repository, test_user_id):
        """
        测试：使用 keywords 字段查询 - 应该成功

        keywords 在 query_fields 中，虽然没有索引，但应该能查询
        """
        # 应该不抛出异常（因为 keywords 在 query_fields 中）
        result = await memcell_repository.model.find_one(
            {"keywords": {"$in": ["test"]}}
        )
        # 结果可能为 None（没有数据），但不应该抛异常
        assert True  # 如果到这里没抛异常就成功了

    async def test_error_message_provides_fix_instructions(
        self, memcell_repository, test_user_id
    ):
        """
        测试：错误消息提供清晰的修复指导

        验证错误消息包含：
        1. 缺失的字段列表
        2. 如何修复的说明
        3. 当前的 indexed_fields
        """
        with pytest.raises(LiteStorageQueryError) as exc_info:
            await memcell_repository.model.find_one(
                {"unknown_field_1": "value1", "unknown_field_2": "value2"}
            )

        error_msg = str(exc_info.value)

        # 验证包含缺失字段
        assert "unknown_field_1" in error_msg
        assert "unknown_field_2" in error_msg

        # 验证包含修复说明
        assert "Settings" in error_msg
        assert "query_fields" in error_msg

        # 验证包含当前 indexed_fields 信息
        assert "Current indexed fields" in error_msg

    async def test_complex_query_validation(self, memcell_repository, test_user_id):
        """
        测试：复杂查询条件的字段验证

        测试包含 $and, $or 等逻辑操作符的查询
        """
        # 有效的复杂查询 - 应该成功
        try:
            await memcell_repository.model.find_one(
                {
                    "$and": [
                        {"user_id": test_user_id},
                        {"timestamp": {"$gt": datetime.now()}},
                    ]
                }
            )
            assert True  # 不抛异常就成功
        except LiteStorageQueryError:
            pytest.fail("Valid complex query should not raise error")

        # 包含无效字段的复杂查询 - 应该抛出异常
        with pytest.raises(LiteStorageQueryError) as exc_info:
            await memcell_repository.model.find_one(
                {
                    "$or": [
                        {"user_id": test_user_id},
                        {"invalid_field": "value"},
                    ]
                }
            )

        error_msg = str(exc_info.value)
        assert "invalid_field" in error_msg

    async def test_find_method_validation(self, memcell_repository, test_user_id):
        """
        测试：find() 方法也会进行字段验证
        """
        # 有效查询 - 应该成功
        cursor = memcell_repository.model.find({"user_id": test_user_id})
        assert cursor is not None

        # 无效查询 - 应该抛出异常
        with pytest.raises(LiteStorageQueryError):
            memcell_repository.model.find({"invalid_field": "value"})

    async def test_delete_many_validation(self, memcell_repository, test_user_id):
        """
        测试：delete_many() 方法也会进行字段验证
        """
        # 有效查询 - 应该成功
        try:
            await memcell_repository.model.delete_many({"user_id": test_user_id})
            assert True
        except LiteStorageQueryError:
            pytest.fail("Valid delete_many should not raise error")

        # 无效查询 - 应该抛出异常
        with pytest.raises(LiteStorageQueryError):
            await memcell_repository.model.delete_many({"invalid_field": "value"})
