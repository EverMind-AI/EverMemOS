"""
租户数据库初始化模块

此模块用于初始化特定租户的 MongoDB 和 Milvus 数据库。
通过环境变量 TENANT_SINGLE_TENANT_ID 指定租户ID：
1. 创建租户信息并设置租户上下文
2. 调用 MongoDB 的 lifespan startup 逻辑
3. 调用 Milvus 的 lifespan startup 逻辑

使用方式：
    通过 manage.py 调用：
    export TENANT_SINGLE_TENANT_ID=tenant_001
    python src/manage.py tenant-init

注意：
    - 必须设置环境变量 TENANT_SINGLE_TENANT_ID，否则会报错
    - 数据库名称会根据租户ID自动生成（格式：{tenant_id}_memsys）
    - 数据库连接配置从默认环境变量获取
"""

from core.observation.logger import get_logger
from core.tenants.tenant_config import get_tenant_config
from core.lifespan.mongodb_lifespan import MongoDBLifespanProvider
from core.lifespan.milvus_lifespan import MilvusLifespanProvider

logger = get_logger(__name__)


async def init_mongodb() -> bool:
    """
    初始化租户的 MongoDB 数据库

    Args:
        tenant_info: 租户信息

    Returns:
        是否初始化成功
    """
    logger.info("=" * 60)
    logger.info("开始初始化租户的 MongoDB 数据库...")
    logger.info("=" * 60)

    try:
        # 创建 MongoDB lifespan provider
        mongodb_provider = MongoDBLifespanProvider()

        # 创建一个模拟的 FastAPI app 对象（只需要 state 属性）
        class MockApp:
            class State:
                pass

            state = State()

        mock_app = MockApp()

        # 调用 startup 逻辑
        await mongodb_provider.startup(mock_app)

        logger.info("=" * 60)
        logger.info("✅ 租户的 MongoDB 数据库初始化成功")
        logger.info("=" * 60)

        # 关闭连接
        await mongodb_provider.shutdown(mock_app)

        return True

    except Exception as e:
        logger.error("=" * 60)
        logger.error("❌ 租户的 MongoDB 数据库初始化失败: %s", e)
        logger.error("=" * 60)
        return False


async def init_milvus() -> bool:
    """
    初始化租户的 Milvus 数据库

    Args:
        tenant_info: 租户信息

    Returns:
        是否初始化成功
    """
    logger.info("=" * 60)
    logger.info("开始初始化租户的 Milvus 数据库...")
    logger.info("=" * 60)

    try:
        # 创建 Milvus lifespan provider
        milvus_provider = MilvusLifespanProvider()

        # 创建一个模拟的 FastAPI app 对象（只需要 state 属性）
        class MockApp:
            class State:
                pass

            state = State()

        mock_app = MockApp()

        # 调用 startup 逻辑
        await milvus_provider.startup(mock_app)

        logger.info("=" * 60)
        logger.info("✅ 租户的 Milvus 数据库初始化成功")
        logger.info("=" * 60)

        # 关闭连接
        await milvus_provider.shutdown(mock_app)

        return True

    except Exception as e:
        logger.error("=" * 60)
        logger.error("❌ 租户的 Milvus 数据库初始化失败: %s", e)
        logger.error("=" * 60)
        return False


async def run_tenant_init() -> bool:
    """
    执行租户数据库初始化

    从环境变量 TENANT_SINGLE_TENANT_ID 读取租户ID。
    如果未设置该环境变量，则抛出错误。

    Returns:
        是否全部初始化成功

    Raises:
        ValueError: 如果未设置 TENANT_SINGLE_TENANT_ID 环境变量

    Examples:
        export TENANT_SINGLE_TENANT_ID=tenant_001
        python src/manage.py tenant-init
    """
    logger.info("*" * 60)
    logger.info("租户数据库初始化工具")
    logger.info("*" * 60)

    # 从配置中获取租户ID
    tenant_config = get_tenant_config()
    tenant_id = tenant_config.single_tenant_id

    # 如果没有配置租户ID，报错
    if not tenant_id:
        error_msg = (
            "未设置租户ID！\n"
            "请设置环境变量 TENANT_SINGLE_TENANT_ID，例如：\n"
            "  export TENANT_SINGLE_TENANT_ID=tenant_001\n"
            "  python src/manage.py tenant-init"
        )
        logger.error(error_msg)
        raise ValueError("未设置环境变量 TENANT_SINGLE_TENANT_ID")

    logger.info("租户ID: %s", tenant_id)
    logger.info("*" * 60)

    # 初始化 MongoDB
    mongodb_success = await init_mongodb()

    # 初始化 Milvus
    milvus_success = await init_milvus()

    # 输出总结
    logger.info("")
    logger.info("*" * 60)
    logger.info("初始化结果总结")
    logger.info("*" * 60)
    logger.info("租户ID: %s", tenant_id)
    logger.info("MongoDB: %s", "✅ 成功" if mongodb_success else "❌ 失败")
    logger.info("Milvus: %s", "✅ 成功" if milvus_success else "❌ 失败")
    logger.info("*" * 60)

    # 返回是否全部成功
    return mongodb_success and milvus_success
