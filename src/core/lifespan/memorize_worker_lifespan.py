"""
记忆提取 Worker 生命周期管理

在应用启动时启动 Worker 服务，应用关闭时停止 Worker 服务。
"""

from core.lifespan.lifespan_interface import LifespanProvider
from core.observation.logger import get_logger
from core.di.decorators import component

logger = get_logger(__name__)


@component(name="memorize_worker_lifespan_provider")
class MemorizeWorkerLifespanProvider(LifespanProvider):
    """记忆提取 Worker 生命周期提供者"""
    
    def __init__(self, name: str = "memorize_worker", order: int = 100):
        super().__init__(name, order)
        self.worker_service = None
    
    async def startup(self, app):
        """应用启动时执行"""
        from biz_layer.memorize_worker_service import MemorizeWorkerService
        
        logger.info("[MemorizeWorkerLifespan] 正在启动 Worker 服务...")
        
        # 获取单例实例并启动
        self.worker_service = await MemorizeWorkerService.get_instance(num_workers=3)
        await self.worker_service.start()
        
        logger.info("[MemorizeWorkerLifespan] ✅ Worker 服务已启动")
    
    async def shutdown(self, app):
        """应用关闭时执行"""
        if self.worker_service:
            logger.info("[MemorizeWorkerLifespan] 正在停止 Worker 服务...")
            
            try:
                await self.worker_service.stop(timeout=30.0)
                logger.info("[MemorizeWorkerLifespan] ✅ Worker 服务已停止")
            except Exception as e:
                logger.error(f"[MemorizeWorkerLifespan] ❌ 停止 Worker 服务失败: {e}")
        else:
            logger.warning("[MemorizeWorkerLifespan] Worker 服务未初始化")

