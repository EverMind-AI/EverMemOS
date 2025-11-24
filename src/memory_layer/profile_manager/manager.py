"""ProfileManager - Core component for automatic profile extraction and management."""

import asyncio
from typing import Any, Dict, List, Optional, Set
from pathlib import Path

from memory_layer.llm.llm_provider import LLMProvider
from memory_layer.memory_extractor.profile_memory_extractor import (
    ProfileMemoryExtractor,
    ProfileMemoryExtractRequest,
)
from memory_layer.profile_manager.config import ProfileManagerConfig, ScenarioType
from memory_layer.profile_manager.discriminator import ValueDiscriminator, DiscriminatorConfig
from memory_layer.profile_manager.storage import ProfileStorage, InMemoryProfileStorage
from core.observation.logger import get_logger

logger = get_logger(__name__)


class ProfileManager:
    """Automatic profile extraction and management integrated with clustering.
    
    ProfileManager monitors memcell clustering and automatically extracts/updates
    user profiles when high-value information is detected.
    
    Key Features:
    - Automatic profile extraction triggered by cluster updates
    - Value discrimination to filter high-quality updates
    - Incremental profile merging with version history
    - Flexible storage backends (in-memory, file-based, or custom)
    - Seamless integration with ConvMemCellExtractor
    
    Example:
        ```python
        # Initialize
        config = ProfileManagerConfig(
            scenario="group_chat",
            min_confidence=0.6,
            enable_versioning=True
        )
        
        profile_mgr = ProfileManager(llm_provider, config)
        
        # Option 1: Attach to extractor for automatic updates
        memcell_extractor = ConvMemCellExtractor(llm_provider)
        profile_mgr.attach_to_extractor(memcell_extractor)
        
        # Option 2: Manual updates
        await profile_mgr.on_memcell_clustered(
            memcell=memcell,
            cluster_id="cluster_001",
            recent_memcells=[...]
        )
        
        # Access profiles
        profile = await profile_mgr.get_profile(user_id)
        all_profiles = await profile_mgr.get_all_profiles()
        ```
    """
    
    def __init__(
        self,
        llm_provider: LLMProvider,
        config: Optional[ProfileManagerConfig] = None,
        storage: Optional[ProfileStorage] = None,
        group_id: Optional[str] = None,
        group_name: Optional[str] = None,
    ):
        """Initialize ProfileManager.
        
        Args:
            llm_provider: LLM provider for profile extraction and discrimination
            config: Manager configuration (uses defaults if None)
            storage: Profile storage backend (uses InMemoryProfileStorage if None)
            group_id: Group/conversation identifier
            group_name: Group/conversation name
        """
        self.llm_provider = llm_provider
        self.config = config or ProfileManagerConfig()
        self.group_id = group_id or "default"
        self.group_name = group_name
        
        # Initialize components
        self._profile_extractor = ProfileMemoryExtractor(llm_provider=llm_provider)
        
        # 💡 设置 MemCell 阈值：降低到 2 个，让 Profile 更容易提取
        self._min_memcells_threshold = 1  # 从默认的 3 降低到 2
        
        discriminator_config = DiscriminatorConfig(
            min_confidence=self.config.min_confidence,
            use_context=True,
            context_window=2
        )
        scenario_str = self.config.scenario.value if isinstance(self.config.scenario, ScenarioType) else str(self.config.scenario)
        self._discriminator = ValueDiscriminator(
            llm_provider=llm_provider,
            config=discriminator_config,
            scenario=scenario_str
        )
        
        # Storage
        if storage is None:
            storage = InMemoryProfileStorage(
                enable_persistence=False,
                enable_versioning=self.config.enable_versioning
            )
        self._storage = storage
        
        # Internal state for cluster tracking
        self._cluster_memcells: Dict[str, List[Any]] = {}  # cluster_id -> memcells
        self._watched_clusters: Set[str] = set()  # clusters flagged for profile extraction
        self._recent_memcells: List[Any] = []  # rolling window for context
        
        # Statistics
        self._stats = {
            "total_memcells": 0,
            "high_value_memcells": 0,
            "profile_extractions": 0,
            "failed_extractions": 0,
        }
        
        # 可配置的最小 MemCells 阈值（默认 1）
        self._min_memcells_threshold = 1
    
    async def on_memcell_clustered(
        self,
        memcell: Any,
        cluster_id: str,
        recent_memcells: Optional[List[Any]] = None,
        user_id_list: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Handle a newly clustered memcell and conditionally extract/update profiles.
        
        This method should be called when a memcell has been assigned to a cluster.
        It will:
        1. Add the memcell to the cluster's collection
        2. Evaluate if the memcell contains high-value profile information
        3. If high-value, mark the cluster as "watched" and extract profiles
        4. If cluster is already watched, update profiles incrementally
        
        Args:
            memcell: The memcell that was just clustered
            cluster_id: The cluster it was assigned to
            recent_memcells: Recent memcells for context (optional, uses internal if None)
            user_id_list: List of user IDs to extract profiles for (optional)
        
        Returns:
            Dict with extraction results:
            {
                "cluster_id": str,
                "is_high_value": bool,
                "confidence": float,
                "reason": str,
                "watched": bool,
                "profiles_updated": int,
                "updated_user_ids": List[str]
            }
        """
        self._stats["total_memcells"] += 1
        
        # Add to cluster collection
        if cluster_id not in self._cluster_memcells:
            self._cluster_memcells[cluster_id] = []
        self._cluster_memcells[cluster_id].append(memcell)
        print(f"[ProfileManager] 已添加 MemCell 到 _cluster_memcells[{cluster_id}]，当前数量: {len(self._cluster_memcells[cluster_id])}")
        
        # Update recent memcells window
        self._recent_memcells.append(memcell)
        if len(self._recent_memcells) > 10:
            self._recent_memcells = self._recent_memcells[-10:]
        
        # Use provided context or internal window
        context = recent_memcells if recent_memcells is not None else self._recent_memcells[-2:]
        
        # Value discrimination
        is_high_value, confidence, reason = await self._discriminator.is_high_value(
            memcell,
            context
        )
        logger.info(f"is_high_value: {is_high_value}, confidence: {confidence}, reason: {reason}")
        
        # 💡 强制设为 True，跳过价值判别（用于调试）
        is_high_value = True
        confidence = 1.0
        reason = f"[FORCED] {reason}"
        
        if is_high_value:
            self._stats["high_value_memcells"] += 1
            self._watched_clusters.add(cluster_id)
            log_msg = (
                f"High-value memcell detected in cluster {cluster_id}: "
                f"confidence={confidence:.2f}, reason='{reason}'"
            )
            logger.info(log_msg)
            print(f"[ProfileManager] {log_msg}")  # 确保控制台能看到
        
        # Extract/update profiles if cluster is watched and auto_extract is enabled
        updated_user_ids = []
        profiles_updated = 0
        
        if cluster_id in self._watched_clusters and self.config.auto_extract:
            # ✅ 参考原代码：只有当簇内 MemCells 达到一定数量才提取
            # 避免单个 MemCell 提取导致 Profile 为空
            cluster_memcell_count = len(self._cluster_memcells.get(cluster_id, []))
            
            # 使用实例属性或默认值
            min_memcells_for_extraction = getattr(self, '_min_memcells_threshold', 3)
            
            print(f"[ProfileManager] Cluster {cluster_id}: {cluster_memcell_count} MemCells, 阈值: {min_memcells_for_extraction}")
            
            if cluster_memcell_count < min_memcells_for_extraction:
                log_msg = (
                    f"Cluster {cluster_id} only has {cluster_memcell_count} memcells, "
                    f"waiting for {min_memcells_for_extraction} before extraction"
                )
                logger.debug(log_msg)
                print(f"[ProfileManager] {log_msg}")
            else:
                try:
                    print(f"[ProfileManager] 🚀 开始提取 Cluster {cluster_id} 的 Profile...")
                    updated_profiles = await self._extract_profiles_for_cluster(
                        cluster_id=cluster_id,
                        user_id_list=user_id_list
                    )
                    
                    profiles_updated = len(updated_profiles)
                    updated_user_ids = [
                        getattr(prof, "user_id", None)
                        for prof in updated_profiles
                        if hasattr(prof, "user_id")
                    ]
                    
                    self._stats["profile_extractions"] += 1
                    
                    if profiles_updated > 0:
                        log_msg = f"Updated {profiles_updated} profiles for cluster {cluster_id}"
                        logger.info(log_msg)
                        print(f"[ProfileManager] ✅ {log_msg}")
                
                except Exception as e:
                    error_msg = f"Failed to extract profiles for cluster {cluster_id}: {e}"
                    logger.error(error_msg)
                    print(f"[ProfileManager] ❌ {error_msg}")
                    import traceback
                    print(traceback.format_exc())
                    self._stats["failed_extractions"] += 1
        
        return {
            "cluster_id": cluster_id,
            "is_high_value": is_high_value,
            "confidence": confidence,
            "reason": reason,
            "watched": cluster_id in self._watched_clusters,
            "profiles_updated": profiles_updated,
            "updated_user_ids": updated_user_ids,
        }
    
    async def _extract_profiles_for_cluster(
        self,
        cluster_id: str,
        user_id_list: Optional[List[str]] = None
    ) -> List[Any]:
        """Extract profiles for a specific cluster using all its memcells.
        
        Args:
            cluster_id: Cluster identifier
            user_id_list: List of user IDs to extract for (optional)
        
        Returns:
            List of extracted/updated ProfileMemory objects
        """
        print(f"[ProfileManager] _extract_profiles_for_cluster 被调用")
        print(f"[ProfileManager]   cluster_id: {cluster_id}")
        print(f"[ProfileManager]   _cluster_memcells keys: {list(self._cluster_memcells.keys())}")
        print(f"[ProfileManager]   _cluster_memcells[{cluster_id}]: {len(self._cluster_memcells.get(cluster_id, []))} 个")
        
        memcells = self._cluster_memcells.get(cluster_id, [])
        if not memcells:
            print(f"[ProfileManager] ❌ _cluster_memcells[{cluster_id}] 为空！返回空列表")
            return []
        
        print(f"[ProfileManager] ✅ 找到 {len(memcells)} 个 MemCell")
        
        # ✅ 直接使用传入的字典格式 MemCell
        # biz_layer 已经准备好了完整数据,包含 episode (已保存到 MongoDB)
        
        memcells = full_memcells = self._cluster_memcells.get(cluster_id, [])
        
        for mc in memcells:
            event_id = mc.get('event_id', 'unknown') if isinstance(mc, dict) else getattr(mc, 'event_id', 'unknown')
            episode = mc.get('episode', '') if isinstance(mc, dict) else getattr(mc, 'episode', '')
            episode_len = len(episode) if episode else 0
            print(f"[ProfileManager] 使用 MemCell: {event_id}, episode 长度: {episode_len} 字符")
        
        if not full_memcells:
            print(f"[ProfileManager] ❌ 没有加载到任何完整的 MemCell")
            return []
        
        print(f"[ProfileManager] ✅ 加载了 {len(full_memcells)} 个完整的 MemCell，开始提取...")
        memcells = full_memcells  # 使用完整的 MemCell 对象
        
        # Limit batch size
        if len(memcells) > self.config.batch_size:
            logger.warning(
                f"Cluster {cluster_id} has {len(memcells)} memcells, "
                f"limiting to {self.config.batch_size} most recent"
            )
            memcells = memcells[-self.config.batch_size:]
        
        # Get old profiles for incremental merging
        old_profiles = []
        all_current_profiles = await self._storage.get_all_profiles()
        for profile in all_current_profiles.values():
            old_profiles.append(profile)
        
        # Build extraction request
        request = ProfileMemoryExtractRequest(
            memcell_list=memcells,
            user_id_list=user_id_list or [],
            group_id=self.group_id,
            group_name=self.group_name,
            old_memory_list=old_profiles if old_profiles else None,
        )
        
        # Extract profiles with retry logic
        for attempt in range(self.config.max_retries):
            try:
                print(f"[ProfileManager] 开始调用 LLM 提取 Profile (场景: {self.config.scenario.value})...")
                
                if self.config.scenario == ScenarioType.ASSISTANT:
                    result = await self._profile_extractor.extract_profile_companion(request)
                else:
                    result = await self._profile_extractor.extract_memory(request)
                
                print(f"[ProfileManager] LLM 调用完成，返回结果: {result}")
                print(f"[ProfileManager] 结果类型: {type(result)}, 长度: {len(result) if result else 0}")
                
                if not result:
                    logger.warning(f"Profile extraction returned empty result for cluster {cluster_id}")
                    print(f"[ProfileManager] ❌ LLM 返回空结果")
                    return []
                
                print(f"[ProfileManager] ✅ LLM 返回了 {len(result)} 个 Profile")
                
                # 🚀 并行保存所有 profiles 到 storage
                updated_profiles = []
                save_tasks = []
                
                for i, profile in enumerate(result):
                    user_id = getattr(profile, "user_id", None)
                    print(f"[ProfileManager] Profile {i+1}/{len(result)}: user_id={user_id}, 类型={type(profile).__name__}")
                    
                    if user_id:
                        metadata = {
                            "group_id": self.group_id,  # 🔧 添加 group_id
                            "cluster_id": cluster_id,
                            "memcell_count": len(memcells),
                            "scenario": self.config.scenario.value,
                        }
                        
                        save_tasks.append((
                            user_id,
                            profile,
                            self._storage.save_profile(
                                user_id=user_id,
                                profile=profile,
                                metadata=metadata
                            )
                        ))
                    else:
                        print(f"[ProfileManager] ⚠️  Profile {i+1} 没有 user_id，跳过保存")
                
                print(f"[ProfileManager] 准备保存 {len(save_tasks)} 个 Profile...")
                
                # 并行执行所有保存任务
                if save_tasks:
                    save_results = await asyncio.gather(
                        *[task[2] for task in save_tasks],
                        return_exceptions=True
                    )
                    
                    print(f"[ProfileManager] 保存结果: {save_results}")
                    
                    # 处理结果
                    for (user_id, profile, _), success in zip(save_tasks, save_results):
                        if isinstance(success, Exception):
                            error_msg = f"Failed to save profile for user {user_id}: {success}"
                            logger.warning(error_msg)
                            print(f"[ProfileManager] ❌ {error_msg}")
                            import traceback
                            print(traceback.format_exc())
                        elif success:
                            updated_profiles.append(profile)
                            print(f"[ProfileManager] ✅ 成功保存 Profile: user_id={user_id}")
                        else:
                            logger.warning(f"Failed to save profile for user {user_id}")
                            print(f"[ProfileManager] ❌ 保存失败（返回 False）: user_id={user_id}")
                
                print(f"[ProfileManager] 最终成功保存了 {len(updated_profiles)} 个 Profile")
                return updated_profiles
            
            except Exception as e:
                logger.warning(
                    f"Profile extraction attempt {attempt + 1}/{self.config.max_retries} failed: {e}"
                )
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(0.5 * (attempt + 1))  # Exponential backoff
                else:
                    logger.error(f"All profile extraction attempts failed for cluster {cluster_id}")
                    raise
        
        return []
    
    async def get_profile(self, user_id: str) -> Optional[Any]:
        """Get the latest profile for a user.
        
        Args:
            user_id: User identifier
        
        Returns:
            ProfileMemory object if found, None otherwise
        """
        return await self._storage.get_profile(user_id)
    
    async def get_all_profiles(self) -> Dict[str, Any]:
        """Get all user profiles.
        
        Returns:
            Dictionary mapping user_id to ProfileMemory
        """
        return await self._storage.get_all_profiles()
    
    async def get_profile_history(
        self,
        user_id: str,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get profile version history for a user.
        
        Args:
            user_id: User identifier
            limit: Maximum number of versions to return
        
        Returns:
            List of profile versions with metadata, newest first
        """
        return await self._storage.get_profile_history(user_id, limit)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get usage statistics.
        
        Returns:
            Dictionary with statistics:
            {
                "total_memcells": int,
                "high_value_memcells": int,
                "profile_extractions": int,
                "failed_extractions": int,
                "watched_clusters": int,
                "total_clusters": int
            }
        """
        return {
            **self._stats,
            "watched_clusters": len(self._watched_clusters),
            "total_clusters": len(self._cluster_memcells),
        }
    
    def attach_to_cluster_manager(self, cluster_manager: Any) -> None:
        """Attach ProfileManager to ClusterManager for automatic updates.
        
        This is the recommended approach when using the new ClusterManager component.
        
        Args:
            cluster_manager: ClusterManager instance
        """
        async def on_cluster_callback(group_id: str, memcell: Dict[str, Any], cluster_id: str):
            """Callback for cluster assignment events."""
            print(f"[ProfileManager] 🔔 收到聚类回调: cluster_id={cluster_id}, group_id={group_id}")
            
            # Create wrapper object
            class MemCellWrapper:
                def __init__(self, data: Dict[str, Any]):
                    for k, v in data.items():
                        setattr(self, k, v)
            
            mc_obj = MemCellWrapper(memcell)
            
            # 🔧 优先使用 participants，如果为空则使用 user_id_list
            user_id_list = memcell.get("participants", []) or memcell.get("user_id_list", [])
            print(f"[ProfileManager] 原始参与者列表: {user_id_list}")
            
            # 🔧 如果是 assistant 场景，过滤掉 robot/assistant 用户
            if self.config.scenario == ScenarioType.ASSISTANT:
                # 过滤掉 robot 和 assistant 用户
                filtered_user_ids = [
                    uid for uid in user_id_list 
                    if uid and not (uid.startswith("robot_") or uid.startswith("assistant_"))
                ]
                print(f"[ProfileManager] assistant 场景，过滤前: {user_id_list}, 过滤后: {filtered_user_ids}")
                user_id_list = filtered_user_ids
            
            if not user_id_list:
                print(f"[ProfileManager] ⚠️  过滤后 user_id_list 为空，跳过 Profile 提取")
                return
            
            print(f"[ProfileManager] 调用 on_memcell_clustered，user_id_list={user_id_list}...")
            # Trigger profile update
            result = await self.on_memcell_clustered(
                memcell=mc_obj,
                cluster_id=cluster_id,
                user_id_list=user_id_list
            )
            print(f"[ProfileManager] on_memcell_clustered 返回: {result}")
        
        # Register callback with ClusterManager
        print(f"[ProfileManager] 注册回调到 ClusterManager")
        cluster_manager.on_cluster_assigned(on_cluster_callback)
        print(f"[ProfileManager] 回调注册成功")
        
        logger.info("ProfileManager successfully attached to ClusterManager")
    
    def attach_to_extractor(self, memcell_extractor: Any) -> None:
        """Attach this ProfileManager to a MemCellExtractor for automatic profile updates.
        
        This method integrates the ProfileManager with the clustering worker of a
        ConvMemCellExtractor, so profiles are automatically extracted as conversations
        are processed.
        
        NOTE: This method uses monkey-patching. For cleaner integration, use
        ClusterManager and attach_to_cluster_manager() instead.
        
        Args:
            memcell_extractor: ConvMemCellExtractor instance
        
        Raises:
            AttributeError: If extractor doesn't have a cluster_worker attribute
        """
        if not hasattr(memcell_extractor, "_cluster_worker"):
            raise AttributeError(
                "MemCellExtractor does not have a _cluster_worker attribute. "
                "Only ConvMemCellExtractor with clustering is supported."
            )
        
        cluster_worker = memcell_extractor._cluster_worker
        
        # Check if this is a ClusterManager wrapper
        if hasattr(cluster_worker, "_cluster_mgr"):
            # It's already using ClusterManager, attach directly
            self.attach_to_cluster_manager(cluster_worker._cluster_mgr)
            return
        
        # Monkey-patch the clustering worker to notify us after each memcell is processed
        original_submit = cluster_worker.submit
        
        def patched_submit(group_id: Optional[str], memcell: Dict[str, Any]) -> None:
            # Call original submit first
            original_submit(group_id, memcell)
            
            # Get cluster assignment
            gid = group_id or "__default__"
            state = cluster_worker._states.get(gid)
            if state:
                event_id = str(memcell.get("event_id"))
                cluster_id = state.eventid_to_cluster.get(event_id)
                
                if cluster_id:
                    # Create a simple wrapper object for the memcell dict
                    class MemCellWrapper:
                        def __init__(self, data: Dict[str, Any]):
                            for k, v in data.items():
                                setattr(self, k, v)
                    
                    mc_obj = MemCellWrapper(memcell)
                    
                    # Schedule profile update (run in background)
                    asyncio.create_task(
                        self.on_memcell_clustered(
                            memcell=mc_obj,
                            cluster_id=cluster_id,
                            user_id_list=memcell.get("user_id_list", [])
                        )
                    )
        
        # Replace the submit method
        cluster_worker.submit = patched_submit
        
        logger.info("ProfileManager successfully attached to MemCellExtractor (legacy mode)")
    
    async def export_profiles(
        self,
        output_dir: Path,
        include_history: bool = True
    ) -> int:
        """Export all profiles to JSON files.
        
        Args:
            output_dir: Directory to save profiles
            include_history: Whether to export version history
        
        Returns:
            Number of profiles exported
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        profiles = await self._storage.get_all_profiles()
        count = 0
        
        for user_id, profile in profiles.items():
            try:
                # Convert profile to dict
                if hasattr(profile, "to_dict"):
                    profile_dict = profile.to_dict()
                elif hasattr(profile, "__dict__"):
                    profile_dict = dict(profile.__dict__)
                else:
                    profile_dict = profile
                
                # Save latest profile
                import json
                latest_file = output_dir / f"profile_{user_id}.json"
                with open(latest_file, "w", encoding="utf-8") as f:
                    json.dump(profile_dict, f, ensure_ascii=False, indent=2, default=str)
                
                count += 1
                
                # Export history if requested
                if include_history:
                    history = await self._storage.get_profile_history(user_id)
                    if history:
                        history_dir = output_dir / "history" / user_id
                        history_dir.mkdir(parents=True, exist_ok=True)
                        
                        for i, entry in enumerate(history):
                            version_file = history_dir / f"version_{i:03d}.json"
                            with open(version_file, "w", encoding="utf-8") as f:
                                json.dump(entry, f, ensure_ascii=False, indent=2, default=str)
            
            except Exception as e:
                logger.error(f"Failed to export profile for user {user_id}: {e}")
        
        logger.info(f"Exported {count} profiles to {output_dir}")
        return count

