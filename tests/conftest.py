"""
Pytest configuration file for test suite

This file is automatically loaded by pytest before running tests.
It configures the Python path and initializes the database connection.
"""

import sys
import asyncio
import pytest
import pytest_asyncio
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))


@pytest_asyncio.fixture(scope="function", autouse=True)
async def init_database():
    """Initialize database connection and DI container for all tests."""
    import os
    from dotenv import load_dotenv
    from motor.motor_asyncio import AsyncIOMotorClient
    from beanie import init_beanie
    from infra_layer.adapters.out.persistence.document.memory.memcell import MemCell
    from infra_layer.adapters.out.persistence.document.memory.memcell_lite import MemCellLite
    from infra_layer.adapters.out.persistence.document.memory.episodic_memory import EpisodicMemory
    from infra_layer.adapters.out.persistence.document.memory.episodic_memory_lite import EpisodicMemoryLite
    from infra_layer.adapters.out.persistence.document.memory.event_log_record import EventLogRecord
    from infra_layer.adapters.out.persistence.document.memory.event_log_record_lite import EventLogRecordLite
    from infra_layer.adapters.out.persistence.document.memory.foresight_record import ForesightRecord
    from infra_layer.adapters.out.persistence.document.memory.foresight_record_lite import ForesightRecordLite
    from infra_layer.adapters.out.persistence.document.memory.cluster_state import ClusterState
    from infra_layer.adapters.out.persistence.document.memory.cluster_state_lite import ClusterStateLite
    from core.di import get_container

    # Load environment variables from .env file
    load_dotenv()

    # Get MongoDB configuration from environment
    mongo_host = os.getenv("MONGODB_HOST", "localhost")
    mongo_port = os.getenv("MONGODB_PORT", "27017")
    mongo_username = os.getenv("MONGODB_USERNAME", "")
    mongo_password = os.getenv("MONGODB_PASSWORD", "")
    db_name = os.getenv("MONGODB_DATABASE", "memsys")

    # Build MongoDB URI with authentication if credentials provided
    if mongo_username and mongo_password:
        mongo_uri = f"mongodb://{mongo_username}:{mongo_password}@{mongo_host}:{mongo_port}"
    else:
        mongo_uri = f"mongodb://{mongo_host}:{mongo_port}"

    # Connect to MongoDB
    client = AsyncIOMotorClient(mongo_uri)
    database = client[db_name]

    # Initialize Beanie with all document models
    await init_beanie(database=database, document_models=[MemCell, MemCellLite, EpisodicMemory, EpisodicMemoryLite, EventLogRecord, EventLogRecordLite, ForesightRecord, ForesightRecordLite, ClusterState, ClusterStateLite])

    # Initialize DI container and manually register repositories
    # (Avoid full scan which loads unnecessary components)
    container = get_container()
    from infra_layer.adapters.out.persistence.repository.memcell_raw_repository import MemCellRawRepository
    from infra_layer.adapters.out.persistence.repository.episodic_memory_raw_repository import EpisodicMemoryRawRepository
    from infra_layer.adapters.out.persistence.repository.event_log_record_raw_repository import EventLogRecordRawRepository
    from infra_layer.adapters.out.persistence.repository.foresight_record_repository import ForesightRecordRawRepository
    from infra_layer.adapters.out.persistence.repository.cluster_state_raw_repository import ClusterStateRawRepository
    from infra_layer.adapters.out.persistence.kv_storage.in_memory_kv_storage import InMemoryKVStorage
    from infra_layer.adapters.out.persistence.kv_storage.kv_storage_interface import KVStorageInterface

    # Register KV-Storage (single instance for all memory types)
    try:
        container.get_bean("memcell_kv_storage")
    except:
        kv_storage_instance = InMemoryKVStorage()
        container.register_bean(
            bean_type=KVStorageInterface,
            bean_name="memcell_kv_storage",
            instance=kv_storage_instance
        )
        container.register_bean(
            bean_type=KVStorageInterface,
            bean_name="episodic_memory_kv_storage",
            instance=kv_storage_instance
        )
        container.register_bean(
            bean_type=KVStorageInterface,
            bean_name="kv_storage",
            instance=kv_storage_instance
        )

    # Register MemCell repository manually (only if not already registered)
    try:
        container.get_bean("MemCellRawRepository")
    except:
        container.register_bean(
            bean_type=MemCellRawRepository,
            bean_name="MemCellRawRepository",
            instance=MemCellRawRepository()
        )

    # Register EpisodicMemory repository manually (only if not already registered)
    try:
        container.get_bean("EpisodicMemoryRawRepository")
    except:
        container.register_bean(
            bean_type=EpisodicMemoryRawRepository,
            bean_name="EpisodicMemoryRawRepository",
            instance=EpisodicMemoryRawRepository()
        )

    # Register EventLogRecord repository manually (only if not already registered)
    try:
        container.get_bean("EventLogRecordRawRepository")
    except:
        container.register_bean(
            bean_type=EventLogRecordRawRepository,
            bean_name="EventLogRecordRawRepository",
            instance=EventLogRecordRawRepository()
        )

    # Register ForesightRecord repository manually (only if not already registered)
    try:
        container.get_bean("ForesightRecordRawRepository")
    except:
        container.register_bean(
            bean_type=ForesightRecordRawRepository,
            bean_name="ForesightRecordRawRepository",
            instance=ForesightRecordRawRepository()
        )

    # Register ClusterState repository manually (only if not already registered)
    try:
        container.get_bean("ClusterStateRawRepository")
    except:
        container.register_bean(
            bean_type=ClusterStateRawRepository,
            bean_name="ClusterStateRawRepository",
            instance=ClusterStateRawRepository()
        )

    yield

    # Cleanup: close MongoDB connection
    client.close()
