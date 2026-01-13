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

    # Initialize Beanie with MemCell and MemCellLite documents
    await init_beanie(database=database, document_models=[MemCell, MemCellLite])

    # Initialize DI container and manually register repository
    # (Avoid full scan which loads unnecessary components)
    container = get_container()
    from infra_layer.adapters.out.persistence.repository.memcell_raw_repository import MemCellRawRepository

    # Register repository manually (only if not already registered)
    try:
        container.get_bean("MemCellRawRepository")
    except:
        container.register_bean(
            bean_type=MemCellRawRepository,
            bean_name="MemCellRawRepository",
            instance=MemCellRawRepository()
        )

    yield

    # Cleanup: close MongoDB connection
    client.close()
