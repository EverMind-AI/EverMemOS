"""
Memory Controller API Test Script
Verify input and output structures of all endpoints under /api/v1/memories

Usage:
    # Run all tests
    python tests/test_memory_controller.py
    
    # Specify API address
    python tests/test_memory_controller.py --base-url http://localhost:1995
    
    # Specify test user
    python tests/test_memory_controller.py --base-url http://dev-server:1995 --user-id test_user_123
    
    # Test a specific method
    python tests/test_memory_controller.py --test-method memorize
    python tests/test_memory_controller.py --test-method fetch_episodic
    python tests/test_memory_controller.py --test-method fetch_event_log
    python tests/test_memory_controller.py --test-method search_keyword
    
    # Test all methods except certain ones (parameters separated by commas)
    python tests/test_memory_controller.py --except-test-method memorize
    python tests/test_memory_controller.py --except-test-method memorize,fetch_episodic
    python tests/test_memory_controller.py --except-test-method save_meta,patch_meta
    
    # Disable sync mode (use background mode)
    python tests/test_memory_controller.py --sync-mode false
"""

import argparse
import json
from zoneinfo import ZoneInfo
import uuid
from datetime import datetime, timedelta
import requests


class MemoryControllerTester:
    """Memory Controller API Test Class"""

    # Default tenant information
    DEFAULT_ORGANIZATION_ID = "test_memory_api_organization"
    DEFAULT_SPACE_ID = "test_memory_api_space"
    DEFAULT_HASH_KEY = "test_memory_api_hash_key"

    def __init__(
        self,
        base_url: str,
        user_id: str,
        group_id: str,
        organization_id: str = None,
        space_id: str = None,
        hash_key: str = None,
        timeout: int = 180,
        sync_mode: bool = True,
    ):
        """
        Initialize tester

        Args:
            base_url: API base URL
            user_id: Test user ID
            group_id: Test group ID
            organization_id: Organization ID (default: test_memory_api_organization)
            space_id: Space ID (default: test_memory_api_space)
            hash_key: Hash key (default: test_memory_api_hash_key)
            timeout: Request timeout in seconds, default 180 seconds (3 minutes)
            sync_mode: Whether to enable sync mode (default: True, disable background mode to ensure sequential test effectiveness)
        """
        self.base_url = base_url
        self.api_prefix = "/api/v1/memories"
        self.user_id = user_id
        self.group_id = group_id
        self.organization_id = organization_id or self.DEFAULT_ORGANIZATION_ID
        self.space_id = space_id or self.DEFAULT_SPACE_ID
        self.hash_key = hash_key or self.DEFAULT_HASH_KEY
        self.timeout = timeout
        self.sync_mode = sync_mode

    def get_tenant_headers(self) -> dict:
        """
        Get tenant-related request headers

        Returns:
            dict: Dictionary containing X-Organization-Id, X-Space-Id, and optional X-Hash-Key
        """
        headers = {
            "X-Organization-Id": self.organization_id,
            "X-Space-Id": self.space_id,
        }
        if self.hash_key:
            headers["X-Hash-Key"] = self.hash_key
        return headers

    def init_database(self) -> bool:
        """
        Initialize tenant database

        Call /internal/tenant/init-db endpoint to initialize database.

        Returns:
            bool: Whether initialization was successful
        """
        url = f"{self.base_url}/internal/tenant/init-db"
        headers = self.get_tenant_headers()

        print("\n" + "=" * 80)
        print("  Initialize Tenant Database")
        print("=" * 80)
        print(f"📍 URL: POST {url}")
        print(
            f"📤 Tenant Info: organization_id={self.organization_id}, space_id={self.space_id}"
        )
        print(
            f"📤 Request Headers: {json.dumps(headers, indent=2, ensure_ascii=False)}"
        )

        try:
            response = requests.post(url, headers=headers, timeout=self.timeout)
            print(f"\n📥 Response Status Code: {response.status_code}")
            response_json = response.json()
            print("📥 Response Data:")
            print(json.dumps(response_json, indent=2, ensure_ascii=False))

            if response.status_code == 200 and response_json.get("success"):
                print(
                    f"\n✅ Database initialization successful: tenant_id={response_json.get('tenant_id')}"
                )
                return True
            else:
                print(
                    f"\n⚠️  Database initialization returned: {response_json.get('message', 'Unknown')}"
                )
                # Continue even if failed, possibly database already exists
                return True
        except Exception as e:  # noqa: BLE001
            print(f"\n❌ Database initialization failed: {e}")
            return False

    def print_section(self, title: str):
        """Print section separator"""
        print("\n" + "=" * 80)
        print(f"  {title}")
        print("=" * 80)

    def _get_sync_mode_params(self) -> dict:
        """
        Get query parameters for sync mode

        Returns:
            dict: Dictionary containing sync_mode parameter (if sync mode is enabled)
        """
        if self.sync_mode:
            return {"sync_mode": "true"}
        return {}

    def call_post_api(self, endpoint: str, data: dict):
        """
        Call POST API and print results

        Args:
            endpoint: API endpoint
            data: Request data

        Returns:
            (status_code, response_json)
        """
        # If it's the memorize endpoint and sender is not provided, generate one randomly
        if endpoint == "" and "sender" not in data:
            data["sender"] = f"user_{uuid.uuid4().hex[:12]}"
            print(f"⚠️  Sender not provided, auto-generated: {data['sender']}")

        url = f"{self.base_url}{self.api_prefix}{endpoint}"
        headers = self.get_tenant_headers()
        params = self._get_sync_mode_params()

        print(f"\n📍 URL: POST {url}")
        if params:
            print(f"📤 Query Parameters: {params}")
        print("📤 Request Data:")
        print(json.dumps(data, indent=2, ensure_ascii=False))

        try:
            response = requests.post(
                url, json=data, headers=headers, params=params, timeout=self.timeout
            )
            print(f"\n📥 Response Status Code: {response.status_code}")
            print("📥 Response Data:")
            response_json = response.json()
            print(json.dumps(response_json, indent=2, ensure_ascii=False))
            return response.status_code, response_json
        except (
            Exception
        ) as e:  # noqa: BLE001 Need to catch all exceptions to ensure script continues
            print(f"\n❌ Request failed: {e}")
            return None, None

    def call_get_api(self, endpoint: str, params: dict = None):
        """
        Call GET API and print results

        Args:
            endpoint: API endpoint
            params: Query parameters

        Returns:
            (status_code, response_json)
        """
        url = f"{self.base_url}{self.api_prefix}{endpoint}"
        headers = self.get_tenant_headers()

        # Merge sync mode parameters
        merged_params = self._get_sync_mode_params()
        if params:
            merged_params.update(params)

        print(f"\n📍 URL: GET {url}")
        if merged_params:
            print("📤 Query Parameters:")
            print(json.dumps(merged_params, indent=2, ensure_ascii=False))

        try:
            response = requests.get(
                url, params=merged_params, headers=headers, timeout=self.timeout
            )
            print(f"\n📥 Response Status Code: {response.status_code}")
            print("📥 Response Data:")
            response_json = response.json()
            print(json.dumps(response_json, indent=2, ensure_ascii=False))
            return response.status_code, response_json
        except (
            Exception
        ) as e:  # noqa: BLE001 Need to catch all exceptions to ensure script continues
            print(f"\n❌ Request failed: {e}")
            return None, None

    def call_get_with_body_api(self, endpoint: str, data: dict):
        """
        Call GET API (with body) and print results

        Although uncommon, some search interfaces (e.g., Elasticsearch) use GET + body to pass complex parameters

        Args:
            endpoint: API endpoint
            data: Request data (placed in body)

        Returns:
            (status_code, response_json)
        """
        url = f"{self.base_url}{self.api_prefix}{endpoint}"
        headers = self.get_tenant_headers()
        params = self._get_sync_mode_params()

        print(f"\n📍 URL: GET {url} (with body)")
        if params:
            print(f"📤 Query Parameters: {params}")
        print("📤 Request Data:")
        print(json.dumps(data, indent=2, ensure_ascii=False))

        try:
            # GET request with body (requests library supports this, though not common)
            response = requests.request(
                "GET",
                url,
                json=data,
                headers=headers,
                params=params,
                timeout=self.timeout,
            )
            print(f"\n📥 Response Status Code: {response.status_code}")
            print("📥 Response Data:")
            response_json = response.json()
            print(json.dumps(response_json, indent=2, ensure_ascii=False))
            return response.status_code, response_json
        except (
            Exception
        ) as e:  # noqa: BLE001 Need to catch all exceptions to ensure script continues
            print(f"\n❌ Request failed: {e}")
            return None, None

    def call_patch_api(self, endpoint: str, data: dict):
        """
        Call PATCH API and print results

        Args:
            endpoint: API endpoint
            data: Request data

        Returns:
            (status_code, response_json)
        """
        url = f"{self.base_url}{self.api_prefix}{endpoint}"
        headers = self.get_tenant_headers()
        params = self._get_sync_mode_params()

        print(f"\n📍 URL: PATCH {url}")
        if params:
            print(f"📤 Query Parameters: {params}")
        print("📤 Request Data:")
        print(json.dumps(data, indent=2, ensure_ascii=False))

        try:
            response = requests.patch(
                url, json=data, headers=headers, params=params, timeout=self.timeout
            )
            print(f"\n📥 Response Status Code: {response.status_code}")
            print("📥 Response Data:")
            response_json = response.json()
            print(json.dumps(response_json, indent=2, ensure_ascii=False))
            return response.status_code, response_json
        except (
            Exception
        ) as e:  # noqa: BLE001 Need to catch all exceptions to ensure script continues
            print(f"\n❌ Request failed: {e}")
            return None, None

    def test_memorize_single_message(self):
        """Test 1: POST /api/v1/memories - Store conversation memory (send multiple messages to trigger boundary detection)"""
        self.print_section("Test 1: POST /api/v1/memories - Store Conversation Memory")

        # Prepare a simple conversation to simulate user and assistant interaction
        # Sending multiple messages can trigger boundary detection and extract memories
        base_time = datetime.now(ZoneInfo("UTC"))

        # Build conversation sequence, triggering boundary detection through:
        # 1. First scenario: Discussion about coffee preferences (4 messages)
        # 2. Second scenario: Start new topic (trigger boundary via time gap + topic switch)
        messages = [
            # Scenario 1: Discuss coffee preferences (complete conversation episode)
            {
                "group_id": self.group_id,
                "group_name": "Test Group",
                "message_id": "msg_001",
                "create_time": base_time.isoformat(),
                "sender": self.user_id,
                "sender_name": "Test User",
                "content": "I recently want to develop a habit of drinking coffee, do you have any suggestions?",
                "refer_list": [],
            },
            {
                "group_id": self.group_id,
                "group_name": "Test Group",
                "message_id": "msg_002",
                "create_time": (base_time + timedelta(seconds=30)).isoformat(),
                "sender": "assistant_001",
                "sender_name": "AI Assistant",
                "content": "Of course! Coffee comes in many varieties, from strong espresso to mild Americano. You can choose based on your taste. I suggest starting with Americano.",
                "refer_list": [],
            },
            {
                "group_id": self.group_id,
                "group_name": "Test Group",
                "message_id": "msg_003",
                "create_time": (base_time + timedelta(minutes=1)).isoformat(),
                "sender": self.user_id,
                "sender_name": "Test User",
                "content": "I like drinking Americano, no sugar, no milk, the stronger the better.",
                "refer_list": [],
            },
            {
                "group_id": self.group_id,
                "group_name": "Test Group",
                "message_id": "msg_004",
                "create_time": (
                    base_time + timedelta(minutes=1, seconds=30)
                ).isoformat(),
                "sender": "assistant_001",
                "sender_name": "AI Assistant",
                "content": "I understand your preference! Black Americano can fully experience the flavor of coffee beans. I suggest choosing dark roasted beans for a stronger taste.",
                "refer_list": [],
            },
            # Scenario 2: Start new topic (trigger boundary via longer time gap + topic switch)
            # According to boundary detection rules: time gap over 4 hours and content unrelated will trigger boundary
            {
                "group_id": self.group_id,
                "group_name": "Test Group",
                "message_id": "msg_005",
                "create_time": (base_time + timedelta(hours=24)).isoformat(),
                "sender": self.user_id,
                "sender_name": "Test User",
                "content": "By the way, how is the weekend project progressing?",
                "refer_list": [],
            },
            {
                "group_id": self.group_id,
                "group_name": "Test Group",
                "message_id": "msg_006",
                "create_time": (
                    base_time + timedelta(hours=24, seconds=30)
                ).isoformat(),
                "sender": "assistant_001",
                "sender_name": "AI Assistant",
                "content": "The project is progressing smoothly, main features are 80% complete, expected to submit for testing next week.",
                "refer_list": [],
            },
        ]

        # Send messages one by one
        print("\n📨 Starting to send conversation sequence...")
        print(
            "💡 Strategy Explanation: First 4 messages form complete scenario 1 (coffee preference discussion)"
        )
        print(
            "💡 5th message triggers boundary detection via 5-hour time gap + new topic"
        )
        print("💡 This ensures memory from scenario 1 is successfully extracted")

        last_response = None
        for i, msg in enumerate(messages, 1):
            if i == 5:
                print(
                    f"\n🔄 --- Scenario Switch: Sending message {i}/{len(messages)} (triggering boundary) ---"
                )
            else:
                print(f"\n--- Sending message {i}/{len(messages)} ---")

            status_code, response = self.call_post_api("", msg)

            # Verify each message is successfully processed
            assert (
                status_code == 200
            ), f"Message {i} status code should be 200, actual: {status_code}"
            assert response.get("status") == "ok", f"Message {i} status should be ok"

            last_response = response

        # Use the response from the last message for validation
        status_code = 200
        response = last_response

        # Assert: Validate result structure
        print("\n📊 Validating conversation memory extraction results...")
        assert "result" in response, "Successful response should contain result field"
        result = response["result"]
        assert "saved_memories" in result, "result should contain saved_memories field"
        assert "count" in result, "result should contain count field"
        assert "status_info" in result, "result should contain status_info field"

        # Validate saved_memories is a list
        assert isinstance(
            result["saved_memories"], list
        ), "saved_memories should be a list"
        assert result["count"] >= 0, "count should be >= 0"
        assert result["status_info"] in [
            "accumulated",
            "extracted",
        ], "status_info should be accumulated or extracted"

        # If there are extracted memories, validate each memory's structure
        if result["count"] > 0:
            print(f"\n✅ Successfully extracted {result['count']} memories!")
            print(
                f"✅ Boundary detection successful: triggered by time gap (5 hours) + topic switch"
            )
            for idx, memory in enumerate(result["saved_memories"], 1):
                assert isinstance(memory, dict), f"Memory {idx} should be a dictionary"
                # Note: Different memory types may have different field structures
                # Here only basic field existence is validated
                memory_type = memory.get('memory_type', 'unknown')
                summary = memory.get('summary', memory.get('content', 'no summary'))[
                    :50
                ]
                print(f"  Memory {idx}: {memory_type} - {summary}...")
        else:
            print(
                f"\n⚠️  Messages accumulated, waiting for boundary detection (status_info: {result['status_info']})"
            )
            print(
                f"   Sent {len(messages)} messages, but boundary detection conditions may not be met"
            )
            print(
                f"   💡 Tip: Boundary detection requires one of the following conditions:"
            )
            print(
                f"      1. Cross-day (new message date differs from previous message)"
            )
            print(f"      2. Long interruption (over 4 hours) + topic switch")
            print(f"      3. Clear scene/topic switch signal")

        print(f"\n✅ Memorize Test Completed")
        return status_code, response

    def test_fetch_episodic(self):
        """Test 2: GET /api/v1/memories - Fetch user episodic memory (episodic_memory type, pass parameters via body)"""
        self.print_section("Test 2: GET /api/v1/memories - Fetch User Episodic Memory")

        data = {
            "user_id": self.user_id,
            "memory_type": "episodic_memory",
            "limit": 10,
            "offset": 0,
        }

        status_code, response = self.call_get_with_body_api("", data)

        # Assert: Precisely validate response structure
        assert status_code == 200, f"Status code should be 200, actual: {status_code}"
        assert (
            response.get("status") == "ok"
        ), f"Status should be ok, actual: {response.get('status')}"
        assert "result" in response, "Response should contain result field"

        result = response["result"]
        assert "memories" in result, "result should contain memories field"
        assert "total_count" in result, "result should contain total_count field"
        assert "has_more" in result, "result should contain has_more field"
        assert "metadata" in result, "result should contain metadata field"

        # Validate data types
        assert isinstance(result["memories"], list), "memories should be a list"
        assert result["total_count"] >= 0, "total_count should be >= 0"
        assert isinstance(result["has_more"], bool), "has_more should be boolean"

        # Validate metadata structure
        metadata = result["metadata"]
        assert isinstance(metadata, dict), "metadata should be a dictionary"
        assert "source" in metadata, "metadata should contain source field"
        assert "user_id" in metadata, "metadata should contain user_id field"
        assert "memory_type" in metadata, "metadata should contain memory_type field"
        assert metadata.get("user_id") == self.user_id, "metadata user_id should match"

        # If there are memories, deeply validate structure
        if result["total_count"] > 0 and len(result["memories"]) > 0:
            for idx, memory in enumerate(result["memories"]):
                assert isinstance(memory, dict), f"Memory {idx} should be a dictionary"
                assert "user_id" in memory, f"Memory {idx} should contain user_id"
                assert "timestamp" in memory, f"Memory {idx} should contain timestamp"
                assert (
                    memory.get("user_id") == self.user_id
                ), f"Memory {idx} user_id should match"

            print(
                f"✅ Fetch Episodic successful, returned {result['total_count']} episodic memories, deep structure validated"
            )
        else:
            print(
                f"✅ Fetch Episodic successful, returned {result['total_count']} episodic memories"
            )

        return status_code, response

    def test_fetch_foresight(self):
        """Test 3: GET /api/v1/memories - Fetch personal foresight (foresight type, pass parameters via body)"""
        self.print_section("Test 3: GET /api/v1/memories - Fetch Personal Foresight")

        data = {
            "user_id": self.user_id,
            "memory_type": "foresight",
            "limit": 10,
            "offset": 0,
        }

        status_code, response = self.call_get_with_body_api("", data)

        # Assert: Precisely validate response structure
        assert status_code == 200, f"Status code should be 200, actual: {status_code}"
        assert (
            response.get("status") == "ok"
        ), f"Status should be ok, actual: {response.get('status')}"
        assert "result" in response, "Response should contain result field"

        result = response["result"]
        assert "memories" in result, "result should contain memories field"
        assert "total_count" in result, "result should contain total_count field"
        assert "has_more" in result, "result should contain has_more field"
        assert "metadata" in result, "result should contain metadata field"

        # Validate data types
        assert isinstance(result["memories"], list), "memories should be a list"
        assert result["total_count"] >= 0, "total_count should be >= 0"
        assert isinstance(result["has_more"], bool), "has_more should be boolean"

        # Validate metadata structure
        metadata = result["metadata"]
        assert isinstance(metadata, dict), "metadata should be a dictionary"
        assert "source" in metadata, "metadata should contain source field"
        assert "user_id" in metadata, "metadata should contain user_id field"
        assert "memory_type" in metadata, "metadata should contain memory_type field"
        assert metadata.get("user_id") == self.user_id, "metadata user_id should match"

        # If there are memories, deeply validate structure
        if result["total_count"] > 0 and len(result["memories"]) > 0:
            for idx, memory in enumerate(result["memories"]):
                assert isinstance(memory, dict), f"Memory {idx} should be a dictionary"
                assert "content" in memory, f"Memory {idx} should contain content"
                assert (
                    "parent_episode_id" in memory
                ), f"Memory {idx} should contain parent_episode_id"
                # Personal foresight user_id may be None (group scenario), so not enforced

            print(
                f"✅ Fetch Personal Foresight successful, returned {result['total_count']} personal foresights, deep structure validated"
            )
        else:
            print(
                f"✅ Fetch Personal Foresight successful, returned {result['total_count']} personal foresights"
            )

        return status_code, response

    def test_fetch_event_log(self):
        """Test 4: GET /api/v1/memories - Fetch user event log (event_log type, pass parameters via body)"""
        self.print_section("Test 4: GET /api/v1/memories - Fetch User Event Log")

        data = {
            "user_id": self.user_id,
            "memory_type": "event_log",
            "limit": 10,
            "offset": 0,
        }

        status_code, response = self.call_get_with_body_api("", data)

        # Assert: Precisely validate response structure
        assert status_code == 200, f"Status code should be 200, actual: {status_code}"
        assert (
            response.get("status") == "ok"
        ), f"Status should be ok, actual: {response.get('status')}"
        assert "result" in response, "Response should contain result field"

        result = response["result"]
        assert "memories" in result, "result should contain memories field"
        assert "total_count" in result, "result should contain total_count field"
        assert "has_more" in result, "result should contain has_more field"
        assert "metadata" in result, "result should contain metadata field"

        # Validate data types
        assert isinstance(result["memories"], list), "memories should be a list"
        assert result["total_count"] >= 0, "total_count should be >= 0"
        assert isinstance(result["has_more"], bool), "has_more should be boolean"

        # Validate metadata structure
        metadata = result["metadata"]
        assert isinstance(metadata, dict), "metadata should be a dictionary"
        assert "source" in metadata, "metadata should contain source field"
        assert "user_id" in metadata, "metadata should contain user_id field"
        assert "memory_type" in metadata, "metadata should contain memory_type field"
        assert metadata.get("user_id") == self.user_id, "metadata user_id should match"

        # If there are event logs, deeply validate structure
        if result["total_count"] > 0 and len(result["memories"]) > 0:
            for idx, memory in enumerate(result["memories"]):
                assert isinstance(memory, dict), f"Memory {idx} should be a dictionary"
                assert (
                    "atomic_fact" in memory
                ), f"Memory {idx} should contain atomic_fact"
                assert "timestamp" in memory, f"Memory {idx} should contain timestamp"
                assert "user_id" in memory, f"Memory {idx} should contain user_id"
                assert (
                    memory.get("user_id") == self.user_id
                ), f"Memory {idx} user_id should match"

            print(
                f"✅ Fetch Event Log successful, returned {result['total_count']} event logs, deep structure validated"
            )
        else:
            print(
                f"✅ Fetch Event Log successful, returned {result['total_count']} event logs"
            )

        return status_code, response

    def test_search_memories_keyword(self):
        """Test 5: GET /api/v1/memories/search - Keyword search (pass parameters via body)"""
        self.print_section("Test 5: GET /api/v1/memories/search - Keyword Search")

        # Note: Although route is defined as GET, actual implementation reads parameters from body
        # Similar to Elasticsearch search API, GET requests can carry body
        data = {
            "user_id": self.user_id,
            "query": "coffee",
            "top_k": 10,
            "retrieve_method": "keyword",
        }

        status_code, response = self.call_get_with_body_api("/search", data)

        # Assert: Precisely validate response structure
        assert status_code == 200, f"Status code should be 200, actual: {status_code}"
        assert (
            response.get("status") == "ok"
        ), f"Status should be ok, actual: {response.get('status')}"
        assert "result" in response, "Response should contain result field"

        result = response["result"]
        assert "memories" in result, "result should contain memories field"
        assert "scores" in result, "result should contain scores field"
        assert "total_count" in result, "result should contain total_count field"
        assert "has_more" in result, "result should contain has_more field"
        assert "metadata" in result, "result should contain metadata field"

        # Validate data types
        assert isinstance(result["memories"], list), "memories should be a list"
        assert isinstance(result["scores"], list), "scores should be a list"
        assert result["total_count"] >= 0, "total_count should be >= 0"

        # Validate metadata
        metadata = result["metadata"]
        assert metadata.get("user_id") == self.user_id, "metadata user_id should match"

        # If there are results, deeply validate nested structure
        if result["total_count"] > 0 and len(result["memories"]) > 0:
            # Validate memories and scores have same length
            assert len(result["memories"]) == len(
                result["scores"]
            ), "memories and scores should have same length"

            # Iterate through each group's memories
            for group_idx, memory_group in enumerate(result["memories"]):
                assert isinstance(
                    memory_group, dict
                ), f"Memory group {group_idx} should be a dictionary"

                # Iterate through memories within group
                for group_id, memory_list in memory_group.items():
                    assert isinstance(group_id, str), "group_id should be string"
                    assert isinstance(
                        memory_list, list
                    ), f"Memory list for group {group_id} should be a list"

                    # Validate basic fields for each memory
                    for mem_idx, mem in enumerate(memory_list):
                        assert isinstance(
                            mem, dict
                        ), f"Memory {mem_idx} should be a dictionary"
                        assert (
                            "memory_type" in mem
                        ), f"Memory {mem_idx} should contain memory_type"
                        assert (
                            "user_id" in mem
                        ), f"Memory {mem_idx} should contain user_id"
                        assert (
                            "timestamp" in mem
                        ), f"Memory {mem_idx} should contain timestamp"

            print(
                f"✅ Search Keyword successful, returned {result['total_count']} groups of memories"
            )
        else:
            print(
                f"✅ Search Keyword successful, returned {result['total_count']} groups of memories"
            )

        return status_code, response

    def test_search_memories_vector(self):
        """Test 6: GET /api/v1/memories/search - Vector search (pass parameters via body)"""
        self.print_section("Test 6: GET /api/v1/memories/search - Vector Search")

        data = {
            "user_id": self.user_id,
            "query": "user's dietary preferences",
            "top_k": 10,
            "retrieve_method": "vector",
        }

        status_code, response = self.call_get_with_body_api("/search", data)

        # Assert: Precisely validate response structure
        assert status_code == 200, f"Status code should be 200, actual: {status_code}"
        assert (
            response.get("status") == "ok"
        ), f"Status should be ok, actual: {response.get('status')}"
        assert "result" in response, "Response should contain result field"

        result = response["result"]
        assert "memories" in result, "result should contain memories field"
        assert "scores" in result, "result should contain scores field"
        assert "total_count" in result, "result should contain total_count field"
        assert "has_more" in result, "result should contain has_more field"
        assert "metadata" in result, "result should contain metadata field"

        # Vector search should have importance_scores
        if result["total_count"] > 0:
            assert (
                "importance_scores" in result
            ), "Vector search result should contain importance_scores field"
            assert isinstance(
                result["importance_scores"], list
            ), "importance_scores should be a list"

        print(
            f"✅ Search Vector successful, returned {result['total_count']} groups of memories"
        )

        return status_code, response

    def test_search_memories_hybrid(self):
        """Test 7: GET /api/v1/memories/search - Hybrid search (pass parameters via body)"""
        self.print_section("Test 7: GET /api/v1/memories/search - Hybrid Search")

        now = datetime.now(ZoneInfo("UTC"))
        data = {
            "user_id": self.user_id,
            "query": "coffee preference",
            "top_k": 10,
            "retrieve_method": "hybrid",
            "start_time": (now - timedelta(days=60)).isoformat(),
            "end_time": now.isoformat(),
        }

        status_code, response = self.call_get_with_body_api("/search", data)

        # Assert: Precisely validate response structure
        assert status_code == 200, f"Status code should be 200, actual: {status_code}"
        assert (
            response.get("status") == "ok"
        ), f"Status should be ok, actual: {response.get('status')}"
        assert "result" in response, "Response should contain result field"

        result = response["result"]
        assert "memories" in result, "result should contain memories field"
        assert "scores" in result, "result should contain scores field"
        assert "total_count" in result, "result should contain total_count field"
        assert "has_more" in result, "result should contain has_more field"
        assert "metadata" in result, "result should contain metadata field"

        # Hybrid search should have importance_scores
        if result["total_count"] > 0:
            assert (
                "importance_scores" in result
            ), "Hybrid search result should contain importance_scores field"
            assert isinstance(
                result["importance_scores"], list
            ), "importance_scores should be a list"

            # Validate source in metadata
            metadata = result["metadata"]
            assert (
                metadata.get("source") == "hybrid_retrieval"
            ), "Hybrid search source should be hybrid_retrieval"

        print(
            f"✅ Search Hybrid successful, returned {result['total_count']} groups of memories"
        )

        return status_code, response

    def test_save_conversation_meta(self):
        """Test 8: POST /api/v1/memories/conversation-meta - Save conversation metadata"""
        self.print_section(
            "Test 8: POST /api/v1/memories/conversation-meta - Save Conversation Metadata"
        )

        now = datetime.now(ZoneInfo("UTC"))
        data = {
            "version": "1.0",
            "scene": "assistant",
            "scene_desc": {
                "description": "Project collaboration group chat",
                "bot_ids": ["bot_001"],
                "extra": {"category": "test"},
            },
            "name": "Test Project Discussion Group",
            "description": "Project discussion group for testing",
            "group_id": self.group_id,
            "created_at": now.isoformat(),
            "default_timezone": "UTC",
            "user_details": {
                self.user_id: {
                    "full_name": "Test User",
                    "role": "developer",
                    "extra": {"department": "Engineering"},
                }
            },
            "tags": ["test", "project"],
        }

        status_code, response = self.call_post_api("/conversation-meta", data)

        # Assert: Precisely validate response structure
        assert status_code == 200, f"Status code should be 200, actual: {status_code}"
        assert (
            response.get("status") == "ok"
        ), f"Status should be ok, actual: {response.get('status')}"
        assert "result" in response, "Response should contain result field"

        result = response["result"]
        assert "id" in result, "result should contain id field"
        assert "group_id" in result, "result should contain group_id field"
        assert "scene" in result, "result should contain scene field"
        assert "name" in result, "result should contain name field"
        assert "version" in result, "result should contain version field"

        # Validate value correctness
        assert result["group_id"] == self.group_id, "Returned group_id should match"
        assert result["scene"] == "assistant", "Returned scene should match"
        assert (
            result["name"] == "Test Project Discussion Group"
        ), "Returned name should match"

        print(f"✅ Save Conversation Meta successful, id={result['id']}")

        return status_code, response

    def test_patch_conversation_meta(self):
        """Test 9: PATCH /api/v1/memories/conversation-meta - Partially update conversation metadata"""
        self.print_section(
            "Test 9: PATCH /api/v1/memories/conversation-meta - Partially Update Conversation Metadata"
        )

        data = {
            "group_id": self.group_id,
            "name": "Updated Test Project Discussion Group",
            "tags": ["test", "project", "update"],
        }

        status_code, response = self.call_patch_api("/conversation-meta", data)

        # Assert: Precisely validate response structure
        if status_code == 200:
            assert (
                response.get("status") == "ok"
            ), f"Status should be ok, actual: {response.get('status')}"
            assert "result" in response, "Response should contain result field"

            result = response["result"]
            assert "id" in result, "result should contain id field"
            assert "group_id" in result, "result should contain group_id field"
            assert (
                "updated_fields" in result
            ), "result should contain updated_fields field"

            # Validate updated fields
            assert result["group_id"] == self.group_id, "Returned group_id should match"
            assert isinstance(
                result["updated_fields"], list
            ), "updated_fields should be a list"

            if len(result["updated_fields"]) > 0:
                print(
                    f"✅ Patch Conversation Meta successful, updated {len(result['updated_fields'])} fields: {result['updated_fields']}"
                )
            else:
                print("✅ Patch Conversation Meta successful, no fields needed update")
        elif status_code == 404:
            print(
                f"⚠️  Patch Conversation Meta: Conversation metadata does not exist (need to call POST first to create)"
            )
        else:
            print(
                f"⚠️  Patch Conversation Meta failed: {response.get('message', 'Unknown error')}"
            )

        return status_code, response

    def run_all_tests(self, test_method: str = "all", except_test_methods: str = None):
        """
        Run tests

        Args:
            test_method: Specify test method to run, options:
                - all: Run all tests
                - memorize: Test storing conversation memory
                - fetch_episodic: Test fetching episodic memory
                - fetch_event_log: Test fetching event log
                - fetch_profile: Test fetching user profile
                - search_keyword: Test keyword search
                - search_vector: Test vector search
                - search_hybrid: Test hybrid search
                - save_meta: Test saving conversation metadata
                - patch_meta: Test updating conversation metadata
            except_test_methods: Specify test methods to exclude (comma-separated), e.g.: "memorize,fetch_episodic"
                When specified, run all tests except these methods
        """
        print("\n" + "=" * 80)
        print("  Starting Memory Controller API Tests")
        print("=" * 80)
        print(f"  API Address: {self.base_url}")
        print(f"  Test User: {self.user_id}")
        print(f"  Test Group: {self.group_id}")
        print(f"  Organization ID: {self.organization_id}")
        print(f"  Space ID: {self.space_id}")
        print(f"  Hash Key: {self.hash_key}")
        print(f"  Sync Mode: {self.sync_mode}")
        print(f"  Test Method: {test_method}")
        if except_test_methods:
            print(f"  Excluded Methods: {except_test_methods}")
        print("=" * 80)

        # First initialize database
        if not self.init_database():
            print("\n❌ Database initialization failed, terminating tests")
            return

        # Define test method mapping
        test_methods = {
            "memorize": self.test_memorize_single_message,
            "fetch_episodic": self.test_fetch_episodic,
            "fetch_foresight": self.test_fetch_foresight,
            "fetch_event_log": self.test_fetch_event_log,
            "search_keyword": self.test_search_memories_keyword,
            "search_vector": self.test_search_memories_vector,
            "search_hybrid": self.test_search_memories_hybrid,
            "save_meta": self.test_save_conversation_meta,
            "patch_meta": self.test_patch_conversation_meta,
        }

        # Parse excluded test methods list
        excluded_methods = set()
        if except_test_methods:
            excluded_list = [m.strip() for m in except_test_methods.split(",")]
            for method_name in excluded_list:
                if method_name not in test_methods:
                    print(
                        f"\n⚠️  Warning: Unknown test method '{method_name}', will be ignored"
                    )
                else:
                    excluded_methods.add(method_name)

        # Execute tests
        try:
            if except_test_methods:
                # except-test-method mode: Run all tests except specified ones
                methods_to_run = [
                    (name, method)
                    for name, method in test_methods.items()
                    if name not in excluded_methods
                ]
                if not methods_to_run:
                    print("\n⚠️  No test methods to run (all methods excluded)")
                    return

                print(
                    f"\n📋 Will run {len(methods_to_run)} test methods (excluded {len(excluded_methods)} methods)"
                )
                for name, method in methods_to_run:
                    method()
            elif test_method == "all":
                # Run all tests
                for method in test_methods.values():
                    method()
            elif test_method in test_methods:
                # Run specified single test
                test_methods[test_method]()
            else:
                print(f"\n❌ Unknown test method: {test_method}")
                return
        except AssertionError as e:
            print(f"\n❌ Test failed: {e}")
            raise
        except Exception as e:  # noqa: BLE001
            print(f"\n❌ Test exception: {e}")
            raise

        # Tests completed
        self.print_section("Tests Completed")
        if except_test_methods:
            print(f"\n✅ Completed all tests except [{except_test_methods}]!")
        elif test_method == "all":
            print("\n✅ All interface structure validations passed!")
        else:
            print(f"\n✅ Test method [{test_method}] validation passed!")
        print(
            "💡 Tip: If an interface fails, check if input/output structure has changed\n"
        )


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Memory Controller API Test Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage Examples:
  # Test local service with default configuration
  python tests/test_memory_controller.py

  # Specify API address
  python tests/test_memory_controller.py --base-url http://localhost:1995

  # Specify API address and test user
  python tests/test_memory_controller.py --base-url http://dev-server:1995 --user-id test_user_123

  # Specify tenant information
  python tests/test_memory_controller.py --organization-id my_org --space-id my_space

  # Test a specific method
  python tests/test_memory_controller.py --test-method memorize
  python tests/test_memory_controller.py --test-method fetch_episodic
  python tests/test_memory_controller.py --test-method fetch_event_log
  python tests/test_memory_controller.py --test-method search_keyword

  # Test all methods except certain ones (parameters separated by commas)
  python tests/test_memory_controller.py --except-test-method memorize
  python tests/test_memory_controller.py --except-test-method memorize,fetch_episodic
  python tests/test_memory_controller.py --except-test-method save_meta,patch_meta

  # Disable sync mode (use background mode)
  python tests/test_memory_controller.py --sync-mode false

  # Specify API Key for authentication
  python tests/test_memory_controller.py --hash-key your_hash_key_here

  # Specify all parameters
  python tests/test_memory_controller.py --base-url http://dev-server:1995 --user-id test_user --group-id test_group --organization-id my_org --space-id my_space --hash-key your_hash_key --timeout 60 --sync-mode true
        """,
    )

    parser.add_argument(
        "--base-url",
        default="http://localhost:1995",
        help="API base URL (default: http://localhost:1995)",
    )

    parser.add_argument(
        "--user-id", default=None, help="Test user ID (default: randomly generated)"
    )

    parser.add_argument(
        "--group-id", default=None, help="Test group ID (default: randomly generated)"
    )

    parser.add_argument(
        "--organization-id",
        default=None,
        help=f"Organization ID (default: {MemoryControllerTester.DEFAULT_ORGANIZATION_ID})",
    )

    parser.add_argument(
        "--space-id",
        default=None,
        help=f"Space ID (default: {MemoryControllerTester.DEFAULT_SPACE_ID})",
    )

    parser.add_argument(
        "--hash-key",
        default=None,
        help=f"Hash key for authentication (default: {MemoryControllerTester.DEFAULT_HASH_KEY})",
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=180,
        help="Request timeout in seconds (default: 180)",
    )

    parser.add_argument(
        "--test-method",
        default="all",
        choices=[
            "all",
            "memorize",
            "fetch_episodic",
            "fetch_foresight",
            "fetch_event_log",
            "search_keyword",
            "search_vector",
            "search_hybrid",
            "save_meta",
            "patch_meta",
        ],
        help="Specify test method to run (default: all runs all tests)",
    )

    parser.add_argument(
        "--except-test-method",
        default=None,
        help="Specify test methods to exclude (comma-separated), runs all tests except these. Example: --except-test-method memorize,fetch_episodic",
    )

    parser.add_argument(
        "--sync-mode",
        type=lambda x: x.lower() in ("true", "1", "yes"),
        default=True,
        help="Whether to enable sync mode (default: true). Set to true to disable background mode, ensuring sequential test effectiveness; set to false to use background mode",
    )

    return parser.parse_args()


def main():
    """Main function"""
    # Parse command line arguments
    args = parse_args()

    # Check parameter conflict: cannot specify both --test-method and --except-test-method
    if args.test_method != "all" and args.except_test_method:
        print("❌ Error: Cannot use both --test-method and --except-test-method")
        print("   Please choose one:")
        print("   - Use --test-method to specify a single test to run")
        print(
            "   - Use --except-test-method to specify tests to exclude (run all others)"
        )
        return

    # If user_id not provided, generate randomly
    user_id = args.user_id if args.user_id else f"user_{uuid.uuid4().hex[:12]}"

    # If group_id not provided, generate randomly
    group_id = args.group_id if args.group_id else f"group_{uuid.uuid4().hex[:12]}"

    # Use default values for organization_id and space_id (if not provided)
    organization_id = args.organization_id
    space_id = args.space_id

    # Output used ID information
    if not args.user_id:
        print(f"⚠️  --user-id not provided, auto-generated: {user_id}")
    if not args.group_id:
        print(f"⚠️  --group-id not provided, auto-generated: {group_id}")
    if not args.organization_id:
        print(
            f"⚠️  --organization-id not provided, using default: {MemoryControllerTester.DEFAULT_ORGANIZATION_ID}"
        )
    if not args.space_id:
        print(
            f"⚠️  --space-id not provided, using default: {MemoryControllerTester.DEFAULT_SPACE_ID}"
        )
    if not args.hash_key:
        print(
            f"⚠️  --hash-key not provided, using default: {MemoryControllerTester.DEFAULT_HASH_KEY}"
        )

    # Create tester instance
    tester = MemoryControllerTester(
        base_url=args.base_url,
        user_id=user_id,
        group_id=group_id,
        organization_id=organization_id,
        space_id=space_id,
        hash_key=args.hash_key,
        timeout=args.timeout,
        sync_mode=args.sync_mode,
    )

    # Run tests (decide to run all, single, or exclude certain tests based on parameters)
    tester.run_all_tests(
        test_method=args.test_method, except_test_methods=args.except_test_method
    )


if __name__ == "__main__":
    main()
