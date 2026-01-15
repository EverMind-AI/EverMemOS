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
    
    # Test by category (batch execution)
    python tests/test_memory_controller.py --test-method fetch      # Run all fetch tests
    python tests/test_memory_controller.py --test-method retrieve   # Run all retrieve/search tests
    python tests/test_memory_controller.py --test-method search     # Same as retrieve
    python tests/test_memory_controller.py --test-method memorize   # Run memorization tests
    python tests/test_memory_controller.py --test-method meta       # Run metadata tests
    python tests/test_memory_controller.py --test-method delete     # Run delete tests
    
    # Test a specific method
    python tests/test_memory_controller.py --test-method fetch_episodic
    python tests/test_memory_controller.py --test-method fetch_event_log
    python tests/test_memory_controller.py --test-method fetch_group_filter
    python tests/test_memory_controller.py --test-method fetch_time_range
    python tests/test_memory_controller.py --test-method fetch_combined_filters
    python tests/test_memory_controller.py --test-method fetch_all_types
    python tests/test_memory_controller.py --test-method search_keyword
    python tests/test_memory_controller.py --test-method delete_memories
    
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

    def call_delete_api(self, endpoint: str, data: dict = None, params: dict = None):
        """
        Call DELETE API and print results

        Args:
            endpoint: API endpoint
            data: Request data (placed in body, preferred method)
            params: Query parameters (for compatibility)

        Returns:
            (status_code, response_json)
        """
        url = f"{self.base_url}{self.api_prefix}{endpoint}"
        headers = self.get_tenant_headers()

        # Merge sync mode parameters
        merged_params = self._get_sync_mode_params()
        if params:
            merged_params.update(params)

        print(f"\n📍 URL: DELETE {url}")
        if merged_params:
            print("📤 Query Parameters:")
            print(json.dumps(merged_params, indent=2, ensure_ascii=False))
        if data:
            print("📤 Request Data (Body):")
            print(json.dumps(data, indent=2, ensure_ascii=False))

        try:
            response = requests.delete(
                url,
                json=data,
                params=merged_params,
                headers=headers,
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
                "role": "assistant",
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
                "role": "assistant",
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
                "role": "user",
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
        """Test 2: GET /api/v1/memories - Fetch user episodic memory (episodic_memory type, pass parameters via body)

        Tests multiple scenarios:
        1. Only user_id (group_id NOT provided in request)
        2. user_id + group_id=None (explicitly null)
        3. user_id + group_id="" (explicitly empty string)
        4. user_id + group_id both have valid values
        5. user_id="__all__" + valid group_id
        """
        self.print_section("Test 2: GET /api/v1/memories - Fetch User Episodic Memory")

        # Scenario 1: Only user_id, group_id NOT provided (parameter doesn't exist)
        print("\n--- Scenario 1: Only user_id (group_id NOT provided) ---")
        data = {
            "user_id": self.user_id,
            "memory_type": "episodic_memory",
            "limit": 10,
            "offset": 0,
            # group_id is NOT in the request at all
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
                f"✅ Scenario 1 successful, returned {result['total_count']} episodic memories"
            )
        else:
            print(
                f"✅ Scenario 1 successful, returned {result['total_count']} episodic memories"
            )

        # Scenario 2: user_id + group_id=None (explicitly null)
        print("\n--- Scenario 2: user_id + group_id=None (explicitly null) ---")
        data = {
            "user_id": self.user_id,
            "group_id": None,  # Explicitly set to None
            "memory_type": "episodic_memory",
            "limit": 10,
            "offset": 0,
        }

        status_code, response = self.call_get_with_body_api("", data)

        assert status_code == 200, f"Status code should be 200, actual: {status_code}"
        assert response.get("status") == "ok", f"Status should be ok"
        result = response["result"]

        # Validate that returned memories have null or empty group_id
        if result["total_count"] > 0 and len(result["memories"]) > 0:
            for idx, memory in enumerate(result["memories"]):
                group_id_value = memory.get("group_id")
                assert group_id_value in (
                    None,
                    "",
                ), f"Memory {idx} group_id should be None or empty string, actual: {group_id_value}"
            print(
                f"✅ Scenario 2 successful, returned {result['total_count']} episodic memories with null/empty group_id"
            )
        else:
            print(
                f"✅ Scenario 2 successful, returned {result['total_count']} episodic memories"
            )

        # Scenario 3: user_id + group_id="" (explicitly empty string)
        print("\n--- Scenario 3: user_id + group_id='' (explicitly empty string) ---")
        data = {
            "user_id": self.user_id,
            "group_id": "",  # Explicitly set to empty string
            "memory_type": "episodic_memory",
            "limit": 10,
            "offset": 0,
        }

        status_code, response = self.call_get_with_body_api("", data)

        assert status_code == 200, f"Status code should be 200, actual: {status_code}"
        assert response.get("status") == "ok", f"Status should be ok"
        result = response["result"]

        # Validate that returned memories have null or empty group_id
        if result["total_count"] > 0 and len(result["memories"]) > 0:
            for idx, memory in enumerate(result["memories"]):
                group_id_value = memory.get("group_id")
                assert group_id_value in (
                    None,
                    "",
                ), f"Memory {idx} group_id should be None or empty string, actual: {group_id_value}"
            print(
                f"✅ Scenario 3 successful, returned {result['total_count']} episodic memories with null/empty group_id"
            )
        else:
            print(
                f"✅ Scenario 3 successful, returned {result['total_count']} episodic memories"
            )

        # Scenario 4: user_id + group_id both have valid values
        print("\n--- Scenario 4: user_id + group_id both have valid values ---")
        data = {
            "user_id": self.user_id,
            "group_id": self.group_id,
            "memory_type": "episodic_memory",
            "limit": 10,
            "offset": 0,
        }

        status_code, response = self.call_get_with_body_api("", data)

        assert status_code == 200, f"Status code should be 200, actual: {status_code}"
        assert response.get("status") == "ok", f"Status should be ok"
        result = response["result"]

        # Validate metadata includes both user_id and group_id
        metadata = result["metadata"]
        assert metadata.get("user_id") == self.user_id, "metadata user_id should match"
        assert (
            metadata.get("group_id") == self.group_id
        ), "metadata group_id should match"

        # Validate that returned memories have the requested group_id
        if result["total_count"] > 0 and len(result["memories"]) > 0:
            for idx, memory in enumerate(result["memories"]):
                assert (
                    memory.get("group_id") == self.group_id
                ), f"Memory {idx} group_id should be {self.group_id}, actual: {memory.get('group_id')}"
                assert (
                    memory.get("user_id") == self.user_id
                ), f"Memory {idx} user_id should be {self.user_id}, actual: {memory.get('user_id')}"
            print(
                f"✅ Scenario 4 successful, returned {result['total_count']} episodic memories with matching filters"
            )
        else:
            print(
                f"✅ Scenario 4 successful, returned {result['total_count']} episodic memories"
            )

        # Scenario 5: user_id="__all__" + valid group_id
        print("\n--- Scenario 5: user_id='__all__' + valid group_id ---")
        data = {
            "user_id": "__all__",
            "group_id": self.group_id,
            "memory_type": "episodic_memory",
            "limit": 10,
            "offset": 0,
        }

        status_code, response = self.call_get_with_body_api("", data)

        assert status_code == 200, f"Status code should be 200, actual: {status_code}"
        assert response.get("status") == "ok", f"Status should be ok"
        result = response["result"]

        # Validate metadata includes group_id
        metadata = result["metadata"]
        assert (
            metadata.get("group_id") == self.group_id
        ), "metadata group_id should match"

        # Validate that returned memories have the requested group_id
        if result["total_count"] > 0 and len(result["memories"]) > 0:
            for idx, memory in enumerate(result["memories"]):
                assert (
                    memory.get("group_id") == self.group_id
                ), f"Memory {idx} group_id should be {self.group_id}, actual: {memory.get('group_id')}"
            print(
                f"✅ Scenario 5 successful, returned {result['total_count']} episodic memories with group_id={self.group_id}"
            )
        else:
            print(
                f"✅ Scenario 5 successful, returned {result['total_count']} episodic memories"
            )

        return status_code, response

    def test_fetch_foresight(self):
        """Test 3: GET /api/v1/memories - Fetch foresight (foresight type, pass parameters via body)

        Tests multiple scenarios:
        1. Only user_id (group_id NOT provided in request)
        2. user_id + group_id=None (explicitly null)
        3. user_id + group_id="" (explicitly empty string)
        4. user_id + group_id both have valid values
        5. user_id="__all__" + valid group_id
        """
        self.print_section("Test 3: GET /api/v1/memories - Fetch Foresight")

        # Scenario 1: Only user_id, group_id NOT provided (parameter doesn't exist)
        print("\n--- Scenario 1: Only user_id (group_id NOT provided) ---")
        data = {
            "user_id": self.user_id,
            "memory_type": "foresight",
            "limit": 10,
            "offset": 0,
            # group_id is NOT in the request at all
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
                    "parent_type" in memory
                ), f"Memory {idx} should contain parent_type"
                assert "parent_id" in memory, f"Memory {idx} should contain parent_id"
                # Foresight user_id may be None (group scenario), so not enforced

            print(
                f"✅ Scenario 1 successful, returned {result['total_count']} foresights"
            )
        else:
            print(
                f"✅ Scenario 1 successful, returned {result['total_count']} foresights"
            )

        # Scenario 2: user_id + group_id=None (explicitly null)
        print("\n--- Scenario 2: user_id + group_id=None (explicitly null) ---")
        data = {
            "user_id": self.user_id,
            "group_id": None,  # Explicitly set to None
            "memory_type": "foresight",
            "limit": 10,
            "offset": 0,
        }

        status_code, response = self.call_get_with_body_api("", data)

        assert status_code == 200, f"Status code should be 200, actual: {status_code}"
        assert response.get("status") == "ok", f"Status should be ok"
        result = response["result"]

        # Validate that returned memories have null or empty group_id
        if result["total_count"] > 0 and len(result["memories"]) > 0:
            for idx, memory in enumerate(result["memories"]):
                group_id_value = memory.get("group_id")
                assert group_id_value in (
                    None,
                    "",
                ), f"Memory {idx} group_id should be None or empty string, actual: {group_id_value}"
            print(
                f"✅ Scenario 2 successful, returned {result['total_count']} foresights with null/empty group_id"
            )
        else:
            print(
                f"✅ Scenario 2 successful, returned {result['total_count']} foresights"
            )

        # Scenario 3: user_id + group_id="" (explicitly empty string)
        print("\n--- Scenario 3: user_id + group_id='' (explicitly empty string) ---")
        data = {
            "user_id": self.user_id,
            "group_id": "",  # Explicitly set to empty string
            "memory_type": "foresight",
            "limit": 10,
            "offset": 0,
        }

        status_code, response = self.call_get_with_body_api("", data)

        assert status_code == 200, f"Status code should be 200, actual: {status_code}"
        assert response.get("status") == "ok", f"Status should be ok"
        result = response["result"]

        # Validate that returned memories have null or empty group_id
        if result["total_count"] > 0 and len(result["memories"]) > 0:
            for idx, memory in enumerate(result["memories"]):
                group_id_value = memory.get("group_id")
                assert group_id_value in (
                    None,
                    "",
                ), f"Memory {idx} group_id should be None or empty string, actual: {group_id_value}"
            print(
                f"✅ Scenario 3 successful, returned {result['total_count']} foresights with null/empty group_id"
            )
        else:
            print(
                f"✅ Scenario 3 successful, returned {result['total_count']} foresights"
            )

        # Scenario 4: user_id + group_id both have valid values
        print("\n--- Scenario 4: user_id + group_id both have valid values ---")
        data = {
            "user_id": self.user_id,
            "group_id": self.group_id,
            "memory_type": "foresight",
            "limit": 10,
            "offset": 0,
        }

        status_code, response = self.call_get_with_body_api("", data)

        assert status_code == 200, f"Status code should be 200, actual: {status_code}"
        assert response.get("status") == "ok", f"Status should be ok"
        result = response["result"]

        # Validate metadata includes both user_id and group_id
        metadata = result["metadata"]
        assert metadata.get("user_id") == self.user_id, "metadata user_id should match"
        assert (
            metadata.get("group_id") == self.group_id
        ), "metadata group_id should match"

        # Validate that returned memories have the requested group_id
        # Note: foresight user_id may be None in some cases, so only validate group_id
        if result["total_count"] > 0 and len(result["memories"]) > 0:
            for idx, memory in enumerate(result["memories"]):
                assert (
                    memory.get("group_id") == self.group_id
                ), f"Memory {idx} group_id should be {self.group_id}, actual: {memory.get('group_id')}"
            print(
                f"✅ Scenario 4 successful, returned {result['total_count']} foresights with group_id={self.group_id}"
            )
        else:
            print(
                f"✅ Scenario 4 successful, returned {result['total_count']} foresights"
            )

        # Scenario 5: user_id="__all__" + valid group_id
        print("\n--- Scenario 5: user_id='__all__' + valid group_id ---")
        data = {
            "user_id": "__all__",
            "group_id": self.group_id,
            "memory_type": "foresight",
            "limit": 10,
            "offset": 0,
        }

        status_code, response = self.call_get_with_body_api("", data)

        assert status_code == 200, f"Status code should be 200, actual: {status_code}"
        assert response.get("status") == "ok", f"Status should be ok"
        result = response["result"]

        # Validate metadata includes group_id
        metadata = result["metadata"]
        assert (
            metadata.get("group_id") == self.group_id
        ), "metadata group_id should match"

        # Validate that returned memories have the requested group_id
        if result["total_count"] > 0 and len(result["memories"]) > 0:
            for idx, memory in enumerate(result["memories"]):
                assert (
                    memory.get("group_id") == self.group_id
                ), f"Memory {idx} group_id should be {self.group_id}, actual: {memory.get('group_id')}"
            print(
                f"✅ Scenario 5 successful, returned {result['total_count']} foresights with group_id={self.group_id}"
            )
        else:
            print(
                f"✅ Scenario 5 successful, returned {result['total_count']} foresights"
            )

        return status_code, response

    def test_fetch_event_log(self):
        """Test 4: GET /api/v1/memories - Fetch user event log (event_log type, pass parameters via body)

        Tests multiple scenarios:
        1. Only user_id (group_id NOT provided in request)
        2. user_id + group_id=None (explicitly null)
        3. user_id + group_id="" (explicitly empty string)
        4. user_id + group_id both have valid values
        5. user_id="__all__" + valid group_id
        """
        self.print_section("Test 4: GET /api/v1/memories - Fetch User Event Log")

        # Scenario 1: Only user_id, group_id NOT provided (parameter doesn't exist)
        print("\n--- Scenario 1: Only user_id (group_id NOT provided) ---")
        data = {
            "user_id": self.user_id,
            "memory_type": "event_log",
            "limit": 10,
            "offset": 0,
            # group_id is NOT in the request at all
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
                f"✅ Scenario 1 successful, returned {result['total_count']} event logs"
            )
        else:
            print(
                f"✅ Scenario 1 successful, returned {result['total_count']} event logs"
            )

        # Scenario 2: user_id + group_id=None (explicitly null)
        print("\n--- Scenario 2: user_id + group_id=None (explicitly null) ---")
        data = {
            "user_id": self.user_id,
            "group_id": None,  # Explicitly set to None
            "memory_type": "event_log",
            "limit": 10,
            "offset": 0,
        }

        status_code, response = self.call_get_with_body_api("", data)

        assert status_code == 200, f"Status code should be 200, actual: {status_code}"
        assert response.get("status") == "ok", f"Status should be ok"
        result = response["result"]

        # Validate that returned memories have null or empty group_id
        if result["total_count"] > 0 and len(result["memories"]) > 0:
            for idx, memory in enumerate(result["memories"]):
                group_id_value = memory.get("group_id")
                assert group_id_value in (
                    None,
                    "",
                ), f"Memory {idx} group_id should be None or empty string, actual: {group_id_value}"
            print(
                f"✅ Scenario 2 successful, returned {result['total_count']} event logs with null/empty group_id"
            )
        else:
            print(
                f"✅ Scenario 2 successful, returned {result['total_count']} event logs"
            )

        # Scenario 3: user_id + group_id="" (explicitly empty string)
        print("\n--- Scenario 3: user_id + group_id='' (explicitly empty string) ---")
        data = {
            "user_id": self.user_id,
            "group_id": "",  # Explicitly set to empty string
            "memory_type": "event_log",
            "limit": 10,
            "offset": 0,
        }

        status_code, response = self.call_get_with_body_api("", data)

        assert status_code == 200, f"Status code should be 200, actual: {status_code}"
        assert response.get("status") == "ok", f"Status should be ok"
        result = response["result"]

        # Validate that returned memories have null or empty group_id
        if result["total_count"] > 0 and len(result["memories"]) > 0:
            for idx, memory in enumerate(result["memories"]):
                group_id_value = memory.get("group_id")
                assert group_id_value in (
                    None,
                    "",
                ), f"Memory {idx} group_id should be None or empty string, actual: {group_id_value}"
            print(
                f"✅ Scenario 3 successful, returned {result['total_count']} event logs with null/empty group_id"
            )
        else:
            print(
                f"✅ Scenario 3 successful, returned {result['total_count']} event logs"
            )

        # Scenario 4: user_id + group_id both have valid values
        print("\n--- Scenario 4: user_id + group_id both have valid values ---")
        data = {
            "user_id": self.user_id,
            "group_id": self.group_id,
            "memory_type": "event_log",
            "limit": 10,
            "offset": 0,
        }

        status_code, response = self.call_get_with_body_api("", data)

        assert status_code == 200, f"Status code should be 200, actual: {status_code}"
        assert response.get("status") == "ok", f"Status should be ok"
        result = response["result"]

        # Validate metadata includes both user_id and group_id
        metadata = result["metadata"]
        assert metadata.get("user_id") == self.user_id, "metadata user_id should match"
        assert (
            metadata.get("group_id") == self.group_id
        ), "metadata group_id should match"

        # Validate that returned memories have the requested group_id
        if result["total_count"] > 0 and len(result["memories"]) > 0:
            for idx, memory in enumerate(result["memories"]):
                assert (
                    memory.get("group_id") == self.group_id
                ), f"Memory {idx} group_id should be {self.group_id}, actual: {memory.get('group_id')}"
                assert (
                    memory.get("user_id") == self.user_id
                ), f"Memory {idx} user_id should be {self.user_id}, actual: {memory.get('user_id')}"
            print(
                f"✅ Scenario 4 successful, returned {result['total_count']} event logs with matching filters"
            )
        else:
            print(
                f"✅ Scenario 4 successful, returned {result['total_count']} event logs"
            )

        # Scenario 5: user_id="__all__" + valid group_id
        print("\n--- Scenario 5: user_id='__all__' + valid group_id ---")
        data = {
            "user_id": "__all__",
            "group_id": self.group_id,
            "memory_type": "event_log",
            "limit": 10,
            "offset": 0,
        }

        status_code, response = self.call_get_with_body_api("", data)

        assert status_code == 200, f"Status code should be 200, actual: {status_code}"
        assert response.get("status") == "ok", f"Status should be ok"
        result = response["result"]

        # Validate metadata includes group_id
        metadata = result["metadata"]
        assert (
            metadata.get("group_id") == self.group_id
        ), "metadata group_id should match"

        # Validate that returned memories have the requested group_id
        if result["total_count"] > 0 and len(result["memories"]) > 0:
            for idx, memory in enumerate(result["memories"]):
                assert (
                    memory.get("group_id") == self.group_id
                ), f"Memory {idx} group_id should be {self.group_id}, actual: {memory.get('group_id')}"
            print(
                f"✅ Scenario 5 successful, returned {result['total_count']} event logs with group_id={self.group_id}"
            )
        else:
            print(
                f"✅ Scenario 5 successful, returned {result['total_count']} event logs"
            )

        return status_code, response

    def test_fetch_with_group_filter(self):
        """Test: GET /api/v1/memories - Fetch memories with group_id filter"""
        self.print_section("Test: GET /api/v1/memories - Fetch with group_id Filter")

        # Test different memory types with group_id filter
        memory_types = ["episodic_memory", "event_log", "foresight"]

        for memory_type in memory_types:
            print(f"\n--- Testing memory_type: {memory_type} with group_id ---")

            data = {
                "user_id": "__all__",  # Query all users
                "group_id": self.group_id,  # Filter by group
                "memory_type": memory_type,
                "limit": 10,
                "offset": 0,
            }

            status_code, response = self.call_get_with_body_api("", data)

            # Assert: Validate response structure
            assert (
                status_code == 200
            ), f"Status code should be 200, actual: {status_code}"
            assert response.get("status") == "ok", f"Status should be ok"
            assert "result" in response, "Response should contain result field"

            result = response["result"]
            assert "memories" in result, "result should contain memories field"
            assert "total_count" in result, "result should contain total_count field"
            assert "metadata" in result, "result should contain metadata field"

            # Validate metadata
            metadata = result["metadata"]
            assert (
                metadata.get("group_id") == self.group_id
            ), "metadata group_id should match"

            print(
                f"✅ {memory_type} with group_id filter: {result['total_count']} records"
            )

        print(f"\n✅ Group Filter Test Completed")
        return status_code, response

    def test_fetch_with_time_range(self):
        """Test: GET /api/v1/memories - Fetch memories with time range filter"""
        self.print_section("Test: GET /api/v1/memories - Fetch with Time Range Filter")

        now = datetime.now(ZoneInfo("UTC"))
        start_time = (now - timedelta(days=30)).isoformat()
        end_time = now.isoformat()

        # Test different memory types with time range
        memory_types = ["episodic_memory", "event_log", "foresight"]

        for memory_type in memory_types:
            print(f"\n--- Testing memory_type: {memory_type} with time_range ---")

            data = {
                "user_id": self.user_id,
                "memory_type": memory_type,
                "start_time": start_time,
                "end_time": end_time,
                "limit": 10,
                "offset": 0,
            }

            status_code, response = self.call_get_with_body_api("", data)

            # Assert: Validate response structure
            assert (
                status_code == 200
            ), f"Status code should be 200, actual: {status_code}"
            assert response.get("status") == "ok", f"Status should be ok"
            assert "result" in response, "Response should contain result field"

            result = response["result"]
            assert "memories" in result, "result should contain memories field"
            assert "total_count" in result, "result should contain total_count field"

            # Validate time range in metadata
            metadata = result["metadata"]
            if "start_time" in metadata:
                assert (
                    metadata.get("start_time") == start_time
                ), "metadata start_time should match"
            if "end_time" in metadata:
                assert (
                    metadata.get("end_time") == end_time
                ), "metadata end_time should match"

            print(
                f"✅ {memory_type} with time_range [{start_time[:10]} to {end_time[:10]}]: {result['total_count']} records"
            )

        print(f"\n✅ Time Range Filter Test Completed")
        return status_code, response

    def test_fetch_with_combined_filters(self):
        """Test: GET /api/v1/memories - Fetch memories with combined filters (user_id + group_id + time_range)"""
        self.print_section("Test: GET /api/v1/memories - Fetch with Combined Filters")

        now = datetime.now(ZoneInfo("UTC"))
        start_time = (now - timedelta(days=7)).isoformat()
        end_time = now.isoformat()

        # Test episodic_memory with all filters
        print("\n--- Testing episodic_memory with user_id + group_id + time_range ---")

        data = {
            "user_id": self.user_id,
            "group_id": self.group_id,
            "memory_type": "episodic_memory",
            "start_time": start_time,
            "end_time": end_time,
            "limit": 10,
            "offset": 0,
        }

        status_code, response = self.call_get_with_body_api("", data)

        # Assert: Validate response structure
        assert status_code == 200, f"Status code should be 200, actual: {status_code}"
        assert response.get("status") == "ok", f"Status should be ok"
        assert "result" in response, "Response should contain result field"

        result = response["result"]
        assert "memories" in result, "result should contain memories field"
        assert "total_count" in result, "result should contain total_count field"

        # Validate all filters are reflected in metadata
        metadata = result["metadata"]
        assert metadata.get("user_id") == self.user_id, "metadata user_id should match"
        assert (
            metadata.get("group_id") == self.group_id
        ), "metadata group_id should match"

        print(
            f"✅ Combined filters (user + group + time): {result['total_count']} records"
        )
        print(f"\n✅ Combined Filters Test Completed")
        return status_code, response

    def test_fetch_profile_memory(self):
        """Test: GET /api/v1/memories - Fetch user profile memory

        Tests multiple scenarios:
        1. Only user_id (group_id NOT provided)
        2. user_id + group_id=None (explicitly null)
        3. user_id + group_id="" (explicitly empty string)
        4. user_id + group_id both have valid values
        5. user_id="__all__" + valid group_id
        """
        self.print_section("Test: GET /api/v1/memories - Fetch User Profile Memory")

        # Scenario 1: Only user_id, group_id NOT provided
        print("\n--- Scenario 1: Only user_id (group_id NOT provided) ---")
        data = {
            "user_id": self.user_id,
            "memory_type": "profile",
            "limit": 10,
            "offset": 0,
        }

        status_code, response = self.call_get_with_body_api("", data)
        assert status_code == 200, f"Status code should be 200, actual: {status_code}"
        assert response.get("status") == "ok", f"Status should be ok"

        result = response["result"]
        print(f"✅ Scenario 1 successful, returned {result['total_count']} profiles")

        # Scenario 2: user_id + group_id=None (explicitly null)
        print("\n--- Scenario 2: user_id + group_id=None (explicitly null) ---")
        data = {
            "user_id": self.user_id,
            "group_id": None,  # Explicitly set to None
            "memory_type": "profile",
            "limit": 10,
            "offset": 0,
        }

        status_code, response = self.call_get_with_body_api("", data)
        assert status_code == 200, f"Status code should be 200, actual: {status_code}"
        assert response.get("status") == "ok", f"Status should be ok"

        result = response["result"]

        # Validate that returned memories have null or empty group_id
        if result["total_count"] > 0 and len(result["memories"]) > 0:
            for idx, memory in enumerate(result["memories"]):
                group_id_value = memory.get("group_id")
                assert group_id_value in (
                    None,
                    "",
                ), f"Memory {idx} group_id should be None or empty string, actual: {group_id_value}"
            print(
                f"✅ Scenario 2 successful, returned {result['total_count']} profiles with null/empty group_id"
            )
        else:
            print(
                f"✅ Scenario 2 successful, returned {result['total_count']} profiles"
            )

        # Scenario 3: user_id + group_id="" (explicitly empty string)
        print("\n--- Scenario 3: user_id + group_id='' (explicitly empty string) ---")
        data = {
            "user_id": self.user_id,
            "group_id": "",  # Explicitly set to empty string
            "memory_type": "profile",
            "limit": 10,
            "offset": 0,
        }

        status_code, response = self.call_get_with_body_api("", data)
        assert status_code == 200, f"Status code should be 200, actual: {status_code}"
        assert response.get("status") == "ok", f"Status should be ok"

        result = response["result"]

        # Validate that returned memories have null or empty group_id
        if result["total_count"] > 0 and len(result["memories"]) > 0:
            for idx, memory in enumerate(result["memories"]):
                group_id_value = memory.get("group_id")
                assert group_id_value in (
                    None,
                    "",
                ), f"Memory {idx} group_id should be None or empty string, actual: {group_id_value}"
            print(
                f"✅ Scenario 3 successful, returned {result['total_count']} profiles with null/empty group_id"
            )
        else:
            print(
                f"✅ Scenario 3 successful, returned {result['total_count']} profiles"
            )

        # Scenario 4: user_id + group_id both have valid values
        print("\n--- Scenario 4: user_id + group_id both have valid values ---")
        data = {
            "user_id": self.user_id,
            "group_id": self.group_id,
            "memory_type": "profile",
            "limit": 10,
            "offset": 0,
        }

        status_code, response = self.call_get_with_body_api("", data)
        assert status_code == 200, f"Status code should be 200, actual: {status_code}"
        assert response.get("status") == "ok", f"Status should be ok"

        result = response["result"]

        # Validate that returned memories have the requested group_id
        if result["total_count"] > 0 and len(result["memories"]) > 0:
            for idx, memory in enumerate(result["memories"]):
                assert (
                    memory.get("group_id") == self.group_id
                ), f"Memory {idx} group_id should be {self.group_id}, actual: {memory.get('group_id')}"
                assert (
                    memory.get("user_id") == self.user_id
                ), f"Memory {idx} user_id should be {self.user_id}, actual: {memory.get('user_id')}"
            print(
                f"✅ Scenario 4 successful, returned {result['total_count']} profiles with matching filters"
            )
        else:
            print(
                f"✅ Scenario 4 successful, returned {result['total_count']} profiles"
            )

        # Scenario 5: user_id="__all__" + valid group_id
        print("\n--- Scenario 5: user_id='__all__' + valid group_id ---")
        data = {
            "user_id": "__all__",
            "group_id": self.group_id,
            "memory_type": "profile",
            "limit": 10,
            "offset": 0,
        }

        status_code, response = self.call_get_with_body_api("", data)
        assert status_code == 200, f"Status code should be 200, actual: {status_code}"
        assert response.get("status") == "ok", f"Status should be ok"

        result = response["result"]

        # Validate that returned memories have the requested group_id
        if result["total_count"] > 0 and len(result["memories"]) > 0:
            for idx, memory in enumerate(result["memories"]):
                assert (
                    memory.get("group_id") == self.group_id
                ), f"Memory {idx} group_id should be {self.group_id}, actual: {memory.get('group_id')}"
            print(
                f"✅ Scenario 5 successful, returned {result['total_count']} profiles with group_id={self.group_id}"
            )
        else:
            print(
                f"✅ Scenario 5 successful, returned {result['total_count']} profiles"
            )

        print(f"\n✅ Profile Memory Test Completed")
        return status_code, response

    def test_fetch_all_memory_types(self):
        """Test: GET /api/v1/memories - Fetch all supported memory types

        Note: Only tests with user_id as some memory types don't support group_id filtering:
        - base_memory: Personal data, only supports specific user_id
        - preference: Personal preferences, only supports specific user_id
        - behavior_history: Personal behavior logs, only supports specific user_id
        - entity: Not supported with group_id in current implementation
        - relation: Not supported with group_id in current implementation

        Memory types that support group_id are tested separately in:
        - test_fetch_episodic() for episodic_memory
        - test_fetch_event_log() for event_log
        - test_fetch_foresight() for foresight
        - test_fetch_profile_memory() for profile
        """
        self.print_section("Test: GET /api/v1/memories - Fetch All Memory Types")

        # All supported memory types
        memory_types = [
            "episodic_memory",
            "event_log",
            "foresight",
            "profile",
            "base_memory",
            "preference",
            "entity",
            "relation",
            "behavior_history",
        ]

        results_summary = []

        for memory_type in memory_types:
            print(f"\n--- Testing memory_type: {memory_type} ---")

            data = {
                "user_id": self.user_id,
                "memory_type": memory_type,
                "limit": 10,
                "offset": 0,
            }

            status_code, response = self.call_get_with_body_api("", data)

            # Assert: Basic validation
            assert status_code == 200, f"Status code should be 200 for {memory_type}"
            assert (
                response.get("status") == "ok"
            ), f"Status should be ok for {memory_type}"
            assert (
                "result" in response
            ), f"Response should contain result field for {memory_type}"

            result = response["result"]
            assert (
                "memories" in result
            ), f"result should contain memories for {memory_type}"
            assert (
                "total_count" in result
            ), f"result should contain total_count for {memory_type}"

            results_summary.append(
                {"memory_type": memory_type, "total_count": result["total_count"]}
            )

            print(f"✅ {memory_type}: {result['total_count']} records")

        # Print summary
        print("\n" + "=" * 80)
        print("  Memory Types Summary")
        print("=" * 80)
        for item in results_summary:
            print(f"  {item['memory_type']:20s}: {item['total_count']:5d} records")
        print("=" * 80)

        print(f"\n✅ All Memory Types Test Completed")
        return status_code, response

    def _validate_search_response_structure(
        self, status_code, response, scenario_name: str
    ):
        """
        Validate the basic structure of a search response.

        Args:
            status_code: HTTP status code
            response: Response JSON
            scenario_name: Name of the test scenario for error messages

        Returns:
            result dict from response
        """
        assert (
            status_code == 200
        ), f"[{scenario_name}] Status code should be 200, actual: {status_code}"
        assert (
            response.get("status") == "ok"
        ), f"[{scenario_name}] Status should be ok, actual: {response.get('status')}"
        assert (
            "result" in response
        ), f"[{scenario_name}] Response should contain result field"

        result = response["result"]
        assert (
            "memories" in result
        ), f"[{scenario_name}] result should contain memories field"
        assert (
            "scores" in result
        ), f"[{scenario_name}] result should contain scores field"
        assert (
            "total_count" in result
        ), f"[{scenario_name}] result should contain total_count field"
        assert (
            "has_more" in result
        ), f"[{scenario_name}] result should contain has_more field"
        assert (
            "metadata" in result
        ), f"[{scenario_name}] result should contain metadata field"
        assert (
            "pending_messages" in result
        ), f"[{scenario_name}] result should contain pending_messages field"

        # Validate data types
        assert isinstance(
            result["memories"], list
        ), f"[{scenario_name}] memories should be a list"
        assert isinstance(
            result["scores"], list
        ), f"[{scenario_name}] scores should be a list"
        assert (
            result["total_count"] >= 0
        ), f"[{scenario_name}] total_count should be >= 0"
        assert isinstance(
            result["pending_messages"], list
        ), f"[{scenario_name}] pending_messages should be a list"

        # Validate pending_messages structure if not empty
        if len(result["pending_messages"]) > 0:
            for idx, msg in enumerate(result["pending_messages"]):
                assert isinstance(
                    msg, dict
                ), f"[{scenario_name}] pending_messages[{idx}] should be a dictionary"
                # Required fields
                assert (
                    "id" in msg
                ), f"[{scenario_name}] pending_messages[{idx}] should contain id field"
                assert (
                    "request_id" in msg
                ), f"[{scenario_name}] pending_messages[{idx}] should contain request_id field"
                # Optional fields validation (check type if present)
                if "message_id" in msg and msg["message_id"] is not None:
                    assert isinstance(
                        msg["message_id"], str
                    ), f"[{scenario_name}] pending_messages[{idx}].message_id should be string"
                if "content" in msg and msg["content"] is not None:
                    assert isinstance(
                        msg["content"], str
                    ), f"[{scenario_name}] pending_messages[{idx}].content should be string"
            print(
                f"    [{scenario_name}] Found {len(result['pending_messages'])} pending messages"
            )

        return result

    def _validate_search_memories_filter(
        self,
        result: dict,
        scenario_name: str,
        expected_user_id: str = None,
        expected_group_id: str = None,
        user_id_filter_type: str = "exact",  # "exact", "null_or_empty", "any"
        group_id_filter_type: str = "exact",  # "exact", "null_or_empty", "any"
    ):
        """
        Validate that search results match expected user_id/group_id filters.

        Search API returns memories grouped by group_id:
        {
            "memories": [
                {
                    "group_id_key": [
                        {
                            "memory_type": "episodic_memory",
                            "user_id": "user_123",
                            "group_id": "group_456",
                            "timestamp": "...",
                            "data": [...]
                        }
                    ]
                }
            ]
        }

        Args:
            result: Result dict from search response
            scenario_name: Name of the test scenario
            expected_user_id: Expected user_id value (for exact matching)
            expected_group_id: Expected group_id value (for exact matching)
            user_id_filter_type: How to validate user_id ("exact", "null_or_empty", "any")
            group_id_filter_type: How to validate group_id ("exact", "null_or_empty", "any")
        """
        memories_checked = 0

        if result["total_count"] > 0 and len(result["memories"]) > 0:
            # Iterate through each group's memories
            for group_idx, memory_group in enumerate(result["memories"]):
                assert isinstance(
                    memory_group, dict
                ), f"[{scenario_name}] Memory group {group_idx} should be a dictionary"

                # Iterate through memories within group (key is the group_id)
                for group_id_key, memory_list in memory_group.items():
                    assert isinstance(
                        group_id_key, str
                    ), f"[{scenario_name}] group_id_key should be string"
                    assert isinstance(
                        memory_list, list
                    ), f"[{scenario_name}] Memory list for group {group_id_key} should be a list"

                    # Validate group_id from the grouping key itself
                    if group_id_filter_type == "exact" and expected_group_id:
                        assert group_id_key == expected_group_id, (
                            f"[{scenario_name}] Memory group key should be {expected_group_id}, "
                            f"actual: {group_id_key}"
                        )
                    elif group_id_filter_type == "null_or_empty":
                        assert (
                            group_id_key in ("", "null", "None") or group_id_key == ""
                        ), (
                            f"[{scenario_name}] Memory group key should be empty/null, "
                            f"actual: {group_id_key}"
                        )

                    # Validate each memory in the group
                    for mem_idx, mem in enumerate(memory_list):
                        assert isinstance(
                            mem, dict
                        ), f"[{scenario_name}] Memory {mem_idx} should be a dictionary"
                        assert (
                            "memory_type" in mem
                        ), f"[{scenario_name}] Memory {mem_idx} should contain memory_type"

                        memories_checked += 1

                        # Validate user_id filter from the memory object
                        mem_user_id = mem.get("user_id")
                        if user_id_filter_type == "exact" and expected_user_id:
                            assert mem_user_id == expected_user_id, (
                                f"[{scenario_name}] Memory {mem_idx} user_id should be {expected_user_id}, "
                                f"actual: {mem_user_id}"
                            )
                        elif user_id_filter_type == "null_or_empty":
                            assert mem_user_id in (None, ""), (
                                f"[{scenario_name}] Memory {mem_idx} user_id should be None or empty, "
                                f"actual: {mem_user_id}"
                            )
                        # "any" means no validation needed for user_id

                        # Validate group_id filter from the memory object (if present)
                        mem_group_id = mem.get("group_id")
                        if group_id_filter_type == "exact" and expected_group_id:
                            # Also check the memory's own group_id field if it exists
                            if mem_group_id is not None:
                                assert mem_group_id == expected_group_id, (
                                    f"[{scenario_name}] Memory {mem_idx} group_id should be {expected_group_id}, "
                                    f"actual: {mem_group_id}"
                                )
                        elif group_id_filter_type == "null_or_empty":
                            if mem_group_id is not None:
                                assert mem_group_id in (None, ""), (
                                    f"[{scenario_name}] Memory {mem_idx} group_id should be None or empty, "
                                    f"actual: {mem_group_id}"
                                )
                        # "any" means no validation needed for group_id

        print(f"    [Debug] Checked {memories_checked} memories in {scenario_name}")

    def test_search_memories_keyword(self):
        """Test 5: GET /api/v1/memories/search - Keyword search (pass parameters via body)

        Tests multiple scenarios for user_id/group_id parameter behavior:
        Note: user_id and group_id cannot BOTH be MAGIC_ALL (not provided or "__all__")

        1. Neither user_id nor group_id provided - should return 400 error
        2. Only user_id provided (group_id NOT in request, query_all for group_id)
        3. user_id + group_id=None or "" (filter for null/empty group_id)
        4. user_id + group_id both have valid values (exact match)
        5. user_id="__all__" + valid group_id (query_all for user_id)
        6. user_id=None or "" + valid group_id (filter for null/empty user_id)
        """
        self.print_section("Test 5: GET /api/v1/memories/search - Keyword Search")

        # =================================================================
        # Scenario 1: Neither user_id nor group_id provided - should return 400 error
        # (user_id and group_id cannot both be MAGIC_ALL)
        # =================================================================
        print(
            "\n--- Scenario 1: Neither user_id nor group_id provided (should return 400) ---"
        )
        data = {
            "query": "coffee",
            "top_k": 10,
            "retrieve_method": "keyword",
            # user_id and group_id are NOT in the request at all
        }

        status_code, response = self.call_get_with_body_api("/search", data)

        # Should return 400 error because user_id and group_id cannot both be MAGIC_ALL
        assert (
            status_code == 400
        ), f"[Scenario 1] Status code should be 400, actual: {status_code}"
        assert (
            response.get("status") == "failed"
        ), f"[Scenario 1] Status should be failed"
        assert "user_id and group_id cannot both be MAGIC_ALL" in response.get(
            "message", ""
        ), f"[Scenario 1] Error message should mention the constraint, actual: {response.get('message')}"

        print(
            f"✅ Scenario 1 successful, correctly returned 400 error for invalid request"
        )

        # =================================================================
        # Scenario 2: Only user_id provided (group_id NOT in request)
        # =================================================================
        print(
            "\n--- Scenario 2: Only user_id provided (group_id NOT in request, query_all for group_id) ---"
        )
        data = {
            "user_id": self.user_id,
            "query": "coffee",
            "top_k": 10,
            "retrieve_method": "keyword",
            # group_id is NOT in the request at all
        }

        status_code, response = self.call_get_with_body_api("/search", data)
        result = self._validate_search_response_structure(
            status_code, response, "Scenario 2"
        )

        # Validate metadata
        metadata = result["metadata"]
        assert (
            metadata.get("user_id") == self.user_id
        ), "[Scenario 2] metadata user_id should match"

        # When group_id is not provided, should return memories from all groups for this user
        self._validate_search_memories_filter(
            result,
            "Scenario 2",
            expected_user_id=self.user_id,
            user_id_filter_type="exact",
            group_id_filter_type="any",
        )

        print(
            f"✅ Scenario 2 successful, returned {result['total_count']} groups of memories for user_id={self.user_id}"
        )

        # =================================================================
        # Scenario 3: user_id + group_id=None or "" (filter for null/empty group_id)
        # =================================================================
        print(
            "\n--- Scenario 3: user_id + group_id='' (filter for null/empty group_id) ---"
        )
        data = {
            "user_id": self.user_id,
            "group_id": "",  # Empty string, equivalent to None
            "query": "coffee",
            "top_k": 10,
            "retrieve_method": "keyword",
        }

        status_code, response = self.call_get_with_body_api("/search", data)
        result = self._validate_search_response_structure(
            status_code, response, "Scenario 3"
        )

        # Validate metadata
        metadata = result["metadata"]
        assert (
            metadata.get("user_id") == self.user_id
        ), "[Scenario 3] metadata user_id should match"

        # When group_id is "" or None, should only return memories with null/empty group_id
        self._validate_search_memories_filter(
            result,
            "Scenario 3",
            expected_user_id=self.user_id,
            user_id_filter_type="exact",
            group_id_filter_type="null_or_empty",
        )

        print(
            f"✅ Scenario 3 successful, returned {result['total_count']} groups of memories with null/empty group_id"
        )

        # =================================================================
        # Scenario 4: user_id + group_id both have valid values (exact match)
        # =================================================================
        print(
            "\n--- Scenario 4: user_id + group_id both have valid values (exact match) ---"
        )
        data = {
            "user_id": self.user_id,
            "group_id": self.group_id,
            "query": "coffee",
            "top_k": 10,
            "retrieve_method": "keyword",
        }

        status_code, response = self.call_get_with_body_api("/search", data)
        result = self._validate_search_response_structure(
            status_code, response, "Scenario 4"
        )

        # Validate metadata (user_id should be present)
        metadata = result["metadata"]
        assert (
            metadata.get("user_id") == self.user_id
        ), "[Scenario 4] metadata user_id should match"
        # Note: metadata.group_id may not be returned by backend, so we verify via actual memories

        # When both have valid values, should only return exact matches - THIS IS THE KEY VALIDATION
        self._validate_search_memories_filter(
            result,
            "Scenario 4",
            expected_user_id=self.user_id,
            expected_group_id=self.group_id,
            user_id_filter_type="exact",
            group_id_filter_type="exact",
        )

        print(
            f"✅ Scenario 4 successful, returned {result['total_count']} groups of memories with exact user_id and group_id"
        )

        # =================================================================
        # Scenario 5: user_id="__all__" + valid group_id (query_all for user_id)
        # =================================================================
        print(
            "\n--- Scenario 5: user_id='__all__' + valid group_id (query_all for user_id) ---"
        )
        data = {
            "user_id": "__all__",
            "group_id": self.group_id,
            "query": "coffee",
            "top_k": 10,
            "retrieve_method": "keyword",
        }

        status_code, response = self.call_get_with_body_api("/search", data)
        result = self._validate_search_response_structure(
            status_code, response, "Scenario 5"
        )

        # Note: metadata.group_id may not be returned by backend, so we verify via actual memories

        # When user_id is "__all__", should return memories from all users in this group - THIS IS THE KEY VALIDATION
        self._validate_search_memories_filter(
            result,
            "Scenario 5",
            expected_group_id=self.group_id,
            user_id_filter_type="any",
            group_id_filter_type="exact",
        )

        print(
            f"✅ Scenario 5 successful, returned {result['total_count']} groups of memories with group_id={self.group_id} (user_id=__all__)"
        )

        # =================================================================
        # Scenario 6: user_id=None or "" + valid group_id (filter for null/empty user_id)
        # =================================================================
        print(
            "\n--- Scenario 6: user_id='' + valid group_id (filter for null/empty user_id) ---"
        )
        data = {
            "user_id": "",  # Empty string, equivalent to None
            "group_id": self.group_id,
            "query": "coffee",
            "top_k": 10,
            "retrieve_method": "keyword",
        }

        status_code, response = self.call_get_with_body_api("/search", data)
        result = self._validate_search_response_structure(
            status_code, response, "Scenario 6"
        )

        # Note: metadata.group_id may not be returned by backend, so we verify via actual memories

        # When user_id is "" or None, should only return memories with null/empty user_id - THIS IS THE KEY VALIDATION
        self._validate_search_memories_filter(
            result,
            "Scenario 6",
            expected_group_id=self.group_id,
            user_id_filter_type="null_or_empty",
            group_id_filter_type="exact",
        )

        print(
            f"✅ Scenario 6 successful, returned {result['total_count']} groups of memories with null/empty user_id"
        )

        print(f"\n✅ All Keyword Search Scenarios Completed Successfully")
        return status_code, response

    def test_search_memories_vector(self):
        """Test 6: GET /api/v1/memories/search - Vector search (pass parameters via body)

        Tests multiple scenarios for user_id/group_id parameter behavior:
        Note: user_id and group_id cannot BOTH be MAGIC_ALL (not provided or "__all__")

        1. Neither user_id nor group_id provided - should return 400 error
        2. Only user_id provided (group_id NOT in request, query_all for group_id)
        3. user_id + group_id=None or "" (filter for null/empty group_id)
        4. user_id + group_id both have valid values (exact match)
        5. user_id="__all__" + valid group_id (query_all for user_id)
        6. user_id=None or "" + valid group_id (filter for null/empty user_id)
        """
        self.print_section("Test 6: GET /api/v1/memories/search - Vector Search")

        # =================================================================
        # Scenario 1: Neither user_id nor group_id provided - should return 400 error
        # =================================================================
        print(
            "\n--- Scenario 1: Neither user_id nor group_id provided (should return 400) ---"
        )
        data = {
            "query": "user's dietary preferences",
            "top_k": 10,
            "retrieve_method": "vector",
            # user_id and group_id are NOT in the request at all
        }

        status_code, response = self.call_get_with_body_api("/search", data)

        # Should return 400 error because user_id and group_id cannot both be MAGIC_ALL
        assert (
            status_code == 400
        ), f"[Scenario 1] Status code should be 400, actual: {status_code}"
        assert (
            response.get("status") == "failed"
        ), f"[Scenario 1] Status should be failed"
        assert "user_id and group_id cannot both be MAGIC_ALL" in response.get(
            "message", ""
        ), f"[Scenario 1] Error message should mention the constraint, actual: {response.get('message')}"

        print(
            f"✅ Scenario 1 successful, correctly returned 400 error for invalid request"
        )

        # =================================================================
        # Scenario 2: Only user_id provided (group_id NOT in request)
        # =================================================================
        print(
            "\n--- Scenario 2: Only user_id provided (group_id NOT in request, query_all for group_id) ---"
        )
        data = {
            "user_id": self.user_id,
            "query": "user's dietary preferences",
            "top_k": 10,
            "retrieve_method": "vector",
            # group_id is NOT in the request at all
        }

        status_code, response = self.call_get_with_body_api("/search", data)
        result = self._validate_search_response_structure(
            status_code, response, "Scenario 2"
        )

        # Validate metadata
        metadata = result["metadata"]
        assert (
            metadata.get("user_id") == self.user_id
        ), "[Scenario 2] metadata user_id should match"

        # Vector search should have importance_scores when there are results
        if result["total_count"] > 0:
            assert (
                "importance_scores" in result
            ), "[Scenario 2] Vector search result should contain importance_scores field"

        self._validate_search_memories_filter(
            result,
            "Scenario 2",
            expected_user_id=self.user_id,
            user_id_filter_type="exact",
            group_id_filter_type="any",
        )

        print(
            f"✅ Scenario 2 successful, returned {result['total_count']} groups of memories for user_id={self.user_id}"
        )

        # =================================================================
        # Scenario 3: user_id + group_id=None or "" (filter for null/empty group_id)
        # =================================================================
        print(
            "\n--- Scenario 3: user_id + group_id='' (filter for null/empty group_id) ---"
        )
        data = {
            "user_id": self.user_id,
            "group_id": "",  # Empty string, equivalent to None
            "query": "user's dietary preferences",
            "top_k": 10,
            "retrieve_method": "vector",
        }

        status_code, response = self.call_get_with_body_api("/search", data)
        result = self._validate_search_response_structure(
            status_code, response, "Scenario 3"
        )

        # Validate metadata
        metadata = result["metadata"]
        assert (
            metadata.get("user_id") == self.user_id
        ), "[Scenario 3] metadata user_id should match"

        self._validate_search_memories_filter(
            result,
            "Scenario 3",
            expected_user_id=self.user_id,
            user_id_filter_type="exact",
            group_id_filter_type="null_or_empty",
        )

        print(
            f"✅ Scenario 3 successful, returned {result['total_count']} groups of memories with null/empty group_id"
        )

        # =================================================================
        # Scenario 4: user_id + group_id both have valid values (exact match)
        # =================================================================
        print(
            "\n--- Scenario 4: user_id + group_id both have valid values (exact match) ---"
        )
        data = {
            "user_id": self.user_id,
            "group_id": self.group_id,
            "query": "user's dietary preferences",
            "top_k": 10,
            "retrieve_method": "vector",
        }

        status_code, response = self.call_get_with_body_api("/search", data)
        result = self._validate_search_response_structure(
            status_code, response, "Scenario 4"
        )

        # Validate metadata (user_id should be present)
        metadata = result["metadata"]
        assert (
            metadata.get("user_id") == self.user_id
        ), "[Scenario 4] metadata user_id should match"
        # Note: metadata.group_id may not be returned by backend, so we verify via actual memories

        # When both have valid values, should only return exact matches - THIS IS THE KEY VALIDATION
        self._validate_search_memories_filter(
            result,
            "Scenario 4",
            expected_user_id=self.user_id,
            expected_group_id=self.group_id,
            user_id_filter_type="exact",
            group_id_filter_type="exact",
        )

        print(
            f"✅ Scenario 4 successful, returned {result['total_count']} groups of memories with exact user_id and group_id"
        )

        # =================================================================
        # Scenario 5: user_id="__all__" + valid group_id (query_all for user_id)
        # =================================================================
        print(
            "\n--- Scenario 5: user_id='__all__' + valid group_id (query_all for user_id) ---"
        )
        data = {
            "user_id": "__all__",
            "group_id": self.group_id,
            "query": "user's dietary preferences",
            "top_k": 10,
            "retrieve_method": "vector",
        }

        status_code, response = self.call_get_with_body_api("/search", data)
        result = self._validate_search_response_structure(
            status_code, response, "Scenario 5"
        )

        # Note: metadata.group_id may not be returned by backend, so we verify via actual memories

        # When user_id is "__all__", should return memories from all users in this group - THIS IS THE KEY VALIDATION
        self._validate_search_memories_filter(
            result,
            "Scenario 5",
            expected_group_id=self.group_id,
            user_id_filter_type="any",
            group_id_filter_type="exact",
        )

        print(
            f"✅ Scenario 5 successful, returned {result['total_count']} groups of memories with group_id={self.group_id}"
        )

        # =================================================================
        # Scenario 6: user_id=None or "" + valid group_id (filter for null/empty user_id)
        # =================================================================
        print(
            "\n--- Scenario 6: user_id='' + valid group_id (filter for null/empty user_id) ---"
        )
        data = {
            "user_id": "",  # Empty string, equivalent to None
            "group_id": self.group_id,
            "query": "user's dietary preferences",
            "top_k": 10,
            "retrieve_method": "vector",
        }

        status_code, response = self.call_get_with_body_api("/search", data)
        result = self._validate_search_response_structure(
            status_code, response, "Scenario 6"
        )

        # Note: metadata.group_id may not be returned by backend, so we verify via actual memories

        # When user_id is "" or None, should only return memories with null/empty user_id - THIS IS THE KEY VALIDATION
        self._validate_search_memories_filter(
            result,
            "Scenario 6",
            expected_group_id=self.group_id,
            user_id_filter_type="null_or_empty",
            group_id_filter_type="exact",
        )

        print(
            f"✅ Scenario 6 successful, returned {result['total_count']} groups of memories with null/empty user_id"
        )

        print(f"\n✅ All Vector Search Scenarios Completed Successfully")
        return status_code, response

    def test_search_memories_hybrid(self):
        """Test 7: GET /api/v1/memories/search - Hybrid search (pass parameters via body)

        Tests multiple scenarios for user_id/group_id parameter behavior:
        Note: user_id and group_id cannot BOTH be MAGIC_ALL (not provided or "__all__")

        1. Neither user_id nor group_id provided - should return 400 error
        2. Only user_id provided (group_id NOT in request, query_all for group_id)
        3. user_id + group_id=None or "" (filter for null/empty group_id)
        4. user_id + group_id both have valid values (exact match)
        5. user_id="__all__" + valid group_id (query_all for user_id)
        6. user_id=None or "" + valid group_id (filter for null/empty user_id)
        """
        self.print_section("Test 7: GET /api/v1/memories/search - Hybrid Search")

        now = datetime.now(ZoneInfo("UTC"))
        start_time = (now - timedelta(days=60)).isoformat()
        end_time = now.isoformat()

        # =================================================================
        # Scenario 1: Neither user_id nor group_id provided - should return 400 error
        # =================================================================
        print(
            "\n--- Scenario 1: Neither user_id nor group_id provided (should return 400) ---"
        )
        data = {
            "query": "coffee preference",
            "top_k": 10,
            "retrieve_method": "hybrid",
            "start_time": start_time,
            "end_time": end_time,
            # user_id and group_id are NOT in the request at all
        }

        status_code, response = self.call_get_with_body_api("/search", data)

        # Should return 400 error because user_id and group_id cannot both be MAGIC_ALL
        assert (
            status_code == 400
        ), f"[Scenario 1] Status code should be 400, actual: {status_code}"
        assert (
            response.get("status") == "failed"
        ), f"[Scenario 1] Status should be failed"
        assert "user_id and group_id cannot both be MAGIC_ALL" in response.get(
            "message", ""
        ), f"[Scenario 1] Error message should mention the constraint, actual: {response.get('message')}"

        print(
            f"✅ Scenario 1 successful, correctly returned 400 error for invalid request"
        )

        # =================================================================
        # Scenario 2: Only user_id provided (group_id NOT in request)
        # =================================================================
        print(
            "\n--- Scenario 2: Only user_id provided (group_id NOT in request, query_all for group_id) ---"
        )
        data = {
            "user_id": self.user_id,
            "query": "coffee preference",
            "top_k": 10,
            "retrieve_method": "hybrid",
            "start_time": start_time,
            "end_time": end_time,
            # group_id is NOT in the request at all
        }

        status_code, response = self.call_get_with_body_api("/search", data)
        result = self._validate_search_response_structure(
            status_code, response, "Scenario 2"
        )

        # Validate metadata
        metadata = result["metadata"]
        assert (
            metadata.get("user_id") == self.user_id
        ), "[Scenario 2] metadata user_id should match"

        # Hybrid search should have importance_scores when there are results
        if result["total_count"] > 0:
            assert (
                "importance_scores" in result
            ), "[Scenario 2] Hybrid search result should contain importance_scores field"
            assert (
                metadata.get("source") == "hybrid"
            ), "[Scenario 2] Hybrid search source should be hybrid"

        self._validate_search_memories_filter(
            result,
            "Scenario 2",
            expected_user_id=self.user_id,
            user_id_filter_type="exact",
            group_id_filter_type="any",
        )

        print(
            f"✅ Scenario 2 successful, returned {result['total_count']} groups of memories for user_id={self.user_id}"
        )

        # =================================================================
        # Scenario 3: user_id + group_id=None or "" (filter for null/empty group_id)
        # =================================================================
        print(
            "\n--- Scenario 3: user_id + group_id='' (filter for null/empty group_id) ---"
        )
        data = {
            "user_id": self.user_id,
            "group_id": "",  # Empty string, equivalent to None
            "query": "coffee preference",
            "top_k": 10,
            "retrieve_method": "hybrid",
            "start_time": start_time,
            "end_time": end_time,
        }

        status_code, response = self.call_get_with_body_api("/search", data)
        result = self._validate_search_response_structure(
            status_code, response, "Scenario 3"
        )

        # Validate metadata
        metadata = result["metadata"]
        assert (
            metadata.get("user_id") == self.user_id
        ), "[Scenario 3] metadata user_id should match"

        self._validate_search_memories_filter(
            result,
            "Scenario 3",
            expected_user_id=self.user_id,
            user_id_filter_type="exact",
            group_id_filter_type="null_or_empty",
        )

        print(
            f"✅ Scenario 3 successful, returned {result['total_count']} groups of memories with null/empty group_id"
        )

        # =================================================================
        # Scenario 4: user_id + group_id both have valid values (exact match)
        # =================================================================
        print(
            "\n--- Scenario 4: user_id + group_id both have valid values (exact match) ---"
        )
        data = {
            "user_id": self.user_id,
            "group_id": self.group_id,
            "query": "coffee preference",
            "top_k": 10,
            "retrieve_method": "hybrid",
            "start_time": start_time,
            "end_time": end_time,
        }

        status_code, response = self.call_get_with_body_api("/search", data)
        result = self._validate_search_response_structure(
            status_code, response, "Scenario 4"
        )

        # Validate metadata (user_id should be present)
        metadata = result["metadata"]
        assert (
            metadata.get("user_id") == self.user_id
        ), "[Scenario 4] metadata user_id should match"
        # Note: metadata.group_id may not be returned by backend, so we verify via actual memories

        # When both have valid values, should only return exact matches - THIS IS THE KEY VALIDATION
        self._validate_search_memories_filter(
            result,
            "Scenario 4",
            expected_user_id=self.user_id,
            expected_group_id=self.group_id,
            user_id_filter_type="exact",
            group_id_filter_type="exact",
        )

        print(
            f"✅ Scenario 4 successful, returned {result['total_count']} groups of memories with exact user_id and group_id"
        )

        # =================================================================
        # Scenario 5: user_id="__all__" + valid group_id (query_all for user_id)
        # =================================================================
        print(
            "\n--- Scenario 5: user_id='__all__' + valid group_id (query_all for user_id) ---"
        )
        data = {
            "user_id": "__all__",
            "group_id": self.group_id,
            "query": "coffee preference",
            "top_k": 10,
            "retrieve_method": "hybrid",
            "start_time": start_time,
            "end_time": end_time,
        }

        status_code, response = self.call_get_with_body_api("/search", data)
        result = self._validate_search_response_structure(
            status_code, response, "Scenario 5"
        )

        # Note: metadata.group_id may not be returned by backend, so we verify via actual memories

        # When user_id is "__all__", should return memories from all users in this group - THIS IS THE KEY VALIDATION
        self._validate_search_memories_filter(
            result,
            "Scenario 5",
            expected_group_id=self.group_id,
            user_id_filter_type="any",
            group_id_filter_type="exact",
        )

        print(
            f"✅ Scenario 5 successful, returned {result['total_count']} groups of memories with group_id={self.group_id}"
        )

        # =================================================================
        # Scenario 6: user_id=None or "" + valid group_id (filter for null/empty user_id)
        # =================================================================
        print(
            "\n--- Scenario 6: user_id='' + valid group_id (filter for null/empty user_id) ---"
        )
        data = {
            "user_id": "",  # Empty string, equivalent to None
            "group_id": self.group_id,
            "query": "coffee preference",
            "top_k": 10,
            "retrieve_method": "hybrid",
            "start_time": start_time,
            "end_time": end_time,
        }

        status_code, response = self.call_get_with_body_api("/search", data)
        result = self._validate_search_response_structure(
            status_code, response, "Scenario 6"
        )

        # Note: metadata.group_id may not be returned by backend, so we verify via actual memories

        # When user_id is "" or None, should only return memories with null/empty user_id - THIS IS THE KEY VALIDATION
        self._validate_search_memories_filter(
            result,
            "Scenario 6",
            expected_group_id=self.group_id,
            user_id_filter_type="null_or_empty",
            group_id_filter_type="exact",
        )

        print(
            f"✅ Scenario 6 successful, returned {result['total_count']} groups of memories with null/empty user_id"
        )

        print(f"\n✅ All Hybrid Search Scenarios Completed Successfully")
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

    def test_delete_memories(self):
        """Test 10: DELETE /api/v1/memories - Soft delete memories (pass parameters via body)

        Tests multiple scenarios for combined filter conditions:
        1. All MAGIC_ALL - should return 422 error (Pydantic validation)
        2. Only event_id (single record deletion)
        3. Only user_id (batch deletion by user)
        4. Only group_id (batch deletion by group)
        5. user_id + group_id (combined filters)
        6. Query parameters compatibility test
        """
        self.print_section("Test 10: DELETE /api/v1/memories - Soft Delete Memories")

        # =================================================================
        # Scenario 1: All MAGIC_ALL - should return 422 error (Pydantic validation)
        # =================================================================
        print("\n--- Scenario 1: All parameters are MAGIC_ALL (should return 422) ---")
        data = {"event_id": "__all__", "user_id": "__all__", "group_id": "__all__"}

        status_code, response = self.call_delete_api("", data)

        # Should return 422 error (Pydantic model validation failed)
        assert (
            status_code == 422
        ), f"[Scenario 1] Status code should be 422, actual: {status_code}"
        # Pydantic validation errors return "detail" instead of "status"
        assert (
            "detail" in response
        ), f"[Scenario 1] Response should contain detail field for validation error"

        print(
            "✅ Scenario 1 successful, correctly returned 422 error for invalid request"
        )

        # =================================================================
        # Scenario 2: Only event_id (single record deletion)
        # =================================================================
        print("\n--- Scenario 2: Only event_id (delete single memory) ---")

        # First, memorize a message to get an event_id for deletion
        print("  Step 1: Create a test memory...")
        base_time = datetime.now(ZoneInfo("UTC"))
        memorize_data = {
            "group_id": self.group_id,
            "group_name": "Delete Test Group",
            "message_id": f"msg_delete_test_{uuid.uuid4().hex[:8]}",
            "create_time": base_time.isoformat(),
            "sender": self.user_id,
            "sender_name": "Delete Test User",
            "content": "This is a test message for deletion.",
            "refer_list": [],
        }

        memorize_status, _ = self.call_post_api("", memorize_data)
        assert (
            memorize_status == 200
        ), f"Failed to create test memory: {memorize_status}"

        # Get some memories to find an event_id to delete
        print("  Step 2: Fetch memories to get event_id...")
        fetch_data = {
            "user_id": self.user_id,
            "group_id": self.group_id,
            "memory_type": "episodic_memory",
            "limit": 1,
        }

        fetch_status, fetch_response = self.call_get_with_body_api("", fetch_data)

        event_id_to_delete = None
        if (
            fetch_status == 200
            and fetch_response.get("result", {}).get("total_count", 0) > 0
        ):
            memories = fetch_response["result"]["memories"]
            if len(memories) > 0:
                # Get memcell event_id from memcell_event_id_list
                # Note: Delete API only supports deleting memcells, not episodic_memory
                memcell_event_id_list = memories[0].get("memcell_event_id_list", [])
                if memcell_event_id_list:
                    event_id_to_delete = str(memcell_event_id_list[0])
                else:
                    # Fallback: try event_id or id (for backward compatibility)
                    event_id_to_delete = memories[0].get("event_id") or memories[0].get(
                        "id"
                    )

        if event_id_to_delete:
            print(f"  Step 3: Delete memory with event_id={event_id_to_delete}...")
            delete_data = {
                "event_id": event_id_to_delete,
                "user_id": "__all__",
                "group_id": "__all__",
            }

            status_code, response = self.call_delete_api("", delete_data)

            # Validate response
            # Note: 404 is acceptable if memory was already soft-deleted by previous test runs
            if status_code == 200:
                assert (
                    response.get("status") == "ok"
                ), f"[Scenario 2] Status should be ok"
                assert (
                    "result" in response
                ), "[Scenario 2] Response should contain result"

                result = response["result"]
                assert (
                    "filters" in result
                ), "[Scenario 2] result should contain filters field"
                assert (
                    "count" in result
                ), "[Scenario 2] result should contain count field"
                assert (
                    "event_id" in result["filters"]
                ), "[Scenario 2] filters should contain event_id"
                assert result["count"] >= 0, "[Scenario 2] count should be >= 0"

                print(
                    f"✅ Scenario 2 successful, deleted {result['count']} memory by event_id"
                )
            elif status_code == 404:
                print(
                    "⚠️  Scenario 2: Memory already deleted or not found (this is okay if memory was soft-deleted by previous test runs)"
                )
            else:
                print(
                    f"⚠️  Scenario 2: Unexpected status code {status_code}: {response.get('message')}"
                )
        else:
            print(
                "⚠️  Scenario 2 skipped: No memories found to delete (this is okay for fresh test environment)"
            )

        # =================================================================
        # Scenario 3: Only user_id (batch deletion by user)
        # =================================================================
        print("\n--- Scenario 3: Only user_id (batch delete by user) ---")

        # Create a unique test user for this scenario
        test_user_id = f"delete_test_user_{uuid.uuid4().hex[:8]}"

        # Create some test memories for this user
        print(f"  Step 1: Create test memories for user_id={test_user_id}...")
        for i in range(3):
            memorize_data = {
                "group_id": self.group_id,
                "group_name": "Delete Test Group",
                "message_id": f"msg_user_delete_{uuid.uuid4().hex[:8]}",
                "create_time": (
                    datetime.now(ZoneInfo("UTC")) + timedelta(seconds=i)
                ).isoformat(),
                "sender": test_user_id,
                "sender_name": "Batch Delete Test User",
                "content": f"Test message {i+1} for batch deletion by user.",
                "refer_list": [],
            }
            self.call_post_api("", memorize_data)

        print(f"  Step 2: Delete all memories for user_id={test_user_id}...")
        delete_data = {
            "event_id": "__all__",
            "user_id": test_user_id,
            "group_id": "__all__",
        }

        status_code, response = self.call_delete_api("", delete_data)

        # Validate response
        if status_code == 200:
            assert response.get("status") == "ok", f"[Scenario 3] Status should be ok"
            result = response["result"]
            assert (
                "user_id" in result["filters"]
            ), "[Scenario 3] filters should contain user_id"
            print(
                f"✅ Scenario 3 successful, deleted {result['count']} memories by user_id"
            )
        elif status_code == 404:
            print(
                "⚠️  Scenario 3: No memories found to delete (this is okay if boundary detection didn't extract memories yet)"
            )
        else:
            print(
                f"⚠️  Scenario 3: Unexpected status code {status_code}: {response.get('message')}"
            )

        # =================================================================
        # Scenario 4: Only group_id (batch deletion by group)
        # =================================================================
        print("\n--- Scenario 4: Only group_id (batch delete by group) ---")

        # Create a unique test group for this scenario
        test_group_id = f"delete_test_group_{uuid.uuid4().hex[:8]}"

        # Create some test memories for this group
        print(f"  Step 1: Create test memories for group_id={test_group_id}...")
        for i in range(2):
            memorize_data = {
                "group_id": test_group_id,
                "group_name": "Batch Delete Test Group",
                "message_id": f"msg_group_delete_{uuid.uuid4().hex[:8]}",
                "create_time": (
                    datetime.now(ZoneInfo("UTC")) + timedelta(seconds=i)
                ).isoformat(),
                "sender": self.user_id,
                "sender_name": "Group Delete Test User",
                "content": f"Test message {i+1} for batch deletion by group.",
                "refer_list": [],
            }
            self.call_post_api("", memorize_data)

        print(f"  Step 2: Delete all memories for group_id={test_group_id}...")
        delete_data = {
            "event_id": "__all__",
            "user_id": "__all__",
            "group_id": test_group_id,
        }

        status_code, response = self.call_delete_api("", delete_data)

        # Validate response
        if status_code == 200:
            assert response.get("status") == "ok", f"[Scenario 4] Status should be ok"
            result = response["result"]
            assert (
                "group_id" in result["filters"]
            ), "[Scenario 4] filters should contain group_id"
            print(
                f"✅ Scenario 4 successful, deleted {result['count']} memories by group_id"
            )
        elif status_code == 404:
            print(
                "⚠️  Scenario 4: No memories found to delete (this is okay if boundary detection didn't extract memories yet)"
            )
        else:
            print(
                f"⚠️  Scenario 4: Unexpected status code {status_code}: {response.get('message')}"
            )

        # =================================================================
        # Scenario 5: user_id + group_id (combined filters)
        # =================================================================
        print("\n--- Scenario 5: user_id + group_id (combined filter deletion) ---")

        # Create a unique test user and group for this scenario
        combined_user_id = f"combined_user_{uuid.uuid4().hex[:8]}"
        combined_group_id = f"combined_group_{uuid.uuid4().hex[:8]}"

        # Create test memories with specific user + group combination
        print(
            f"  Step 1: Create test memories for user_id={combined_user_id}, group_id={combined_group_id}..."
        )
        for i in range(2):
            memorize_data = {
                "group_id": combined_group_id,
                "group_name": "Combined Filter Test Group",
                "message_id": f"msg_combined_delete_{uuid.uuid4().hex[:8]}",
                "create_time": (
                    datetime.now(ZoneInfo("UTC")) + timedelta(seconds=i)
                ).isoformat(),
                "sender": combined_user_id,
                "sender_name": "Combined Filter Test User",
                "content": f"Test message {i+1} for combined filter deletion.",
                "refer_list": [],
            }
            self.call_post_api("", memorize_data)

        print(
            f"  Step 2: Delete memories matching both user_id={combined_user_id} AND group_id={combined_group_id}..."
        )
        delete_data = {
            "event_id": "__all__",
            "user_id": combined_user_id,
            "group_id": combined_group_id,
        }

        status_code, response = self.call_delete_api("", delete_data)

        # Validate response
        if status_code == 200:
            assert response.get("status") == "ok", f"[Scenario 5] Status should be ok"
            result = response["result"]
            assert (
                "user_id" in result["filters"] and "group_id" in result["filters"]
            ), "[Scenario 5] filters should contain both user_id and group_id"
            print(
                f"✅ Scenario 5 successful, deleted {result['count']} memories with combined filters"
            )
        elif status_code == 404:
            print(
                "⚠️  Scenario 5: No memories found to delete (this is okay if boundary detection didn't extract memories yet)"
            )
        else:
            print(
                f"⚠️  Scenario 5: Unexpected status code {status_code}: {response.get('message')}"
            )

        # =================================================================
        # Scenario 6: Test with query parameters (compatibility)
        # =================================================================
        print(
            "\n--- Scenario 6: Delete using query parameters (compatibility test) ---"
        )

        # Create a unique test user for query params test
        query_user_id = f"query_delete_user_{uuid.uuid4().hex[:8]}"

        # Create test memory
        print(f"  Step 1: Create test memory for user_id={query_user_id}...")
        memorize_data = {
            "group_id": self.group_id,
            "group_name": "Query Params Test Group",
            "message_id": f"msg_query_delete_{uuid.uuid4().hex[:8]}",
            "create_time": datetime.now(ZoneInfo("UTC")).isoformat(),
            "sender": query_user_id,
            "sender_name": "Query Params Test User",
            "content": "Test message for query params deletion.",
            "refer_list": [],
        }
        self.call_post_api("", memorize_data)

        print(f"  Step 2: Delete using query parameters...")
        # Use query params instead of body
        params = {"user_id": query_user_id}

        status_code, response = self.call_delete_api("", params=params)

        # Validate response
        if status_code == 200:
            assert response.get("status") == "ok", f"[Scenario 6] Status should be ok"
            result = response["result"]
            print(
                f"✅ Scenario 6 successful, deleted {result['count']} memories using query params"
            )
        elif status_code == 404:
            print(
                "⚠️  Scenario 6: No memories found to delete (this is okay if boundary detection didn't extract memories yet)"
            )
        else:
            print(
                f"⚠️  Scenario 6: Unexpected status code {status_code}: {response.get('message')}"
            )

        print(f"\n✅ All Delete Memory Scenarios Completed")
        return status_code, response

    def run_all_tests(self, test_method: str = "all", except_test_methods: str = None):
        """
        Run tests

        Args:
            test_method: Specify test method to run, options:
                - all: Run all tests
                - fetch: Run all fetch-related tests (batch)
                - retrieve/search: Run all retrieve/search-related tests (batch)
                - memorize: Test storing conversation memory / Run memorization tests (batch)
                - meta: Run metadata-related tests (batch)
                - delete: Run delete-related tests (batch)
                - fetch_episodic: Test fetching episodic memory
                - fetch_foresight: Test fetching foresight memory
                - fetch_event_log: Test fetching event log
                - fetch_group_filter: Test fetching with group_id filter
                - fetch_time_range: Test fetching with time range filter
                - fetch_combined_filters: Test fetching with combined filters
                - fetch_profile: Test fetching user profile
                - fetch_all_types: Test fetching all memory types
                - search_keyword: Test keyword search
                - search_vector: Test vector search
                - search_hybrid: Test hybrid search
                - save_meta: Test saving conversation metadata
                - patch_meta: Test updating conversation metadata
                - delete_memories: Test soft delete memories
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
            "fetch_group_filter": self.test_fetch_with_group_filter,
            "fetch_time_range": self.test_fetch_with_time_range,
            "fetch_combined_filters": self.test_fetch_with_combined_filters,
            "fetch_profile": self.test_fetch_profile_memory,
            "fetch_all_types": self.test_fetch_all_memory_types,
            "search_keyword": self.test_search_memories_keyword,
            "search_vector": self.test_search_memories_vector,
            "search_hybrid": self.test_search_memories_hybrid,
            "save_meta": self.test_save_conversation_meta,
            "patch_meta": self.test_patch_conversation_meta,
            "delete_memories": self.test_delete_memories,
        }

        # Define test type grouping
        test_type_groups = {
            "fetch": [
                "fetch_episodic",
                "fetch_foresight",
                "fetch_event_log",
                "fetch_group_filter",
                "fetch_time_range",
                "fetch_combined_filters",
                "fetch_profile",
                "fetch_all_types",
            ],
            "retrieve": ["search_keyword", "search_vector", "search_hybrid"],
            "search": [  # Alias for retrieve
                "search_keyword",
                "search_vector",
                "search_hybrid",
            ],
            "memorize": ["memorize"],
            "meta": ["save_meta", "patch_meta"],
            "delete": ["delete_memories"],
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
            if test_method in test_type_groups:
                # Batch mode: Run tests by category (fetch, retrieve, search, memorize, meta)
                method_names = test_type_groups[test_method]
                methods_to_run = [
                    (name, test_methods[name])
                    for name in method_names
                    if name in test_methods
                ]

                print(
                    f"\n📋 Will run {len(methods_to_run)} test methods in [{test_method}] category"
                )
                for name, method in methods_to_run:
                    method()

            elif except_test_methods:
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
        if test_method in test_type_groups:
            print(f"\n✅ All [{test_method}] category tests passed!")
        elif except_test_methods:
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
            # Batch categories
            "fetch",
            "retrieve",
            "search",
            "memorize",
            "meta",
            "delete",
            # Individual methods
            "fetch_episodic",
            "fetch_foresight",
            "fetch_event_log",
            "fetch_group_filter",
            "fetch_time_range",
            "fetch_combined_filters",
            "fetch_profile",
            "fetch_all_types",
            "search_keyword",
            "search_vector",
            "search_hybrid",
            "save_meta",
            "patch_meta",
            "delete_memories",
        ],
        help="Specify test method to run (default: all). Supports batch categories (fetch, retrieve/search, memorize, meta, delete) or individual test methods",
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
        print("   - Use --test-method to specify a test or category to run")
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

    # Run tests (decide to run all, single, by category, or exclude certain tests based on parameters)
    tester.run_all_tests(
        test_method=args.test_method, except_test_methods=args.except_test_method
    )


if __name__ == "__main__":
    main()
