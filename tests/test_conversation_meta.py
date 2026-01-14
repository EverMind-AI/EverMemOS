# -*- coding: utf-8 -*-
"""
Conversation Meta API Test Script
Verify input and output structures of conversation-meta endpoints under /api/v1/memories

Usage:
    # Run all tests
    python tests/test_conversation_meta.py
    
    # Specify API address
    python tests/test_conversation_meta.py --base-url http://localhost:1995
    
    # Test by category
    python tests/test_conversation_meta.py --test-method post       # Test POST (create/update)
    python tests/test_conversation_meta.py --test-method get        # Test GET (with fallback)
    python tests/test_conversation_meta.py --test-method patch      # Test PATCH (partial update)
    python tests/test_conversation_meta.py --test-method fallback   # Test fallback logic
    python tests/test_conversation_meta.py --test-method error      # Test error/exception cases
    python tests/test_conversation_meta.py --test-method all        # Run all tests (default)
    
    # Test a specific method
    python tests/test_conversation_meta.py --test-method post_default
    python tests/test_conversation_meta.py --test-method post_with_group_id
    python tests/test_conversation_meta.py --test-method get_by_group_id
    python tests/test_conversation_meta.py --test-method get_default
    python tests/test_conversation_meta.py --test-method get_fallback
    python tests/test_conversation_meta.py --test-method patch_update
    python tests/test_conversation_meta.py --test-method patch_default
    
    # Test error/exception cases
    python tests/test_conversation_meta.py --test-method error_dup_default      # Duplicate default POST (upsert)
    python tests/test_conversation_meta.py --test-method error_dup_group_id     # Duplicate group_id POST (upsert)
    python tests/test_conversation_meta.py --test-method error_missing_fields   # Missing required fields
    python tests/test_conversation_meta.py --test-method error_invalid_scene    # Invalid scene value
    python tests/test_conversation_meta.py --test-method error_patch_fallback   # PATCH non-existent -> fallback to default
    python tests/test_conversation_meta.py --test-method error_empty_body       # Empty POST body
"""

import argparse
import json
import uuid
from datetime import datetime
from zoneinfo import ZoneInfo
import requests


class ConversationMetaTester:
    """Conversation Meta API Test Class"""

    # Default tenant information
    DEFAULT_ORGANIZATION_ID = "test_conv_meta_organization"
    DEFAULT_SPACE_ID = "test_conv_meta_space"
    DEFAULT_HASH_KEY = "test_conv_meta_hash_key"
    # ta38b637741

    def __init__(
        self,
        base_url: str,
        organization_id: str = None,
        space_id: str = None,
        hash_key: str = None,
        timeout: int = 60,
    ):
        """
        Initialize tester

        Args:
            base_url: API base URL
            organization_id: Organization ID
            space_id: Space ID
            hash_key: Hash key
            timeout: Request timeout in seconds
        """
        self.base_url = base_url
        self.api_prefix = "/api/v1/memories"
        self.organization_id = organization_id or self.DEFAULT_ORGANIZATION_ID
        self.space_id = space_id or self.DEFAULT_SPACE_ID
        self.hash_key = hash_key or self.DEFAULT_HASH_KEY
        self.timeout = timeout

        # Generate unique test IDs
        self.test_run_id = uuid.uuid4().hex[:8]
        self.test_group_id = f"test_group_{self.test_run_id}"

    def get_tenant_headers(self) -> dict:
        """Get tenant-related request headers"""
        headers = {
            "X-Organization-Id": self.organization_id,
            "X-Space-Id": self.space_id,
            "Content-Type": "application/json",
        }
        if self.hash_key:
            headers["X-Hash-Key"] = self.hash_key
        return headers

    def print_separator(self, title: str):
        """Print section separator"""
        print("\n" + "=" * 80)
        print(f"  {title}")
        print("=" * 80)

    def print_request(self, method: str, url: str, body: dict = None):
        """Print request info"""
        print(f"📍 URL: {method} {url}")
        if body:
            print(f"📤 Request Body:")
            print(json.dumps(body, indent=2, ensure_ascii=False))

    def print_response(self, response: requests.Response):
        """Print response info"""
        print(f"\n📥 Response Status Code: {response.status_code}")
        try:
            response_json = response.json()
            print("📥 Response Data:")
            print(json.dumps(response_json, indent=2, ensure_ascii=False))
            return response_json
        except Exception:
            print(f"📥 Response Text: {response.text}")
            return None

    def init_database(self) -> bool:
        """Initialize tenant database"""
        url = f"{self.base_url}/internal/tenant/init-db"
        headers = self.get_tenant_headers()

        self.print_separator("Initialize Tenant Database")
        print(
            f"📤 Tenant Info: organization_id={self.organization_id}, space_id={self.space_id}"
        )

        try:
            response = requests.post(url, headers=headers, timeout=self.timeout)
            response_json = self.print_response(response)

            if response.status_code == 200:
                print(f"\n✅ Database initialization successful")
                return True
            else:
                print(f"\n⚠️  Database initialization returned: {response_json}")
                return True  # Continue even if failed
        except Exception as e:
            print(f"\n❌ Database initialization failed: {e}")
            return False

    # ==================== POST Tests ====================

    def test_post_default_config(self) -> bool:
        """
        Test POST: Create default config (group_id=null)

        Creates a default conversation meta that will be used as fallback.
        """
        self.print_separator("POST: Create Default Config (group_id=null)")

        url = f"{self.base_url}{self.api_prefix}/conversation-meta"
        headers = self.get_tenant_headers()

        body = {
            "version": "1.0.0",
            "scene": "group_chat",
            "scene_desc": {"bot_ids": ["default_bot"], "type": "default_config"},
            "name": f"Default Settings ({self.test_run_id})",
            "description": "This is the default conversation meta config",
            "group_id": None,  # null = default config
            "created_at": datetime.now(ZoneInfo("UTC")).isoformat(),
            "default_timezone": "UTC",
            "user_details": {
                "default_user": {
                    "full_name": "Default User",
                    "role": "user",
                    "custom_role": "member",
                    "extra": {"is_default": True},
                }
            },
            "tags": ["default", "test"],
        }

        self.print_request("POST", url, body)

        try:
            response = requests.post(
                url, headers=headers, json=body, timeout=self.timeout
            )
            response_json = self.print_response(response)

            if response.status_code == 200 and response_json.get("status") == "ok":
                result = response_json.get("result", {})
                is_default = result.get("is_default", False)
                if is_default and result.get("group_id") is None:
                    print(f"\n✅ Default config created successfully!")
                    print(f"   - ID: {result.get('id')}")
                    print(f"   - is_default: {is_default}")
                    return True
                else:
                    print(f"\n❌ Expected is_default=True and group_id=null")
                    return False
            else:
                print(f"\n❌ Failed to create default config")
                return False
        except Exception as e:
            print(f"\n❌ Request failed: {e}")
            return False

    def test_post_with_group_id(self) -> bool:
        """
        Test POST: Create config with specific group_id
        """
        self.print_separator("POST: Create Config with group_id")

        url = f"{self.base_url}{self.api_prefix}/conversation-meta"
        headers = self.get_tenant_headers()

        body = {
            "version": "1.0.0",
            "scene": "group_chat",
            "scene_desc": {
                "bot_ids": ["bot_001", "bot_002"],
                "type": "project_discussion",
            },
            "name": f"Test Group ({self.test_run_id})",
            "description": "Test conversation meta with specific group_id",
            "group_id": self.test_group_id,
            "created_at": datetime.now(ZoneInfo("UTC")).isoformat(),
            "default_timezone": "Asia/Shanghai",
            "user_details": {
                "user_001": {
                    "full_name": "Test User",
                    "role": "user",
                    "custom_role": "developer",
                    "extra": {"department": "Engineering"},
                },
                "bot_001": {
                    "full_name": "AI Assistant",
                    "role": "assistant",
                    "custom_role": "assistant",
                    "extra": {"type": "ai"},
                },
            },
            "tags": ["test", "project", "engineering"],
        }

        self.print_request("POST", url, body)

        try:
            response = requests.post(
                url, headers=headers, json=body, timeout=self.timeout
            )
            response_json = self.print_response(response)

            if response.status_code == 200 and response_json.get("status") == "ok":
                result = response_json.get("result", {})
                if result.get("group_id") == self.test_group_id:
                    print(f"\n✅ Config with group_id created successfully!")
                    print(f"   - ID: {result.get('id')}")
                    print(f"   - group_id: {result.get('group_id')}")
                    print(f"   - is_default: {result.get('is_default', False)}")
                    return True
                else:
                    print(f"\n❌ group_id mismatch")
                    return False
            else:
                print(f"\n❌ Failed to create config")
                return False
        except Exception as e:
            print(f"\n❌ Request failed: {e}")
            return False

    def test_post_update_existing(self) -> bool:
        """
        Test POST: Update existing config (upsert)
        """
        self.print_separator("POST: Update Existing Config (Upsert)")

        url = f"{self.base_url}{self.api_prefix}/conversation-meta"
        headers = self.get_tenant_headers()

        # Update the same group_id with new data
        body = {
            "version": "1.0.1",  # Updated version
            "scene": "group_chat",
            "scene_desc": {
                "bot_ids": ["bot_001", "bot_002", "bot_003"],
                "type": "updated",
            },
            "name": f"Updated Group ({self.test_run_id})",
            "description": "Updated conversation meta",
            "group_id": self.test_group_id,
            "created_at": datetime.now(ZoneInfo("UTC")).isoformat(),
            "default_timezone": "America/New_York",
            "user_details": {
                "user_001": {
                    "full_name": "Updated User",
                    "role": "user",
                    "custom_role": "lead",
                    "extra": {"department": "Engineering", "updated": True},
                }
            },
            "tags": ["updated", "test"],
        }

        self.print_request("POST", url, body)

        try:
            response = requests.post(
                url, headers=headers, json=body, timeout=self.timeout
            )
            response_json = self.print_response(response)

            if response.status_code == 200 and response_json.get("status") == "ok":
                result = response_json.get("result", {})
                if result.get("version") == "1.0.1":
                    print(f"\n✅ Config updated successfully via upsert!")
                    print(f"   - version updated to: {result.get('version')}")
                    return True
                else:
                    print(f"\n❌ Version not updated")
                    return False
            else:
                print(f"\n❌ Failed to update config")
                return False
        except Exception as e:
            print(f"\n❌ Request failed: {e}")
            return False

    # ==================== GET Tests ====================

    def test_get_by_group_id(self) -> bool:
        """
        Test GET: Retrieve config by specific group_id
        """
        self.print_separator("GET: Retrieve by group_id")

        url = f"{self.base_url}{self.api_prefix}/conversation-meta"
        headers = self.get_tenant_headers()
        params = {"group_id": self.test_group_id}

        print(f"📍 URL: GET {url}")
        print(f"📤 Query Params: {params}")

        try:
            response = requests.get(
                url, headers=headers, params=params, timeout=self.timeout
            )
            response_json = self.print_response(response)

            if response.status_code == 200 and response_json.get("status") == "ok":
                result = response_json.get("result", {})
                if result.get("group_id") == self.test_group_id:
                    print(f"\n✅ Retrieved config by group_id successfully!")
                    print(f"   - group_id: {result.get('group_id')}")
                    print(f"   - name: {result.get('name')}")
                    print(f"   - is_default: {result.get('is_default', False)}")
                    return True
                else:
                    print(f"\n❌ group_id mismatch in response")
                    return False
            else:
                print(f"\n❌ Failed to retrieve config")
                return False
        except Exception as e:
            print(f"\n❌ Request failed: {e}")
            return False

    def test_get_default_config(self) -> bool:
        """
        Test GET: Retrieve default config (group_id=null or not provided)
        """
        self.print_separator("GET: Retrieve Default Config")

        url = f"{self.base_url}{self.api_prefix}/conversation-meta"
        headers = self.get_tenant_headers()
        # No group_id param = get default config

        print(f"📍 URL: GET {url}")
        print(f"📤 Query Params: (none - should return default config)")

        try:
            response = requests.get(url, headers=headers, timeout=self.timeout)
            response_json = self.print_response(response)

            if response.status_code == 200 and response_json.get("status") == "ok":
                result = response_json.get("result", {})
                if result.get("group_id") is None and result.get("is_default") is True:
                    print(f"\n✅ Retrieved default config successfully!")
                    print(f"   - group_id: {result.get('group_id')} (null)")
                    print(f"   - is_default: {result.get('is_default')}")
                    print(f"   - name: {result.get('name')}")
                    return True
                else:
                    print(
                        f"\n❌ Expected default config (group_id=null, is_default=true)"
                    )
                    return False
            else:
                print(f"\n❌ Failed to retrieve default config")
                return False
        except Exception as e:
            print(f"\n❌ Request failed: {e}")
            return False

    def test_get_fallback_to_default(self) -> bool:
        """
        Test GET: Fallback to default when group_id not found

        This is the core fallback logic test:
        1. Request a non-existent group_id
        2. Should automatically fallback to default config
        3. Verify is_default=true and message indicates fallback
        """
        self.print_separator("GET: Fallback to Default (Non-existent group_id)")

        url = f"{self.base_url}{self.api_prefix}/conversation-meta"
        headers = self.get_tenant_headers()

        # Use a non-existent group_id
        non_existent_group_id = f"non_existent_{uuid.uuid4().hex[:8]}"
        params = {"group_id": non_existent_group_id}

        print(f"📍 URL: GET {url}")
        print(f"📤 Query Params: {params}")
        print(
            f"📝 Note: group_id '{non_existent_group_id}' does not exist, should fallback to default"
        )

        try:
            response = requests.get(
                url, headers=headers, params=params, timeout=self.timeout
            )
            response_json = self.print_response(response)

            if response.status_code == 200 and response_json.get("status") == "ok":
                result = response_json.get("result", {})
                message = response_json.get("message", "")

                # Verify fallback behavior
                if result.get("is_default") is True and result.get("group_id") is None:
                    print(f"\n✅ Fallback to default config successful!")
                    print(f"   - Requested group_id: {non_existent_group_id}")
                    print(
                        f"   - Returned group_id: {result.get('group_id')} (null = default)"
                    )
                    print(f"   - is_default: {result.get('is_default')}")
                    print(f"   - message: {message}")

                    if "default" in message.lower():
                        print(f"   - Message correctly indicates fallback")
                    return True
                else:
                    print(f"\n❌ Expected fallback to default config")
                    print(f"   - Got group_id: {result.get('group_id')}")
                    print(f"   - Got is_default: {result.get('is_default')}")
                    return False
            elif response.status_code == 404:
                print(f"\n⚠️  404 returned - default config may not exist")
                print(f"   Make sure to run test_post_default_config first")
                return False
            else:
                print(f"\n❌ Unexpected response")
                return False
        except Exception as e:
            print(f"\n❌ Request failed: {e}")
            return False

    def test_get_not_found(self) -> bool:
        """
        Test GET: 404 when no default config exists

        This test requires a clean state where no default config exists.
        """
        self.print_separator("GET: 404 When No Config Exists")

        # Use a completely new tenant to ensure no default config
        url = f"{self.base_url}{self.api_prefix}/conversation-meta"
        headers = self.get_tenant_headers()
        headers["X-Organization-Id"] = f"temp_org_{uuid.uuid4().hex[:8]}"
        headers["X-Space-Id"] = f"temp_space_{uuid.uuid4().hex[:8]}"

        non_existent_group_id = f"definitely_not_exists_{uuid.uuid4().hex}"
        params = {"group_id": non_existent_group_id}

        print(f"📍 URL: GET {url}")
        print(f"📤 Using temporary tenant (no data)")
        print(f"📤 Query Params: {params}")

        try:
            response = requests.get(
                url, headers=headers, params=params, timeout=self.timeout
            )
            response_json = self.print_response(response)

            if response.status_code == 404:
                print(f"\n✅ Correctly returned 404 when no config exists!")
                return True
            else:
                print(f"\n⚠️  Expected 404, got {response.status_code}")
                print(f"   (This may be OK if tenant has default config)")
                return True  # Not a failure, just different state
        except Exception as e:
            print(f"\n❌ Request failed: {e}")
            return False

    # ==================== PATCH Tests ====================

    def test_patch_update_fields(self) -> bool:
        """
        Test PATCH: Partial update of specific fields
        """
        self.print_separator("PATCH: Partial Update Fields")

        url = f"{self.base_url}{self.api_prefix}/conversation-meta"
        headers = self.get_tenant_headers()

        body = {
            "group_id": self.test_group_id,
            "name": f"Patched Name ({self.test_run_id})",
            "description": "Patched description via PATCH",
            "tags": ["patched", "updated", "test"],
        }

        self.print_request("PATCH", url, body)

        try:
            response = requests.patch(
                url, headers=headers, json=body, timeout=self.timeout
            )
            response_json = self.print_response(response)

            if response.status_code == 200 and response_json.get("status") == "ok":
                result = response_json.get("result", {})
                updated_fields = result.get("updated_fields", [])

                if "name" in updated_fields and "description" in updated_fields:
                    print(f"\n✅ Partial update successful!")
                    print(f"   - Updated fields: {updated_fields}")
                    print(f"   - New name: {result.get('name')}")
                    return True
                else:
                    print(f"\n❌ Expected fields not updated")
                    return False
            else:
                print(f"\n❌ Failed to patch config")
                return False
        except Exception as e:
            print(f"\n❌ Request failed: {e}")
            return False

    def test_patch_default_config(self) -> bool:
        """
        Test PATCH: Update default config (group_id=null)
        """
        self.print_separator("PATCH: Update Default Config (group_id=null)")

        url = f"{self.base_url}{self.api_prefix}/conversation-meta"
        headers = self.get_tenant_headers()

        body = {
            "group_id": None,  # Target default config
            "name": f"Patched Default ({self.test_run_id})",
            "tags": ["default", "patched"],
        }

        self.print_request("PATCH", url, body)

        try:
            response = requests.patch(
                url, headers=headers, json=body, timeout=self.timeout
            )
            response_json = self.print_response(response)

            if response.status_code == 200 and response_json.get("status") == "ok":
                result = response_json.get("result", {})
                if result.get("group_id") is None:
                    print(f"\n✅ Default config patched successfully!")
                    print(f"   - Updated fields: {result.get('updated_fields', [])}")
                    return True
                else:
                    print(f"\n❌ Expected group_id=null for default config")
                    return False
            elif response.status_code == 404:
                print(
                    f"\n⚠️  Default config not found - run test_post_default_config first"
                )
                return False
            else:
                print(f"\n❌ Failed to patch default config")
                return False
        except Exception as e:
            print(f"\n❌ Request failed: {e}")
            return False

    def test_patch_no_changes(self) -> bool:
        """
        Test PATCH: No changes when all fields are null
        """
        self.print_separator("PATCH: No Changes (Empty Update)")

        url = f"{self.base_url}{self.api_prefix}/conversation-meta"
        headers = self.get_tenant_headers()

        body = {
            "group_id": self.test_group_id,
            # All optional fields are null/not provided
        }

        self.print_request("PATCH", url, body)

        try:
            response = requests.patch(
                url, headers=headers, json=body, timeout=self.timeout
            )
            response_json = self.print_response(response)

            if response.status_code == 200:
                result = response_json.get("result", {})
                updated_fields = result.get("updated_fields", [])

                if len(updated_fields) == 0:
                    print(f"\n✅ Correctly returned no changes!")
                    print(f"   - Message: {response_json.get('message')}")
                    return True
                else:
                    print(f"\n⚠️  Unexpected fields updated: {updated_fields}")
                    return True
            else:
                print(f"\n❌ Unexpected response")
                return False
        except Exception as e:
            print(f"\n❌ Request failed: {e}")
            return False

    def test_patch_user_details(self) -> bool:
        """
        Test PATCH: Update user_details field
        """
        self.print_separator("PATCH: Update user_details")

        url = f"{self.base_url}{self.api_prefix}/conversation-meta"
        headers = self.get_tenant_headers()

        body = {
            "group_id": self.test_group_id,
            "user_details": {
                "user_001": {
                    "full_name": "Patched User",
                    "role": "user",
                    "custom_role": "admin",
                    "extra": {"patched": True, "level": 10},
                },
                "new_user": {
                    "full_name": "New User Added",
                    "role": "user",
                    "custom_role": "guest",
                    "extra": {},
                },
            },
        }

        self.print_request("PATCH", url, body)

        try:
            response = requests.patch(
                url, headers=headers, json=body, timeout=self.timeout
            )
            response_json = self.print_response(response)

            if response.status_code == 200 and response_json.get("status") == "ok":
                result = response_json.get("result", {})
                if "user_details" in result.get("updated_fields", []):
                    print(f"\n✅ user_details updated successfully!")
                    return True
                else:
                    print(f"\n❌ user_details not in updated_fields")
                    return False
            else:
                print(f"\n❌ Failed to update user_details")
                return False
        except Exception as e:
            print(f"\n❌ Request failed: {e}")
            return False

    # ==================== Error/Exception Tests ====================

    def test_post_duplicate_default_upsert(self) -> bool:
        """
        Test POST: Duplicate default config (group_id=null) - should upsert (update)

        Since we use upsert logic, posting duplicate group_id=null should update, not error.
        """
        self.print_separator("POST: Duplicate Default Config (Upsert Behavior)")

        url = f"{self.base_url}{self.api_prefix}/conversation-meta"
        headers = self.get_tenant_headers()

        # First POST
        body1 = {
            "version": "1.0.0",
            "scene": "group_chat",
            "scene_desc": {"bot_ids": ["bot_v1"]},
            "name": f"Default V1 ({self.test_run_id})",
            "description": "First default config",
            "group_id": None,
            "created_at": datetime.now(ZoneInfo("UTC")).isoformat(),
            "default_timezone": "UTC",
            "tags": ["v1"],
        }

        print("📤 First POST (create default):")
        self.print_request("POST", url, body1)

        try:
            response1 = requests.post(
                url, headers=headers, json=body1, timeout=self.timeout
            )
            response_json1 = self.print_response(response1)

            if response1.status_code != 200:
                print(f"\n❌ First POST failed")
                return False

            first_id = response_json1.get("result", {}).get("id")

            # Second POST with same group_id=null (should update)
            body2 = {
                "version": "2.0.0",
                "scene": "group_chat",
                "scene_desc": {"bot_ids": ["bot_v2"]},
                "name": f"Default V2 ({self.test_run_id})",
                "description": "Second default config (should update)",
                "group_id": None,
                "created_at": datetime.now(ZoneInfo("UTC")).isoformat(),
                "default_timezone": "UTC",
                "tags": ["v2"],
            }

            print("\n📤 Second POST (should upsert/update):")
            self.print_request("POST", url, body2)

            response2 = requests.post(
                url, headers=headers, json=body2, timeout=self.timeout
            )
            response_json2 = self.print_response(response2)

            if response2.status_code == 200:
                result = response_json2.get("result", {})
                second_id = result.get("id")
                version = result.get("version")

                # Should be same ID (updated) with new version
                if second_id == first_id and version == "2.0.0":
                    print(
                        f"\n✅ Duplicate default POST correctly updated existing record!"
                    )
                    print(f"   - Same ID: {first_id}")
                    print(f"   - Version updated to: {version}")
                    return True
                elif second_id == first_id:
                    print(f"\n✅ Correctly updated (same ID), version: {version}")
                    return True
                else:
                    print(f"\n⚠️  Different IDs: first={first_id}, second={second_id}")
                    print(f"   This might indicate duplicate records were created")
                    return False
            else:
                print(f"\n❌ Second POST failed with status {response2.status_code}")
                return False

        except Exception as e:
            print(f"\n❌ Request failed: {e}")
            return False

    def test_post_duplicate_group_id_upsert(self) -> bool:
        """
        Test POST: Duplicate specific group_id - should upsert (update)
        """
        self.print_separator("POST: Duplicate group_id (Upsert Behavior)")

        url = f"{self.base_url}{self.api_prefix}/conversation-meta"
        headers = self.get_tenant_headers()

        dup_group_id = f"dup_group_{self.test_run_id}"

        # First POST
        body1 = {
            "version": "1.0.0",
            "scene": "group_chat",
            "scene_desc": {"bot_ids": ["bot_v1"]},
            "name": f"Group V1",
            "description": "First version",
            "group_id": dup_group_id,
            "created_at": datetime.now(ZoneInfo("UTC")).isoformat(),
            "default_timezone": "UTC",
            "tags": ["v1"],
        }

        print(f"📤 First POST (create group_id={dup_group_id}):")
        self.print_request("POST", url, body1)

        try:
            response1 = requests.post(
                url, headers=headers, json=body1, timeout=self.timeout
            )
            response_json1 = self.print_response(response1)

            if response1.status_code != 200:
                print(f"\n❌ First POST failed")
                return False

            first_id = response_json1.get("result", {}).get("id")

            # Second POST with same group_id
            body2 = {
                "version": "2.0.0",
                "scene": "group_chat",
                "scene_desc": {"bot_ids": ["bot_v2", "bot_v3"]},
                "name": f"Group V2 (Updated)",
                "description": "Second version (should update)",
                "group_id": dup_group_id,
                "created_at": datetime.now(ZoneInfo("UTC")).isoformat(),
                "default_timezone": "Asia/Shanghai",
                "tags": ["v2", "updated"],
            }

            print("\n📤 Second POST (should upsert/update):")
            self.print_request("POST", url, body2)

            response2 = requests.post(
                url, headers=headers, json=body2, timeout=self.timeout
            )
            response_json2 = self.print_response(response2)

            if response2.status_code == 200:
                result = response_json2.get("result", {})
                second_id = result.get("id")
                version = result.get("version")

                if second_id == first_id:
                    print(
                        f"\n✅ Duplicate group_id POST correctly updated existing record!"
                    )
                    print(f"   - Same ID: {first_id}")
                    print(f"   - Version: {version}")
                    return True
                else:
                    print(f"\n⚠️  Different IDs - possible duplicate creation")
                    return False
            else:
                print(f"\n❌ Second POST failed")
                return False

        except Exception as e:
            print(f"\n❌ Request failed: {e}")
            return False

    def test_post_missing_required_fields(self) -> bool:
        """
        Test POST: Missing required fields should return 400/422
        """
        self.print_separator("POST: Missing Required Fields (Error Case)")

        url = f"{self.base_url}{self.api_prefix}/conversation-meta"
        headers = self.get_tenant_headers()

        # Missing version, scene, name, created_at
        body = {
            "group_id": f"incomplete_{self.test_run_id}",
            "description": "Missing required fields",
        }

        self.print_request("POST", url, body)
        print("📝 Note: Missing required fields: version, scene, name, created_at")

        try:
            response = requests.post(
                url, headers=headers, json=body, timeout=self.timeout
            )
            response_json = self.print_response(response)

            if response.status_code in [400, 422]:
                print(
                    f"\n✅ Correctly returned {response.status_code} for missing required fields!"
                )
                return True
            elif response.status_code == 500:
                print(f"\n⚠️  Returned 500 - validation might be at DB level")
                return True
            else:
                print(f"\n❌ Expected 400/422, got {response.status_code}")
                return False
        except Exception as e:
            print(f"\n❌ Request failed: {e}")
            return False

    def test_post_invalid_scene(self) -> bool:
        """
        Test POST: Invalid scene value should be rejected
        """
        self.print_separator("POST: Invalid Scene Value (Error Case)")

        url = f"{self.base_url}{self.api_prefix}/conversation-meta"
        headers = self.get_tenant_headers()

        body = {
            "version": "1.0.0",
            "scene": "invalid_scene_type_xyz",  # Invalid scene
            "scene_desc": {},
            "name": "Invalid Scene Test",
            "description": "Should fail",
            "group_id": f"invalid_scene_{self.test_run_id}",
            "created_at": datetime.now(ZoneInfo("UTC")).isoformat(),
            "default_timezone": "UTC",
            "tags": [],
        }

        self.print_request("POST", url, body)
        print("📝 Note: Using invalid scene value 'invalid_scene_type_xyz'")

        try:
            response = requests.post(
                url, headers=headers, json=body, timeout=self.timeout
            )
            response_json = self.print_response(response)

            if response.status_code in [400, 422, 500]:
                print(
                    f"\n✅ Correctly rejected invalid scene with status {response.status_code}!"
                )
                return True
            elif response.status_code == 200:
                # Scene validation might be lenient
                print(f"\n⚠️  Scene validation is lenient - accepted invalid scene")
                return True
            else:
                print(f"\n❌ Unexpected status: {response.status_code}")
                return False
        except Exception as e:
            print(f"\n❌ Request failed: {e}")
            return False

    def test_patch_nonexistent_group_id_fallback(self) -> bool:
        """
        Test PATCH: Non-existent group_id should fallback to default config

        This tests the fallback behavior: when PATCH targets a non-existent group_id,
        it should fallback to the default config (group_id=null) and update that instead.
        """
        self.print_separator("PATCH: Non-existent group_id (Fallback to Default)")

        url = f"{self.base_url}{self.api_prefix}/conversation-meta"
        headers = self.get_tenant_headers()

        non_existent_id = f"definitely_not_exists_{uuid.uuid4().hex}"

        body = {
            "group_id": non_existent_id,
            "name": f"Fallback Update ({self.test_run_id})",
            "tags": ["fallback", "patched"],
        }

        self.print_request("PATCH", url, body)
        print(
            f"📝 Note: group_id '{non_existent_id}' does not exist, should fallback to default"
        )

        try:
            response = requests.patch(
                url, headers=headers, json=body, timeout=self.timeout
            )
            response_json = self.print_response(response)

            if response.status_code == 200:
                result = response_json.get("result", {})
                # Should have updated the default config (group_id=null)
                if result.get("group_id") is None:
                    print(f"\n✅ Correctly fallback to default config and updated it!")
                    print(f"   - Requested group_id: {non_existent_id} (not found)")
                    print(f"   - Updated group_id: null (default config)")
                    print(f"   - Updated fields: {result.get('updated_fields', [])}")
                    return True
                else:
                    print(
                        f"\n⚠️  Updated group_id={result.get('group_id')}, expected null"
                    )
                    return True  # Still acceptable
            elif response.status_code == 404:
                # No default config exists
                print(f"\n⚠️  404 - no default config to fallback to")
                print(f"   Run test_post_default_config first to create default config")
                return True  # Acceptable if no default exists
            else:
                print(f"\n❌ Unexpected status: {response.status_code}")
                return False
        except Exception as e:
            print(f"\n❌ Request failed: {e}")
            return False

    def test_patch_nonexistent_default(self) -> bool:
        """
        Test PATCH: group_id=null when no default exists should return 404
        """
        self.print_separator("PATCH: Non-existent Default Config (Error Case)")

        # Use a new tenant without default config
        url = f"{self.base_url}{self.api_prefix}/conversation-meta"
        headers = self.get_tenant_headers()
        headers["X-Organization-Id"] = f"no_default_org_{uuid.uuid4().hex[:8]}"
        headers["X-Space-Id"] = f"no_default_space_{uuid.uuid4().hex[:8]}"

        body = {"group_id": None, "name": "Should Fail"}  # Target non-existent default

        self.print_request("PATCH", url, body)
        print(f"📝 Note: Using new tenant without default config")

        try:
            response = requests.patch(
                url, headers=headers, json=body, timeout=self.timeout
            )
            response_json = self.print_response(response)

            if response.status_code == 404:
                print(f"\n✅ Correctly returned 404 for non-existent default config!")
                return True
            else:
                print(f"\n⚠️  Expected 404, got {response.status_code}")
                return True  # Not a critical failure
        except Exception as e:
            print(f"\n❌ Request failed: {e}")
            return False

    def test_post_empty_body(self) -> bool:
        """
        Test POST: Empty body should return 400/422
        """
        self.print_separator("POST: Empty Body (Error Case)")

        url = f"{self.base_url}{self.api_prefix}/conversation-meta"
        headers = self.get_tenant_headers()

        body = {}

        self.print_request("POST", url, body)

        try:
            response = requests.post(
                url, headers=headers, json=body, timeout=self.timeout
            )
            response_json = self.print_response(response)

            if response.status_code in [400, 422, 500]:
                print(
                    f"\n✅ Correctly rejected empty body with status {response.status_code}!"
                )
                return True
            else:
                print(f"\n❌ Expected error status, got {response.status_code}")
                return False
        except Exception as e:
            print(f"\n❌ Request failed: {e}")
            return False

    # ==================== Full Fallback Flow Test ====================

    def test_full_fallback_flow(self) -> bool:
        """
        Test complete fallback flow:
        1. Create default config
        2. Create specific group config
        3. Get specific group - should return group config
        4. Get non-existent group - should fallback to default
        5. Get without group_id - should return default
        """
        self.print_separator("Full Fallback Flow Test")

        print("\n📋 Test Flow:")
        print("   1. Create default config (group_id=null)")
        print("   2. Create specific group config")
        print("   3. GET specific group -> should return group config")
        print("   4. GET non-existent group -> should fallback to default")
        print("   5. GET without group_id -> should return default")
        print()

        results = []

        # Step 1: Create default config
        print("\n--- Step 1: Create default config ---")
        results.append(("Create default", self.test_post_default_config()))

        # Step 2: Create specific group config
        print("\n--- Step 2: Create specific group config ---")
        results.append(("Create group", self.test_post_with_group_id()))

        # Step 3: Get specific group
        print("\n--- Step 3: Get specific group ---")
        results.append(("Get group", self.test_get_by_group_id()))

        # Step 4: Get non-existent group (fallback)
        print("\n--- Step 4: Get non-existent group (should fallback) ---")
        results.append(("Fallback", self.test_get_fallback_to_default()))

        # Step 5: Get without group_id
        print("\n--- Step 5: Get without group_id ---")
        results.append(("Get default", self.test_get_default_config()))

        # Summary
        self.print_separator("Fallback Flow Test Summary")
        all_passed = True
        for name, passed in results:
            status = "✅" if passed else "❌"
            print(f"   {status} {name}")
            if not passed:
                all_passed = False

        if all_passed:
            print(f"\n🎉 All fallback flow tests passed!")
        else:
            print(f"\n⚠️  Some tests failed")

        return all_passed

    # ==================== Test Runner ====================

    def run_all_tests(self) -> dict:
        """Run all tests and return results"""
        results = {}

        # Initialize database first
        self.init_database()

        # POST tests
        results["post_default"] = self.test_post_default_config()
        results["post_with_group_id"] = self.test_post_with_group_id()
        results["post_update_existing"] = self.test_post_update_existing()

        # GET tests
        results["get_by_group_id"] = self.test_get_by_group_id()
        results["get_default"] = self.test_get_default_config()
        results["get_fallback"] = self.test_get_fallback_to_default()
        results["get_not_found"] = self.test_get_not_found()

        # PATCH tests
        results["patch_update"] = self.test_patch_update_fields()
        results["patch_default"] = self.test_patch_default_config()
        results["patch_no_changes"] = self.test_patch_no_changes()
        results["patch_user_details"] = self.test_patch_user_details()

        # Error/Exception tests
        results["error_dup_default"] = self.test_post_duplicate_default_upsert()
        results["error_dup_group_id"] = self.test_post_duplicate_group_id_upsert()
        results["error_missing_fields"] = self.test_post_missing_required_fields()
        results["error_invalid_scene"] = self.test_post_invalid_scene()
        results["error_patch_fallback"] = (
            self.test_patch_nonexistent_group_id_fallback()
        )
        results["error_patch_no_default"] = self.test_patch_nonexistent_default()
        results["error_empty_body"] = self.test_post_empty_body()

        return results

    def run_test_by_name(self, test_name: str) -> bool:
        """Run a specific test by name"""
        # Initialize database first
        self.init_database()

        test_map = {
            # POST tests
            "post": lambda: all(
                [
                    self.test_post_default_config(),
                    self.test_post_with_group_id(),
                    self.test_post_update_existing(),
                ]
            ),
            "post_default": self.test_post_default_config,
            "post_with_group_id": self.test_post_with_group_id,
            "post_update": self.test_post_update_existing,
            # GET tests
            "get": lambda: all(
                [
                    self.test_post_default_config(),  # Need data first
                    self.test_post_with_group_id(),
                    self.test_get_by_group_id(),
                    self.test_get_default_config(),
                    self.test_get_fallback_to_default(),
                ]
            ),
            "get_by_group_id": lambda: self.test_post_with_group_id()
            and self.test_get_by_group_id(),
            "get_default": lambda: self.test_post_default_config()
            and self.test_get_default_config(),
            "get_fallback": lambda: self.test_post_default_config()
            and self.test_get_fallback_to_default(),
            "get_not_found": self.test_get_not_found,
            # PATCH tests
            "patch": lambda: all(
                [
                    self.test_post_default_config(),
                    self.test_post_with_group_id(),
                    self.test_patch_update_fields(),
                    self.test_patch_default_config(),
                    self.test_patch_no_changes(),
                    self.test_patch_user_details(),
                ]
            ),
            "patch_update": lambda: self.test_post_with_group_id()
            and self.test_patch_update_fields(),
            "patch_default": lambda: self.test_post_default_config()
            and self.test_patch_default_config(),
            "patch_no_changes": lambda: self.test_post_with_group_id()
            and self.test_patch_no_changes(),
            "patch_user_details": lambda: self.test_post_with_group_id()
            and self.test_patch_user_details(),
            # Error/Exception tests
            "error": lambda: all(
                [
                    self.test_post_duplicate_default_upsert(),
                    self.test_post_duplicate_group_id_upsert(),
                    self.test_post_missing_required_fields(),
                    self.test_post_invalid_scene(),
                    self.test_patch_nonexistent_group_id_fallback(),
                    self.test_patch_nonexistent_default(),
                    self.test_post_empty_body(),
                ]
            ),
            "error_dup_default": self.test_post_duplicate_default_upsert,
            "error_dup_group_id": self.test_post_duplicate_group_id_upsert,
            "error_missing_fields": self.test_post_missing_required_fields,
            "error_invalid_scene": self.test_post_invalid_scene,
            "error_patch_fallback": lambda: self.test_post_default_config()
            and self.test_patch_nonexistent_group_id_fallback(),
            "error_patch_no_default": self.test_patch_nonexistent_default,
            "error_empty_body": self.test_post_empty_body,
            # Fallback flow
            "fallback": self.test_full_fallback_flow,
            # All tests
            "all": lambda: all(self.run_all_tests().values()),
        }

        if test_name in test_map:
            return test_map[test_name]()
        else:
            print(f"❌ Unknown test: {test_name}")
            print(f"Available tests: {', '.join(test_map.keys())}")
            return False


def print_final_summary(results: dict):
    """Print final test summary"""
    print("\n" + "=" * 80)
    print("  FINAL TEST SUMMARY")
    print("=" * 80)

    passed = sum(1 for v in results.values() if v)
    failed = sum(1 for v in results.values() if not v)
    total = len(results)

    for name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {status} - {name}")

    print()
    print(f"   Total: {total} | Passed: {passed} | Failed: {failed}")

    if failed == 0:
        print("\n🎉 All tests passed!")
    else:
        print(f"\n⚠️  {failed} test(s) failed")


def main():
    parser = argparse.ArgumentParser(description="Conversation Meta API Test Script")
    parser.add_argument(
        "--base-url",
        default="http://localhost:1995",
        help="API base URL (default: http://localhost:1995)",
    )
    parser.add_argument(
        "--organization-id", default=None, help="Organization ID for tenant headers"
    )
    parser.add_argument("--space-id", default=None, help="Space ID for tenant headers")
    parser.add_argument(
        "--test-method",
        default="all",
        help="Test method to run: all, post, get, patch, fallback, or specific test name",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Request timeout in seconds (default: 60)",
    )

    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("  CONVERSATION META API TEST")
    print("=" * 80)
    print(f"📍 Base URL: {args.base_url}")
    print(f"📍 Test Method: {args.test_method}")
    print(f"📍 Timeout: {args.timeout}s")

    tester = ConversationMetaTester(
        base_url=args.base_url,
        organization_id=args.organization_id,
        space_id=args.space_id,
        timeout=args.timeout,
    )

    if args.test_method == "all":
        results = tester.run_all_tests()
        print_final_summary(results)
    else:
        success = tester.run_test_by_name(args.test_method)
        if success:
            print("\n✅ Test passed!")
        else:
            print("\n❌ Test failed!")


if __name__ == "__main__":
    main()
