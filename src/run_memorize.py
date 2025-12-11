#!/usr/bin/env python3
"""
Group chat memory storage script

Read JSON files in GroupChatFormat format, convert and call memorize interface to store memories

Usage:
    # Call memorize interface: simple direct single message format, process one by one
    python src/bootstrap.py src/run_memorize.py --input data/group_chat.json --api-url http://localhost:1995/api/v1/memories

    # Validate format only
    python src/bootstrap.py src/run_memorize.py --input data/example.json --validate-only
"""

import json
import argparse
import sys
import asyncio
import time
from pathlib import Path
from typing import Dict, Any, Optional

ALLOWED_SCENES: tuple[str, str] = ("assistant", "companion")
# todo xinzegao originally was group_chat

from infra_layer.adapters.input.api.mapper.group_chat_converter import (
    validate_group_chat_format_input,
)
from core.observation.logger import get_logger
from common_utils.datetime_utils import get_timezone

logger = get_logger(__name__)


class GroupChatMemorizer:
    """Group chat memory storage processing class"""

    def __init__(self, api_url: str, scene: str = "assistant"):
        """
        Initialize

        Args:
            api_url: memorize API address (required)
            scene: memory extraction scene (default "assistant")
        """
        self.api_url = api_url
        self.scene = scene

    def validate_input_file(self, file_path: str) -> bool:
        """
        Validate input file format

        Args:
            file_path: input file path

        Returns:
            whether validation passed
        """
        logger.info("=" * 70)
        logger.info("Validating input file format")
        logger.info("=" * 70)

        try:
            # Read file
            logger.info(f"Reading file: {file_path}")
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Validate format
            logger.info("Validating GroupChatFormat format...")
            is_valid = validate_group_chat_format_input(data)

            if is_valid:
                logger.info("✓ Format validation passed!")

                # Output statistics
                meta = data.get("conversation_meta", {})
                messages = data.get("conversation_list", [])

                logger.info("\n=== Data Statistics ===")
                logger.info(f"Format version: {data.get('version', 'N/A')}")
                logger.info(f"Group name: {meta.get('name', 'N/A')}")
                logger.info(f"Group ID: {meta.get('group_id', 'N/A')}")
                logger.info(f"Number of users: {len(meta.get('user_details', {}))}")
                logger.info(f"Number of messages: {len(messages)}")

                if messages:
                    first_time = messages[0].get('create_time', 'N/A')
                    last_time = messages[-1].get('create_time', 'N/A')
                    logger.info(f"Time range: {first_time} ~ {last_time}")

                return True
            else:
                logger.error("✗ Format validation failed!")
                logger.error(
                    "Please ensure the input file conforms to the GroupChatFormat specification"
                )
                return False

        except json.JSONDecodeError as e:
            logger.error(f"✗ JSON parsing failed: {e}")
            return False
        except Exception as e:
            logger.error(f"✗ Validation failed: {e}")
            import traceback

            traceback.print_exc()
            return False

    async def process_with_api(self, group_chat_data: Dict[str, Any]) -> bool:
        """
        Process one by one via API (using simple direct single message format)

        Args:
            group_chat_data: data in GroupChatFormat format

        Returns:
            whether successful
        """
        logger.info("\n" + "=" * 70)
        logger.info("Starting to call memorize API one by one")
        logger.info("=" * 70)

        try:
            import httpx

            meta = group_chat_data.get("conversation_meta", {})
            messages = group_chat_data.get("conversation_list", [])

            group_id = meta.get("group_id")
            group_name = meta.get("name")

            logger.info(f"Group name: {group_name or 'N/A'}")
            logger.info(f"Group ID: {group_id or 'N/A'}")
            logger.info(f"Number of messages: {len(messages)}")
            logger.info(f"API address: {self.api_url}")

            # ========== Step 1: First call conversation-meta interface to save scene ==========
            async with httpx.AsyncClient(timeout=30.0) as client:
                logger.info(
                    "\n--- Saving conversation metadata (conversation-meta) ---"
                )

                # Build conversation-meta request data
                conversation_meta_request = {
                    "version": group_chat_data.get("version", "1.0.0"),
                    "scene": self.scene,  # Use scene passed from command line
                    "scene_desc": meta.get("scene_desc", {}),
                    "name": meta.get("name", "Unnamed conversation"),
                    "description": meta.get("description", ""),
                    "group_id": group_id,
                    "created_at": meta.get("created_at", ""),
                    "default_timezone": meta.get(
                        "default_timezone", get_timezone().key
                    ),
                    "user_details": meta.get("user_details", {}),
                    "tags": meta.get("tags", []),
                }

                # Get conversation-meta API address (constructed based on memories API)
                # Assume memories API is http://host:port/api/v1/memories
                # Then conversation-meta API is http://host:port/api/v1/memories/conversation-meta
                conversation_meta_url = f"{self.api_url}/conversation-meta"

                logger.info(f"Saving conversation metadata to: {conversation_meta_url}")
                logger.info(f"Scene: {self.scene}, Group ID: {group_id}")

                try:
                    response = await client.post(
                        conversation_meta_url,
                        json=conversation_meta_request,
                        headers={"Content-Type": "application/json"},
                    )

                    if response.status_code == 200:
                        result = response.json()
                        logger.info(f"  ✓ Conversation metadata saved successfully")
                        logger.info(f"  Scene: {self.scene}")
                    else:
                        logger.warning(
                            f"  ⚠ Failed to save conversation metadata: {response.status_code}"
                        )
                        logger.warning(f"  Response content: {response.text}")
                        logger.warning(f"  Continuing to process messages...")

                except Exception as e:
                    logger.warning(f"  ⚠ Error saving conversation metadata: {e}")
                    logger.warning(f"  Continuing to process messages...")

            # ========== Step 2: Process messages one by one ==========

            total_memories = 0
            success_count = 0

            async with httpx.AsyncClient(timeout=300.0) as client:
                for i, message in enumerate(messages):
                    logger.info(f"\n--- Processing message {i+1}/{len(messages)} ---")

                    # Build simple direct single message format
                    request_data = {
                        "message_id": message.get("message_id"),
                        "create_time": message.get("create_time"),
                        "sender": message.get("sender"),
                        "sender_name": message.get("sender_name"),
                        "content": message.get("content"),
                        "refer_list": message.get("refer_list", []),
                    }

                    # Add optional group information
                    if group_id:
                        request_data["group_id"] = group_id
                    if group_name:
                        request_data["group_name"] = group_name

                    # Send request
                    try:
                        response = await client.post(
                            self.api_url,
                            json=request_data,
                            headers={"Content-Type": "application/json"},
                        )

                        if response.status_code == 200:
                            result = response.json()
                            result_data = result.get('result', {})
                            memory_count = result_data.get('count', 0)

                            total_memories += memory_count
                            success_count += 1
                            if memory_count > 0:
                                logger.info(
                                    f"  ✓ Successfully saved {memory_count} memories"
                                )
                            else:
                                logger.info(f"  ⏳ Waiting for scene boundary")
                            # Add delay to avoid processing too fast
                            time.sleep(0.1)

                        else:
                            logger.error(f"  ✗ API call failed: {response.status_code}")
                            logger.error(f"  Response content: {response.text}")

                    except Exception as e:
                        logger.error(f"  ✗ Processing failed: {e}")

            # Output summary
            logger.info("\n" + "=" * 70)
            logger.info("Processing completed")
            logger.info("=" * 70)
            logger.info(
                f"✓ Successfully processed: {success_count}/{len(messages)} messages"
            )
            logger.info(f"✓ Total saved: {total_memories} memories")

            return success_count == len(messages)

        except ImportError:
            logger.error("✗ httpx library is required: pip install httpx")
            return False
        except Exception as e:
            logger.error(f"✗ Processing failed: {e}")
            import traceback

            traceback.print_exc()
            return False

    async def process_file(self, file_path: str) -> bool:
        """
        Process group chat file

        Args:
            file_path: input file path

        Returns:
            whether successful
        """
        # First validate format
        if not self.validate_input_file(file_path):
            return False

        # Check API address
        if not self.api_url:
            logger.error(
                "✗ API address not provided, please specify using --api-url parameter"
            )
            return False

        try:
            # Read file
            logger.info("\n" + "=" * 70)
            logger.info("Reading group chat data")
            logger.info("=" * 70)
            logger.info(f"Reading file: {file_path}")
            with open(file_path, 'r', encoding='utf-8') as f:
                group_chat_data = json.load(f)

            # Sequential interface: directly send GroupChatFormat format, process one by one
            logger.info(
                "Using simple direct single message format, processing one by one"
            )
            return await self.process_with_api(group_chat_data)

        except Exception as e:
            logger.error(f"✗ Failed to read or process: {e}")
            import traceback

            traceback.print_exc()
            return False


async def async_main():
    """Asynchronous main function"""
    parser = argparse.ArgumentParser(
        description='Group chat memory storage script',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # Call memorize interface: simple direct single message format, process one by one
  python src/bootstrap.py src/run_memorize.py --input data/group_chat.json --api-url http://localhost:1995/api/v1/memories
  
  # Validate format only (API address not required)
  python src/bootstrap.py src/run_memorize.py --input data/group_chat.json --validate-only
  
Input file format:
  Input file must conform to GroupChatFormat specification, refer to data_format/group_chat/group_chat_format.py
        """,
    )

    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input group chat JSON file path (GroupChatFormat format)',
    )
    parser.add_argument(
        '--api-url',
        type=str,
        help='memorize API address (required, unless using --validate-only)',
    )
    parser.add_argument(
        '--scene',
        type=str,
        choices=ALLOWED_SCENES,
        required=True,
        help='Memory extraction scene (required, only supports assistant or companion)',
    )
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Validate input file format only, do not call API',
    )

    args = parser.parse_args()

    # Process input file path
    input_file = Path(args.input)
    if not input_file.is_absolute():
        # Relative path, relative to current working directory
        input_file = Path.cwd() / input_file

    if not input_file.exists():
        logger.error(f"Error: Input file does not exist: {input_file}")
        sys.exit(1)

    logger.info("🚀 Group chat memory storage script")
    logger.info("=" * 70)
    logger.info(f"📄 Input file: {input_file}")
    logger.info(f"🔍 Validation mode: {'Yes' if args.validate_only else 'No'}")
    if args.api_url:
        logger.info(f"🌐 API address: {args.api_url}")
    logger.info("=" * 70)

    # If validation mode only, validate and exit
    if args.validate_only:
        # Validation mode does not require API address
        memorizer = GroupChatMemorizer(
            api_url="", scene=args.scene
        )  # Pass empty string as placeholder
        success = memorizer.validate_input_file(str(input_file))
        if success:
            logger.info("\n✓ Validation completed, file format is correct!")
            sys.exit(0)
        else:
            logger.error("\n✗ Validation failed, file format is incorrect!")
            sys.exit(1)

    # Non-validation mode, API address must be provided
    if not args.api_url:
        logger.error("✗ Error: --api-url parameter must be provided")
        logger.error("   Usage:")
        logger.error(
            "     python src/bootstrap.py src/run_memorize.py --input <file> --api-url http://localhost:1995/api/v1/memories"
        )
        logger.error("   Or use --validate-only to validate format only")
        sys.exit(1)

    # Create processor and process file
    memorizer = GroupChatMemorizer(api_url=args.api_url, scene=args.scene)
    success = await memorizer.process_file(str(input_file))

    if success:
        logger.info("\n" + "=" * 70)
        logger.info("✓ Processing completed!")
        logger.info("=" * 70)
    else:
        logger.error("\n" + "=" * 70)
        logger.error("✗ Processing failed!")
        logger.error("=" * 70)


def main():
    """Synchronous main function entry"""
    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        logger.warning("\n⚠️ User interrupted execution")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n❌ Execution failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
