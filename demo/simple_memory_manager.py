"""Simple Memory Manager - Simplified Memory Manager (HTTP API Version)

Encapsulates all HTTP API call details and provides the simplest interface.
"""

import asyncio
import httpx
from typing import List, Dict, Any
from common_utils.datetime_utils import get_now_with_timezone, to_iso_format


class SimpleMemoryManager:
    """Super Simple Memory Manager
    
    Uses HTTP API, no need to worry about internal implementation.
    
    Usage:
        memory = SimpleMemoryManager()
        await memory.store("I love playing soccer")
        results = await memory.search("What sports does the user like?")
    """
    
    def __init__(self, base_url: str = "http://localhost:8001", group_id: str = "default_group"):
        """Initialize the manager
        
        Args:
            base_url: API server address (default: localhost:8001)
            group_id: Group ID (default: default_group)
        """
        self.base_url = base_url
        self.group_id = group_id
        self.group_name = "Simple Demo Group"
        self.memorize_url = f"{base_url}/api/v3/agentic/memorize"
        self.retrieve_url = f"{base_url}/api/v3/agentic/retrieve_lightweight"
        self._message_counter = 0
    
    async def store(self, content: str, sender: str = "User") -> bool:
        """Store a message
        
        Args:
            content: Message content
            sender: Sender name (default: "User")
        
        Returns:
            Success status
        """
        # Generate unique message ID
        self._message_counter += 1
        now = get_now_with_timezone()  # Use project's unified time utility (with timezone)
        message_id = f"msg_{self._message_counter}_{int(now.timestamp() * 1000)}"
        
        # Build message data (completely consistent with test_v3_api_http.py format)
        message_data = {
            "message_id": message_id,
            "create_time": to_iso_format(now),  # Use project's unified time formatting (with timezone)
            "sender": sender,
            "sender_name": sender,  # Consistent with JSON data format
            "type": "text",  # Message type
            "content": content,
            "group_id": self.group_id,
            "group_name": self.group_name,
        }
        
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(self.memorize_url, json=message_data)
                response.raise_for_status()
                result = response.json()
                
                if result.get("status") == "ok":
                    count = result.get("result", {}).get("count", 0)
                    if count > 0:
                        print(f"  ✅ Stored: {content[:40]}... (Extracted {count} memories)")
                    else:
                        print(f"  📝 Recorded: {content[:40]}... (Waiting for more context to extract memories)")
                    return True
                else:
                    print(f"  ❌ Storage failed: {result.get('message')}")
                    return False
                    
        except httpx.ConnectError:
            print(f"  ❌ Cannot connect to API server ({self.base_url})")
            print(f"     Please start first: uv run python src/bootstrap.py start_server.py")
            return False
        except Exception as e:
            print(f"  ❌ Storage failed: {e}")
            return False
    
    async def search(
        self, 
        query: str, 
        top_k: int = 3,
        mode: str = "rrf",
        show_details: bool = True
    ) -> List[Dict[str, Any]]:
        """Search memories
        
        Args:
            query: Query text
            top_k: Number of results to return (default: 3)
            mode: Retrieval mode (default: "rrf")
                - "rrf": RRF fusion (recommended)
                - "embedding": Vector retrieval
                - "bm25": Keyword retrieval
            show_details: Whether to show detailed information (default: True)
        
        Returns:
            List of memories
        """
        payload = {
            "query": query,
            "user_id": "demo_user",
            "top_k": top_k,
            "data_source": "memcell",
            "retrieval_mode": mode,
            "memory_scope": "all",
        }
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(self.retrieve_url, json=payload)
                response.raise_for_status()
                result = response.json()
                
                if result.get("status") == "ok":
                    memories = result.get("result", {}).get("memories", [])
                    metadata = result.get("result", {}).get("metadata", {})
                    latency = metadata.get("total_latency_ms", 0)
                    
                    if show_details:
                        print(f"  🔍 Found {len(memories)} memories (took {latency:.2f}ms)")
                        self._print_memories(memories)
                    
                    return memories
                else:
                    print(f"  ❌ Search failed: {result.get('message')}")
                    return []
                    
        except httpx.ConnectError:
            print(f"  ❌ Cannot connect to API server ({self.base_url})")
            return []
        except Exception as e:
            print(f"  ❌ Search failed: {e}")
            return []
    
    def _print_memories(self, memories: List[Dict[str, Any]]):
        """Print memory details (internal method)"""
        if not memories:
            print("     💡 Tip: No related memories found")
            print("         Possible reasons:")
            print("         - Too little conversation input, system hasn't generated memories yet")
            print("           (This simple demo only demonstrates retrieval, not full memory generation)")
            return
        
        for i, mem in enumerate(memories, 1):
            score = mem.get('score', 0)
            timestamp = mem.get('timestamp', '')[:10]
            subject = mem.get('subject', '')
            summary = mem.get('summary', '')
            episode = mem.get('episode', '')
            
            print(f"\n     [{i}] Relevance: {score:.4f} | Time: {timestamp}")
            if subject:
                print(f"         Subject: {subject}")
            if summary:
                print(f"         Summary: {summary[:60]}...")
            if episode:
                print(f"         Details: {episode[:80]}...")
    
    async def wait_for_index(self, seconds: int = 10):
        """Wait for index building
        
        Args:
            seconds: Wait time in seconds (default: 10)
        """
        print("  💡 Tip: Memory extraction requires sufficient context")
        print("     - Short conversations may only record messages, not generate memories immediately")
        print("     - Multi-turn conversations with specific information are easier to extract memories from")
        print("     - System extracts memories at conversation boundaries (topic changes, time gaps)")
        print(f"  ⏳ Waiting {seconds} seconds to ensure data is written...")
        await asyncio.sleep(seconds)
        print(f"  ✅ Index building completed")
    
    def print_separator(self, text: str = ""):
        """Print separator line"""
        if text:
            print(f"\n{'='*60}")
            print(f"{text}")
            print('='*60)
        else:
            print('-'*60)
    
    def print_summary(self):
        """Print usage summary and tips"""
        print("\n" + "="*60)
        print("✅ Demo completed!")
        print("="*60)
        print("\n📚 About Memory Extraction:")
        print("   The memory system uses intelligent extraction strategy, not recording all conversations:")
        print("   - ✅ Will extract: Conversations with specific info, opinions, preferences, events")
        print("   - ❌ Won't extract: Too brief, low-information small talk")
        print("   - 🎯 Best practice: Multi-turn conversations, rich context, specific details")