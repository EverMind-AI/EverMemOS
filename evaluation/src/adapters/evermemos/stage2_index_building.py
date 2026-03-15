import json
import os
import sys
import pickle
from pathlib import Path
from datetime import datetime
from typing import Any

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from rank_bm25 import BM25Okapi
import asyncio




from evaluation.src.adapters.evermemos.config import ExperimentConfig
from agentic_layer import vectorize_service
from api_specs.memory_types import MemCell, RawDataType
from common_utils.datetime_utils import (
    from_iso_format,
    from_timestamp,
    get_now_with_timezone,
)
from memory_layer.llm.llm_provider import LLMProvider
from memory_layer.profile_manager import ProfileManager, ProfileManagerConfig


def ensure_nltk_data():
    """Ensure required NLTK data is downloaded."""
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        print("Downloading punkt...")
        nltk.download("punkt", quiet=True)
    
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        print("Downloading punkt_tab...")
        nltk.download("punkt_tab", quiet=True)

    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        print("Downloading stopwords...")
        nltk.download("stopwords", quiet=True)
    
    # Verify stopwords availability
    try:
        from nltk.corpus import stopwords
        test_stopwords = stopwords.words("english")
        if not test_stopwords:
            raise ValueError("Stopwords is empty")
    except Exception as e:
        print(f"Warning: NLTK stopwords error: {e}")
        print("Re-downloading stopwords...")
        nltk.download("stopwords", quiet=False, force=True)


def build_searchable_text(doc: dict) -> str:
    """
    Build searchable text from a document with weighted fields.

    Priority:
    1. If event_log exists, use atomic_fact for indexing
    2. Otherwise, fall back to original fields:
       - "subject" corresponds to "title" (weight * 3)
       - "summary" corresponds to "summary" (weight * 2)
       - "episode" corresponds to "content" (weight * 1)
    """
    parts = []

    # Prefer event_log's atomic_fact (if exists)
    if doc.get("event_log") and doc["event_log"].get("atomic_fact"):
        atomic_facts = doc["event_log"]["atomic_fact"]
        if isinstance(atomic_facts, list):
            # Handle nested atomic_fact structure
            # atomic_fact can be list of strings or list of dicts (containing "fact" and "embedding")
            for fact in atomic_facts:
                if isinstance(fact, dict) and "fact" in fact:
                    # New format: {"fact": "...", "embedding": [...]}
                    parts.append(fact["fact"])
                elif isinstance(fact, str):
                    # Old format: pure string list (backward compatible)
                    parts.append(fact)
            # Continue to add other fields for better recall
    
    # Fall back to original fields (maintain backward compatibility)
    # Title has highest weight (use once, rely on BM25 length normalization)
    if doc.get("subject"):
        parts.append(doc["subject"])

    # Summary (use once)
    if doc.get("summary"):
        parts.append(doc["summary"])

    # Content - only add if we have no other content or if specifically desired
    # For now, let's keep it additive but maybe not replicate episode if facts exist to avoid noise?
    # The original logic was exclusive. 
    # If we have atomic facts, usually we don't need raw episode. 
    # But subject/summary are high value.
    if not parts and doc.get("episode"):
        parts.append(doc["episode"])

    return " ".join(str(part) for part in parts if part)


def tokenize(text: str, stemmer, stop_words: set) -> list[str]:
    """
    NLTK-based tokenization with stemming and stopword removal.
    """
    if not text:
        return []

    tokens = word_tokenize(text.lower())

    processed_tokens = [
        stemmer.stem(token)
        for token in tokens
        if token.isalpha() and len(token) >= 2 and token not in stop_words
    ]

    return processed_tokens


def extract_atomic_facts(doc: dict) -> list[str]:
    """
    Extract atomic_fact texts from a document.
    Also includes high-value fields like subject and summary to improve retrieval.
    
    Returns:
        List of strings (facts, subject, summary).
    """
    facts = []

    # 1. Atomic Facts
    if doc.get("event_log") and doc["event_log"].get("atomic_fact"):
        atomic_facts = doc["event_log"]["atomic_fact"]
        if isinstance(atomic_facts, list):
            for fact in atomic_facts:
                if isinstance(fact, dict) and "fact" in fact:
                    facts.append(fact["fact"])
                elif isinstance(fact, str):
                    facts.append(fact)

    # 2. Subject (Title) - High value for topic matching
    # Optimization: Do not replicate multiple times. 
    # BM25 algorithm inherently favors shorter fields (Length Normalization).
    # Adding it once is sufficient and avoids skewing the term frequency excessively.
    if doc.get("subject"):
        facts.append(doc["subject"])

    # 3. Summary - High value for overview matching
    # Optimization: Add once. Summary provides rich context.
    if doc.get("summary"):
        facts.append(doc["summary"])
    
    # 4. Fallback: use episode only if no other info is available
    if not facts and doc.get("episode"):
        return [doc["episode"]]
    
    return facts


def _parse_timestamp(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, (int, float)):
        return from_timestamp(value)
    if isinstance(value, str) and value.strip():
        return from_iso_format(value)
    return get_now_with_timezone()


def _filter_user_ids(user_ids: list[str]) -> list[str]:
    filtered = []
    for uid in user_ids:
        if not uid:
            continue
        uid_lower = str(uid).lower()
        if "assistant" in uid_lower or "robot" in uid_lower:
            continue
        filtered.append(str(uid))
    return filtered


def _memcell_from_dict(data: dict) -> MemCell | None:
    original_data = data.get("original_data") or []
    if not original_data:
        return None

    timestamp = _parse_timestamp(data.get("timestamp"))
    raw_type = data.get("type")
    memcell_type = RawDataType.from_string(raw_type) if raw_type else None

    return MemCell(
        user_id_list=data.get("user_id_list") or [],
        original_data=original_data,
        timestamp=timestamp,
        summary=data.get("summary"),
        event_id=data.get("event_id"),
        group_id=data.get("group_id"),
        group_name=data.get("group_name"),
        participants=data.get("participants"),
        type=memcell_type,
        keywords=data.get("keywords"),
        subject=data.get("subject"),
        linked_entities=data.get("linked_entities"),
        episode=data.get("episode"),
        foresights=data.get("foresights"),
        event_log=data.get("event_log"),
        extend=data.get("extend"),
    )


def build_bm25_index(
    config: ExperimentConfig, data_dir: Path, bm25_save_dir: Path
) -> list[list[float]]:
    """
    Build BM25 index with MaxSim support.
    
    Index structure:
    - fact_corpus: List of tokenized atomic_facts (one entry per fact)
    - fact_to_doc_idx: Maps fact index to document index
    - docs: Original documents
    
    This enables MaxSim strategy: search at fact level, aggregate by document.
    """
    # --- NLTK Setup ---
    print("Ensuring NLTK data is available...")
    ensure_nltk_data()
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words("english"))

    print(f"Reading data from: {data_dir}")

    for i in range(config.num_conv):
        file_path = data_dir / f"memcell_list_conv_{i}.json"
        if not file_path.exists():
            print(f"Warning: File not found, skipping: {file_path}")
            continue

        print(f"\nProcessing {file_path.name}...")

        # Fact-level corpus for MaxSim BM25
        fact_corpus = []           # Tokenized facts
        fact_to_doc_idx = []       # Maps fact index -> doc index
        original_docs = []

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

            for doc_idx, doc in enumerate(data):
                original_docs.append(doc)
                
                # Extract atomic_facts from document
                facts = extract_atomic_facts(doc)
                
                for fact_text in facts:
                    tokenized_fact = tokenize(fact_text, stemmer, stop_words)
                    if tokenized_fact:  # Only add non-empty facts
                        fact_corpus.append(tokenized_fact)
                        fact_to_doc_idx.append(doc_idx)

        if not fact_corpus:
            print(
                f"Warning: No facts found in {file_path.name}. Skipping index creation."
            )
            continue

        print(f"Processed {len(original_docs)} documents, {len(fact_corpus)} atomic_facts from {file_path.name}.")
        print(f"  Average facts per document: {len(fact_corpus) / len(original_docs):.1f}")

        # --- BM25 Indexing (fact-level) ---
        print(f"Building BM25 MaxSim index for {file_path.name}...")
        bm25 = BM25Okapi(fact_corpus)

        # --- Saving the Index ---
        # New format: includes fact_to_doc_idx for MaxSim aggregation
        index_data = {
            "bm25": bm25,
            "docs": original_docs,
            "fact_to_doc_idx": fact_to_doc_idx,  # For MaxSim aggregation
            "index_type": "maxsim",  # Mark as MaxSim index
        }

        output_path = bm25_save_dir / f"bm25_index_conv_{i}.pkl"
        print(f"Saving index to: {output_path}")
        with open(output_path, "wb") as f:
            pickle.dump(index_data, f)


async def build_emb_index(config: ExperimentConfig, data_dir: Path, emb_save_dir: Path):
    """
    Build Embedding index (stable version).
    
    Performance optimization strategy:
    1. Controlled concurrency: strictly follow API Semaphore(5) limit
    2. Conservative batch size: 256 texts/batch (avoid timeouts)
    3. Serial batch submission: grouped submission to avoid queue buildup
    4. Progress monitoring: real-time progress and speed display
    
    Optimization effects:
    - Stability first, avoid timeouts and API overload
    - API concurrency: 5 (controlled by vectorize_service.Semaphore)
    - Batch size: 256 (balance stability and efficiency)
    """
    # Conservative batch size (avoid timeouts)
    BATCH_SIZE = 256  # Use larger batches (single API call processes more, reduce request count)
    MAX_CONCURRENT_BATCHES = 5  # Strictly control concurrency (match Semaphore(5))
    
    import time  # For performance statistics
    
    for i in range(config.num_conv):
        file_path = data_dir / f"memcell_list_conv_{i}.json"
        if not file_path.exists():
            print(f"Warning: File not found, skipping: {file_path}")
            continue

        print(f"\n{'='*60}")
        print(f"Processing {file_path.name} for embedding...")
        print(f"{'='*60}")

        with open(file_path, "r", encoding="utf-8") as f:
            original_docs = json.load(f)

        texts_to_embed = []
        doc_field_map = []
        for doc_idx, doc in enumerate(original_docs):
            # 1. Subject (Title) - High value for topic matching
            if doc.get("subject"):
                texts_to_embed.append(doc["subject"])
                doc_field_map.append((doc_idx, "subject"))

            # 2. Summary - High value for overview/context matching
            if doc.get("summary"):
                summary_text = doc["summary"]
                # Optimization: Simple noise reduction for summary
                # Remove common meta-prefixes that dilute embedding quality
                prefixes_to_remove = [
                    "In this conversation,", "The conversation is about", 
                    "This dialogue discusses", "Here, "
                ]
                clean_summary = summary_text
                for prefix in prefixes_to_remove:
                    if clean_summary.startswith(prefix):
                        clean_summary = clean_summary[len(prefix):].strip()
                
                # Check cleanliness again
                if clean_summary:
                    texts_to_embed.append(clean_summary)
                    doc_field_map.append((doc_idx, "summary"))

            # 3. Atomic Facts - Detailed matching
            if doc.get("event_log") and doc["event_log"].get("atomic_fact"):
                atomic_facts = doc["event_log"]["atomic_fact"]
                if isinstance(atomic_facts, list) and atomic_facts:
                    # calculate embedding for each atomic_fact separately (MaxSim strategy)
                    # This precisely matches specific atomic facts, avoiding semantic dilution
                    for fact_idx, fact in enumerate(atomic_facts):
                        # compatible with both formats (string / dict)
                        fact_text = None
                        if isinstance(fact, dict) and "fact" in fact:
                            # New format: {"fact": "...", "embedding": [...]}
                            fact_text = fact["fact"]
                        elif isinstance(fact, str):
                            # Old format: pure string
                            fact_text = fact
                        
                        # Ensure fact is non-empty
                        if fact_text and fact_text.strip():
                            texts_to_embed.append(fact_text)
                            doc_field_map.append((doc_idx, f"atomic_fact_{fact_idx}"))
                    
                    # If we have facts, we continue to next doc (we already added subject/summary above)
                    continue

            # 4. Fallback: Episode (only if no facts)
            # Note: We already added subject/summary if they exist.
            # If no atomic facts, use episode as a fallback "fact".
            if doc.get("episode"):
                texts_to_embed.append(doc["episode"])
                doc_field_map.append((doc_idx, "episode"))

        if not texts_to_embed:
            print(
                f"Warning: No documents found in {file_path.name}. Skipping embedding creation."
            )
            continue

        total_texts = len(texts_to_embed)
        total_batches = (total_texts + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"Total texts to embed: {total_texts}")
        print(f"Batch size: {BATCH_SIZE}")
        print(f"Total batches: {total_batches}")
        print(f"Max concurrent batches: {MAX_CONCURRENT_BATCHES}")
        print(f"\nStarting parallel embedding generation...")
        
        # Stable batch processing (avoid timeouts)
        start_time = time.time()
        
        async def process_batch_with_retry(batch_idx: int, batch_texts: list, max_retries: int = 3) -> tuple[int, list]:
            """Process single batch (async + retry)."""
            for attempt in range(max_retries):
                try:
                    # Call API to get embeddings (concurrency controlled by Semaphore(5))
                    batch_embeddings = await vectorize_service.get_vectorize_service().get_embeddings(batch_texts)
                    return (batch_idx, batch_embeddings)
                except Exception as e:
                    if attempt < max_retries - 1:
                        wait_time = 2.0 * (2 ** attempt)  # Exponential backoff: 2s, 4s
                        print(f"  ⚠️  Batch {batch_idx + 1}/{total_batches} failed (attempt {attempt + 1}), retrying in {wait_time:.1f}s: {e}")
                        await asyncio.sleep(wait_time)
                    else:
                        print(f"  ❌ Batch {batch_idx + 1}/{total_batches} failed after {max_retries} attempts: {e}")
                        return (batch_idx, [])
        
        #Grouped serial submission (avoid queue buildup causing timeouts)
        print(f"Processing {total_batches} batches in groups of {MAX_CONCURRENT_BATCHES}...")
        
        batch_results = []
        completed = 0
        
        # Grouped submission, max MAX_CONCURRENT_BATCHES concurrent per group
        for group_start in range(0, total_texts, BATCH_SIZE * MAX_CONCURRENT_BATCHES):
            # Calculate batch range for current group
            group_end = min(group_start + BATCH_SIZE * MAX_CONCURRENT_BATCHES, total_texts)
            group_tasks = []
            
            for j in range(group_start, group_end, BATCH_SIZE):
                batch_idx = j // BATCH_SIZE
                batch_texts = texts_to_embed[j : j + BATCH_SIZE]
                task = process_batch_with_retry(batch_idx, batch_texts)
                group_tasks.append(task)
            
            # Process current group concurrently (max MAX_CONCURRENT_BATCHES)
            print(f"  Group {group_start//BATCH_SIZE//MAX_CONCURRENT_BATCHES + 1}: Processing {len(group_tasks)} batches concurrently...")
            group_results = await asyncio.gather(*group_tasks, return_exceptions=False)
            batch_results.extend(group_results)
            
            completed += len(group_tasks)
            progress = (completed / total_batches) * 100
            print(f"  Progress: {completed}/{total_batches} batches ({progress:.1f}%)")
            
            # Inter-group delay (give API server breathing room)
            if group_end < total_texts:
                await asyncio.sleep(1.0)  # 1s inter-group delay
        
        # Reorganize results by batch order
        all_embeddings = []
        for batch_idx, batch_embeddings in sorted(batch_results, key=lambda x: x[0]):
            all_embeddings.extend(batch_embeddings)
        
        elapsed_time = time.time() - start_time
        speed = total_texts / elapsed_time if elapsed_time > 0 else 0
        print(f"\n✅ Embedding generation complete!")
        print(f"   - Total texts: {total_texts}")
        print(f"   - Total embeddings: {len(all_embeddings)}")
        print(f"   - Time elapsed: {elapsed_time:.2f}s")
        print(f"   - Speed: {speed:.1f} texts/sec")
        print(f"   - Average batch time: {elapsed_time/total_batches:.2f}s")
        
        # Verify result completeness
        if len(all_embeddings) != total_texts:
            print(f"   ⚠️  Warning: Expected {total_texts} embeddings, got {len(all_embeddings)}")
        else:
            print(f"   ✓ All embeddings generated successfully")

        # Re-associate embeddings with their original documents and fields
        # Support multiple atomic_fact embeddings per document (for MaxSim strategy)
        doc_embeddings = [{"doc": doc, "embeddings": {}} for doc in original_docs]
        
        for (doc_idx, field), emb in zip(doc_field_map, all_embeddings):
            # If atomic_fact field, save as list (support multiple atomic_facts)
            if field.startswith("atomic_fact_"):
                if "atomic_facts" not in doc_embeddings[doc_idx]["embeddings"]:
                    doc_embeddings[doc_idx]["embeddings"]["atomic_facts"] = []
                doc_embeddings[doc_idx]["embeddings"]["atomic_facts"].append(emb)
            else:
                # Save other fields directly
                doc_embeddings[doc_idx]["embeddings"][field] = emb

        # The final structure of the saved .pkl file will be a list of dicts:
        # [
        #     {
        #         "doc": { ... original document ... },
        #         "embeddings": {
        #             "atomic_facts": [  # New: atomic_fact embeddings list (for MaxSim)
        #                 [ ... embedding vector for fact 0 ... ],
        #                 [ ... embedding vector for fact 1 ... ],
        #                 ...
        #             ],
        #             "subject": [ ... embedding vector ... ],  # Backward compatible legacy fields
        #             "summary": [ ... embedding vector ... ],
        #             "episode": [ ... embedding vector ... ]
        #         }
        #     },
        #     ...
        # ]
        output_path = emb_save_dir / f"embedding_index_conv_{i}.pkl"
        emb_save_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving embeddings to: {output_path}")
        with open(output_path, "wb") as f:
            pickle.dump(doc_embeddings, f)


def build_scene_index(
    config: ExperimentConfig,
    data_dir: Path,
    cluster_dir: Path,
    scene_save_dir: Path,
) -> None:
    """
    Build scene index from clustering results (hypergraph hyperedges).
    
    This function reads the clustering results generated by ClusterManager in Stage 1
    and builds a scene index for scene-first retrieval.
    
    Hypergraph architecture:
    - Nodes: MemCells (with episode, event_log, foresight attributes)
    - Hyperedges: Scenes (clustering results connecting semantically related MemCells)
    
    Input:
    - memcells/memcell_list_conv_{i}.json: MemCell data
    - clusters/conv_{i}/cluster_state_*.json: Clustering results from Stage 1
    
    Output:
    - scenes/scene_index_conv_{i}.pkl: Scene index
    
    Scene index structure:
    {
        "scenes": [
            {
                "scene_id": "cluster_001",
                "centroid": [...],  # Scene vector (cluster centroid)
                "memcell_ids": ["event_id_1", "event_id_2", ...],
                "memcell_count": 3,
                "last_timestamp": 1699999999.0
            },
            ...
        ],
        "memcell_to_scene": {
            "event_id_1": "cluster_001",
            "event_id_2": "cluster_001",
            ...
        },
        "total_scenes": 5,
        "total_memcells": 20
    }
    
    Args:
        config: Experiment configuration
        data_dir: Directory containing memcell JSON files
        cluster_dir: Directory containing cluster state JSON files
        scene_save_dir: Directory to save scene index files
    """
    import numpy as np
    
    print(f"\n{'='*60}")
    print(f"Building Scene Index (Hypergraph Hyperedges)")
    print(f"{'='*60}")
    
    scene_save_dir.mkdir(parents=True, exist_ok=True)
    
    for i in range(config.num_conv):
        # Check if memcell file exists
        memcell_file = data_dir / f"memcell_list_conv_{i}.json"
        if not memcell_file.exists():
            print(f"Warning: MemCell file not found, skipping: {memcell_file}")
            continue
        
        # Check if cluster state file exists
        conv_cluster_dir = cluster_dir / f"conv_{i}"
        cluster_state_file = conv_cluster_dir / f"cluster_state_conv_{i}.json"
        
        if not cluster_state_file.exists():
            print(f"Warning: Cluster state not found for conv_{i}, skipping scene index")
            print(f"  Expected: {cluster_state_file}")
            print(f"  Hint: Make sure enable_clustering=True in Stage 1")
            continue
        
        print(f"\nProcessing conv_{i}...")
        
        # Load memcell data (for validation and metadata)
        with open(memcell_file, "r", encoding="utf-8") as f:
            memcells = json.load(f)
        
        # Build event_id to memcell mapping
        memcell_map = {mc.get("event_id"): mc for mc in memcells if mc.get("event_id")}
        
        # Load cluster state
        with open(cluster_state_file, "r", encoding="utf-8") as f:
            cluster_state = json.load(f)
        
        # Extract clustering information
        eventid_to_cluster = cluster_state.get("eventid_to_cluster", {})
        cluster_centroids = cluster_state.get("cluster_centroids", {})
        cluster_counts = cluster_state.get("cluster_counts", {})
        cluster_last_ts = cluster_state.get("cluster_last_ts", {})
        
        if not eventid_to_cluster:
            print(f"  Warning: No cluster assignments found for conv_{i}, skipping")
            continue
        
        # Build scene index structure
        # Group memcells by cluster_id
        cluster_to_memcells = {}
        for event_id, cluster_id in eventid_to_cluster.items():
            if cluster_id not in cluster_to_memcells:
                cluster_to_memcells[cluster_id] = []
            cluster_to_memcells[cluster_id].append(event_id)
        
        # Build scenes list
        scenes = []
        for cluster_id, memcell_ids in cluster_to_memcells.items():
            # Get centroid vector (convert from list to numpy array for storage)
            centroid = cluster_centroids.get(cluster_id, [])
            if isinstance(centroid, list):
                centroid = np.array(centroid, dtype=np.float32)
            
            scene = {
                "scene_id": cluster_id,
                "centroid": centroid.tolist() if isinstance(centroid, np.ndarray) else centroid,
                "memcell_ids": memcell_ids,
                "memcell_count": len(memcell_ids),
                "last_timestamp": cluster_last_ts.get(cluster_id),
            }
            scenes.append(scene)
        
        # Sort scenes by scene_id for consistent ordering
        scenes.sort(key=lambda x: x["scene_id"])
        
        # Build scene index
        scene_index = {
            "scenes": scenes,
            "memcell_to_scene": eventid_to_cluster,
            "total_scenes": len(scenes),
            "total_memcells": len(eventid_to_cluster),
        }
        
        # Save scene index
        output_path = scene_save_dir / f"scene_index_conv_{i}.pkl"
        print(f"  Saving scene index to: {output_path}")
        print(f"    - Total scenes: {len(scenes)}")
        print(f"    - Total memcells: {len(eventid_to_cluster)}")
        
        with open(output_path, "wb") as f:
            pickle.dump(scene_index, f)
    
    print(f"\n✅ Scene index building complete!")


async def build_scene_profiles(
    config: ExperimentConfig,
    data_dir: Path,
    cluster_dir: Path,
    scene_profile_dir: Path,
    profile_output_dir: Path,
) -> None:
    """
    Build scene-based profile extraction results.

    Uses clustering results (scenes) to group MemCells and extract profiles
    within each scene. This mirrors the online scene-based extraction flow.
    """
    if not config.enable_profile_extraction:
        print("Scene profile extraction disabled (enable_profile_extraction=False).")
        return
    if getattr(config, "profile_extraction_mode", "conversation") != "scene":
        print(
            "Scene profile extraction skipped "
            "(profile_extraction_mode != 'scene')."
        )
        return

    llm_cfg = config.llm_config.get(config.llm_service, {})
    llm_provider = LLMProvider(
        provider_type=llm_cfg.get("llm_provider", config.llm_service),
        model=llm_cfg.get("model", "gpt-4o-mini"),
        api_key=llm_cfg.get("api_key", ""),
        base_url=llm_cfg.get("base_url", "https://api.openai.com/v1"),
        temperature=llm_cfg.get("temperature", 0.3),
        max_tokens=llm_cfg.get("max_tokens", 16384),
    )

    profile_config = ProfileManagerConfig(
        scenario=config.profile_scenario,
        min_confidence=config.profile_min_confidence,
        batch_size=50,
        max_retries=getattr(config, "max_retries", 3),
    )
    profile_mgr = ProfileManager(llm_provider=llm_provider, config=profile_config)

    min_memcells = getattr(config, "profile_min_memcells", 1)
    life_max_items = getattr(config, "profile_life_max_items", 25)

    print(f"\n{'='*60}")
    print("Building Scene Profiles")
    print(f"{'='*60}")
    print(
        f"Profile scenario: {profile_config.scenario.value}, min_memcells={min_memcells}"
    )

    scene_profile_dir.mkdir(parents=True, exist_ok=True)
    profile_output_dir.mkdir(parents=True, exist_ok=True)

    for i in range(config.num_conv):
        memcell_file = data_dir / f"memcell_list_conv_{i}.json"
        if not memcell_file.exists():
            print(f"Warning: MemCell file not found, skipping: {memcell_file}")
            continue

        conv_cluster_dir = cluster_dir / f"conv_{i}"
        cluster_state_file = conv_cluster_dir / f"cluster_state_conv_{i}.json"
        if not cluster_state_file.exists():
            print(f"Warning: Cluster state not found, skipping: {cluster_state_file}")
            continue

        print(f"\nProcessing conv_{i} scene profiles...")

        with open(memcell_file, "r", encoding="utf-8") as f:
            memcell_dicts = json.load(f)

        memcell_map: dict[str, MemCell] = {}
        for raw in memcell_dicts:
            memcell = _memcell_from_dict(raw)
            if not memcell:
                continue
            event_id = str(memcell.event_id) if memcell.event_id is not None else ""
            if event_id:
                memcell_map[event_id] = memcell

        with open(cluster_state_file, "r", encoding="utf-8") as f:
            cluster_state = json.load(f)

        eventid_to_cluster = cluster_state.get("eventid_to_cluster", {}) or {}
        if not eventid_to_cluster:
            print(f"  Warning: No cluster assignments for conv_{i}, skipping")
            continue

        cluster_to_event_ids: dict[str, list[str]] = {}
        for event_id, cluster_id in eventid_to_cluster.items():
            cluster_to_event_ids.setdefault(str(cluster_id), []).append(str(event_id))

        scene_profiles = []
        skipped_scenes = []

        cluster_last_ts = cluster_state.get("cluster_last_ts", {}) or {}
        cluster_items = []
        for cluster_id, event_ids in cluster_to_event_ids.items():
            memcells = [memcell_map.get(eid) for eid in event_ids]
            memcells = [mc for mc in memcells if mc is not None]

            last_ts_value = cluster_last_ts.get(cluster_id)
            if isinstance(last_ts_value, (int, float)):
                last_dt = from_timestamp(last_ts_value)
            else:
                last_dt = None
                if memcells:
                    last_dt = max(
                        (mc.timestamp for mc in memcells if mc.timestamp),
                        default=None,
                    )

            cluster_items.append(
                {
                    "scene_id": str(cluster_id),
                    "event_ids": [str(eid) for eid in event_ids],
                    "memcells": memcells,
                    "last_dt": last_dt,
                    "last_ts": last_dt.timestamp() if isinstance(last_dt, datetime) else 0,
                }
            )

        cluster_items.sort(key=lambda item: (item["last_ts"], item["scene_id"]))

        profile_state: dict[str, Any] = {}
        scene_order = [
            {
                "scene_id": item["scene_id"],
                "last_timestamp": (
                    item["last_dt"].isoformat()
                    if isinstance(item["last_dt"], datetime)
                    else None
                ),
            }
            for item in cluster_items
        ]

        def _profile_user_id(profile_obj: Any) -> str:
            if isinstance(profile_obj, dict):
                return str(profile_obj.get("user_id") or "")
            return str(getattr(profile_obj, "user_id", "") or "")

        for item in cluster_items:
            cluster_id = item["scene_id"]
            event_ids = item["event_ids"]
            memcells = item["memcells"]

            if len(memcells) < min_memcells:
                skipped_scenes.append(
                    {
                        "scene_id": cluster_id,
                        "reason": f"memcells<{min_memcells}",
                        "memcell_count": len(memcells),
                    }
                )
                continue

            user_id_set = set()
            for memcell in memcells:
                user_id_set.update(memcell.user_id_list or [])
            user_id_list = _filter_user_ids(sorted(user_id_set))
            if not user_id_list:
                skipped_scenes.append(
                    {
                        "scene_id": cluster_id,
                        "reason": "no_valid_user_ids",
                        "memcell_count": len(memcells),
                    }
                )
                continue

            memcells_sorted = sorted(
                memcells, key=lambda m: m.timestamp or get_now_with_timezone()
            )

            group_id = f"conv_{i}_{cluster_id}"
            try:
                if profile_config.scenario.value == "assistant":
                    profiles = await profile_mgr.extract_profiles_life(
                        memcells=memcells_sorted,
                        old_profiles=list(profile_state.values()),
                        user_id_list=user_id_list,
                        group_id=group_id,
                        max_items=life_max_items,
                    )
                else:
                    profiles = await profile_mgr.extract_profiles(
                        memcells=memcells_sorted,
                        old_profiles=list(profile_state.values()),
                        user_id_list=user_id_list,
                        group_id=group_id,
                    )
            except Exception as e:
                skipped_scenes.append(
                    {
                        "scene_id": cluster_id,
                        "reason": f"extract_failed: {e}",
                        "memcell_count": len(memcells),
                    }
                )
                continue

            profile_payloads = []
            for profile in profiles or []:
                user_id = _profile_user_id(profile)
                if user_id:
                    profile_state[user_id] = profile

                if hasattr(profile, "to_dict"):
                    profile_payloads.append(profile.to_dict())
                elif isinstance(profile, dict):
                    profile_payloads.append(profile)
                elif hasattr(profile, "__dict__"):
                    profile_payloads.append(profile.__dict__)
                else:
                    profile_payloads.append({"value": str(profile)})

            scene_profiles.append(
                {
                    "scene_id": cluster_id,
                    "scene_last_timestamp": (
                        item["last_dt"].isoformat()
                        if isinstance(item["last_dt"], datetime)
                        else None
                    ),
                    "memcell_ids": event_ids,
                    "memcell_count": len(memcells),
                    "user_id_list": user_id_list,
                    "profiles_after_scene": profile_payloads,
                }
            )

        final_profiles = []
        for user_id, profile in profile_state.items():
            if hasattr(profile, "to_dict"):
                payload = profile.to_dict()
            elif isinstance(profile, dict):
                payload = profile
            elif hasattr(profile, "__dict__"):
                payload = profile.__dict__
            else:
                payload = {"value": str(profile)}
            if "user_id" not in payload:
                payload["user_id"] = user_id
            final_profiles.append(payload)

        output_payload = {
            "conversation_id": f"conv_{i}",
            "scenario": profile_config.scenario.value,
            "profile_extraction_mode": "scene",
            "total_scenes": len(cluster_to_event_ids),
            "scene_order": scene_order,
            "processed_scenes": len(scene_profiles),
            "skipped_scenes": skipped_scenes,
            "scene_profiles": scene_profiles,
            "final_profiles": final_profiles,
        }

        output_path = scene_profile_dir / f"scene_profiles_conv_{i}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_payload, f, ensure_ascii=False, indent=2)

        profile_path = profile_output_dir / f"profile_conv_{i}.json"
        with open(profile_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "conversation_id": f"conv_{i}",
                    "scenario": profile_config.scenario.value,
                    "profiles": final_profiles,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        print(
            f"  Saved scene profiles: {output_path} "
            f"(processed={len(scene_profiles)}, skipped={len(skipped_scenes)})"
        )
        print(f"  Saved final profiles: {profile_path} (users={len(final_profiles)})")


async def main():
    """Main function to build and save the BM25 index."""
    # --- Configuration ---
    # The directory containing the JSON files
    config = ExperimentConfig()
    data_dir = Path(__file__).parent / config.experiment_name / "memcells"
    bm25_save_dir = (
        Path(__file__).parent / config.experiment_name / "bm25_index"
    )
    emb_save_dir = (
        Path(__file__).parent / config.experiment_name / "vectors"
    )
    cluster_dir = (
        Path(__file__).parent / config.experiment_name / "memcells" / "clusters"
    )
    scene_save_dir = (
        Path(__file__).parent / config.experiment_name / "scenes"
    )
    scene_profile_dir = (
        Path(__file__).parent / config.experiment_name / "scene_profiles"
    )
    profile_output_dir = (
        Path(__file__).parent / config.experiment_name / "profiles"
    )
    
    os.makedirs(bm25_save_dir, exist_ok=True)
    os.makedirs(emb_save_dir, exist_ok=True)
    build_bm25_index(config, data_dir, bm25_save_dir)
    if config.use_emb:
        await build_emb_index(config, data_dir, emb_save_dir)
    
    # Build scene index if enabled
    if config.enable_scene_retrieval and config.enable_clustering:
        os.makedirs(scene_save_dir, exist_ok=True)
        build_scene_index(config, data_dir, cluster_dir, scene_save_dir)
        if config.enable_profile_extraction:
            os.makedirs(scene_profile_dir, exist_ok=True)
            os.makedirs(profile_output_dir, exist_ok=True)
            await build_scene_profiles(
                config,
                data_dir,
                cluster_dir,
                scene_profile_dir,
                profile_output_dir,
            )

    print("\nAll indexing complete!")


if __name__ == "__main__":
    asyncio.run(main())
