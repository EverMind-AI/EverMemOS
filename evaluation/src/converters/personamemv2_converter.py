"""
PersonaMem v2 Converter - convert PersonaMem v2 dataset to Locomo format.
"""
import ast
import csv
import json
import re
import random
import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any

from evaluation.src.converters.base import BaseConverter
from evaluation.src.converters.registry import register_converter


def _safe_parse_json(value: str) -> Any:
    if not value:
        return None
    try:
        return json.loads(value)
    except Exception:
        try:
            return ast.literal_eval(value)
        except Exception:
            return None


def _extract_persona_name(expanded_persona: str, system_content: str) -> str:
    persona_data = _safe_parse_json(expanded_persona)
    if isinstance(persona_data, dict):
        name = persona_data.get("name")
        if name:
            return str(name).strip()

    if system_content:
        start = system_content.find("{")
        end = system_content.rfind("}")
        if start != -1 and end != -1 and end > start:
            payload = system_content[start : end + 1]
            persona_data = _safe_parse_json(payload)
            if isinstance(persona_data, dict):
                name = persona_data.get("name")
                if name:
                    return str(name).strip()

    return "User"


def _clean_message_prefix(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"^(User|Assistant):\s*", "", text, flags=re.MULTILINE)
    return text.strip()


def _parse_incorrect_answers(value: str) -> List[str]:
    data = _safe_parse_json(value)
    if isinstance(data, list):
        return [str(item).strip() for item in data if str(item).strip()]
    if isinstance(value, str):
        parts = [p.strip() for p in value.split("|") if p.strip()]
        return parts
    return []


def _find_dataset_root(start_path: Path) -> Path:
    for parent in [start_path] + list(start_path.parents):
        if (parent / "data").exists():
            return parent
    return start_path


@register_converter("personamemv2")
class PersonaMemV2Converter(BaseConverter):
    """PersonaMem v2 dataset converter."""

    def get_input_files(self) -> Dict[str, str]:
        """Return required input files."""
        return {
            "questions": "benchmark.csv",
        }

    def get_output_filename(self) -> str:
        """Return output filename."""
        return "personamemv2_32k_locomo_style.json"

    def convert(self, input_paths: Dict[str, str], output_path: str) -> None:
        """
        Execute conversion.

        Args:
            input_paths: {"questions": "path/to/benchmark.csv"}
            output_path: Output file path
        """
        print("🔄 Converting PersonaMem v2 to Locomo format...")

        questions_path = Path(input_paths["questions"])
        data_dir = questions_path.parent
        dataset_root = _find_dataset_root(data_dir)

        print(f"   Loading questions: {questions_path}")
        print(f"   Dataset root: {dataset_root}")
        with open(questions_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        print(f"   Loaded {len(rows)} questions")

        grouped_questions = defaultdict(list)
        for idx, row in enumerate(rows):
            row["_row_idx"] = idx
            chat_path = row.get("chat_history_32k_link", "").strip()
            if not chat_path:
                continue
            grouped_questions[chat_path].append(row)

        print(f"   Grouped into {len(grouped_questions)} chat histories")

        locomo_data = []
        labels = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]

        for chat_rel_path, question_list in grouped_questions.items():
            chat_path = dataset_root / chat_rel_path
            if not chat_path.exists():
                print(f"   Warning: chat history not found: {chat_path}")
                continue

            with open(chat_path, "r", encoding="utf-8") as f:
                chat_data = json.load(f)

            chat_history = chat_data.get("chat_history", [])
            system_content = ""
            if chat_history and chat_history[0].get("role") == "system":
                system_content = chat_history[0].get("content", "")

            persona_name = _extract_persona_name(
                question_list[0].get("expanded_persona", ""),
                system_content,
            )

            locomo_entry = {
                "qa": [],
                "conversation": {
                    "speaker_a": persona_name,
                    "speaker_b": "Assistant",
                    "session_0_date_time": "Unknown",
                    "session_0": [],
                },
            }

            # Build dialogue list (skip system message)
            dialogue_idx = 0
            for msg in chat_history:
                role = msg.get("role")
                if role == "system":
                    continue

                speaker = persona_name if role == "user" else "Assistant"
                cleaned_text = _clean_message_prefix(msg.get("content", ""))
                if not cleaned_text:
                    continue

                locomo_entry["conversation"]["session_0"].append(
                    {
                        "speaker": speaker,
                        "text": cleaned_text,
                        "dia_id": f"D0:{dialogue_idx}",
                    }
                )
                dialogue_idx += 1

            # Add questions for this chat history
            for row in question_list:
                question = (row.get("user_query") or "").strip()
                correct_answer = (row.get("correct_answer") or "").strip()
                incorrect_answers = _parse_incorrect_answers(
                    row.get("incorrect_answers", "")
                )

                incorrect_filtered = [
                    ans
                    for ans in incorrect_answers
                    if ans and ans != correct_answer
                ]
                option_texts = [correct_answer] + incorrect_filtered[: len(labels) - 1]

                option_entries = [
                    {"text": option_texts[0], "is_correct": True}
                ] + [
                    {"text": ans, "is_correct": False}
                    for ans in option_texts[1:]
                ]

                seed_str = f"{row.get('persona_id', '')}-{row.get('_row_idx', '')}"
                seed_int = int(hashlib.md5(seed_str.encode("utf-8")).hexdigest(), 16)
                rng = random.Random(seed_int)
                rng.shuffle(option_entries)

                options = {}
                answer_label = ""
                for idx, entry in enumerate(option_entries):
                    label = labels[idx]
                    options[label] = entry["text"]
                    if entry["is_correct"]:
                        answer_label = label
                question_id = (
                    f"persona{row.get('persona_id', 'unknown')}_q{row.get('_row_idx')}"
                )

                qa_item = {
                    "question_id": question_id,
                    "question": question,
                    "answer": answer_label,
                    "answer_text": correct_answer,
                    "correct_answer_text": correct_answer,
                    "incorrect_answers": incorrect_answers,
                    "all_options": options,
                    "evidence": [],
                    "category": row.get("pref_type") or row.get("conversation_scenario"),
                    "topic": row.get("topic_query"),
                    "persona_id": row.get("persona_id"),
                    "preference": row.get("preference"),
                    "pref_type": row.get("pref_type"),
                    "related_conversation_snippet": row.get(
                        "related_conversation_snippet"
                    ),
                    "conversation_scenario": row.get("conversation_scenario"),
                    "topic_query": row.get("topic_query"),
                    "topic_preference": row.get("topic_preference"),
                }
                locomo_entry["qa"].append(qa_item)

            locomo_data.append(locomo_entry)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(locomo_data, f, indent=2, ensure_ascii=False)

        total_questions = sum(len(entry["qa"]) for entry in locomo_data)
        print(f"   ✅ Saved {len(locomo_data)} entries to {output_path}")
        print(f"   Total questions: {total_questions}")
