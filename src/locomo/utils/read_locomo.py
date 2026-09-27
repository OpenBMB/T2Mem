"""
LoCoMo:
A sample holds a set of QA pairs and one conversation; the conversation in turn
contains a set of sessions. Each returned record is keyed by sample and carries
that sample's QA pairs plus its sessions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterator, List


def load_locomo_samples(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def iter_session_records(samples: List[Dict[str, Any]]) -> Iterator[Dict[str, Any]]:
    for sample in samples:
        sample_id = str(sample["sample_id"])
        conversation = sample["conversation"]
        qa = [
            {
                "question": str(item["question"]),
                "answer": str(item.get("answer", item.get("adversarial_answer", ""))),
                "evidence": [str(value) for value in item.get("evidence", [])],
                "category": item.get("category"),
            }
            for item in sample.get("qa", [])
        ]

        session_keys = [
            key
            for key in conversation
            if key.startswith("session_")
            and not key.endswith("_date_time")
            and key.rsplit("_", 1)[-1].isdigit()
        ]

        sessions: List[Dict[str, Any]] = []
        for session_key in session_keys:
            session_index = int(session_key.rsplit("_", 1)[-1])
            turns = conversation[session_key]
            speakers: List[str] = []
            transcript_lines: List[str] = []

            for turn in turns:
                speaker = str(turn.get("speaker", "")).strip()
                dia_id = str(turn.get("dia_id", "")).strip()
                text = str(turn.get("text", "")).strip()

                if not text:
                    continue

                if speaker and speaker not in speakers:
                    speakers.append(speaker)

                transcript_lines.append(f"speaker: {speaker}")
                if dia_id:
                    transcript_lines.append(f"dia_id: {dia_id}")
                transcript_lines.append(f"text: {text}")

            if transcript_lines:
                sessions.append(
                    {
                        "source_session_id": f"{sample_id}_{session_index}",
                        "session_index": session_index,
                        "wiki_id": f"{sample_id}_session_{session_index}",
                        "title": f"{sample_id} Session {session_index}",
                        "date_time": str(conversation.get(f"{session_key}_date_time", "")),
                        "speakers": speakers,
                        "transcript_text": "\n".join(transcript_lines),
                    }
                )

        if sessions:
            yield {
                "sample_id": sample_id,
                "qa": qa,
                "sessions": sessions,
            }
