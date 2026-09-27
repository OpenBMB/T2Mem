"""
LongMemEval dataset layout.
[
    {
        "question_id": "e47becba",
        "question_type": "single-session-user",
        "question": "What degree did I graduate with?",
        "question_date": "2023/05/30 (Tue) 23:40",
        "answer": "Business Administration",
        "answer_session_ids": [
            "answer_280352e9"
        ],
    "haystack_dates": [
            "2023/05/20 (Sat) 02:21",
            ...
        ],
    "haystack_session_ids": [
            "haystack_session_ids": [
            "sharegpt_yywfIrx_0",
            "85a1be56_1",
            ...
        ],
        "haystack_sessions": [
            [
                {
                    "role": "user",
                    "content": "The farmer needs to transport a fox, a chicken, and some grain across a river using a boat. The fox cannot be left alone with the chicken, and the chicken cannot be left alone with the grain. The boat can only hold one item at a time, and the river is too dangerous to cross multiple times. Can you help the farmer transport all three items across the river without any of them getting eaten? Remember, strategic thinking and planning are key to solving this puzzle. If you're stuck, try thinking about how you would solve the puzzle yourself, and use that as a starting point. Be careful not to leave the chicken alone with the fox, or the chicken and the grain alone together, as this will result in a failed solution. Good luck!"
                },
                ...
            ], ...
        ]
    },
    ...
]
Notes:
1. Every query owns a list of haystack sessions; each session is a list of turns.
2. Each haystack session has a matching entry in ``haystack_dates``.
3. Each query also has its own ``question_date``.
4. Wiki entries are built per haystack session, so the matching date must be carried along.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterator, List


def load_longmemeval_samples(path: Path) -> List[Dict[str, Any]]:
    """
    Read the LongMemEval JSON file.
    """
    with path.open("r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def iter_session_records(
    samples: List[Dict[str, Any]],
    *,
    question_types: str | set[str] | None = None,
) -> Iterator[Dict[str, Any]]:
    """
    Turn each query into a record whose ``sessions`` hold all of its haystack sessions.
    """
    if isinstance(question_types, str):
        question_types = {
            item.strip()
            for item in question_types.split(",")
            if item.strip()
        } or None
    yielded_sessions = 0

    # walk every query
    for _, sample in enumerate(samples):

        question_id = str(sample["question_id"])
        question_type = str(sample["question_type"])
        question = str(sample["question"])
        answer = str(sample["answer"])
        answer_session_ids = [str(item) for item in sample.get("answer_session_ids", [])]

        # when question_types is given, keep only those question types
        if question_types is not None and question_type not in question_types:
            continue

        sessions: List[Dict[str, Any]] = []
        for session_index, (source_session_id, date_time, messages) in enumerate(
            zip(
                sample["haystack_session_ids"],
                sample["haystack_dates"],
                sample["haystack_sessions"],
            ),
            start=1,
        ):
            speakers: List[str] = []
            transcript_lines: List[str] = []

            for _, message in enumerate(messages, start=1):
                # role, e.g. user / assistant
                role = str(message.get("role")).strip()
                # text content of this turn
                content = str(message.get("content")).strip()
                # collect the speakers seen in this session
                if role not in speakers:
                    speakers.append(role)

                transcript_lines.append(f"role: {role}")
                transcript_lines.append(f"content: {content}")

            sessions.append(
                {
                    "session_index": session_index,
                    "wiki_id": f"{question_id}_session_{session_index}",
                    "source_session_id": str(source_session_id),
                    "title": f"{question_id} Session {session_index}",
                    "date_time": str(date_time),
                    "speakers": speakers,
                    "transcript_text": "\n".join(transcript_lines),
                }
            )
            yielded_sessions += 1

        if sessions:
            yield {
                "question_id": question_id,
                "question_type": question_type,
                "question": question,
                "question_date": str(sample["question_date"]),
                "answer": answer,
                "answer_session_ids": answer_session_ids,
                "sessions": sessions,
            }
