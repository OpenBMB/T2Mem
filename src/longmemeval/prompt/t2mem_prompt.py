from __future__ import annotations

# build wiki
SESSION_WIKI_SYSTEM_PROMPT = """You turn a raw conversation session into grounded wiki content for long-term conversational memory.
Use only the provided transcript. Do not invent facts.
Return ONLY valid JSON."""

SESSION_WIKI_USER_PROMPT_TEMPLATE = """Create the content section for a grounded wiki page from this conversation session.

Authoritative session date/time:
{date_time}

Transcript:
{transcript}

1. Use the authoritative conversation date/time as the temporal anchor, resolving relative or ambiguous time expressions (e.g., yesterday, last year, next month) into explicit dates whenever possible.
2. Preserve concrete details that may be useful for later QA, including preferences, objects, locations, plans, family details, activities, and quoted names etc.
3. Write 8-15 atomic facts when the transcript contains enough information.

Return exactly this JSON schema:
{{
  "summary": "2-3 sentence knowledge-style overview of the session's core event",
  "event_core": ["core point 1", "core point 2", ...],
  "atomic_facts": ["specific evidence-backed fact 1 ", "specific evidence-backed fact 2", ...],
  "entities": ["entity 1", "entity 2", ...],
  "temporal_clues": ["temporal clue 1", ...],
  "resolved_temporal_facts": ["event with resolved or anchored time", ...],
  "ongoing_threads": ["thread 1", ...]
}}
"""

# dual reading
ADAPTIVE_READING_SYSTEM_PROMPT = """You are an adaptive controller for long-term memory, choosing the next read action to maximize useful evidence.

Memory sources:
- wiki memories: wiki based on the raw session, highly condensed from the session topics and timeline clues.
- raw transcripts: compared to wikis, raw session content contains more precise evidence.

Allowed actions:
1. SEARCH_WIKI
2. READ_RAW_FROM_WIKI
3. SPAN_LOOKUP
4. ANSWER

Action guidance:
- Use `SEARCH_WIKI` when you need more wiki page for high-level abstract events, preferences, or timeline clues.
- Use `READ_RAW_FROM_WIKI` when provided wiki finds the right session but lacks exact detail, to inspect the raw session corresponding to that wiki entry.
- Use `SPAN_LOOKUP` when you need to precisely locate relevant atomic facts or other fine-grained information in the raw session.
- Terminate with `ANSWER` only when you already have enough support."""

ADAPTIVE_READING_ACTION_PROMPT_TEMPLATE = """Decide the next reading action.

Question time: {question_time}
Question: {question}

Current information:
{current_information}

Return exactly one JSON object with this schema:
{{
  "evidence": [
    {{
      "snippet": "Return copy the exact relevant span from the provided current information.",
      "date_time": "Return the exact date/time expression associated with the snippet, copied verbatim from the provided source text if available.",
    }},
    ...
  ],
  "reasoning": "Thinking about the current information and what action to choose",
  "action": "SEARCH_WIKI | READ_RAW_FROM_WIKI | SPAN_LOOKUP | ANSWER",
  "target_wiki_id":["You need to view the original session's wiki ID.","xxxx",...],
  "keywords": ["xxx", "xxx"],
  "context_lines": xxx
}}

Notes:
For `READ_RAW_FROM_WIKI`, fill `target_wiki_id` with the wiki ID whose raw session should be viewed.
- For `SPAN_LOOKUP`, provide 1-5 keywords and a small non-negative `context_lines` value used for view content.
"""


# dual reading
ADAPTIVE_READING_SYSTEM_PROMPT_NO_LOOK = """You are an adaptive controller for long-term memory, choosing the next read action to maximize useful evidence.

Memory sources:
- wiki memories: wiki based on the raw session, highly condensed from the session topics and timeline clues.
- raw transcripts: compared to wikis, raw session content contains more precise evidence.

Allowed actions:
1. SEARCH_WIKI
2. READ_RAW_FROM_WIKI
3. ANSWER

Action guidance:
- Use `SEARCH_WIKI` when you need more wiki page for high-level abstract events, preferences, or timeline clues.
- Use `READ_RAW_FROM_WIKI` when provided wiki finds the right session but lacks exact detail, to inspect the raw session corresponding to that wiki entry.
- Terminate with `ANSWER` only when you already have enough support."""

ADAPTIVE_READING_ACTION_PROMPT_TEMPLATE_NO_LOOK = """Decide the next reading action.

Question:
{question}

Current information:
{current_information}

Return exactly one JSON object with this schema:
{{
  "evidence": [
    {{
      "snippet": "Return copy the exact relevant span from the provided current information.",
      "date_time": "Return the exact date/time expression associated with the snippet, copied verbatim from the provided source text if available.",
    }},
    ...
  ],
  "reasoning": "Thinking about the current information and what action to choose",
  "action": "SEARCH_WIKI | READ_RAW_FROM_WIKI | ANSWER",
  "target_wiki_id":["You need to view the original session's wiki ID.","xxxx",...]
}}

Notes:
If using READ_RAW_FROM_WIKI, set target_wiki_id to the wiki ID of the raw session to be viewed.
"""


# --- wiki-only variant: raw transcripts unreadable (ablation A) ---
ADAPTIVE_READING_SYSTEM_PROMPT_NO_LOOK_NO_RAW = """You are an adaptive controller for long-term memory, choosing the next read action to maximize useful evidence.

Memory sources:
- wiki memories: wiki based on the raw session, highly condensed from the session topics and timeline clues.

Allowed actions:
1. SEARCH_WIKI
2. ANSWER

Action guidance:
- Use `SEARCH_WIKI` when you need more wiki pages for high-level abstract events, preferences, or timeline clues.
- Terminate with `ANSWER` only when you already have enough support."""

ADAPTIVE_READING_ACTION_PROMPT_TEMPLATE_NO_LOOK_NO_RAW = """Decide the next reading action.

Question:
{question}

Current information:
{current_information}

Return exactly one JSON object with this schema:
{{
  "evidence": [
    {{
      "snippet": "Return copy the exact relevant span from the provided current information.",
      "date_time": "Return the exact date/time expression associated with the snippet, copied verbatim from the provided source text if available.",
    }},
    ...
  ],
  "reasoning": "Thinking about the current information and what action to choose",
  "action": "SEARCH_WIKI | ANSWER"
}}

Notes:
- Raw transcripts are NOT available in this setting. All evidence must come from the wiki pages shown above.
- Use `SEARCH_WIKI` to pull the next wiki page when the current one is insufficient.
"""


# --- raw-only variant: no wiki, raw dialogue only (ablation B) ---
ADAPTIVE_READING_SYSTEM_PROMPT_NO_LOOK_RAW_ONLY = """You are an adaptive controller for long-term memory, choosing the next read action to maximize useful evidence.

Memory sources:
- raw transcripts: the original conversation sessions, containing precise evidence.

Allowed actions:
1. SEARCH_RAW
2. ANSWER

Action guidance:
- Use `SEARCH_RAW` when you need to read the next raw session not yet shown.
- Terminate with `ANSWER` only when you already have enough support."""

ADAPTIVE_READING_ACTION_PROMPT_TEMPLATE_NO_LOOK_RAW_ONLY = """Decide the next reading action.

Question:
{question}

Current information:
{current_information}

Return exactly one JSON object with this schema:
{{
  "evidence": [
    {{
      "snippet": "Return copy the exact relevant span from the provided current information.",
      "date_time": "Return the exact date/time expression associated with the snippet, copied verbatim from the provided source text if available.",
    }},
    ...
  ],
  "reasoning": "Thinking about the current information and what action to choose",
  "action": "SEARCH_RAW | ANSWER"
}}

Notes:
- Wiki pages are NOT available in this setting. All evidence must come from the raw sessions shown above.
- Use `SEARCH_RAW` to read the next raw session when the current ones are insufficient.
"""

# longmemeval bge
QA_ANSWER_LONGMEM_VANILLA_PROMPT_TEMPLATE = """You are an intelligent assistant with access to a memory system. Your task is to answer the user's question by effectively utilizing the provided memory.
Question time: {question_time}
Question: {question}
Please answer the question based on the following memories:\x20
{context}
"""

# longmemeval ada reading
QA_ANSWER_LONGMEM_ADA_PROMPT_TEMPLATE = """You are an intelligent assistant with access to a memory system. Your task is to answer the user's question by effectively utilizing the provided memory evidences.

Question time: {question_time}
Question: {question}
Memory evidences:
{context}

# Memory Evidences Guidelines:
1. Carefully analyze all provided memory evidences.
2. If the evidences contain contradictory information, prioritize the most recent dated evidence.

Think step by step about how to utilize the memory evidences, then provide your answer.
"""
