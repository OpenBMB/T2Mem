from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from prompt.t2mem_prompt import (
    ADAPTIVE_READING_ACTION_PROMPT_TEMPLATE_NO_LOOK,
    ADAPTIVE_READING_ACTION_PROMPT_TEMPLATE_NO_LOOK_NO_RAW,
    ADAPTIVE_READING_ACTION_PROMPT_TEMPLATE_NO_LOOK_RAW_ONLY,
    ADAPTIVE_READING_SYSTEM_PROMPT_NO_LOOK,
    ADAPTIVE_READING_SYSTEM_PROMPT_NO_LOOK_NO_RAW,
    ADAPTIVE_READING_SYSTEM_PROMPT_NO_LOOK_RAW_ONLY,
    QA_ANSWER_LOCOMO_ADA_PROMPT_TEMPLATE,
)
from utils.llm_utils import (
    DEFAULT_DISABLE_THINKING_EXTRA_BODY,
    parse_json_response_lenient,
)

ALLOWED_ACTIONS = {"SEARCH_WIKI", "READ_RAW_FROM_WIKI", "ANSWER"}
# wiki-only ablation: raw transcripts are unreadable, so the action set shrinks to two
ALLOWED_ACTIONS_NO_RAW = {"SEARCH_WIKI", "ANSWER"}
# raw-only ablation: no wiki, read raw sessions along the retrieval order
ALLOWED_ACTIONS_RAW_ONLY = {"SEARCH_RAW", "ANSWER"}
USAGE_COUNT_KEYS = ("requests", "requests_without_usage", "prompt_tokens", "completion_tokens", "total_tokens")
USAGE_PHASES = ("controller", "answer")


def new_usage_counts() -> dict[str, int]:
    return {key: 0 for key in USAGE_COUNT_KEYS}


def new_token_usage() -> dict[str, dict[str, int]]:
    return {
        "total": new_usage_counts(),
        **{phase: new_usage_counts() for phase in USAGE_PHASES},
    }


def _usage_value(usage: Any, key: str) -> int:
    value = usage.get(key) if isinstance(usage, dict) else getattr(usage, key, None)
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def extract_response_token_usage(response: Any) -> dict[str, int] | None:
    usage = getattr(response, "usage", None)
    if usage is None:
        return None

    prompt_tokens = _usage_value(usage, "prompt_tokens")
    completion_tokens = _usage_value(usage, "completion_tokens")
    total_tokens = _usage_value(usage, "total_tokens")
    if not total_tokens:
        total_tokens = prompt_tokens + completion_tokens

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


def add_token_usage(
    aggregate: dict[str, dict[str, int]],
    phase: str,
    usage: dict[str, int] | None,
) -> None:
    if phase not in aggregate:
        aggregate[phase] = new_usage_counts()

    for bucket_name in ("total", phase):
        bucket = aggregate[bucket_name]
        bucket["requests"] += 1
        if usage is None:
            bucket["requests_without_usage"] += 1
            continue
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            bucket[key] += int(usage.get(key, 0) or 0)


def merge_token_usage(
    aggregate: dict[str, dict[str, int]],
    usage: dict[str, Any] | None,
) -> None:
    if not usage:
        return

    for bucket_name, source_counts in usage.items():
        if not isinstance(source_counts, dict):
            continue
        if bucket_name not in aggregate:
            aggregate[bucket_name] = new_usage_counts()
        for key in USAGE_COUNT_KEYS:
            aggregate[bucket_name][key] += int(source_counts.get(key, 0) or 0)


def summarize_token_usage_jsonl(path: Path) -> dict[str, Any]:
    token_usage_by_query: dict[str, dict[str, Any]] = {}
    rows_seen = 0
    rows_without_token_usage = 0

    if path.exists():
        with path.open("r", encoding="utf-8") as file_obj:
            for line in file_obj:
                if not line.strip():
                    continue
                rows_seen += 1
                row = json.loads(line)
                query_id = normalize_string(row.get("query_id"))
                token_usage = row.get("token_usage")
                if not query_id or not token_usage:
                    rows_without_token_usage += 1
                    continue
                token_usage_by_query[query_id] = token_usage

    total_usage = new_token_usage()
    for token_usage in token_usage_by_query.values():
        merge_token_usage(total_usage, token_usage)

    return {
        "path": str(path),
        "rows_seen": rows_seen,
        "rows_without_token_usage": rows_without_token_usage,
        "queries_counted": len(token_usage_by_query),
        "token_usage": total_usage,
    }


def format_token_usage(token_usage: dict[str, dict[str, int]]) -> str:
    total = token_usage.get("total", {})
    return (
        f"requests={total.get('requests', 0)}, "
        f"prompt_tokens={total.get('prompt_tokens', 0)}, "
        f"completion_tokens={total.get('completion_tokens', 0)}, "
        f"total_tokens={total.get('total_tokens', 0)}, "
        f"requests_without_usage={total.get('requests_without_usage', 0)}"
    )


def load_json(path: Path) -> Any:
    """Read a plain JSON file."""
    with path.open("r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file into a list of parsed objects."""
    with path.open("r", encoding="utf-8") as file_obj:
        return [json.loads(line) for line in file_obj if line.strip()]


def load_raw_session_map(data_path: Path) -> dict[str, dict[str, Any]]:
    """Expand the processed dataset into a source_session_id -> raw session lookup."""
    records = load_json(data_path)
    session_map: dict[str, dict[str, Any]] = {}
    for record in records:
        for session in record.get("sessions", []):
            session_map[session["source_session_id"]] = session
    return session_map


def load_query_session_map(dataset: str, data_path: Path) -> dict[str, list[dict[str, Any]]]:
    """Collect each query's full session list, used by READ_RAW_FROM_WIKI."""
    records = load_json(data_path)
    session_map: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        key = record["sample_id"]
        session_map[key] = list(record.get("sessions"))
    return session_map


def load_wiki_map_by_id(wiki_path: Path) -> dict[str, dict[str, Any]]:
    """Build a wiki lookup keyed by wiki_id."""
    rows = load_jsonl(wiki_path)
    by_wiki_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        wiki_id = row.get("wiki_id")
        if wiki_id:
            by_wiki_id[wiki_id] = row
    return by_wiki_id


def build_question_text(dataset: str, row: dict[str, Any]) -> str:
    """Render the question into the text handed to the model."""
    return row["question"]


def build_result_metadata(dataset: str, row: dict[str, Any]) -> dict[str, Any]:
    """Base fields copied into the output JSONL (excludes runtime-generated content)."""
    return {
        "dataset": dataset,
        "sample_id": row["sample_id"],
        "query_id": row["query_id"],
        "question": row["question"],
        "answer": row.get("answer"),
        "evidence": row.get("evidence"),
        "category": row.get("category"),
    }


def should_skip_t2mem_row(dataset: str, row: dict[str, Any]) -> bool:
    """Skip LoCoMo category 5 to stay consistent with prior evaluations."""
    return dataset == "locomo" and normalize_string(row.get("category")) == "5"

# wiki rendering
def wiki_text(row: dict[str, Any]) -> str:
    """Format one wiki row into a text block ready to feed the model."""
    parts = [
        f"Wiki ID: {row.get('wiki_id')}",
        f"Date: {row.get('source_session_date_time')}",
        f"summary: {row.get('summary')}",
        "event_core:",
        *[f"- {item}" for item in row.get("event_core", [])],
        "atomic_facts:",
        *[f"- {item}" for item in row.get("atomic_facts", [])],
        "entities:",
        *[f"- {item}" for item in row.get("entities", [])],
        "temporal_clues:",
        *[f"- {item}" for item in row.get("temporal_clues", [])],
        "resolved_temporal_facts:",
        *[f"- {item}" for item in row.get("resolved_temporal_facts", [])],
        "ongoing_threads:",
        *[f"- {item}" for item in row.get("ongoing_threads", [])],
    ]
    return "\n".join(parts).strip()

# raw rendering
def raw_text(session: dict[str, Any]) -> str:
    """Format a raw session into transcript text ready to feed the model."""
    return "\n".join(
        [
            f"Date: {session.get('date_time')}",
            "Transcript:",
            session.get("transcript_text"),
        ]
    ).strip()


def build_wiki_context(
    topk_sessions: list[dict[str, Any]],
    *,
    wiki_map: dict[str, dict[str, Any]],
    start_index: int = 0,
    end_index: int | None = None,
    heading: str,
) -> tuple[str, list[str], list[str]]:
    """Slice wiki entries out of the top-k list by wiki_id.

    Returns the rendered context together with the source-session ids and wiki ids hit.
    """
    blocks = [heading] if heading else []
    source_session_ids: list[str] = []
    wiki_ids: list[str] = []
    selected = topk_sessions[start_index:end_index]

    for _, item in enumerate(selected, start=start_index + 1):
        wiki_id = normalize_string(item.get("wiki_id"))
        if not wiki_id or wiki_id not in wiki_map:
            continue

        row = wiki_map[wiki_id]
        source_session_id = normalize_string(row.get("source_session_id")) or normalize_string(item.get("source_session_id"))
        source_session_ids.append(source_session_id)
        wiki_ids.append(wiki_id)
        blocks.append(f"{wiki_text(row)}")

    if not blocks:
        blocks.append("No wiki memories available.")

    return "\n\n".join(blocks).strip(), source_session_ids, wiki_ids


def build_raw_context(
    target_ids: list[str],
    *,
    query_sessions: list[dict[str, Any]],
) -> tuple[str, list[str], list[str]]:
    """Resolve each controller-supplied wiki_id back to its raw session."""
    if not target_ids:
        return "No target wiki id was provided for READ_RAW_FROM_WIKI.", [], []

    blocks = ["Raw sessions requested from wiki:"]
    source_session_ids: list[str] = []
    wiki_ids: list[str] = []
    missing_count = 0
    local_raw_by_wiki_id = {
        normalize_string(session.get("wiki_id")): session
        for session in query_sessions
        if normalize_string(session.get("wiki_id"))
    }

    for target in target_ids:
        wiki_id = normalize_string(target)
        if not wiki_id or wiki_id not in local_raw_by_wiki_id:
            missing_count += 1
            continue

        session = local_raw_by_wiki_id[wiki_id]
        source_session_id = normalize_string(session.get("source_session_id"))
        source_session_ids.append(source_session_id)
        wiki_ids.append(wiki_id)
        blocks.append(f"[Raw From Wiki]\nWiki ID: {wiki_id}\n{raw_text(session)}")

    if missing_count:
        blocks.append(f"Missing wiki targets: {missing_count}")

    return "\n\n".join(blocks).strip(), source_session_ids, wiki_ids


def build_raw_direct_context(
    topk_sessions: list[dict[str, Any]],
    *,
    query_sessions: list[dict[str, Any]],
    start_index: int = 0,
    end_index: int | None = None,
    heading: str,
) -> tuple[str, list[str], list[str]]:
    """raw-only mode: slice raw session text along the retrieval order.

    Mirrors build_wiki_context exactly (same top-k ranking, same start/end slicing);
    the only difference is emitting raw_text instead of wiki_text, so the ablation
    changes nothing but *what is read*.
    """
    blocks = [heading] if heading else []
    source_session_ids: list[str] = []
    wiki_ids: list[str] = []
    local_raw_by_wiki_id = {
        normalize_string(session.get("wiki_id")): session
        for session in query_sessions
        if normalize_string(session.get("wiki_id"))
    }

    for item in topk_sessions[start_index:end_index]:
        wiki_id = normalize_string(item.get("wiki_id"))
        if not wiki_id or wiki_id not in local_raw_by_wiki_id:
            continue
        session = local_raw_by_wiki_id[wiki_id]
        source_session_ids.append(
            normalize_string(session.get("source_session_id"))
            or normalize_string(item.get("source_session_id"))
        )
        wiki_ids.append(wiki_id)
        blocks.append(f"[Raw]\nWiki ID: {wiki_id}\n{raw_text(session)}")

    if not blocks:
        blocks.append("No raw sessions available.")

    return "\n\n".join(blocks).strip(), source_session_ids, wiki_ids


def normalize_string(value: Any) -> str:
    """Coerce any value to a stripped string."""
    return "" if value is None else str(value).strip()


def normalize_string_list(value: Any) -> list[str]:
    """Normalize a scalar-or-list model output into a list of strings."""
    if isinstance(value, list):
        return [normalize_string(item) for item in value if normalize_string(item)]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def normalize_action_response(
    response: dict[str, Any],
    *,
    allowed_actions: set[str] | None = None,
) -> dict[str, Any]:
    """Clean controller output: fall back on illegal actions, normalize evidence / target ids.

    ``allowed_actions`` backs the ablation switches: when an action is removed from
    the allowed set, a model that still emits it is treated as illegal and falls back to ANSWER.
    """
    allowed = ALLOWED_ACTIONS if allowed_actions is None else allowed_actions
    action = normalize_string(response.get("action")).upper()
    fallback = action not in allowed
    if fallback:
        action = "ANSWER"

    evidence_items: list[dict[str, str]] = []
    for item in response.get("evidence", []) if isinstance(response.get("evidence"), list) else []:
        if not isinstance(item, dict):
            continue
        snippet = normalize_string(item.get("snippet"))
        date_time = normalize_string(item.get("date_time"))
        #reasoning = normalize_string(item.get("reasoning"))
        if not snippet:
            continue
        evidence_items.append(
            {
                "snippet": snippet,
                "date_time": date_time,
                #"reasoning": reasoning,
            }
        )

    return {
        "evidence": evidence_items,
        "reasoning": normalize_string(response.get("reasoning")),
        "action": action,
        "action_fallback": fallback,
        "target_wiki_id": normalize_string_list(response.get("target_wiki_id")),
    }


def canonicalize_wiki_ids(
    target_ids: list[str],
    *,
    query_sessions: list[dict[str, Any]],
) -> list[str]:
    """Keep only wiki_ids that really belong to the current query."""
    canonical_ids: list[str] = []
    seen: set[str] = set()
    local_wiki_ids = {
        normalize_string(session.get("wiki_id"))
        for session in query_sessions
        if normalize_string(session.get("wiki_id"))
    }

    for target in target_ids:
        normalized = normalize_string(target)
        if not normalized or normalized not in local_wiki_ids or normalized in seen:
            continue
        seen.add(normalized)
        canonical_ids.append(normalized)

    return canonical_ids


def extend_evidence(
    accumulated_evidence: list[dict[str, str]],
    new_evidence: list[dict[str, str]],
    seen_keys: set[tuple[str, str, str]],
) -> None:
    """Append this step's evidence to the running pool, dropping duplicates."""
    for item in new_evidence:
        key = (
            normalize_string(item.get("snippet")),
            normalize_string(item.get("date_time")),
            #normalize_string(item.get("reasoning")),
        )
        if key in seen_keys:
            continue
        seen_keys.add(key)
        accumulated_evidence.append(
            {
                "snippet": key[0],
                "date_time": key[1],
                #"reasoning": key[2],
            }
        )


def format_evidence_list(evidence_items: list[dict[str, str]], *, heading: str) -> str:
    """Render the evidence list into stable, readable model input."""
    if not evidence_items:
        return f"{heading}\nNone."

    blocks = [heading]
    for index, item in enumerate(evidence_items, start=1):
        blocks.append(
            "\n".join(
                [
                    f"[Evidence {index}]",
                    f"Date: {normalize_string(item.get('date_time')) or 'N/A'}",
                    f"Snippet: {normalize_string(item.get('snippet'))}",
                    #f"Reasoning: {normalize_string(item.get('reasoning')) or 'N/A'}",
                ]
            )
        )
    return "\n\n".join(blocks).strip()


def compose_current_information(
    *,
    observation: str,
    accumulated_evidence: list[dict[str, str]],
) -> str:
    """Compose the full context the controller sees on the current turn."""
    sections = [observation]
    if accumulated_evidence:
        sections.append(format_evidence_list(accumulated_evidence, heading="Accumulated evidence so far:"))
    return "\n\n".join(section for section in sections if section.strip()).strip()


def answer_prompt_pair(dataset: str) -> tuple[str | None, str]:
    """Pick the adaptive-answer prompt for the current dataset."""
    return None, QA_ANSWER_LOCOMO_ADA_PROMPT_TEMPLATE


def call_json_with_retries(
    *,
    client: Any,
    model_name: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    temperature: float,
    max_retries: int,
    token_usage: dict[str, dict[str, int]],
) -> tuple[dict[str, Any], int | None, str]:
    """Call the JSON LLM with bounded retries on transient failure.

    Returns (parsed_json, completion_tokens, raw_text) so callers can track the output
    """
    last_error = ""
    for attempt in range(1, max_retries + 1):
        try:
            messages = [{"role": "user", "content": user_prompt}]
            if system_prompt:
                messages.insert(0, {"role": "system", "content": system_prompt})

            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                extra_body=DEFAULT_DISABLE_THINKING_EXTRA_BODY,
            )
            usage = extract_response_token_usage(response)
            add_token_usage(token_usage, "controller", usage)
            completion_tokens = usage.get("completion_tokens") if usage else None
            raw_content = response.choices[0].message.content or ""
            parsed = parse_json_response_lenient(raw_content)
            # third value: raw model output (analysis only; never feeds decisions)
            return parsed, completion_tokens, raw_content
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < max_retries:
                time.sleep(1.0)
    raise RuntimeError(last_error or "Model call failed without error details.")


def build_output_budget_info(remaining: int | None, answer_reserve: int) -> str:
    """Build the budget notice shown to the controller."""
    if remaining is None:
        return "Unlimited."
    if remaining <= 0:
        return (
            f"Exhausted ({remaining} tokens left). You may only output an empty response; "
            "the question will be answered from already accumulated evidence."
        )
    reserve_note = (
        f" Reserve at least {answer_reserve} tokens for the final answer." if answer_reserve else ""
    )
    return (
        f"You have {remaining} output tokens left for the controller's reading actions. "
        f"Keep reasoning, evidence snippets and JSON structure within this budget.{reserve_note}"
    )


def evidence_from_wikis(
    topk_sessions: list[dict[str, Any]],
    wiki_map_by_id: dict[str, dict[str, Any]],
    *,
    max_snippets: int = 5,
) -> list[dict[str, str]]:
    """Fallback when the budget is too low for the controller to emit evidence:
    synthesize evidence from the initial top-k wiki pages.
    """
    evidence: list[dict[str, str]] = []
    for session in (topk_sessions or [])[:max_snippets]:
        row = wiki_map_by_id.get(session.get("wiki_id")) or {}
        snippet = normalize_string(row.get("summary"))[:400]
        if not snippet:
            continue
        evidence.append(
            {
                "snippet": snippet,
                "date_time": normalize_string(row.get("source_session_date_time")) or "N/A",
            }
        )
    return evidence


def _extract_logprobs(response: Any) -> dict[str, Any] | None:
    """Extract token-level logprobs from a response."""
    try:
        logprobs = response.choices[0].logprobs
        if logprobs is None:
            return None
        tokens = []
        for item in (logprobs.content or []):
            tokens.append({
                "token": item.token,
                "logprob": item.logprob,
            })
        return {"tokens": tokens}
    except Exception:
        return None


def call_text_with_retries(
    *,
    client: Any,
    model_name: str,
    system_prompt: str | None,
    user_prompt: str,
    max_tokens: int,
    temperature: float,
    max_retries: int,
    token_usage: dict[str, dict[str, int]],
) -> tuple[str, dict[str, Any] | None]:
    """Answer stage: return raw model text together with logprobs."""
    last_error = ""
    for attempt in range(1, max_retries + 1):
        try:
            messages = [{"role": "user", "content": user_prompt}]
            if system_prompt:
                messages.insert(0, {"role": "system", "content": system_prompt})

            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                extra_body=DEFAULT_DISABLE_THINKING_EXTRA_BODY,
                logprobs=True,
                top_logprobs=1,
            )
            add_token_usage(token_usage, "answer", extract_response_token_usage(response))
            text = response.choices[0].message.content or ""
            logprobs = _extract_logprobs(response)
            return text, logprobs
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < max_retries:
                time.sleep(1.0)
    raise RuntimeError(last_error or "Answer model call failed without error details.")


def compute_teacher_forcing_logprobs(
    *,
    client: Any,
    model_name: str,
    system_prompt: str | None,
    user_prompt: str,
    gold_answer: str,
    max_retries: int,
    token_usage: dict[str, dict[str, int]],
) -> dict[str, Any] | None:
    """Compute teacher-forcing PPL via prompt_logprobs.

    The gold answer is appended as an assistant message. Two requests (with and without
    gold) differ in prompt length, which locates the gold token positions; the logprob
    of each gold token is then read off prompt_logprobs.
    """
    gold = gold_answer.strip()
    if not gold:
        return None

    messages_no_gold = [{"role": "user", "content": user_prompt}]
    if system_prompt:
        messages_no_gold.insert(0, {"role": "system", "content": system_prompt})

    messages_with_gold = messages_no_gold + [{"role": "assistant", "content": gold}]

    for attempt in range(1, max_retries + 1):
        try:
            # Step 1: prompt token count without gold
            resp1 = client.chat.completions.create(
                model=model_name,
                messages=messages_no_gold,
                max_tokens=1,
                temperature=0.0,
                extra_body=DEFAULT_DISABLE_THINKING_EXTRA_BODY,
            )
            add_token_usage(token_usage, "answer", extract_response_token_usage(resp1))
            n_prompt = resp1.usage.prompt_tokens

            # Step 2: with gold, enable prompt_logprobs
            resp2 = client.chat.completions.create(
                model=model_name,
                messages=messages_with_gold,
                max_tokens=1,
                temperature=0.0,
                logprobs=True,
                top_logprobs=1,
                extra_body={**DEFAULT_DISABLE_THINKING_EXTRA_BODY, "prompt_logprobs": 1},
            )
            add_token_usage(token_usage, "answer", extract_response_token_usage(resp2))
            plp = getattr(resp2, "prompt_logprobs", None)
            if not plp:
                plp = resp2.prompt_logprobs if hasattr(resp2, "prompt_logprobs") else None
            if not plp:
                return None

            # Step 3: gold-answer token logprobs (positions >= n_prompt, stop at the first <|im_end|>)
            tf_tokens: list[dict[str, Any]] = []
            for i in range(n_prompt, len(plp)):
                item = plp[i]
                if item is None:
                    continue
                # each position maps token ids to info; keep only the first entry (model top-1)
                for _tid, info in item.items():
                    tok = info.get("decoded_token", str(_tid))
                    lp = info.get("logprob", 0.0)
                    if tok in ("<|im_end|>", "assistant", "user"):
                        # stop at the end-of-turn marker or the next role tag
                        break
                    if tok.startswith("<") and tok.endswith(">"):
                        continue
                    if tok in ("\n", "\n\n"):
                        continue
                    tf_tokens.append({"token": tok, "logprob": lp})
                    break  # only the first entry per position

            return {"tokens": tf_tokens} if tf_tokens else None

        except Exception:
            if attempt >= max_retries:
                return None
            time.sleep(1.0)

    return None


def answer_from_evidence(
    *,
    dataset: str,
    row: dict[str, Any],
    accumulated_evidence: list[dict[str, str]],
    client: Any,
    model_name: str,
    max_tokens: int,
    temperature: float,
    max_retries: int,
    token_usage: dict[str, dict[str, int]],
) -> dict[str, str | dict[str, Any] | None]:
    """Generate the final answer from accumulated evidence, with logprobs and TF logprobs."""
    system_prompt, prompt_template = answer_prompt_pair(dataset)
    context = format_evidence_list(accumulated_evidence, heading="Evidences:") if accumulated_evidence else "No evidence available."
    user_prompt = prompt_template.format(question=build_question_text(dataset, row), context=context)

    answer_text, logprobs = call_text_with_retries(
        client=client,
        model_name=model_name,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        max_retries=max_retries,
        token_usage=token_usage,
    )

    # Teacher forcing: PPL(gold_answer | evidence)
    gold_answer = row.get("answer", "")
    tf_logprobs = compute_teacher_forcing_logprobs(
        client=client,
        model_name=model_name,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        gold_answer=gold_answer,
        max_retries=max_retries,
        token_usage=token_usage,
    )

    return {
        "answer": answer_text.strip(),
        "answer_logprobs": logprobs,
        "gold_answer_logprobs": tf_logprobs,
        "answer_system_prompt": system_prompt,
        "answer_user_prompt": user_prompt,
    }


def build_reasoning_instruction(budget: int | None) -> str:
    """Build the controller reasoning instruction from --reasoning-budget.

    budget=None: unlimited (default); budget=0: no reasoning; budget>0: at most N words.
    """
    if budget is None:
        return "Thinking about the current information and what action to choose."
    if budget <= 0:
        return "Do NOT include any reasoning. Set the reasoning field to an empty string."
    return (
        "Thinking about the current information and what action to choose, "
        f"but keep the reasoning concise: at most {budget} words."
    )


def run_t2mem_query(
    row: dict[str, Any],
    *,
    dataset: str,
    args: Any,
    client: Any,
    query_sessions_map: dict[str, list[dict[str, Any]]],
    wiki_map_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Run the T2Mem loop for one query; return the full step trace and final answer."""
    token_usage = new_token_usage()

    if should_skip_t2mem_row(dataset, row):
        result = build_result_metadata(dataset, row)
        result.update(
            {
                "prediction_wiki_context": [],
                "initial_prediction_wiki_context": [],
                "steps_taken": 0,
                "stop_reason": "SKIPPED_CATEGORY_5",
                "controller_steps": [],
                "accumulated_evidence": [],
                #"reasoning": "",
                "prediction": "",
                "skipped": True,
                "token_usage": token_usage,
            }
        )
        return result

    # Per-query entry point:
    # 1. feed the top-k retrieval results as the step-1 observation;
    # 2. let the controller pick the next action;
    # 3. follow that action to fetch more observation and accumulate evidence;
    # 4. finally answer from the accumulated evidence.
    result = build_result_metadata(dataset, row)
    topk_sessions = list(row.get("topk_sessions") or [])
    query_key = row["sample_id"]
    query_sessions = list(query_sessions_map.get(query_key, []))

    # step 1: render the top-k entries as the controller's initial observation.
    read_unit = str(getattr(args, "read_unit", "wiki") or "wiki")
    if read_unit == "raw":
        initial_observation, initial_source_ids, initial_wiki_ids = build_raw_direct_context(
            topk_sessions,
            query_sessions=query_sessions,
            start_index=0,
            end_index=args.top_k,
            heading="",
        )
    else:
        initial_observation, initial_source_ids, initial_wiki_ids = build_wiki_context(
            topk_sessions,
            wiki_map=wiki_map_by_id,
            start_index=0,
            end_index=args.top_k,
            heading="",
        )

    # visited_* records which sessions/wiki entries this query touched (for analysis).
    # accumulated_evidence is the pool the answer stage actually consumes.
    visited_source_ids = list(dict.fromkeys(initial_source_ids))
    visited_wiki_ids = list(dict.fromkeys(initial_wiki_ids))
    accumulated_evidence: list[dict[str, str]] = []
    evidence_seen: set[tuple[str, str, str]] = set()
    step_trace: list[dict[str, Any]] = []
    current_observation = initial_observation
    current_observation_kind = "INITIAL_TOPK_RAW" if read_unit == "raw" else "INITIAL_TOPK_WIKI"
    current_source_ids = list(initial_source_ids)
    current_wiki_ids = list(initial_wiki_ids)
    next_wiki_cursor = min(args.top_k, len(topk_sessions))
    stop_reason = "max_steps"

    # Total output budget: constrains controller output only, not the number of rounds.
    output_budget = getattr(args, "output_budget_tokens", None)
    remaining_output = output_budget
    answer_reserve = 0  # the answer stage has its own max_tokens; a reserve of 100 only deadlocked the gate
    min_call_tokens = 50  # below this remaining budget no further call is issued; degrade instead
    controller_calls = 0
    controller_output_tokens = 0
    controller_fallback = False  # truncation broke parsing -> skip reading, fall back to the initial pages
    # Analysis mode: dump the exact controller / answer prompts (never affects decisions).
    dump_prompts = bool(getattr(args, "dump_prompts", False))

    # Ablation switches: narrow the action set and select the matching prompt pair.
    allow_read_raw = bool(getattr(args, "allow_read_raw", True))
    if read_unit == "raw":
        controller_allowed_actions = ALLOWED_ACTIONS_RAW_ONLY
        controller_system_prompt_template = ADAPTIVE_READING_SYSTEM_PROMPT_NO_LOOK_RAW_ONLY
        controller_action_prompt_template = ADAPTIVE_READING_ACTION_PROMPT_TEMPLATE_NO_LOOK_RAW_ONLY
    elif allow_read_raw:
        controller_allowed_actions = ALLOWED_ACTIONS
        controller_system_prompt_template = ADAPTIVE_READING_SYSTEM_PROMPT_NO_LOOK
        controller_action_prompt_template = ADAPTIVE_READING_ACTION_PROMPT_TEMPLATE_NO_LOOK
    else:
        controller_allowed_actions = ALLOWED_ACTIONS_NO_RAW
        controller_system_prompt_template = ADAPTIVE_READING_SYSTEM_PROMPT_NO_LOOK_NO_RAW
        controller_action_prompt_template = ADAPTIVE_READING_ACTION_PROMPT_TEMPLATE_NO_LOOK_NO_RAW
    action_fallback_count = 0

    try:
        if current_observation.strip():
            for step in range(1, args.max_steps + 1):
                # Hard budget gate: remaining budget cannot fund even the smallest call -> stop reading.
                if output_budget is not None:
                    available_for_call = remaining_output - answer_reserve
                    if available_for_call < min_call_tokens:
                        stop_reason = "output_budget_exhausted"
                        break
                    call_max_tokens = min(args.action_max_tokens, available_for_call)
                else:
                    call_max_tokens = args.action_max_tokens
                # What the controller sees this turn:
                # 1. the observation just fetched;
                # 2. the evidence accumulated so far.
                current_information = compose_current_information(
                    observation=current_observation,
                    accumulated_evidence=accumulated_evidence,
                )
                # The controller only decides what to look at next; it never writes the final answer.
                controller_system_prompt = controller_system_prompt_template
                controller_user_prompt = controller_action_prompt_template.format(
                    question=build_question_text(dataset, row),
                    current_information=current_information,
                    reasoning_instruction=build_reasoning_instruction(
                        getattr(args, "reasoning_budget", None)
                    ),
                    output_budget_info=build_output_budget_info(
                        remaining_output, answer_reserve
                    ),
                )
                try:
                    controller_response, controller_completion_tokens, controller_raw_output = call_json_with_retries(
                        client=client,
                        model_name=args.controller_model_name,
                        system_prompt=controller_system_prompt,
                        user_prompt=controller_user_prompt,
                        max_tokens=call_max_tokens,
                        temperature=args.temperature,
                        max_retries=args.max_retries,
                        token_usage=token_usage,
                    )
                except RuntimeError:
                    # Budget truncation broke the JSON -> degrade gracefully instead of failing the query.
                    if output_budget is not None:
                        controller_fallback = True
                        stop_reason = "output_budget_parse_fallback"
                        break
                    raise
                controller_response = normalize_action_response(
                    controller_response, allowed_actions=controller_allowed_actions
                )
                controller_calls += 1
                if controller_response.get("action_fallback"):
                    action_fallback_count += 1
                if controller_completion_tokens is not None:
                    controller_output_tokens += controller_completion_tokens
                    if output_budget is not None:
                        remaining_output -= controller_completion_tokens
                if controller_response["action"] == "READ_RAW_FROM_WIKI":
                    controller_response["target_wiki_id"] = canonicalize_wiki_ids(
                        controller_response["target_wiki_id"],
                        query_sessions=query_sessions,
                    )
                else:
                    controller_response["target_wiki_id"] = []

                # Each turn may yield new evidence; deduplicate and append to the pool.
                step_evidence = list(controller_response["evidence"])
                evidence_count_before_step = len(accumulated_evidence)
                extend_evidence(accumulated_evidence, step_evidence, evidence_seen)

                # step_record is written verbatim to the output JSONL for later per-turn inspection.
                step_record: dict[str, Any] = {
                    "step": step,
                    "observation_kind": current_observation_kind,
                    "observation_source_ids": current_source_ids,
                    "observation_wiki_ids": current_wiki_ids,
                    "accumulated_evidence_count_before_step": evidence_count_before_step,
                    "accumulated_evidence_count_after_step": len(accumulated_evidence),
                    "reasoning": controller_response["reasoning"],
                    "action": controller_response["action"],
                    "evidence": step_evidence,
                    "target_wiki_id": controller_response["target_wiki_id"],
                    "controller_completion_tokens": controller_completion_tokens,
                    "remaining_output_budget": remaining_output,
                }
                if dump_prompts:
                    # Exact input and raw output of this controller turn (for analysis).
                    step_record["controller_system_prompt"] = controller_system_prompt
                    step_record["controller_user_prompt"] = controller_user_prompt
                    step_record["controller_raw_output"] = controller_raw_output

                # The controller judged the evidence sufficient: stop iterating and answer.
                if controller_response["action"] == "ANSWER":
                    stop_reason = "ANSWER"
                    step_trace.append(step_record)
                    break

                followup_kind = controller_response["action"]
                followup_source_ids: list[str] = []
                followup_wiki_ids: list[str] = []

                if controller_response["action"] in ("SEARCH_WIKI", "SEARCH_RAW"):
                    # Both advance a cursor over the existing top-k list; only the payload differs.
                    # Equivalent to Top-k(q, I \ I_seen): the ranking is fixed and the seen set is always
                    # a prefix, so the next unseen entry is R[|I_seen|] -- exactly next_wiki_cursor.
                    if read_unit == "raw":
                        current_observation, followup_source_ids, followup_wiki_ids = build_raw_direct_context(
                            topk_sessions,
                            query_sessions=query_sessions,
                            start_index=next_wiki_cursor,
                            end_index=next_wiki_cursor + 1,
                            heading="",
                        )
                        empty_msg = (
                            "Additional raw session requested, but there are no more retrievable sessions left."
                        )
                    else:
                        current_observation, followup_source_ids, followup_wiki_ids = build_wiki_context(
                            topk_sessions,
                            wiki_map=wiki_map_by_id,
                            start_index=next_wiki_cursor,
                            end_index=next_wiki_cursor + 1,
                            heading="",
                        )
                        empty_msg = (
                            "Additional wiki memory requested, but there are no more retrievable wiki pages left."
                        )
                    if not followup_source_ids:
                        current_observation = empty_msg
                    next_wiki_cursor = min(next_wiki_cursor + 1, len(topk_sessions))
                elif controller_response["action"] == "READ_RAW_FROM_WIKI":
                    # READ_RAW_FROM_WIKI: resolve each controller-supplied wiki_id back to its raw session.
                    current_observation, followup_source_ids, followup_wiki_ids = build_raw_context(
                        controller_response["target_wiki_id"],
                        query_sessions=query_sessions,
                    )

                # Whatever the action, record the sessions/wiki entries newly visited this turn.
                visited_source_ids = list(dict.fromkeys(visited_source_ids + followup_source_ids))
                visited_wiki_ids = list(dict.fromkeys(visited_wiki_ids + followup_wiki_ids))
                current_observation_kind = followup_kind
                current_source_ids = list(followup_source_ids)
                current_wiki_ids = list(followup_wiki_ids)

                step_record["followup_observation_kind"] = followup_kind
                step_record["followup_source_ids"] = followup_source_ids
                step_record["followup_wiki_ids"] = followup_wiki_ids
                step_trace.append(step_record)
            else:
                stop_reason = "MAX_STEPS"
        else:
            stop_reason = "NO_INITIAL_CONTEXT"

        # Budget too low for the controller to ever emit evidence -> fall back to the initial top-k.
        if controller_fallback or (output_budget is not None and not accumulated_evidence):
            accumulated_evidence = evidence_from_wikis(
                topk_sessions, wiki_map_by_id, max_snippets=args.top_k
            )

        # After the loop the answer model sees only accumulated evidence, never raw context.
        answer_result = answer_from_evidence(
            dataset=dataset,
            row=row,
            accumulated_evidence=accumulated_evidence,
            client=client,
            model_name=args.answer_model_name,
            max_tokens=args.answer_max_tokens,
            temperature=args.temperature,
            max_retries=args.max_retries,
            token_usage=token_usage,
        )

        final_prediction = answer_result["answer"]
        final_logprobs = answer_result.get("answer_logprobs")
        gold_logprobs = answer_result.get("gold_answer_logprobs")

        # Keep the controller step trace, accumulated evidence and final answer in the output.
        result.update(
            {
                "prediction_wiki_context": visited_wiki_ids,
                "initial_prediction_wiki_context": initial_wiki_ids,
                "steps_taken": len(step_trace),
                "stop_reason": stop_reason,
                "controller_steps": step_trace,
                "accumulated_evidence": accumulated_evidence,
                "prediction": final_prediction,
                "answer_logprobs": final_logprobs,
                "gold_answer_logprobs": gold_logprobs,
                "token_usage": token_usage,
                "output_budget": output_budget,
                "controller_calls": controller_calls,
                "controller_output_tokens": controller_output_tokens,
                "read_mode": str(getattr(args, "read_mode", "full")),
                "allow_read_raw": allow_read_raw,
                "read_unit": read_unit,
                "action_fallback_count": action_fallback_count,
            }
        )
        if dump_prompts:
            # Exact input of the final answer stage (for analysis).
            result["answer_system_prompt"] = answer_result.get("answer_system_prompt")
            result["answer_user_prompt"] = answer_result.get("answer_user_prompt")
        return result
    except Exception as exc:
        # On failure, emit how far the loop got and what evidence was collected, to aid debugging.
        result.update(
            {
                "prediction_wiki_context": visited_wiki_ids,
                "initial_prediction_wiki_context": initial_wiki_ids,
                "steps_taken": len(step_trace),
                "stop_reason": "ERROR",
                "controller_steps": step_trace,
                "accumulated_evidence": accumulated_evidence,
                #"reasoning": "",
                "prediction": "",
                "error": f"{type(exc).__name__}: {exc}",
                "token_usage": token_usage,
            }
        )
        return result
