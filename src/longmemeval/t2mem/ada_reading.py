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
    QA_ANSWER_LONGMEM_ADA_PROMPT_TEMPLATE,
)
from utils.llm_utils import DEFAULT_DISABLE_THINKING_EXTRA_BODY, call_json_llm, call_llm

ALLOWED_ACTIONS = {"SEARCH_WIKI", "READ_RAW_FROM_WIKI", "ANSWER"}
# wiki-only ablation: raw transcripts are unreadable
ALLOWED_ACTIONS_NO_RAW = {"SEARCH_WIKI", "ANSWER"}
# raw-only ablation: no wiki, read raw sessions along the retrieval order
ALLOWED_ACTIONS_RAW_ONLY = {"SEARCH_RAW", "ANSWER"}


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
        key = record["question_id"]
        session_map[key] = list(record.get("sessions"))
    return session_map


def load_query_wiki_map(wiki_path: Path) -> dict[str, dict[str, dict[str, Any]]]:
    """Build a per-question wiki lookup: question_id -> wiki_id -> row."""
    rows = load_jsonl(wiki_path)
    query_wiki_map: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        question_id = normalize_string(row.get("question_id"))
        wiki_id = normalize_string(row.get("wiki_id"))
        if not question_id or not wiki_id:
            continue
        query_wiki_map.setdefault(question_id, {})[wiki_id] = row
    return query_wiki_map


def build_result_metadata(dataset: str, row: dict[str, Any]) -> dict[str, Any]:
    """Base fields copied into the output JSONL (excludes runtime-generated content)."""
    return {
        "dataset": dataset,
        "question": row["question"],
        "question_id": row["question_id"],
        "question_type": row.get("question_type"),
        "question_date": row.get("question_date"),
        "answer": row.get("answer"),
        "answer_session_ids": row.get("answer_session_ids"),
    }

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
    query_key: str,
    query_wiki_map: dict[str, dict[str, Any]],
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
        if not wiki_id:
            raise KeyError(f"Missing wiki_id in retrieval row for question_id={query_key}.")
        if not query_wiki_map:
            raise KeyError(f"No wiki rows are available for question_id={query_key}.")
        if wiki_id not in query_wiki_map:
            raise KeyError(
                f"Retrieved wiki_id={wiki_id} is outside the allowed wiki set for question_id={query_key}."
            )

        row = query_wiki_map[wiki_id]
        source_session_id = normalize_string(row.get("source_session_id")) or normalize_string(item.get("source_session_id"))
        retrieved_source_session_id = normalize_string(item.get("source_session_id"))
        if retrieved_source_session_id and source_session_id and retrieved_source_session_id != source_session_id:
            raise ValueError(
                f"Mismatched source_session_id for question_id={query_key}: retrieval row has source_session_id={retrieved_source_session_id}, "
                f"but wiki_id={wiki_id} belongs to source_session_id={source_session_id}."
            )
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
        raise ValueError("No target wiki id was provided for READ_RAW_FROM_WIKI.")

    blocks = ["Raw sessions requested from wiki:"]
    source_session_ids: list[str] = []
    wiki_ids: list[str] = []
    local_raw_by_wiki_id = {
        normalize_string(session.get("wiki_id")): session
        for session in query_sessions
        if normalize_string(session.get("wiki_id"))
    }

    for target in target_ids:
        wiki_id = normalize_string(target)
        if not wiki_id:
            raise KeyError("READ_RAW_FROM_WIKI returned an empty wiki_id.")
        if wiki_id not in local_raw_by_wiki_id:
            raise KeyError(f"READ_RAW_FROM_WIKI requested wiki_id={wiki_id}, but it does not belong to the current question.")

        session = local_raw_by_wiki_id[wiki_id]
        source_session_id = normalize_string(session.get("source_session_id"))
        source_session_ids.append(source_session_id)
        wiki_ids.append(wiki_id)
        blocks.append(f"[Raw From Wiki]\nWiki ID: {wiki_id}\n{raw_text(session)}")

    return "\n\n".join(blocks).strip(), source_session_ids, wiki_ids


def build_raw_direct_context(
    topk_sessions: list[dict[str, Any]],
    *,
    query_key: str,
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
        if not wiki_id:
            raise KeyError(f"Missing wiki_id in retrieval row for question_id={query_key}.")
        if wiki_id not in local_raw_by_wiki_id:
            raise KeyError(
                f"Retrieved wiki_id={wiki_id} is outside the allowed session set for question_id={query_key}."
            )
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
    ANSWER (matching pre-existing semantics), and action_fallback is recorded.
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
    """Keep only wiki_ids that belong to the current query."""
    canonical_ids: list[str] = []
    seen: set[str] = set()
    local_wiki_ids = {
        normalize_string(session.get("wiki_id"))
        for session in query_sessions
        if normalize_string(session.get("wiki_id"))
    }

    for target in target_ids:
        normalized = normalize_string(target)
        if not normalized:
            raise KeyError("READ_RAW_FROM_WIKI returned an empty wiki_id.")
        if normalized not in local_wiki_ids:
            raise KeyError(f"READ_RAW_FROM_WIKI requested wiki_id={normalized}, but it does not belong to the current question.")
        if normalized in seen:
            continue
        seen.add(normalized)
        canonical_ids.append(normalized)

    return canonical_ids


def extend_evidence(
    accumulated_evidence: list[dict[str, str]],
    new_evidence: list[dict[str, str]],
    seen_keys: set[tuple[str, str]],
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


def answer_prompt_pair(dataset: str) -> tuple[str, str]:
    """Pick the adaptive-answer prompt for the current dataset."""
    return "", QA_ANSWER_LONGMEM_ADA_PROMPT_TEMPLATE


def call_json_with_retries(
    *,
    client: Any,
    model_name: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    temperature: float,
    max_retries: int,
) -> dict[str, Any]:
    """Call the JSON LLM with bounded retries on transient failure."""
    last_error = ""
    for attempt in range(1, max_retries + 1):
        try:
            return call_json_llm(
                client=client,
                model_name=model_name,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                extra_body=DEFAULT_DISABLE_THINKING_EXTRA_BODY,
            )
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < max_retries:
                time.sleep(1.0)
    raise RuntimeError(last_error or "Model call failed without error details.")


def call_answer_with_retries(
    *,
    client: Any,
    model_name: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    temperature: float,
    max_retries: int,
) -> str:
    """Answer stage: return the raw model text."""
    last_error = ""
    for attempt in range(1, max_retries + 1):
        try:
            return call_llm(
                client=client,
                model_name=model_name,
                system_prompt=system_prompt or None,
                user_prompt=user_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                extra_body=DEFAULT_DISABLE_THINKING_EXTRA_BODY,
            ).strip()
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < max_retries:
                time.sleep(1.0)
    raise RuntimeError(last_error or "Model call failed without error details.")


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
) -> dict[str, str]:
    """Generate the final answer from accumulated evidence."""
    system_prompt, prompt_template = answer_prompt_pair(dataset)
    context = format_evidence_list(accumulated_evidence, heading="Evidences:") if accumulated_evidence else "No evidence available."
    answer = call_answer_with_retries(
        client=client,
        model_name=model_name,
        system_prompt=system_prompt,
        user_prompt=prompt_template.format(
            question_time=row.get("question_date", ""),
            question=row["question"],
            context=context,
        ),
        max_tokens=max_tokens,
        temperature=temperature,
        max_retries=max_retries,
    )
    return {"answer": normalize_string(answer)}


def run_t2mem_query(
    row: dict[str, Any],
    *,
    dataset: str,
    args: Any,
    client: Any,
    query_sessions_map: dict[str, list[dict[str, Any]]],
    query_wiki_map_by_question: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    """Run the T2Mem loop for one query; return the full step trace and final answer."""
    # Per-query entry point:
    # 1. feed the top-k retrieval results as the step-1 observation;
    # 2. let the controller pick the next action;
    # 3. follow that action to fetch more observation and accumulate evidence;
    # 4. finally answer from the accumulated evidence.
    result = build_result_metadata(dataset, row)
    topk_sessions = list(row.get("topk_sessions") or [])
    query_key = row["question_id"]
    query_sessions = list(query_sessions_map.get(query_key, []))
    query_wiki_map = query_wiki_map_by_question.get(query_key, {})

    # Reading configuration: full | wiki-only (ablation A) | raw-only (ablation B)
    read_mode = str(getattr(args, "read_mode", "full") or "full")
    read_unit = "raw" if read_mode == "raw-only" else "wiki"
    allow_read_raw = read_mode == "full"
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

    # step 1: render the top-k entries as the controller's initial observation (wiki or raw).
    if read_unit == "raw":
        initial_observation, initial_source_ids, initial_wiki_ids = build_raw_direct_context(
            topk_sessions,
            query_key=query_key,
            query_sessions=query_sessions,
            start_index=0,
            end_index=args.top_k,
            heading="",
        )
    else:
        initial_observation, initial_source_ids, initial_wiki_ids = build_wiki_context(
            topk_sessions,
            query_key=query_key,
            query_wiki_map=query_wiki_map,
            start_index=0,
            end_index=args.top_k,
            heading="",
        )

    # visited_* records which sessions/wiki entries this query touched (for analysis).
    # accumulated_evidence is the pool the answer stage actually consumes.
    visited_source_ids = list(dict.fromkeys(initial_source_ids))
    visited_wiki_ids = list(dict.fromkeys(initial_wiki_ids))
    accumulated_evidence: list[dict[str, str]] = []
    evidence_seen: set[tuple[str, str]] = set()
    step_trace: list[dict[str, Any]] = []
    current_observation = initial_observation
    current_observation_kind = "INITIAL_TOPK_RAW" if read_unit == "raw" else "INITIAL_TOPK_WIKI"
    current_source_ids = list(initial_source_ids)
    current_wiki_ids = list(initial_wiki_ids)
    next_wiki_cursor = min(args.top_k, len(topk_sessions))
    stop_reason = "max_steps"

    try:
        if current_observation.strip():
            for step in range(1, args.max_steps + 1):
                # What the controller sees this turn:
                # 1. the observation just fetched;
                # 2. the evidence accumulated so far.
                current_information = compose_current_information(
                    observation=current_observation,
                    accumulated_evidence=accumulated_evidence,
                )
                # The controller only decides what to look at next; it never writes the final answer.
                controller_response = normalize_action_response(
                    call_json_with_retries(
                        client=client,
                        model_name=args.controller_model_name,
                        system_prompt=controller_system_prompt_template,
                        user_prompt=controller_action_prompt_template.format(
                            question=row["question"],
                            current_information=current_information,
                        ),
                        max_tokens=args.action_max_tokens,
                        temperature=args.temperature,
                        max_retries=args.max_retries,
                    ),
                    allowed_actions=controller_allowed_actions,
                )
                if controller_response.get("action_fallback"):
                    action_fallback_count += 1
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
                }

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
                            query_key=query_key,
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
                            query_key=query_key,
                            query_wiki_map=query_wiki_map,
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
        )
        final_prediction = answer_result["answer"]

        # The output also keeps:
        # - the full controller step trace
        # - the final accumulated evidence
        # - the main answer result
        result.update(
            {
                "prediction_wiki_context": visited_wiki_ids,
                "initial_prediction_wiki_context": initial_wiki_ids,
                "steps_taken": len(step_trace),
                "stop_reason": stop_reason,
                "controller_steps": step_trace,
                "accumulated_evidence": accumulated_evidence,
                #"reasoning": final_reasoning,
                "prediction": final_prediction,
                #"adaptive_answer_reasoning": answer_result["reasoning"],
                "adaptive_answer_prediction": answer_result["answer"],
                "read_mode": read_mode,
                "allow_read_raw": allow_read_raw,
                "read_unit": read_unit,
                "action_fallback_count": action_fallback_count,
            }
        )
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
            }
        )
        return result
