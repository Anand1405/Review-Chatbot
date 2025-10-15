from __future__ import annotations

import os
import re
import sqlite3
import uuid
from datetime import date, datetime
from typing import Any, Callable, Dict, List, Optional, TypedDict

import pandas as pd
from dotenv import load_dotenv
from dateutil.relativedelta import relativedelta
from collections import Counter
from conversation_store import ConversationStore

load_dotenv()  # Load environment variables from a .env file if present

# ---------------------------------------------------------------------------
MemorySaver = None  
SQLiteSaver = None  


from langgraph.graph import StateGraph, END   
_LANGGRAPH_AVAILABLE = True

if _LANGGRAPH_AVAILABLE:
    try:
        from langgraph.checkpoint.memory import MemorySaver  
    except Exception:
        MemorySaver = None  

    try:
        from langgraph.checkpoint.sqlite import SqliteSaver  
        SQLiteSaver = SqliteSaver  
    except Exception:
        try:
            from langgraph.checkpoint import SQLiteSaver as _OldSQLiteSaver  
            SQLiteSaver = _OldSQLiteSaver  
        except Exception:
            SQLiteSaver = None  
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langchain.chains import create_sql_query_chain
from langchain_community.utilities import SQLDatabase


# -----------------------------------------------------------------------------
# Data structures for the agent state

class AgentState(TypedDict):
    """The mutable state carried through the LangGraph.

    Attributes
    ----------
    question : str
        The user question.
    filters : Optional[dict]
        Structured filters extracted from the question (rating bucket,
        time range, platform, version).
    sql : Optional[str]
        The SQL query generated to retrieve relevant reviews.
    reviews : Optional[list]
        A list of dictionaries representing the retrieved reviews.  We
        avoid storing a pandas DataFrame here because DataFrames are
        not serialisable by the default LangGraph checkpointers.
    answer : Optional[str]
        The final answer text returned to the user.
    history : List[BaseMessage]
        Conversation history messages for context and memory.
    notes : Optional[List[str]]
        A list of notes or error messages captured during execution.
    """

# -----------------------------------------------------------------------------
# Helper functions

def init_llm(model: str = "gemini-2.5-flash", temperature: float = 0.0):
    """Initialise a ChatGoogleGenerativeAI instance."""
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    return ChatGoogleGenerativeAI(
        model=model,
        temperature=temperature,
        convert_system_message_to_human=True,
        google_api_key=api_key,
    )


def create_sql_chain(db: SQLDatabase, llm: ChatGoogleGenerativeAI):
    """Create a LangChain chain that converts NL questions into SQL queries."""
    
    template = (
        "You are an assistant tasked with generating SQL to answer the user's\n"
        "question about app reviews. The database has a single table called reviews\n"
        "with columns: id, rating, text, date, version, platform.\n\n"
        "Table information:\n{table_info}\n\n"
        "If a suggested row limit is provided ({top_k}), apply it when appropriate.\n"
        "Translate the following natural-language question into a SQL query.\n"
        "The query should return the columns id, rating, text, date, version,\n"
        "platform. Apply any relevant filters (rating thresholds, date ranges,\n"
        "platform, version) specified in the question. Do not hallucinate column\n"
        "names. Only return the SQL query, with no extraneous text.\n\n"
        "Question: {input}"
    )
    prompt = PromptTemplate(input_variables=["input", "top_k", "table_info"], template=template)
    # Build the SQL query chain with our custom prompt.  We rely on
    # create_sql_query_chain to handle top_k and table_info internally.
    return create_sql_query_chain(llm, db, prompt=prompt)


def _enrich_filters_from_question(question: str, filters: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Fill in missing filter slots with simple rule-based parsing."""
    result: Dict[str, Any] = dict(filters or {})
    text = question.lower()

    if not result.get("rating"):
        if match := re.search(r"(?:rated?|ratings?)\s+(?:below|under)\s+(\d)", text):
            result["rating"] = f"<{match.group(1)}"
        elif match := re.search(r"(?:rated?|ratings?)\s+(?:above|over|at least)\s+(\d)", text):
            result["rating"] = f">={match.group(1)}"
        elif match := re.search(r"(\d)[- ]?star", text):
            result["rating"] = match.group(1)

    if not result.get("date_range"):
        if match := re.search(r"last\s+(\d+)\s+month", text):
            months = int(match.group(1))
            end = date.today()
            start = end - relativedelta(months=months)
            result["date_range"] = f"{start.isoformat()} to {end.isoformat()}"
        elif "this quarter" in text:
            today = date.today()
            quarter_start_month = 3 * ((today.month - 1) // 3) + 1
            start = date(today.year, quarter_start_month, 1)
            result["date_range"] = f"{start.isoformat()} to {today.isoformat()}"

    if not result.get("platform"):
        if "android" in text:
            result["platform"] = "Android"
        elif "ios" in text or "iphone" in text or "apple" in text:
            result["platform"] = "iOS"

    if not result.get("version"):
        if match := re.search(r"since v?\s*([0-9]+(?:\.[0-9]+)*)", text):
            result["version"] = f">={match.group(1)}"
        elif match := re.search(r"version\s+([0-9]+(?:\.[0-9]+)*)", text):
            result["version"] = match.group(1)

    return result


def _parse_numeric_version(value: Any) -> Optional[float]:
    """Extract a numeric major/minor version from a version string."""
    if value is None:
        return None
    match = re.search(r"\d+(?:\.\d+)?", str(value))
    if match is None:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def _apply_version_filter(df: pd.DataFrame, version_expr: str) -> tuple[pd.DataFrame, int]:
    """Return a DataFrame filtered by the version expression and count dropped rows."""
    pattern = re.compile(r"^\s*(<=|>=|<|>|=)?\s*([0-9]+(?:\.[0-9]+)?)\s*$")
    match = pattern.match(str(version_expr))
    if match is None:
        return df, 0
    operator = match.group(1) or "="
    target = float(match.group(2))

    def compare(value: Optional[float]) -> Optional[bool]:
        if value is None:
            return None
        if operator == "<":
            return value < target
        if operator == "<=":
            return value <= target
        if operator == ">":
            return value > target
        if operator == ">=":
            return value >= target
        # Treat '=' or bare numbers as equality on the major.minor pattern.
        return value == target

    parsed_series = df["version"].apply(_parse_numeric_version)
    comparison = parsed_series.apply(compare)
    mask = comparison.fillna(False)
    filtered_df = df[mask]
    dropped_unknown = comparison.isna().sum()
    return filtered_df, dropped_unknown

def parse_filters(state: AgentState) -> AgentState:
    """Use the LLM to extract structured filters from the user's question."""
    llm = init_llm()
    today_iso = date.today().isoformat()
    system_prompt = (
        "You are a filter extraction assistant.  Given a user question\n"
        "about mobile app reviews, identify any explicit filters and\n"
        "return them as a JSON object with the following keys:\n"
        "\n"
        "  rating: an expression like '<3', '>=4', or '5', or null\n"
        "  date_range: an ISO date interval like '2025-01-01 to 2025-06-30' or null\n"
        "  platform: 'Android', 'iOS', or null\n"
        "  version: a specific version like '3.0', '>=2.8', etc., or null\n"
        "\n"
        f"Today's date is {today_iso}. Only include keys mentioned. If the user doesn't specify a filter,\n"
        "set the value to null. Do not include any other information.\n"
        "Respond only with JSON."
    )
    system_msg = SystemMessage(content=system_prompt)
    messages: List[BaseMessage] = [system_msg]
    # Include conversation history if present
    history = state.get("history", [])
    messages.extend(history)
    messages.append(HumanMessage(content=state["question"]))
    raw_response = llm.invoke(messages)
    try:
        import json

        filters = json.loads(raw_response.content)
    except Exception:
        filters = {}
    filters = _enrich_filters_from_question(state["question"], filters)
    state["filters"] = filters
    state.setdefault("history", []).append(raw_response)
    return state


def generate_sql_and_retrieve(state: AgentState, db_path: str) -> AgentState:
    """Generate a SQL query via the LLM and run it to obtain reviews.

    Uses a SQL generation chain to translate the question and filters into
    an executable SQL query, stores the query in ``state['sql']`` and
    executes it against the SQLite database at ``db_path``.  The result
    (a pandas DataFrame) is stored in ``state['reviews_df']``.
    """
    # Open the database using LangChain's SQLDatabase wrapper
    db = SQLDatabase.from_uri(f"sqlite:///{db_path}")
    llm = init_llm()
    sql_chain = create_sql_chain(db, llm)

    # Build a natural-language prompt that includes the filters explicitly
    question = state["question"]
    filters = state.get("filters") or {}
    filter_desc_parts = []
    if filters.get("rating"):
        filter_desc_parts.append(f"rating filter {filters['rating']}")
    if filters.get("date_range"):
        filter_desc_parts.append(f"date range {filters['date_range']}")
    if filters.get("platform"):
        filter_desc_parts.append(f"platform {filters['platform']}")
    if filters.get("version"):
        filter_desc_parts.append(f"version {filters['version']}")
    if filter_desc_parts:
        question_with_filters = question + "\nFilters: " + "; ".join(filter_desc_parts)
    else:
        question_with_filters = question

    # Ask the chain to generate a SQL query
    chain_result = sql_chain.invoke({"question": question_with_filters})
    if isinstance(chain_result, dict):
        sql_query = (
            chain_result.get("sql_query")
            or chain_result.get("query")
            or chain_result.get("result")
            or ""
        )
    else:
        sql_query = str(chain_result)
    sql_query = sql_query.replace("SQLQuery:", "").strip()
    if sql_query.startswith("```"):
        sql_query = (
            sql_query.replace("```sql", "")
            .replace("```", "")
            .strip()
        )
    state["sql"] = sql_query

    if not sql_query:
        state.setdefault("notes", []).append("SQL generator returned an empty query.")
        state["reviews"] = []
        return state

    # Execute the SQL query.  Use sqlite3 directly for simplicity.
    conn = sqlite3.connect(db_path)
    try:
        reviews_df = pd.read_sql_query(sql_query, conn)
    except Exception as e:
        # If the query fails, return an empty DataFrame and log the error
        reviews_df = pd.DataFrame(columns=["id", "rating", "text", "date", "version", "platform"])
        state.setdefault("notes", []).append(str(e))
    finally:
        conn.close()

    version_expr = filters.get("version")
    if version_expr and not reviews_df.empty:
        reviews_df, dropped_unknown = _apply_version_filter(reviews_df, version_expr)
        if dropped_unknown:
            state.setdefault("notes", []).append(
                f"Excluded {dropped_unknown} reviews with unknown version values."
            )

    # Convert the DataFrame to a list of dicts for serialisation.
    state["reviews"] = reviews_df.to_dict(orient="records")
    return state


def summarise_answer(state: AgentState) -> AgentState:
    """Generate a final answer summarising the retrieved reviews.

    The summarisation includes:

        * A 2-3 bullet summary of top themes or concerns.
        * Basic statistics (counts, percentages, trends if relevant).
        * 2-3 example review quotes with IDs.
        * Applied filters and the generated SQL query.
        * Notes about fallbacks or errors.

    The final answer is stored in ``state['answer']`` as a
    markdown-formatted string.
    """
    llm = init_llm(temperature=0.3)

    reviews_data = state.get("reviews")
    if reviews_data is None:
        reviews_df = pd.DataFrame()
    else:
        try:
            reviews_df = pd.DataFrame(reviews_data)
        except Exception:
            # If conversion fails, fall back to an empty DataFrame.
            reviews_df = pd.DataFrame()
    filters = state.get("filters", {})
    sql_query = state.get("sql")
    notes = state.get("notes", [])

    # Prepare context for the LLM.  Summarise the review texts and
    # compute simple stats in Python first to guide the model.
    n_reviews = len(reviews_df)
    stats_lines = [f"Total matching reviews: {n_reviews}"]
    # Distribution of ratings
    if not reviews_df.empty:
        rating_counts = reviews_df["rating"].value_counts().sort_index()
        for rating, count in rating_counts.items():
            pct = (count / n_reviews) * 100
            stats_lines.append(f"Rating {rating}: {count} ({pct:.1f}% of results)")
    stats_text = "\n".join(stats_lines)

    # Compose a prompt for summarisation
    prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an analyst assisting a product team with insights\n"
                "from mobile app reviews.  Summarise the main themes and\n"
                "concerns from the provided review snippets, highlighting\n"
                "what users like or dislike.  Keep the summary factual\n"
                "and refrain from inventing data beyond what is given.\n"
                "Use only the information contained in the review excerpts\n"
                "and stats; if a theme is not supported by that evidence,\n"
                "do not mention it. Summaries should cite concrete details\n"
                "from the provided evidence. If information is missing,\n"
                "acknowledge it briefly but do not ask for additional input.",
            ),
            (
                "human",
                "Review excerpts (the only evidence you may use):\n{examples}\n\n"
                "Stats (counts already computed):\n{stats}\n\n"
                "Key terms with frequencies (derived from all matching reviews):\n{keywords}\n\n"
                "Write a concise summary (2-3 bullets) that references the\n"
                "evidence directly.",
            ),
        ]
    )
    # Select up to three example reviews to show as evidence
    keyword_counts: List[str] = []
    if not reviews_df.empty:
        stopwords = {
            "the",
            "and",
            "for",
            "with",
            "that",
            "this",
            "from",
            "your",
            "have",
            "about",
            "been",
            "very",
            "into",
            "their",
            "will",
            "they",
            "these",
            "those",
            "into",
            "just",
            "like",
            "what",
            "when",
            "were",
            "where",
            "which",
            "while",
            "there",
            "because",
            "ever",
            "even",
            "over",
            "more",
            "much",
            "such",
            "it's",
            "its",
            "are",
            "was",
            "but",
            "you",
            "can",
            "all",
            "not",
            "app",
            "apps",
            "liquide",
        }
        token_counter: Counter[str] = Counter()
        text_series = reviews_df["text"].dropna()
        for text in text_series:
            for token in re.findall(r"[a-zA-Z0-9']+", str(text).lower()):
                if len(token) < 3 or token in stopwords:
                    continue
                token_counter[token] += 1
        keyword_counts = [f"{word} ({count})" for word, count in token_counter.most_common(8)]
        chosen_indices: List[int] = []
        for keyword, _count in token_counter.most_common(6):
            mask = text_series.str.contains(keyword, case=False, na=False)
            matches = text_series[mask]
            for idx in matches.index:
                if idx not in chosen_indices:
                    chosen_indices.append(idx)
                    break
            if len(chosen_indices) >= 3:
                break
        if len(chosen_indices) < 3:
            extra = [idx for idx in reviews_df.index[:3] if idx not in chosen_indices]
            chosen_indices.extend(extra)
        examples = [
            f"- ({reviews_df.loc[idx, 'id']}) {reviews_df.loc[idx, 'text']}"
            for idx in chosen_indices[:3]
        ]
    else:
        examples = []

    examples_text = "\n".join(examples)

    keywords_text = ", ".join(keyword_counts) if keyword_counts else "No dominant terms detected"

    summary_prompt = prompt_template.format_messages(
        examples=examples_text, stats=stats_text, keywords=keywords_text
    )
    summary_response = llm.invoke(summary_prompt)

    # Build final answer markdown
    bullets = summary_response.content.strip()
    applied_filters_lines = []
    for key, value in filters.items():
        if value:
            applied_filters_lines.append(f"{key}: {value}")
    applied_filters = ", ".join(applied_filters_lines) if applied_filters_lines else "None"

    evidence_lines = examples
    notes_text = "; ".join(notes) if notes else "None"

    answer_parts = [
        "### Summary",
        bullets,
        "\n### Stats",
        stats_text,
        "\n### Key Terms",
        keywords_text,
        "\n### Evidence",
        "\n".join(evidence_lines) if evidence_lines else "No examples available.",
        "\n### Applied Filters",
        applied_filters,
        "\n### SQL Query",
        f"```sql\n{sql_query}\n```" if sql_query else "Not generated",
        "\n### Notes",
        notes_text,
    ]

    state["answer"] = "\n".join(answer_parts)
    # Append the assistant response to history
    state.setdefault("history", []).append(AIMessage(content=state["answer"]))
    return state

def build_graph(db_path: str) -> Any:
    """Construct a LangGraph graph for the chat agent."""

    if MemorySaver is not None:
        # Use in-memory persistence if available.
        saver = MemorySaver()
    else:
        if SQLiteSaver is None:
            raise ImportError(
                "No suitable SQLite checkpointer found in langgraph.checkpoint."
            )
        saver = SQLiteSaver.from_conn_string("conversation_state.db")  
        
    from langgraph.graph import StateGraph, START  
    class _State(TypedDict, total=False):  
        question: str
        filters: dict
        sql: str
        reviews: list
        answer: str
        history: List[BaseMessage]
        notes: List[str]

    builder = StateGraph(_State)  
    builder.add_node("parse_filters", parse_filters)
    builder.add_node(
        "generate_sql_and_retrieve",
        lambda state: generate_sql_and_retrieve(state, db_path=db_path),
    )
    builder.add_node("summarise_answer", summarise_answer)

        # Define the start, entry, and finish points of the graph
    builder.set_entry_point("parse_filters")
    builder.set_finish_point("summarise_answer")

    # Add edges between nodes to form a linear pipeline
    builder.add_edge("parse_filters", "generate_sql_and_retrieve")
    builder.add_edge("generate_sql_and_retrieve", "summarise_answer")

    compiled = builder.compile(checkpointer=saver)
    return compiled


def answer_question(
    graph: Any,
    question: str,
    thread_id: str,
    store: Optional[ConversationStore] = None,
) -> str:
    """Run the graph for ``question`` using ``thread_id`` and return the answer."""

    history_messages: List[BaseMessage] = []
    if store is not None:
        thread_id = store.ensure_thread(thread_id)
        for record in store.get_messages(thread_id):
            role = record["role"]
            content = record["content"]
            if role == "assistant":
                history_messages.append(AIMessage(content=content))
            else:
                history_messages.append(HumanMessage(content=content))

    state: AgentState = {"question": question, "history": history_messages}
    config = {"configurable": {"thread_id": thread_id}}

    if hasattr(graph, "invoke"):
        final_state = graph.invoke(state, config=config)
    else:
        result_states = list(graph.stream(state, config=config))
        final_state = result_states[-1]

    answer = final_state.get("answer")
    if answer is None:
        notes_text = "; ".join(final_state.get("notes", [])) or "No diagnostic notes captured."
        answer = f"Could not generate an answer. Notes: {notes_text}"

    if store is not None:
        store.append_message(thread_id, "user", question)
        thread_meta = store.get_thread(thread_id)
        if thread_meta and not thread_meta["title"].strip():
            store.update_thread_title(thread_id, question.strip()[:80])
        store.append_message(thread_id, "assistant", answer)

    return answer


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the Liquide review agent")
    parser.add_argument("--db", default="reviews.db", help="Path to the SQLite database")
    parser.add_argument(
        "--conversation-db",
        help="Optional path to the conversation history SQLite database",
    )
    parser.add_argument(
        "--thread-id",
        help="Existing conversation thread identifier (if omitted a new one is created)",
    )
    parser.add_argument("question", help="User question to answer")
    args = parser.parse_args()

    store = ConversationStore(args.conversation_db) if args.conversation_db else None
    thread_id = args.thread_id or (store.create_thread() if store else str(uuid.uuid4()))

    graph = build_graph(args.db)
    answer = answer_question(graph, args.question, thread_id=thread_id, store=store)
    print(answer)
    if store is not None:
        print(f"[thread_id={thread_id}]")
