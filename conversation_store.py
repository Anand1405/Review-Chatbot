# conversation_store.py
"""
Utility helpers for persisting chat conversations in a local SQLite database.

The database contains two tables:

    threads(id TEXT PRIMARY KEY, title TEXT, created_at TEXT, updated_at TEXT)
    messages(id INTEGER PRIMARY KEY AUTOINCREMENT,
             thread_id TEXT,
             role TEXT,
             content TEXT,
             created_at TEXT,
             FOREIGN KEY(thread_id) REFERENCES threads(id))

All timestamps are stored as ISO formatted UTC strings.  The public API is a
lightweight wrapper around basic CRUD operations used by the Streamlit UI and
the agent runtime.
"""

from __future__ import annotations

import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional


def _utcnow() -> str:
    """Return the current UTC timestamp as an ISO8601 string."""
    return datetime.now(tz=timezone.utc).isoformat()


class ConversationStore:
    """Persist conversation threads and messages in SQLite."""

    def __init__(self, db_path: str | Path = "conversations.db") -> None:
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def _connection(self) -> Iterable[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
        finally:
            conn.commit()
            conn.close()

    def _init_db(self) -> None:
        with self._connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS threads (
                    id TEXT PRIMARY KEY,
                    title TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    thread_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(thread_id) REFERENCES threads(id)
                );
                """
            )

    def list_threads(self) -> List[Dict[str, str]]:
        """Return all threads ordered by most recent activity."""
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT id, COALESCE(title, '') AS title, created_at, updated_at
                FROM threads
                ORDER BY datetime(updated_at) DESC, datetime(created_at) DESC;
                """
            ).fetchall()
        return [
            {
                "id": row[0],
                "title": row[1] or "",
                "created_at": row[2],
                "updated_at": row[3],
            }
            for row in rows
        ]

    def get_thread(self, thread_id: str) -> Optional[Dict[str, str]]:
        with self._connection() as conn:
            row = conn.execute(
                """
                SELECT id, COALESCE(title, ''), created_at, updated_at
                FROM threads
                WHERE id = ?;
                """,
                (thread_id,),
            ).fetchone()
        if row is None:
            return None
        return {
            "id": row[0],
            "title": row[1] or "",
            "created_at": row[2],
            "updated_at": row[3],
        }

    def create_thread(self, title: str = "") -> str:
        """Create a new thread and return its identifier."""
        thread_id = str(uuid.uuid4())
        timestamp = _utcnow()
        with self._connection() as conn:
            conn.execute(
                """
                INSERT INTO threads (id, title, created_at, updated_at)
                VALUES (?, ?, ?, ?);
                """,
                (thread_id, title, timestamp, timestamp),
            )
        return thread_id

    def ensure_thread(self, thread_id: str) -> str:
        """Return ``thread_id`` if it exists, otherwise create a fresh thread."""
        if self.get_thread(thread_id):
            return thread_id
        return self.create_thread()

    def get_messages(self, thread_id: str) -> List[Dict[str, str]]:
        """Return all messages for ``thread_id`` ordered chronologically."""
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT id, role, content, created_at
                FROM messages
                WHERE thread_id = ?
                ORDER BY datetime(created_at) ASC, id ASC;
                """,
                (thread_id,),
            ).fetchall()
        return [
            {"id": row[0], "role": row[1], "content": row[2], "created_at": row[3]}
            for row in rows
        ]

    def append_message(self, thread_id: str, role: str, content: str) -> None:
        """Append a message to the thread and update its timestamp."""
        timestamp = _utcnow()
        with self._connection() as conn:
            conn.execute(
                """
                INSERT INTO messages (thread_id, role, content, created_at)
                VALUES (?, ?, ?, ?);
                """,
                (thread_id, role, content, timestamp),
            )
            conn.execute(
                """
                UPDATE threads
                SET updated_at = ?
                WHERE id = ?;
                """,
                (timestamp, thread_id),
            )

    def update_thread_title(self, thread_id: str, title: str) -> None:
        with self._connection() as conn:
            conn.execute(
                "UPDATE threads SET title = ?, updated_at = ? WHERE id = ?;",
                (title, _utcnow(), thread_id),
            )

    def delete_thread(self, thread_id: str) -> None:
        with self._connection() as conn:
            conn.execute("DELETE FROM messages WHERE thread_id = ?;", (thread_id,))
            conn.execute("DELETE FROM threads WHERE id = ?;", (thread_id,))

