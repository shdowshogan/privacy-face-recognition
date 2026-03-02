import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple

import numpy as np


class PrivacyStore:
    def __init__(self, db_path: str = "face_pipeline/privacy.db"):
        self.db_path = str(Path(db_path))
        self._init_schema()

    def _connect(self):
        return sqlite3.connect(self.db_path)

    def _utc_now(self):
        return datetime.now(timezone.utc).isoformat()

    def _init_schema(self):
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    consent INTEGER NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS embeddings (
                    user_id TEXT NOT NULL,
                    vector BLOB NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(user_id) REFERENCES users(id)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS audit_logs (
                    action TEXT NOT NULL,
                    timestamp TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def _log(self, conn, action: str):
        conn.execute(
            "INSERT INTO audit_logs(action, timestamp) VALUES(?, ?)",
            (action, self._utc_now()),
        )

    def enroll_embedding(self, user_id: str, embedding: np.ndarray, consent: bool):
        if not consent:
            raise ValueError("Explicit consent is required for enrollment.")

        emb = np.asarray(embedding, dtype=np.float32)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO users(id, consent, created_at)
                VALUES(?, 1, ?)
                ON CONFLICT(id) DO UPDATE SET consent=1
                """,
                (user_id, self._utc_now()),
            )
            conn.execute(
                "INSERT INTO embeddings(user_id, vector, created_at) VALUES(?, ?, ?)",
                (user_id, emb.tobytes(), self._utc_now()),
            )
            self._log(conn, f"enroll user_id={user_id}")
            conn.commit()

    def revoke_consent(self, user_id: str):
        with self._connect() as conn:
            conn.execute("DELETE FROM embeddings WHERE user_id=?", (user_id,))
            conn.execute("UPDATE users SET consent=0 WHERE id=?", (user_id,))
            self._log(conn, f"revoke user_id={user_id}")
            conn.commit()

    def delete_user(self, user_id: str):
        with self._connect() as conn:
            conn.execute("DELETE FROM embeddings WHERE user_id=?", (user_id,))
            conn.execute("DELETE FROM users WHERE id=?", (user_id,))
            self._log(conn, f"delete user_id={user_id}")
            conn.commit()

    def get_user(self, user_id: str):
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id, consent, created_at FROM users WHERE id=?",
                (user_id,),
            ).fetchone()
            if row is None:
                return None

            embedding_count = conn.execute(
                "SELECT COUNT(*) FROM embeddings WHERE user_id=?",
                (user_id,),
            ).fetchone()[0]

        return {
            "id": row[0],
            "consent": bool(row[1]),
            "created_at": row[2],
            "embedding_count": int(embedding_count),
        }

    def get_metrics(self):
        with self._connect() as conn:
            total_users = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
            consented_users = conn.execute(
                "SELECT COUNT(*) FROM users WHERE consent=1"
            ).fetchone()[0]
            total_embeddings = conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
            audit_log_count = conn.execute("SELECT COUNT(*) FROM audit_logs").fetchone()[0]

        return {
            "total_users": int(total_users),
            "consented_users": int(consented_users),
            "total_embeddings": int(total_embeddings),
            "audit_log_count": int(audit_log_count),
        }

    def list_active_embeddings(self) -> List[Tuple[str, np.ndarray]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT e.user_id, e.vector
                FROM embeddings e
                JOIN users u ON u.id = e.user_id
                WHERE u.consent = 1
                """
            ).fetchall()

        result = []
        for user_id, blob in rows:
            emb = np.frombuffer(blob, dtype=np.float32)
            result.append((user_id, emb))
        return result
