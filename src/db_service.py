import sqlite3
import json
import time
from pathlib import Path
from typing import Optional, Dict, Any, List
from contextlib import contextmanager

DB_PATH = Path.cwd() / "database" / "pricing.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)


@contextmanager
def get_db_connection():
    """Provides a thread-safe connection context with WAL mode and busy timeout."""
    conn = sqlite3.connect(str(DB_PATH), timeout=30.0)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    try:
        yield conn
    finally:
        conn.close()


def init_catalog_db():
    """Initializes the SQLite schema for background job tracking and inventory records."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
                       CREATE TABLE IF NOT EXISTS catalog_records
                       (
                           product_id
                           TEXT
                           PRIMARY
                           KEY,
                           status
                           TEXT
                           NOT
                           NULL,
                           payload
                           TEXT,
                           error_message
                           TEXT,
                           start_time
                           REAL,
                           execution_time_seconds
                           REAL,
                           created_at
                           TIMESTAMP
                           DEFAULT
                           CURRENT_TIMESTAMP,
                           updated_at
                           TIMESTAMP
                           DEFAULT
                           CURRENT_TIMESTAMP
                       )
                       """)
        conn.commit()


def save_initial_job(product_id: str):
    """Registers an incoming mobile digitization task in PROCESSING state."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO catalog_records (
                product_id,
                status,
                payload,
                error_message,
                start_time,
                execution_time_seconds,
                updated_at
            )
            VALUES (?, 'PROCESSING', NULL, NULL, ?, NULL, CURRENT_TIMESTAMP)
        """, (product_id, time.time()))
        conn.commit()


def complete_job(product_id: str, catalog_json: Dict[str, Any]):
    """Stores the final bilingual card JSON payload and records pipeline duration."""
    with get_db_connection() as conn:
        cursor = conn.cursor()

        # Calculate execution time
        cursor.execute("SELECT start_time FROM catalog_records WHERE product_id = ?", (product_id,))
        row = cursor.fetchone()
        duration = round(time.time() - row[0], 2) if row and row[0] else None

        cursor.execute("""
                       UPDATE catalog_records
                       SET status                 = 'COMPLETED',
                           payload                = ?,
                           execution_time_seconds = ?,
                           updated_at             = CURRENT_TIMESTAMP
                       WHERE product_id = ?
                       """, (json.dumps(catalog_json, ensure_ascii=False), duration, product_id))
        conn.commit()


def fail_job(product_id: str, error_msg: str):
    """Marks a task as FAILED with error trace for debugging."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
                       UPDATE catalog_records
                       SET status        = 'FAILED',
                           error_message = ?,
                           updated_at    = CURRENT_TIMESTAMP
                       WHERE product_id = ?
                       """, (str(error_msg), product_id))
        conn.commit()


def get_catalog_by_id(product_id: str) -> Optional[Dict[str, Any]]:
    """Poll endpoint query helper returning status, payload, or errors."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
                       SELECT status, payload, error_message, execution_time_seconds, created_at
                       FROM catalog_records
                       WHERE product_id = ?
                       """, (product_id,))
        row = cursor.fetchone()

    if not row:
        return None

    status, payload_raw, error_msg, exec_time, created_at = row

    payload = None
    if payload_raw:
        try:
            payload = json.loads(payload_raw)
        except Exception:
            payload = None

    return {
        "status": status,
        "product_id": product_id,
        "catalog": payload,
        "error": error_msg,
        "execution_time_seconds": exec_time,
        "created_at": created_at
    }


def get_all_completed_catalogs(limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
    """Returns inventory list of completed artisan cards for the mobile dashboard."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
                       SELECT product_id, payload, created_at
                       FROM catalog_records
                       WHERE status = 'COMPLETED'
                         AND payload IS NOT NULL
                       ORDER BY created_at DESC LIMIT ?
                       OFFSET ?
                       """, (limit, offset))
        rows = cursor.fetchall()

    records = []
    for product_id, payload_raw, created_at in rows:
        try:
            parsed = json.loads(payload_raw)
            records.append({
                "product_id": product_id,
                "created_at": created_at,
                "card_view": parsed.get("card_view")
            })
        except Exception:
            continue

    return records


# Self-initialize tables on import
init_catalog_db()