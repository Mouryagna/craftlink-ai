import sqlite3
import json
from pathlib import Path
from typing import Optional, Dict, Any, List

DB_PATH = Path.cwd() / "craftlink.db"


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS catalogs (
            product_id TEXT PRIMARY KEY,
            user_id TEXT DEFAULT 'default_artisan',
            status TEXT NOT NULL,
            vision_data TEXT,
            catalog TEXT,
            error TEXT,
            execution_time_seconds REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    # Automatically add user_id column if table was previously created without it
    cursor.execute("PRAGMA table_info(catalogs)")
    columns = [row["name"] for row in cursor.fetchall()]
    if "user_id" not in columns:
        cursor.execute("ALTER TABLE catalogs ADD COLUMN user_id TEXT DEFAULT 'default_artisan'")

    conn.commit()
    conn.close()

def save_initial_job(product_id: str, user_id: str = "default_artisan"):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO catalogs (product_id, user_id, status)
        VALUES (?, ?, 'PROCESSING')
        ON CONFLICT(product_id) DO UPDATE SET
            user_id = excluded.user_id,
            status = 'PROCESSING',
            updated_at = CURRENT_TIMESTAMP
    """, (product_id, user_id))
    conn.commit()
    conn.close()


def save_vision_step(product_id: str, vision_output: Dict[str, Any]):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO catalogs (product_id, status, vision_data)
        VALUES (?, 'VISION_COMPLETED', ?)
        ON CONFLICT(product_id) DO UPDATE SET
            status = 'VISION_COMPLETED',
            vision_data = excluded.vision_data,
            updated_at = CURRENT_TIMESTAMP
    """, (product_id, json.dumps(vision_output)))
    conn.commit()
    conn.close()



def get_vision_step(product_id: str) -> Optional[Dict[str, Any]]:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT vision_data FROM catalogs WHERE product_id = ?", (product_id,))
    row = cursor.fetchone()
    conn.close()
    return json.loads(row["vision_data"]) if row and row["vision_data"] else None


def complete_job(product_id: str, catalog: Dict[str, Any], execution_time: Optional[float] = None):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        UPDATE catalogs
        SET status = 'COMPLETED',
            catalog = ?,
            execution_time_seconds = ?,
            updated_at = CURRENT_TIMESTAMP
        WHERE product_id = ?
    """, (json.dumps(catalog), execution_time, product_id))
    conn.commit()
    conn.close()


def fail_job(product_id: str, error_message: str):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        UPDATE catalogs
        SET status = 'FAILED',
            error = ?,
            updated_at = CURRENT_TIMESTAMP
        WHERE product_id = ?
    """, (error_message, product_id))
    conn.commit()
    conn.close()


def get_catalog_by_id(product_id: str) -> Optional[Dict[str, Any]]:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM catalogs WHERE product_id = ?", (product_id,))
    row = cursor.fetchone()
    conn.close()
    if not row:
        return None

    data = dict(row)
    if data.get("catalog"):
        data["catalog"] = json.loads(data["catalog"])
    if data.get("vision_data"):
        data["vision_data"] = json.loads(data["vision_data"])
    return data


def get_catalogs_by_user(user_id: Optional[str] = None, limit: int = 20, offset: int = 0) -> List[Dict[str, Any]]:
    conn = get_connection()
    cursor = conn.cursor()

    if user_id:
        cursor.execute("""
            SELECT * FROM catalogs 
            WHERE user_id = ? AND status = 'COMPLETED'
            ORDER BY created_at DESC 
            LIMIT ? OFFSET ?
        """, (user_id, limit, offset))
    else:
        cursor.execute("""
            SELECT *FROM catalogs
            WHERE status = 'COMPLETED'
            ORDER BY created_at DESC LIMIT ?OFFSET ?
        """, (limit, offset))

    rows = cursor.fetchall()
    conn.close()

    results = []
    for r in rows:
        item = dict(r)
        if item.get("catalog"):
            item["catalog"] = json.loads(item["catalog"])
        if item.get("vision_data"):
            item["vision_data"] = json.loads(item["vision_data"])
        results.append(item)
    return results


def get_all_completed_catalogs(limit: int = 20, offset: int = 0) -> List[Dict[str, Any]]:
    return get_catalogs_by_user(user_id=None, limit=limit, offset=offset)