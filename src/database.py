from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Iterable, Optional


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATABASE_DIR = PROJECT_ROOT / "database"
DATABASE_DIR.mkdir(parents=True, exist_ok=True)

DATABASE_PATH = DATABASE_DIR / "pricing.db"


# ============================================================
# DATABASE CONNECTION
# ============================================================

def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DATABASE_PATH))
    conn.row_factory = sqlite3.Row
    return conn


# ============================================================
# TABLE HELPERS
# ============================================================

def table_exists(table_name: str) -> bool:
    conn = get_connection()

    try:
        row = conn.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
              AND name = ?
            """,
            (table_name,),
        ).fetchone()

        return row is not None

    finally:
        conn.close()


def get_table_columns(table_name: str) -> set[str]:
    if not table_exists(table_name):
        return set()

    conn = get_connection()

    try:
        rows = conn.execute(
            f"PRAGMA table_info({table_name})"
        ).fetchall()

        return {
            row["name"]
            for row in rows
        }

    finally:
        conn.close()


def add_missing_columns(
    table_name: str,
    columns: dict[str, str],
) -> None:

    existing_columns = get_table_columns(
        table_name
    )

    if not existing_columns:
        return

    conn = get_connection()

    try:

        for column_name, definition in columns.items():

            if column_name not in existing_columns:

                conn.execute(
                    f"""
                    ALTER TABLE {table_name}
                    ADD COLUMN {column_name} {definition}
                    """
                )

        conn.commit()

    finally:
        conn.close()


# ============================================================
# DATABASE INITIALIZATION
# ============================================================

def initialize_database() -> None:

    conn = get_connection()

    try:

        # ====================================================
        # PRODUCTS
        # ====================================================

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS products (

                product_id TEXT PRIMARY KEY,

                product_name TEXT NOT NULL,

                product_type TEXT,

                material TEXT,

                additional_materials TEXT,

                color TEXT,

                size TEXT,

                dimensions_length REAL,

                dimensions_width REAL,

                dimensions_height REAL,

                dimensions TEXT,

                features TEXT,

                craft_features TEXT,

                description TEXT,

                material_cost REAL DEFAULT 0,

                labour_cost REAL DEFAULT 0,

                production_cost REAL DEFAULT 0,

                time_worked_hours REAL DEFAULT 0,

                created_at TEXT,

                updated_at TEXT
            )
            """
        )

        # ====================================================
        # MARKETPLACE SNAPSHOTS
        # ====================================================

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS marketplace_snapshots (

                snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,

                product_id TEXT,

                marketplace TEXT,

                listing_id TEXT,

                title TEXT,

                url TEXT,

                price REAL,

                currency TEXT DEFAULT 'INR',

                rating REAL,

                review_count INTEGER,

                seller_name TEXT,

                handmade_evidence TEXT,

                product_type TEXT,

                material TEXT,

                color TEXT,

                size TEXT,

                dimensions TEXT,

                features TEXT,

                similarity_score REAL,

                snapshot_date TEXT
            )
            """
        )

        # ====================================================
        # PRICING EVENTS
        # ====================================================

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS pricing_events (

                pricing_event_id INTEGER PRIMARY KEY AUTOINCREMENT,

                product_id TEXT,

                event_date TEXT,

                previous_price REAL,

                recommended_price REAL,

                approved_price REAL,

                actual_selling_price REAL,

                base_cost REAL,

                market_minimum REAL,

                market_median REAL,

                market_maximum REAL,

                market_count INTEGER,

                formula_price REAL,

                market_based_price REAL,

                fair_trade_floor REAL,

                time_factor REAL,

                complexity_factor REAL,

                seasonal_index REAL,

                similarity_score REAL,

                pricing_method TEXT,

                views INTEGER DEFAULT 0,

                clicks INTEGER DEFAULT 0,

                wishlists INTEGER DEFAULT 0,

                add_to_cart INTEGER DEFAULT 0,

                enquiries INTEGER DEFAULT 0,

                orders INTEGER DEFAULT 0,

                inventory INTEGER DEFAULT 0,

                conversion_rate REAL,

                created_at TEXT
            )
            """
        )

        conn.commit()

    finally:
        conn.close()

    # ========================================================
    # MIGRATE EXISTING PRODUCTS TABLE
    # ========================================================

    add_missing_columns(
        "products",
        {
            "product_type": "TEXT",
            "material": "TEXT",
            "additional_materials": "TEXT",
            "color": "TEXT",
            "size": "TEXT",

            "dimensions_length": "REAL",
            "dimensions_width": "REAL",
            "dimensions_height": "REAL",

            "dimensions": "TEXT",

            "features": "TEXT",
            "craft_features": "TEXT",

            "description": "TEXT",

            "material_cost": "REAL DEFAULT 0",
            "labour_cost": "REAL DEFAULT 0",
            "production_cost": "REAL DEFAULT 0",

            "time_worked_hours": "REAL DEFAULT 0",

            "created_at": "TEXT",
            "updated_at": "TEXT",
        },
    )

    # ========================================================
    # MIGRATE MARKETPLACE TABLE
    # ========================================================

    add_missing_columns(
        "marketplace_snapshots",
        {
            "product_id": "TEXT",
            "marketplace": "TEXT",
            "listing_id": "TEXT",
            "title": "TEXT",
            "url": "TEXT",
            "price": "REAL",
            "currency": "TEXT DEFAULT 'INR'",

            "rating": "REAL",
            "review_count": "INTEGER",

            "seller_name": "TEXT",
            "handmade_evidence": "TEXT",

            "product_type": "TEXT",
            "material": "TEXT",
            "color": "TEXT",
            "size": "TEXT",

            "dimensions": "TEXT",
            "features": "TEXT",

            "similarity_score": "REAL",

            "snapshot_date": "TEXT",
        },
    )

    # ========================================================
    # MIGRATE PRICING EVENTS TABLE
    # ========================================================

    add_missing_columns(
        "pricing_events",
        {
            "product_id": "TEXT",
            "event_date": "TEXT",

            "previous_price": "REAL",
            "recommended_price": "REAL",
            "approved_price": "REAL",
            "actual_selling_price": "REAL",

            "base_cost": "REAL",

            "market_minimum": "REAL",
            "market_median": "REAL",
            "market_maximum": "REAL",
            "market_count": "INTEGER",

            "formula_price": "REAL",
            "market_based_price": "REAL",

            "fair_trade_floor": "REAL",

            "time_factor": "REAL",
            "complexity_factor": "REAL",
            "seasonal_index": "REAL",

            "similarity_score": "REAL",

            "pricing_method": "TEXT",

            "views": "INTEGER DEFAULT 0",
            "clicks": "INTEGER DEFAULT 0",
            "wishlists": "INTEGER DEFAULT 0",
            "add_to_cart": "INTEGER DEFAULT 0",
            "enquiries": "INTEGER DEFAULT 0",
            "orders": "INTEGER DEFAULT 0",

            "inventory": "INTEGER DEFAULT 0",

            "conversion_rate": "REAL",

            "created_at": "TEXT",
        },
    )

    # ========================================================
    # INDEXES
    # ========================================================

    conn = get_connection()

    try:

        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS
            idx_marketplace_product
            ON marketplace_snapshots(product_id)
            """
        )

        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS
            idx_marketplace_date
            ON marketplace_snapshots(snapshot_date)
            """
        )

        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS
            idx_pricing_product
            ON pricing_events(product_id)
            """
        )

        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS
            idx_pricing_date
            ON pricing_events(event_date)
            """
        )

        conn.commit()

    finally:
        conn.close()


# ============================================================
# GENERIC SQL FUNCTIONS
# ============================================================

def execute(
    query: str,
    parameters: Iterable[Any] = (),
) -> None:

    conn = get_connection()

    try:

        conn.execute(
            query,
            tuple(parameters),
        )

        conn.commit()

    finally:
        conn.close()


def fetch_one(
    query: str,
    parameters: Iterable[Any] = (),
) -> Optional[dict]:

    conn = get_connection()

    try:

        row = conn.execute(
            query,
            tuple(parameters),
        ).fetchone()

        if row is None:
            return None

        return dict(row)

    finally:
        conn.close()


def fetch_all(
    query: str,
    parameters: Iterable[Any] = (),
) -> list[dict]:

    conn = get_connection()

    try:

        rows = conn.execute(
            query,
            tuple(parameters),
        ).fetchall()

        return [
            dict(row)
            for row in rows
        ]

    finally:
        conn.close()


# ============================================================
# SAFE DYNAMIC INSERT
# ============================================================

def _insert_dynamic(
    table_name: str,
    data: dict,
) -> None:
    """
    Inserts only columns that actually exist in the database.

    This makes the code compatible with older versions of
    pricing.db and eliminates manual placeholder counting.
    """

    existing_columns = get_table_columns(
        table_name
    )

    if not existing_columns:
        raise ValueError(
            f"Table '{table_name}' does not exist."
        )

    filtered_data = {
        key: value
        for key, value in data.items()
        if key in existing_columns
    }

    if not filtered_data:
        raise ValueError(
            f"No valid columns found for table '{table_name}'."
        )

    columns = list(
        filtered_data.keys()
    )

    values = [
        filtered_data[column]
        for column in columns
    ]

    column_sql = ", ".join(
        f'"{column}"'
        for column in columns
    )

    placeholders = ", ".join(
        "?"
        for _ in values
    )

    query = f"""
        INSERT INTO "{table_name}"
        ({column_sql})
        VALUES ({placeholders})
    """

    execute(
        query,
        values,
    )


# ============================================================
# PRODUCTS
# ============================================================

def save_product(
    product: dict,
) -> None:

    product_data = {
        "product_id": product["product_id"],

        "product_name": product["product_name"],

        "product_type": product.get(
            "product_type"
        ),

        "material": product.get(
            "material"
        ),

        "additional_materials": product.get(
            "additional_materials"
        ),

        "color": product.get(
            "color"
        ),

        "size": product.get(
            "size"
        ),

        "dimensions_length": product.get(
            "dimensions_length"
        ),

        "dimensions_width": product.get(
            "dimensions_width"
        ),

        "dimensions_height": product.get(
            "dimensions_height"
        ),

        # Legacy database compatibility.
        "dimensions": product.get(
            "dimensions"
        ),

        "features": product.get(
            "features"
        ),

        # Legacy database compatibility.
        "craft_features": product.get(
            "craft_features"
        ),

        "description": product.get(
            "description"
        ),

        "material_cost": product.get(
            "material_cost",
            0,
        ),

        "labour_cost": product.get(
            "labour_cost",
            0,
        ),

        "production_cost": product.get(
            "production_cost",
            0,
        ),

        "time_worked_hours": product.get(
            "time_worked_hours",
            0,
        ),

        "created_at": product.get(
            "created_at"
        ),

        "updated_at": product.get(
            "updated_at"
        ),
    }

    existing = get_product(
        product["product_id"]
    )

    if existing is not None:

        # Update existing product.
        existing_columns = get_table_columns(
            "products"
        )

        update_data = {
            key: value
            for key, value in product_data.items()
            if key in existing_columns
            and key != "product_id"
        }

        assignments = ", ".join(
            f'"{key}" = ?'
            for key in update_data
        )

        values = list(
            update_data.values()
        )

        values.append(
            product["product_id"]
        )

        query = f"""
            UPDATE products
            SET {assignments}
            WHERE product_id = ?
        """

        execute(
            query,
            values,
        )

    else:

        _insert_dynamic(
            "products",
            product_data,
        )


def get_product(
    product_id: str,
) -> Optional[dict]:

    return fetch_one(
        """
        SELECT *
        FROM products
        WHERE product_id = ?
        """,
        (product_id,),
    )


def get_all_products() -> list[dict]:

    return fetch_all(
        """
        SELECT *
        FROM products
        ORDER BY created_at
        """
    )


# ============================================================
# MARKETPLACE SNAPSHOTS
# ============================================================

def save_marketplace_snapshot(
    snapshot: dict,
) -> None:

    snapshot_data = {
        "product_id": snapshot.get(
            "product_id"
        ),

        "marketplace": snapshot.get(
            "marketplace"
        ),

        "listing_id": snapshot.get(
            "listing_id"
        ),

        "title": snapshot.get(
            "title"
        ),

        "url": snapshot.get(
            "url"
        ),

        "price": snapshot.get(
            "price"
        ),

        "currency": snapshot.get(
            "currency",
            "INR",
        ),

        "rating": snapshot.get(
            "rating"
        ),

        "review_count": snapshot.get(
            "review_count"
        ),

        "seller_name": snapshot.get(
            "seller_name"
        ),

        "handmade_evidence": snapshot.get(
            "handmade_evidence"
        ),

        "product_type": snapshot.get(
            "product_type"
        ),

        "material": snapshot.get(
            "material"
        ),

        "color": snapshot.get(
            "color"
        ),

        "size": snapshot.get(
            "size"
        ),

        "dimensions": snapshot.get(
            "dimensions"
        ),

        "features": snapshot.get(
            "features"
        ),

        "similarity_score": snapshot.get(
            "similarity_score"
        ),

        "snapshot_date": snapshot.get(
            "snapshot_date"
        ),
    }

    _insert_dynamic(
        "marketplace_snapshots",
        snapshot_data,
    )


def get_marketplace_snapshots(
    product_id: str,
) -> list[dict]:

    return fetch_all(
        """
        SELECT *
        FROM marketplace_snapshots
        WHERE product_id = ?
        ORDER BY snapshot_date DESC,
                 snapshot_id DESC
        """,
        (product_id,),
    )


def get_latest_marketplace_snapshots(
    product_id: str,
) -> list[dict]:

    return fetch_all(
        """
        SELECT *
        FROM marketplace_snapshots
        WHERE product_id = ?

        AND snapshot_date = (
            SELECT MAX(snapshot_date)
            FROM marketplace_snapshots
            WHERE product_id = ?
        )

        ORDER BY price ASC
        """,
        (
            product_id,
            product_id,
        ),
    )


# ============================================================
# PRICING EVENTS
# ============================================================

def save_pricing_event(
    event: dict,
) -> None:

    event_data = {
        "product_id": event.get(
            "product_id"
        ),

        "event_date": event.get(
            "event_date"
        ),

        "previous_price": event.get(
            "previous_price"
        ),

        "recommended_price": event.get(
            "recommended_price"
        ),

        "approved_price": event.get(
            "approved_price"
        ),

        "actual_selling_price": event.get(
            "actual_selling_price"
        ),

        "base_cost": event.get(
            "base_cost"
        ),

        "market_minimum": event.get(
            "market_minimum"
        ),

        "market_median": event.get(
            "market_median"
        ),

        "market_maximum": event.get(
            "market_maximum"
        ),

        "market_count": event.get(
            "market_count"
        ),

        "formula_price": event.get(
            "formula_price"
        ),

        "market_based_price": event.get(
            "market_based_price"
        ),

        "fair_trade_floor": event.get(
            "fair_trade_floor"
        ),

        "time_factor": event.get(
            "time_factor"
        ),

        "complexity_factor": event.get(
            "complexity_factor"
        ),

        "seasonal_index": event.get(
            "seasonal_index"
        ),

        "similarity_score": event.get(
            "similarity_score"
        ),

        "pricing_method": event.get(
            "pricing_method"
        ),

        "views": event.get(
            "views",
            0,
        ),

        "clicks": event.get(
            "clicks",
            0,
        ),

        "wishlists": event.get(
            "wishlists",
            0,
        ),

        "add_to_cart": event.get(
            "add_to_cart",
            0,
        ),

        "enquiries": event.get(
            "enquiries",
            0,
        ),

        "orders": event.get(
            "orders",
            0,
        ),

        "inventory": event.get(
            "inventory",
            0,
        ),

        "conversion_rate": event.get(
            "conversion_rate"
        ),

        "created_at": event.get(
            "created_at"
        ),
    }

    _insert_dynamic(
        "pricing_events",
        event_data,
    )


def get_pricing_events(
    product_id: str,
) -> list[dict]:

    return fetch_all(
        """
        SELECT *
        FROM pricing_events
        WHERE product_id = ?
        ORDER BY event_date ASC,
                 pricing_event_id ASC
        """,
        (product_id,),
    )


def get_all_pricing_events() -> list[dict]:

    return fetch_all(
        """
        SELECT *
        FROM pricing_events
        ORDER BY event_date ASC,
                 pricing_event_id ASC
        """
    )


def get_latest_pricing_event(
    product_id: str,
) -> Optional[dict]:

    return fetch_one(
        """
        SELECT *
        FROM pricing_events
        WHERE product_id = ?

        ORDER BY
            event_date DESC,
            pricing_event_id DESC

        LIMIT 1
        """,
        (product_id,),
    )


# ============================================================
# TRAINING DATA
# ============================================================

def get_training_data() -> list[dict]:
    """
    Only pricing events with actual selling outcomes
    are returned as supervised ML training data.
    """

    return fetch_all(
        """
        SELECT *
        FROM pricing_events

        WHERE actual_selling_price IS NOT NULL

        ORDER BY
            event_date ASC,
            pricing_event_id ASC
        """
    )


# ============================================================
# DATABASE VALIDATION
# ============================================================

def validate_database() -> dict:

    initialize_database()

    return {
        "database_path": str(
            DATABASE_PATH
        ),

        "database_exists": DATABASE_PATH.exists(),

        "products_table": table_exists(
            "products"
        ),

        "marketplace_snapshots_table": table_exists(
            "marketplace_snapshots"
        ),

        "pricing_events_table": table_exists(
            "pricing_events"
        ),

        "products_columns": sorted(
            get_table_columns(
                "products"
            )
        ),

        "marketplace_columns": sorted(
            get_table_columns(
                "marketplace_snapshots"
            )
        ),

        "pricing_event_columns": sorted(
            get_table_columns(
                "pricing_events"
            )
        ),
    }


# ============================================================
# INITIALIZE DATABASE WHEN MODULE LOADS
# ============================================================

initialize_database()