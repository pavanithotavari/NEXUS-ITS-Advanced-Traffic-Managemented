"""Hybrid resilient storage: MongoDB Atlas with automated SQLite local edge fallback.

Operates autonomously on the edge when disconnected, and synchronizes/connects
to MongoDB Atlas whenever configured.
"""
import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv

load_dotenv()

_client = None
_client_uri = None
_sqlite_instance = None


class SQLiteCursor:
    def __init__(self, rows: List[Dict[str, Any]]):
        self._rows = rows

    def sort(self, field: str, direction: int = -1):
        reverse = direction < 0
        def _get_key(item):
            val = item.get(field)
            if val is None:
                return ""
            if isinstance(val, datetime):
                return val.isoformat()
            return val
        self._rows.sort(key=_get_key, reverse=reverse)
        return self

    def limit(self, count: int):
        self._rows = self._rows[:count]
        return self

    def __iter__(self):
        return iter(self._rows)

    def __len__(self):
        return len(self._rows)


class SQLiteClientAdmin:
    def command(self, cmd: str):
        if cmd == "ping":
            return {"ok": 1.0}
        return {}


class SQLiteClient:
    def __init__(self):
        self.admin = SQLiteClientAdmin()


class SQLiteDatabase:
    def __init__(self):
        self.client = SQLiteClient()


class SQLiteCollection:
    """Drop-in collection interface backed by local SQLite for edge autonomy."""

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            data_dir = Path("data")
            data_dir.mkdir(parents=True, exist_ok=True)
            db_path = str(data_dir / "edge_traffic.db")
        self.db_path = db_path
        self.database = SQLiteDatabase()
        self._init_db()

    def _get_conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._get_conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS telemetry (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    intersection_id TEXT NOT NULL,
                    t INTEGER NOT NULL,
                    phase TEXT NOT NULL,
                    queues TEXT NOT NULL,
                    action INTEGER NOT NULL,
                    recorded_at TEXT NOT NULL,
                    raw_data TEXT
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_intersection_time ON telemetry (intersection_id, recorded_at DESC)")
            conn.commit()

    def insert_one(self, document: Dict[str, Any]):
        doc = dict(document)
        recorded_at = doc.get("recorded_at")
        if isinstance(recorded_at, datetime):
            rec_str = recorded_at.isoformat()
        else:
            rec_str = str(recorded_at or datetime.now(timezone.utc).isoformat())

        queues_str = json.dumps(doc.get("queues", {}))
        intersection_id = str(doc.get("intersection_id", "junction01"))
        t_val = int(doc.get("t", 0))
        phase = str(doc.get("phase", "NS_GREEN"))
        action = int(doc.get("action", 0))

        with self._get_conn() as conn:
            cursor = conn.execute(
                """
                INSERT INTO telemetry (intersection_id, t, phase, queues, action, recorded_at, raw_data)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (intersection_id, t_val, phase, queues_str, action, rec_str, json.dumps(doc, default=str))
            )
            conn.commit()
            row_id = cursor.lastrowid

        doc["_id"] = str(row_id)
        class InsertResult:
            def __init__(self, inserted_id):
                self.inserted_id = inserted_id
        return InsertResult(str(row_id))

    def find(self, query: Optional[Dict[str, Any]] = None):
        query = query or {}
        intersection_id = query.get("intersection_id")

        sql = "SELECT id, intersection_id, t, phase, queues, action, recorded_at, raw_data FROM telemetry"
        params = []
        if intersection_id:
            sql += " WHERE intersection_id = ?"
            params.append(intersection_id)
        sql += " ORDER BY id DESC"

        results = []
        with self._get_conn() as conn:
            cursor = conn.execute(sql, params)
            for row in cursor.fetchall():
                try:
                    queues = json.loads(row["queues"])
                except Exception:
                    queues = {"N": 0, "S": 0, "E": 0, "W": 0}

                rec_at_str = row["recorded_at"]
                try:
                    rec_at = datetime.fromisoformat(rec_at_str)
                    if rec_at.tzinfo is None:
                        rec_at = rec_at.replace(tzinfo=timezone.utc)
                except Exception:
                    rec_at = datetime.now(timezone.utc)

                item = {
                    "_id": str(row["id"]),
                    "intersection_id": row["intersection_id"],
                    "t": row["t"],
                    "phase": row["phase"],
                    "queues": queues,
                    "action": row["action"],
                    "recorded_at": rec_at
                }
                results.append(item)

        return SQLiteCursor(results)

    def find_one(self, query: Optional[Dict[str, Any]] = None, sort=None):
        cursor = self.find(query)
        if sort:
            field, direction = sort[0]
            cursor.sort(field, direction)
        for item in cursor:
            return item
        return None


def get_sqlite_collection():
    global _sqlite_instance
    if _sqlite_instance is None:
        _sqlite_instance = SQLiteCollection()
    return _sqlite_instance


def get_storage_mode() -> str:
    uri = os.getenv("MONGODB_URI")
    if uri and ("mongodb://" in uri or "mongodb+srv://" in uri):
        return "mongodb"
    return "sqlite"


def get_telemetry_collection():
    """Return MongoDB Atlas collection if configured, else fallback to SQLite edge collection."""
    global _client, _client_uri
    uri = os.getenv("MONGODB_URI")
    if not uri or uri.startswith("mongodb+srv://<") or "placeholder" in uri.lower():
        # Edge-resilient default: use SQLite local storage without throwing error
        return get_sqlite_collection()

    try:
        if _client is None or uri != _client_uri:
            if _client is not None:
                _client.close()
            from pymongo import MongoClient
            _client = MongoClient(uri, serverSelectionTimeoutMS=2000)
            _client_uri = uri
            # Test connectivity
            _client.admin.command("ping")

        database = os.getenv("MONGODB_DATABASE", "traffic_management")
        collection = os.getenv("MONGODB_COLLECTION", "telemetry")
        return _client[database][collection]
    except Exception:
        # Graceful fallback to local edge storage if cloud is unreachable
        return get_sqlite_collection()
