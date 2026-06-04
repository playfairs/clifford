from typing import Any, Dict, List, Optional
import json
from pathlib import Path
import sqlite3

from clifford.memory.types import MemoryItem
from clifford.utils import ensure_directory, get_db_path


class PersistentMemory:
    def __init__(self, db_path: Optional[Path] = None):
        if db_path is None:
            db_path = Path(__file__).parent.parent.parent.parent / "memory" / "persistent.db"
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_database()

    def _initialize_database(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS memory_items (
                    id TEXT PRIMARY KEY,
                    content TEXT,
                    metadata TEXT,
                    timestamp TEXT,
                    importance REAL,
                    access_count INTEGER,
                    last_accessed TEXT,
                    tags TEXT
                )
            """)
            conn.commit()

    def add(self, content: Any, metadata: Optional[Dict[str, Any]] = None, tags: Optional[List[str]] = None) -> MemoryItem:
        item = MemoryItem(content=content, metadata=metadata or {}, tags=tags or [])
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO memory_items (id, content, metadata, timestamp, importance, access_count, last_accessed, tags) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (item.id, str(item.content), json.dumps(item.metadata), item.timestamp, item.importance, item.access_count, item.last_accessed, json.dumps(item.tags))
            )
            conn.commit()
        
        return item

    def get(self, item_id: str) -> Optional[MemoryItem]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM memory_items WHERE id = ?", (item_id,))
            row = cursor.fetchone()
            
            if row:
                cursor.execute("UPDATE memory_items SET access_count = access_count + 1, last_accessed = ? WHERE id = ?", (item.last_accessed, item_id))
                conn.commit()
                
                return MemoryItem(
                    id=row[0],
                    content=row[1],
                    metadata=json.loads(row[2]),
                    timestamp=row[3],
                    importance=row[4],
                    access_count=row[5],
                    last_accessed=row[6],
                    tags=json.loads(row[7])
                )
        
        return None

    def search(self, query: str) -> List[MemoryItem]:
        query_lower = query.lower()
        results = []
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM memory_items")
            rows = cursor.fetchall()
            
            for row in rows:
                content_str = row[1].lower()
                metadata_str = row[2].lower()
                
                if query_lower in content_str or query_lower in metadata_str:
                    results.append(MemoryItem(
                        id=row[0],
                        content=row[1],
                        metadata=json.loads(row[2]),
                        timestamp=row[3],
                        importance=row[4],
                        access_count=row[5],
                        last_accessed=row[6],
                        tags=json.loads(row[7])
                    ))
        
        return sorted(results, key=lambda x: x.importance, reverse=True)

    def remove(self, item_id: str) -> None:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM memory_items WHERE id = ?", (item_id,))
            conn.commit()

    def clear(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM memory_items")
            conn.commit()

    def size(self) -> int:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM memory_items")
            return cursor.fetchone()[0]
