"""
SQLite persistence layer for simulation data.

Two tables as specified:
  - signal_quality_log: per-reading SQM results
  - simulation_run_metadata: per-run summary with parameters

Plus helper methods for querying, exporting, and bulk inserts.
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS signal_quality_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL NOT NULL,
    sensor_id TEXT NOT NULL,
    value REAL NOT NULL,
    quality TEXT NOT NULL DEFAULT 'ok',
    confidence REAL NOT NULL DEFAULT 1.0,
    domain TEXT NOT NULL,
    label TEXT DEFAULT 'unknown',
    run_id TEXT
);

CREATE TABLE IF NOT EXISTS simulation_run_metadata (
    run_id TEXT PRIMARY KEY,
    domain TEXT NOT NULL,
    scenario TEXT NOT NULL,
    start_time REAL NOT NULL,
    end_time REAL,
    total_readings INTEGER DEFAULT 0,
    fault_count INTEGER DEFAULT 0,
    params_json TEXT
);

CREATE INDEX IF NOT EXISTS idx_sql_sensor ON signal_quality_log(sensor_id);
CREATE INDEX IF NOT EXISTS idx_sql_domain ON signal_quality_log(domain);
CREATE INDEX IF NOT EXISTS idx_sql_label ON signal_quality_log(label);
CREATE INDEX IF NOT EXISTS idx_sql_run ON signal_quality_log(run_id);
CREATE INDEX IF NOT EXISTS idx_sql_quality ON signal_quality_log(quality);
"""


class SimulationDatabase:
    """
    SQLite database for storing simulation results and SQM outputs.

    Thread-safe for single-writer use (WAL mode).
    Supports bulk inserts for performance with large datasets.
    """

    def __init__(self, db_path: str = "simulation_data.db") -> None:
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("PRAGMA cache_size=-64000")  # 64MB cache
        self._create_tables()

    def _create_tables(self) -> None:
        """Create tables if they don't exist."""
        self.conn.executescript(_SCHEMA_SQL)
        self.conn.commit()

    # -----------------------------------------------------------------
    # Write methods
    # -----------------------------------------------------------------

    def log_reading(
        self,
        timestamp: float,
        sensor_id: str,
        value: float,
        quality: str = "ok",
        confidence: float = 1.0,
        domain: str = "",
        label: str = "unknown",
        run_id: Optional[str] = None,
    ) -> None:
        """Log a single sensor reading with SQM result."""
        self.conn.execute(
            """INSERT INTO signal_quality_log
               (timestamp, sensor_id, value, quality, confidence, domain, label, run_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (timestamp, sensor_id, value, quality, confidence, domain, label, run_id),
        )

    def log_readings_bulk(
        self,
        readings: List[Dict[str, Any]],
        run_id: Optional[str] = None,
    ) -> int:
        """
        Bulk insert readings for performance.
        Each dict should have: timestamp, sensor_id, value, quality, confidence, domain, label
        Returns the number of rows inserted.
        """
        rows = [
            (
                r["timestamp"], r["sensor_id"], r["value"],
                r.get("quality", "ok"), r.get("confidence", 1.0),
                r.get("domain", ""), r.get("label", "unknown"),
                run_id,
            )
            for r in readings
        ]
        self.conn.executemany(
            """INSERT INTO signal_quality_log
               (timestamp, sensor_id, value, quality, confidence, domain, label, run_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            rows,
        )
        self.conn.commit()
        return len(rows)

    def log_run(
        self,
        run_id: str,
        domain: str,
        scenario: str,
        start_time: float,
        end_time: Optional[float] = None,
        total_readings: int = 0,
        fault_count: int = 0,
        params: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log metadata for a simulation run."""
        self.conn.execute(
            """INSERT OR REPLACE INTO simulation_run_metadata
               (run_id, domain, scenario, start_time, end_time,
                total_readings, fault_count, params_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                run_id, domain, scenario, start_time, end_time,
                total_readings, fault_count,
                json.dumps(params) if params else None,
            ),
        )
        self.conn.commit()

    @staticmethod
    def generate_run_id() -> str:
        """Generate a unique run ID."""
        return str(uuid.uuid4())[:12]

    # -----------------------------------------------------------------
    # Query methods
    # -----------------------------------------------------------------

    def query_by_scenario(
        self,
        scenario: str,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Query readings by scenario label."""
        sql = """
            SELECT sl.timestamp, sl.sensor_id, sl.value, sl.quality,
                   sl.confidence, sl.domain, sl.label
            FROM signal_quality_log sl
            JOIN simulation_run_metadata rm ON sl.run_id = rm.run_id
            WHERE rm.scenario = ?
            ORDER BY sl.timestamp
        """
        if limit:
            sql += f" LIMIT {limit}"
        cursor = self.conn.execute(sql, (scenario,))
        columns = [d[0] for d in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def query_by_domain(self, domain: str) -> List[Dict[str, Any]]:
        """Query all readings for a domain."""
        cursor = self.conn.execute(
            """SELECT timestamp, sensor_id, value, quality, confidence, domain, label
               FROM signal_quality_log WHERE domain = ? ORDER BY timestamp""",
            (domain,),
        )
        columns = [d[0] for d in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def query_faults(self, domain: Optional[str] = None) -> List[Dict[str, Any]]:
        """Query all fault readings, optionally filtered by domain."""
        sql = """SELECT timestamp, sensor_id, value, quality, confidence, domain, label
                 FROM signal_quality_log WHERE quality != 'ok'"""
        params: list = []
        if domain:
            sql += " AND domain = ?"
            params.append(domain)
        sql += " ORDER BY timestamp"
        cursor = self.conn.execute(sql, params)
        columns = [d[0] for d in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def get_run_metadata(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a specific run."""
        cursor = self.conn.execute(
            "SELECT * FROM simulation_run_metadata WHERE run_id = ?", (run_id,)
        )
        row = cursor.fetchone()
        if row:
            columns = [d[0] for d in cursor.description]
            return dict(zip(columns, row))
        return None

    def get_all_runs(self) -> List[Dict[str, Any]]:
        """Get metadata for all runs."""
        cursor = self.conn.execute(
            "SELECT * FROM simulation_run_metadata ORDER BY start_time"
        )
        columns = [d[0] for d in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def get_statistics(self) -> Dict[str, Any]:
        """Get summary statistics of the database."""
        stats = {}
        cursor = self.conn.execute("SELECT COUNT(*) FROM signal_quality_log")
        stats["total_readings"] = cursor.fetchone()[0]

        cursor = self.conn.execute("SELECT COUNT(*) FROM simulation_run_metadata")
        stats["total_runs"] = cursor.fetchone()[0]

        cursor = self.conn.execute(
            "SELECT domain, COUNT(*) FROM signal_quality_log GROUP BY domain"
        )
        stats["readings_by_domain"] = dict(cursor.fetchall())

        cursor = self.conn.execute(
            "SELECT quality, COUNT(*) FROM signal_quality_log GROUP BY quality"
        )
        stats["readings_by_quality"] = dict(cursor.fetchall())

        cursor = self.conn.execute(
            "SELECT label, COUNT(*) FROM signal_quality_log GROUP BY label"
        )
        stats["readings_by_label"] = dict(cursor.fetchall())

        return stats

    # -----------------------------------------------------------------
    # Export
    # -----------------------------------------------------------------

    def export_to_csv(
        self,
        filepath: str,
        domain: Optional[str] = None,
        label: Optional[str] = None,
    ) -> int:
        """Export readings to CSV. Returns row count."""
        import csv

        sql = "SELECT timestamp, sensor_id, value, quality, confidence, domain, label FROM signal_quality_log"
        conditions = []
        params: list = []
        if domain:
            conditions.append("domain = ?")
            params.append(domain)
        if label:
            conditions.append("label = ?")
            params.append(label)
        if conditions:
            sql += " WHERE " + " AND ".join(conditions)
        sql += " ORDER BY timestamp"

        cursor = self.conn.execute(sql, params)
        columns = [d[0] for d in cursor.description]

        count = 0
        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(columns)
            for row in cursor:
                writer.writerow(row)
                count += 1

        return count

    # -----------------------------------------------------------------
    # Cleanup
    # -----------------------------------------------------------------

    def close(self) -> None:
        """Close the database connection."""
        self.conn.close()

    def clear_all(self) -> None:
        """Delete all data (for testing)."""
        self.conn.execute("DELETE FROM signal_quality_log")
        self.conn.execute("DELETE FROM simulation_run_metadata")
        self.conn.commit()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
