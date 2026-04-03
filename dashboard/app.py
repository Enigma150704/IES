import os
import sqlite3
import json
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from typing import List, Optional
import uvicorn

app = FastAPI(title="KKC Simulation Dashboard")

# Serve static files from the 'static' directory
# (We'll create this directory and add the frontend files soon)
if not os.path.exists("dashboard/static"):
    os.makedirs("dashboard/static")

# Helper to find .db files in project root
def get_db_files():
    # Look in the root directory (one level up from dashboard/)
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return [f for f in os.listdir(root_dir) if f.endswith(".db")]

def get_db_path(db_name: str):
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    db_path = os.path.join(root_dir, db_name)
    if not os.path.exists(db_path):
        raise HTTPException(status_code=404, detail="Database file not found")
    return db_path

@app.get("/api/dbs")
def list_databases():
    return {"databases": get_db_files()}

@app.get("/api/stats/{db_name}")
def get_stats(db_name: str):
    db_path = get_db_path(db_name)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    try:
        # Check if metadata exists
        cursor.execute("SELECT COUNT(*) as count FROM simulation_run_metadata")
        meta_count = cursor.fetchone()["count"]
        
        if meta_count > 0:
            # Standard path
            cursor.execute("SELECT COUNT(*) as count FROM simulation_run_metadata")
            total_runs = cursor.fetchone()["count"]
            cursor.execute("SELECT SUM(total_readings) as count FROM simulation_run_metadata")
            total_readings = cursor.fetchone()["count"] or 0
            cursor.execute("SELECT COUNT(*) as count FROM simulation_run_metadata WHERE fault_count = 0")
            passed_runs = cursor.fetchone()["count"]
            cursor.execute("SELECT domain, COUNT(*) as count FROM simulation_run_metadata GROUP BY domain")
            domain_dist = {row["domain"]: row["count"] for row in cursor.fetchall()}
            cursor.execute("""
                SELECT run_id, domain, scenario, fault_count, total_readings, start_time, end_time 
                FROM simulation_run_metadata 
                ORDER BY start_time DESC LIMIT 10
            """)
            latest_runs = [dict(row) for row in cursor.fetchall()]
        else:
            # Fallback path: Reconstruct from signal_quality_log
            cursor.execute("SELECT COUNT(DISTINCT run_id) as count FROM signal_quality_log")
            total_runs = cursor.fetchone()["count"]
            cursor.execute("SELECT COUNT(*) as count FROM signal_quality_log")
            total_readings = cursor.fetchone()["count"]
            
            # Approximate pass rate (runs with no 'ok' != quality)
            cursor.execute("""
                SELECT COUNT(*) as count FROM (
                    SELECT run_id FROM signal_quality_log 
                    GROUP BY run_id 
                    HAVING SUM(CASE WHEN quality != 'ok' THEN 1 ELSE 0 END) = 0
                )
            """)
            passed_runs = cursor.fetchone()["count"]
            
            cursor.execute("SELECT domain, COUNT(*) as count FROM signal_quality_log GROUP BY domain")
            # For fallback, we divide by average sensors per reading or just show raw points
            domain_dist = {row["domain"]: row["count"] for row in cursor.fetchall()}
            
            cursor.execute("""
                SELECT run_id, domain, 'unrecorded' as scenario, 
                       SUM(CASE WHEN quality != 'ok' THEN 1 ELSE 0 END) as fault_count,
                       COUNT(*) as total_readings,
                       MIN(timestamp) as start_time, MAX(timestamp) as end_time
                FROM signal_quality_log 
                GROUP BY run_id 
                ORDER BY start_time DESC LIMIT 10
            """)
            latest_runs = [dict(row) for row in cursor.fetchall()]
        
        return {
            "total_runs": total_runs,
            "total_readings": total_readings,
            "passed_runs": passed_runs,
            "pass_rate": (passed_runs / total_runs * 100) if total_runs > 0 else 0,
            "domain_distribution": domain_dist,
            "latest_runs": latest_runs,
            "is_fallback": meta_count == 0
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()

@app.get("/api/runs/{db_name}")
def get_runs(db_name: str, domain: Optional[str] = None):
    db_path = get_db_path(db_name)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    try:
        # Check metadata
        cursor.execute("SELECT COUNT(*) as count FROM simulation_run_metadata")
        meta_count = cursor.fetchone()["count"]
        
        if meta_count > 0:
            query = "SELECT * FROM simulation_run_metadata"
            params = []
            if domain:
                query += " WHERE domain = ?"
                params.append(domain)
            query += " ORDER BY start_time DESC"
            cursor.execute(query, params)
            runs = [dict(row) for row in cursor.fetchall()]
        else:
            # Fallback
            query = """
                SELECT run_id, domain, 'unrecorded' as scenario, 
                       SUM(CASE WHEN quality != 'ok' THEN 1 ELSE 0 END) as fault_count,
                       COUNT(*) as total_readings,
                       MIN(timestamp) as start_time, MAX(timestamp) as end_time
                FROM signal_quality_log
            """
            params = []
            if domain:
                query += " WHERE domain = ?"
                params.append(domain)
            query += " GROUP BY run_id ORDER BY start_time DESC"
            cursor.execute(query, params)
            runs = [dict(row) for row in cursor.fetchall()]
            
        return {"runs": runs, "is_fallback": meta_count == 0}
    finally:
        conn.close()

@app.get("/api/run_details/{db_name}/{run_id}")
def get_run_details(db_name: str, run_id: str):
    db_path = get_db_path(db_name)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    try:
        # Get metadata
        cursor.execute("SELECT * FROM simulation_run_metadata WHERE run_id = ?", (run_id,))
        meta = cursor.fetchone()
        if not meta:
            raise HTTPException(status_code=404, detail="Run not found")
            
        # Get sample sensor data (limit for performance)
        cursor.execute("""
            SELECT timestamp, sensor_id, value, quality, confidence, label 
            FROM signal_quality_log 
            WHERE run_id = ? 
            ORDER BY timestamp ASC LIMIT 500
        """, (run_id,))
        logs = [dict(row) for row in cursor.fetchall()]
        
        return {
            "metadata": dict(meta),
            "logs": logs
        }
    finally:
        conn.close()

# HTML Fallback
@app.get("/")
def read_index():
    return FileResponse("dashboard/static/index.html")

# Mount static files AFTER the routes to avoid conflict with / or /api
app.mount("/static", StaticFiles(directory="dashboard/static"), name="static")

if __name__ == "__main__":
    import sys
    port = 8001 if "--port" in sys.argv else 8000
    uvicorn.run(app, host="127.0.0.1", port=port)
