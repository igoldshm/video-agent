"""Durable run state: runs, checkpointed steps, trace events, creator feedback.

Everything the harness knows about a run lives here, so a process can die at any
point and another one can pick the run up from the last completed step.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs(
  id TEXT PRIMARY KEY, brief TEXT, status TEXT, config TEXT, state TEXT,
  created REAL, updated REAL);
CREATE TABLE IF NOT EXISTS steps(
  run_id TEXT, key TEXT, tool TEXT, output TEXT, cost REAL, ms REAL, created REAL,
  PRIMARY KEY(run_id, key));
CREATE TABLE IF NOT EXISTS events(
  id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT, ts REAL, kind TEXT, key TEXT, data TEXT);
CREATE TABLE IF NOT EXISTS feedback(
  id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT, shot_id TEXT, note TEXT, ts REAL);
CREATE TABLE IF NOT EXISTS preferences(
  id INTEGER PRIMARY KEY AUTOINCREMENT, project TEXT, note TEXT, ts REAL);
"""


class Store:
    def __init__(self, path: str = ":memory:"):
        self.db = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self.db.row_factory = sqlite3.Row
        self.lock = threading.Lock()
        with self.lock:
            self.db.executescript(SCHEMA)

    def _q(self, sql: str, args: tuple = ()) -> list[sqlite3.Row]:
        with self.lock:
            return self.db.execute(sql, args).fetchall()

    # runs -----------------------------------------------------------------
    def create_run(self, brief: str, config: dict, state: dict) -> str:
        run_id = uuid.uuid4().hex[:16]  # unguessable: media URLs are keyed by it
        now = time.time()
        self._q("INSERT INTO runs VALUES(?,?,?,?,?,?,?)",
                (run_id, brief, "pending", json.dumps(config), json.dumps(state), now, now))
        return run_id

    def get_run(self, run_id: str) -> dict:
        rows = self._q("SELECT * FROM runs WHERE id=?", (run_id,))
        if not rows:
            raise KeyError(run_id)
        r = dict(rows[0])
        r["config"], r["state"] = json.loads(r["config"]), json.loads(r["state"])
        r["cost_usd"] = self.total_cost(run_id)
        return r

    def list_runs(self) -> list[dict]:
        return [self.get_run(r["id"]) for r in self._q("SELECT id FROM runs ORDER BY created DESC")]

    def set_status(self, run_id: str, status: str) -> None:
        self._q("UPDATE runs SET status=?, updated=? WHERE id=?", (status, time.time(), run_id))

    def save_state(self, run_id: str, state: dict) -> None:
        self._q("UPDATE runs SET state=?, updated=? WHERE id=?", (json.dumps(state), time.time(), run_id))

    def save_config(self, run_id: str, config: dict) -> None:
        self._q("UPDATE runs SET config=? WHERE id=?", (json.dumps(config), run_id))

    # checkpointed steps -----------------------------------------------------
    def get_step(self, run_id: str, key: str) -> dict | None:
        rows = self._q("SELECT output FROM steps WHERE run_id=? AND key=?", (run_id, key))
        return json.loads(rows[0]["output"]) if rows else None

    def save_step(self, run_id: str, key: str, tool: str, output: dict, cost: float, ms: float) -> None:
        self._q("INSERT INTO steps VALUES(?,?,?,?,?,?,?)",
                (run_id, key, tool, json.dumps(output), cost, ms, time.time()))

    def steps(self, run_id: str) -> list[dict]:
        out = []
        for r in self._q("SELECT * FROM steps WHERE run_id=? ORDER BY created", (run_id,)):
            d = dict(r)
            d["output"] = json.loads(d["output"])
            out.append(d)
        return out

    def total_cost(self, run_id: str) -> float:
        return round(self._q("SELECT COALESCE(SUM(cost),0) c FROM steps WHERE run_id=?", (run_id,))[0]["c"], 4)

    def steps_for_tool(self, tool: str) -> list[dict]:
        """Every recorded call of one tool across all runs: the raw material for learning."""
        return [json.loads(r["output"]) for r in self._q("SELECT output FROM steps WHERE tool=? ORDER BY created", (tool,))]

    def spent_since(self, ts: float) -> float:
        return self._q("SELECT COALESCE(SUM(cost),0) c FROM steps WHERE created>=?", (ts,))[0]["c"]

    def runs_since(self, ts: float) -> int:
        return self._q("SELECT COUNT(*) c FROM runs WHERE created>=?", (ts,))[0]["c"]

    def unfinished_runs(self) -> list[str]:
        return [r["id"] for r in self._q("SELECT id FROM runs WHERE status IN ('pending','running')")]

    # trace ------------------------------------------------------------------
    def emit(self, run_id: str, kind: str, key: str = "", data: dict | None = None) -> None:
        self._q("INSERT INTO events(run_id,ts,kind,key,data) VALUES(?,?,?,?,?)",
                (run_id, time.time(), kind, key, json.dumps(data or {})))

    def events(self, run_id: str) -> list[dict]:
        out = []
        for r in self._q("SELECT * FROM events WHERE run_id=? ORDER BY id", (run_id,)):
            d = dict(r)
            d["data"] = json.loads(d["data"])
            out.append(d)
        return out

    # creator feedback -------------------------------------------------------
    def add_feedback(self, run_id: str, shot_id: str, note: str) -> None:
        self._q("INSERT INTO feedback(run_id,shot_id,note,ts) VALUES(?,?,?,?)",
                (run_id, shot_id, note, time.time()))

    def feedback(self, run_id: str) -> list[dict]:
        return [dict(r) for r in self._q("SELECT * FROM feedback WHERE run_id=? ORDER BY id", (run_id,))]

    def add_preference(self, project: str, note: str) -> None:
        self._q("INSERT INTO preferences(project,note,ts) VALUES(?,?,?)", (project, note, time.time()))

    def remove_preference(self, project: str, note: str) -> None:
        self._q("DELETE FROM preferences WHERE project=? AND note=?", (project, note))

    def preferences(self, project: str) -> list[str]:
        return [r["note"] for r in self._q("SELECT note FROM preferences WHERE project=? ORDER BY id", (project,))]
