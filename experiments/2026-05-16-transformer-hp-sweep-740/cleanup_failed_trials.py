"""Delete FAIL / RUNNING / WAITING trials from the Optuna SQLite study.

Run between sweep restart attempts so the n_trials budget isn't burned
by trials that crashed before recording any usable signal.
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

DB = Path(__file__).resolve().parent / "results" / "optuna.db"


def main() -> int:
    if not DB.exists():
        print(f"no DB at {DB}, nothing to do")
        return 0
    conn = sqlite3.connect(DB)
    cur = conn.cursor()

    # Find trials with no usable result (no recorded value).
    # COMPLETE trials always have a value; PRUNED have intermediate values
    # we want to keep. FAIL/RUNNING/WAITING are the noise.
    cur.execute(
        "SELECT trial_id, number, state FROM trials "
        "WHERE state IN ('FAIL', 'RUNNING', 'WAITING')"
    )
    bad = cur.fetchall()
    if not bad:
        print("no FAIL/RUNNING/WAITING trials — nothing to clean")
        conn.close()
        return 0

    print(f"cleaning {len(bad)} trials: states "
          f"{sorted(set(r[2] for r in bad))}, numbers "
          f"{sorted(r[1] for r in bad)[:5]}...{sorted(r[1] for r in bad)[-5:]}")
    bad_ids = [r[0] for r in bad]
    placeholders = ",".join("?" * len(bad_ids))
    for tbl in ["trial_params", "trial_values", "trial_user_attributes",
                "trial_system_attributes", "trial_intermediate_values",
                "trial_heartbeats"]:
        cur.execute(
            f"DELETE FROM {tbl} WHERE trial_id IN ({placeholders})", bad_ids
        )
    cur.execute(f"DELETE FROM trials WHERE trial_id IN ({placeholders})", bad_ids)
    conn.commit()

    # Report remaining state.
    cur.execute("SELECT COUNT(*), state FROM trials GROUP BY state")
    print("post-cleanup:", cur.fetchall())
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
