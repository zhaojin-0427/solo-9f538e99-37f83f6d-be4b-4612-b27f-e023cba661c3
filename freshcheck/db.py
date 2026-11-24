"""SQLite storage layer.

Design notes
------------
* One database file.  Every upload is written in a single transaction and
  ``upload_batches`` holds an append-only audit trail of what happened.
* "Same kind + same number, latest upload wins" is implemented by keeping a
  *natural key* per table and doing ``INSERT ... ON CONFLICT DO UPDATE``.
* Arrivals keep a *snapshot* of the barcode they were scanned with.  Purchase
  orders keep an *agreed* barcode.  Correction of either can change the
  verdict of an arrival, therefore ``recheck_arrivals`` is used after every
  write to recompute the set of blocking reasons for affected arrivals.
* ``arrival_search`` is an FTS5 *external-content* trigram index over the
  three columns the arrivals lookup searches.  The ``arrivals`` table stays
  the single source of truth; triggers feed the index in the same SQLite
  transaction as the upload, so a failed upload cannot leave a half-updated
  index behind.  Older database files are backfilled once (see ``_migrate``).
"""

from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

DB_PATH = os.environ.get("FRESHCHECK_DB", os.path.join(os.path.dirname(__file__), "..", "data", "freshcheck.db"))

SEARCH_INDEX_KEY = "arrival_search_version"
SEARCH_INDEX_VERSION = "1"

SCHEMA = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS arrivals (
    arrival_no      TEXT PRIMARY KEY,          -- 到货编号 (business key)
    po_line_no      TEXT NOT NULL,             -- 采购行号
    barcode         TEXT NOT NULL,             -- 扫码条码
    arrived_at      TEXT NOT NULL,             -- UTC ISO-8601
    batch_id        INTEGER NOT NULL,
    updated_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS purchase_lines (
    po_line_no      TEXT PRIMARY KEY,          -- 采购行号
    agreed_barcode  TEXT NOT NULL,             -- 约定条码
    batch_id        INTEGER NOT NULL,
    updated_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS scan_records (
    record_no       TEXT PRIMARY KEY,          -- 用码记录编号
    barcode         TEXT NOT NULL,
    product         TEXT NOT NULL,             -- 商品
    starts_at       TEXT NOT NULL,             -- UTC 起（含）
    ends_at         TEXT NOT NULL,             -- UTC 止（不含）
    batch_id        INTEGER NOT NULL,
    updated_at      TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_scan_barcode ON scan_records (barcode);
CREATE INDEX IF NOT EXISTS idx_arrival_line ON arrivals (po_line_no);
CREATE INDEX IF NOT EXISTS idx_arrival_barcode ON arrivals (barcode);

-- Trigram full-text index for the arrivals lookup.  It is *external content*:
-- rows live in ``arrivals`` and are addressed by that table's ``rowid``.  The
-- triggers keep it in lock-step with every insert/update/delete inside the
-- caller's transaction (no separate write path to forget).  Only the three
-- searched columns are indexed; the release flag is never duplicated here.
CREATE VIRTUAL TABLE IF NOT EXISTS arrival_search USING fts5(
    arrival_no, po_line_no, barcode,
    content='arrivals', content_rowid='rowid', tokenize='trigram'
);

CREATE TRIGGER IF NOT EXISTS arrival_search_ai AFTER INSERT ON arrivals BEGIN
    INSERT INTO arrival_search(rowid, arrival_no, po_line_no, barcode)
    VALUES (new.rowid, new.arrival_no, new.po_line_no, new.barcode);
END;

CREATE TRIGGER IF NOT EXISTS arrival_search_ad AFTER DELETE ON arrivals BEGIN
    INSERT INTO arrival_search(arrival_search, rowid, arrival_no, po_line_no, barcode)
    VALUES ('delete', old.rowid, old.arrival_no, old.po_line_no, old.barcode);
END;

CREATE TRIGGER IF NOT EXISTS arrival_search_au
AFTER UPDATE OF arrival_no, po_line_no, barcode ON arrivals BEGIN
    INSERT INTO arrival_search(arrival_search, rowid, arrival_no, po_line_no, barcode)
    VALUES ('delete', old.rowid, old.arrival_no, old.po_line_no, old.barcode);
    INSERT INTO arrival_search(rowid, arrival_no, po_line_no, barcode)
    VALUES (new.rowid, new.arrival_no, new.po_line_no, new.barcode);
END;

-- Blocking reasons are derived, but cached so the list views are cheap and
-- stable.  The cache is recomputed inside the same transaction as the write
-- (only for the affected arrivals), so base tables, worklists and release
-- flags always become visible together.
CREATE TABLE IF NOT EXISTS arrival_issues (
    arrival_no      TEXT NOT NULL,
    side            TEXT NOT NULL,             -- 'purchase' | 'barcode'
    reason          TEXT NOT NULL,
    ref             TEXT NOT NULL,             -- po_line_no (purchase) / barcode (barcode)
    detail          TEXT NOT NULL,
    PRIMARY KEY (arrival_no, reason, ref)
);

CREATE INDEX IF NOT EXISTS idx_issue_side_ref ON arrival_issues (side, ref);

CREATE TABLE IF NOT EXISTS upload_batches (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    kind            TEXT NOT NULL,
    filename        TEXT NOT NULL,
    row_count       INTEGER NOT NULL,
    received_at     TEXT NOT NULL,
    status          TEXT NOT NULL,             -- 'committed' | 'failed'
    note            TEXT                       -- failure reason, if any
);

-- One-off migration bookkeeping (e.g. "the search index was backfilled").
CREATE TABLE IF NOT EXISTS schema_meta (
    key             TEXT PRIMARY KEY,
    value           TEXT NOT NULL
);
"""

REASON_LABELS = {
    "purchase_line_missing": "采购行缺失",
    "barcode_mismatch": "条码不符",
    "barcode_unowned": "到货时该码无归属",
    "barcode_ambiguous": "该码对应多种商品",
}

SIDE_LABELS = {"purchase": "采购/档案侧", "barcode": "用码/档案侧"}


class UploadError(ValueError):
    """Raised for a malformed upload; rolls the whole batch back."""


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def parse_utc(value: str, field: str) -> str:
    """Parse an ISO-8601 UTC timestamp and normalise it for comparisons.

    Storage uses a fixed-width ``YYYY-MM-DDTHH:MM:SS.ffffffZ`` form so that
    plain string comparison is chronological and sub-second boundaries are
    still respected, without relying on SQLite date functions.
    """
    raw = (value or "").strip()
    if not raw:
        raise UploadError(f"{field} 不能为空")
    candidate = raw.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError as exc:
        raise UploadError(f"{field} 不是合法的 ISO-8601 时刻: {raw!r}") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    parsed = parsed.astimezone(timezone.utc)
    # Fixed width keeps plain string comparison chronological in SQLite,
    # while preserving sub-second precision instead of silently truncating
    # an instant that sits exactly on a window boundary.
    return parsed.strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def display_utc(value: str) -> str:
    """Render a stored UTC instant without a trailing ``.000000``."""
    if not value:
        return ""
    if value.endswith(".000000Z"):
        return value[: -len(".000000Z")] + "Z"
    return value


def connect(path: str | None = None) -> sqlite3.Connection:
    conn = sqlite3.connect(path or DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(path: str | None = None) -> None:
    """Create the schema if needed.  Safe to call from concurrent workers."""
    import time

    target = path or DB_PATH
    if target != ":memory:":
        os.makedirs(os.path.dirname(os.path.abspath(target)), exist_ok=True)
    last: Exception | None = None
    for _attempt in range(6):
        try:
            conn = connect(target)
            try:
                # ``executescript`` commits first and runs in autocommit mode;
                # all schema statements are idempotent, so a crash there is
                # harmless.  The data migration (timestamp normalisation and
                # the one-off search backfill with its version marker) then
                # runs in a single explicit transaction: it is applied in full
                # or not at all.
                conn.executescript(SCHEMA)
                conn.execute("BEGIN IMMEDIATE")
                try:
                    _migrate(conn)
                    conn.commit()
                except Exception:
                    conn.rollback()
                    raise
            finally:
                conn.close()
            return
        except sqlite3.OperationalError as exc:  # first-boot race between workers
            last = exc
            time.sleep(0.1)
    raise last  # type: ignore[misc]


def _migrate(conn: sqlite3.Connection) -> None:
    """Bring an older database file up to the current schema/format."""
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(upload_batches)")}
    if "note" not in columns:
        conn.execute("ALTER TABLE upload_batches ADD COLUMN note TEXT")

    # Early builds stored whole-second instants ("...:00Z"); the current
    # format always carries microseconds so that string comparison in SQLite
    # stays chronological even across sub-second window boundaries.
    for table, column in (
        ("arrivals", "arrived_at"),
        ("scan_records", "starts_at"),
        ("scan_records", "ends_at"),
    ):
        conn.execute(
            f"UPDATE {table} SET {column} = substr({column}, 1, 19) || '.000000Z' "
            f"WHERE length({column}) = 20 AND {column} LIKE '%Z'"
        )

    _ensure_search_index(conn)


def _ensure_search_index(conn: sqlite3.Connection) -> None:
    """Backfill the trigram index once for database files created before it.

    Rows written through the triggers after this point are indexed inline.
    The rebuild and the version marker commit together with the rest of the
    migration, so a crash leaves the marker unset and the next start retries.
    """
    row = conn.execute(
        "SELECT value FROM schema_meta WHERE key = ?", (SEARCH_INDEX_KEY,)
    ).fetchone()
    if row is not None and row["value"] == SEARCH_INDEX_VERSION:
        return
    conn.execute("INSERT INTO arrival_search(arrival_search) VALUES('rebuild')")
    conn.execute(
        "INSERT INTO schema_meta (key, value) VALUES (?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (SEARCH_INDEX_KEY, SEARCH_INDEX_VERSION),
    )


@contextmanager
def transaction(path: str | None = None):
    conn = connect(path)
    try:
        conn.execute("BEGIN IMMEDIATE")
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


@contextmanager
def read_conn(path: str | None = None):
    """Short-lived read transaction.

    SQLite in WAL mode gives a read transaction a stable snapshot, so every
    query a page makes (base data, worklists, release flags) sees the same
    committed batch even if an upload lands in between.
    """
    conn = connect(path)
    try:
        conn.execute("BEGIN")
        yield conn
    finally:
        conn.rollback()
        conn.close()
