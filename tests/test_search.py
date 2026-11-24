"""Tests for the torn-label fragment lookup on /arrivals.

A fragment is 3-8 ASCII letters/digits; it is matched case-insensitively as a
substring of arrival number, purchase line number or scanned barcode, ordered
by arrival number, at most 50 rows per page, using a stable "after this
arrival number" cursor.
"""

from __future__ import annotations

import sqlite3

import pytest

from freshcheck import service
from freshcheck.app import app as flask_app
from freshcheck.db import connect, init_db

PURCHASE = "编号,约定条码\nPO-1001,6900000000017\nPO-1002,6900000000024\n"
SCANS = (
    "编号,条码,商品,UTC起,UTC止\n"
    "SC-01,6900000000017,红富士苹果,2026-11-01T00:00:00Z,2026-11-02T00:00:00Z\n"
)
ARRIVALS = (
    "编号,采购行号,扫码条码,UTC时刻\n"
    "AR-001234-XY,PO-1001,6900000000017,2026-11-01T08:30:00Z\n"
    "AR-009999-ZZ,PO-1002,6900000000024,2026-11-01T08:30:00Z\n"
)


@pytest.fixture()
def db(tmp_path):
    path = str(tmp_path / "search.db")
    init_db(path)
    return path


def up(db, kind, body, filename="f.csv"):
    return service.upload(kind, filename, body.encode("utf-8"), db)


def seed(db):
    up(db, "purchase_lines", PURCHASE)
    up(db, "scan_records", SCANS)
    up(db, "arrivals", ARRIVALS)


def search(db, fragment, after=""):
    with connect(db) as conn:
        return service.search_arrivals(conn, fragment, after=after)


def test_fragment_hits_all_three_columns(db):
    seed(db)
    assert [r["arrival_no"] for r in search(db, "1234")["rows"]] == ["AR-001234-XY"]
    assert [r["arrival_no"] for r in search(db, "1001")["rows"]] == ["AR-001234-XY"]
    assert [r["arrival_no"] for r in search(db, "6900000000017")["rows"]] == ["AR-001234-XY"]


def test_fragment_is_case_insensitive_for_ascii_letters(db):
    up(db, "arrivals", "编号,采购行号,扫码条码,UTC时刻\n"
                       "abc-XY-001,P-lin-77,6900ABCDEF,2026-11-01T08:30:00Z\n")
    for fragment in ("abc", "ABC", "AbC", "Lin", "LIN", "abcdef"):
        assert [r["arrival_no"] for r in search(db, fragment)["rows"]] == ["abc-XY-001"], fragment


def test_multi_column_hit_lists_arrival_once(db):
    # AR-777777 has the fragment both in its number and in its po line / barcode.
    up(db, "arrivals", "编号,采购行号,扫码条码,UTC时刻\n"
                       "AR-777777,PO-777777,697777777777,2026-11-01T08:30:00Z\n")
    result = search(db, "777777")
    assert result["total"] == 1
    assert len(result["rows"]) == 1


def test_legacy_queries_unchanged(db):
    seed(db)
    with connect(db) as conn:
        rows = service.list_arrivals(conn, "AR-001")
    assert [row["arrival_no"] for row in rows] == ["AR-001234-XY"]
    with connect(db) as conn:
        assert service.list_arrivals(conn, "ZZZ") == []


def test_fragment_parser_boundaries():
    assert service.parse_fragment("abc") == "abc"
    assert service.parse_fragment("  ABC123  ") == "ABC123"
    assert service.parse_fragment("a1b2c3d4") == "a1b2c3d4"
    # 2 or 9 characters, spaces, punctuation, non-ASCII -> legacy lookup
    assert service.parse_fragment("ab") is None
    assert service.parse_fragment("a1b2c3d4e") is None
    assert service.parse_fragment("ab cd") is None
    assert service.parse_fragment("ab-1") is None
    assert service.parse_fragment("编号") is None
    assert service.parse_fragment("") is None


def test_page_size_and_ordering(db):
    body = "编号,采购行号,扫码条码,UTC时刻\n" + "".join(
        f"AR-{i:05d},PO-X,6900000000007,2026-11-01T08:30:00Z\n" for i in range(120)
    )
    up(db, "arrivals", body)
    first = search(db, "AR-0")
    assert first["total"] == 120
    assert len(first["rows"]) == 50
    assert [r["arrival_no"] for r in first["rows"]] == [f"AR-{i:05d}" for i in range(50)]
    assert first["has_more"] is True
    assert first["next_after"] == "AR-00049"

    second = search(db, "AR-0", after=first["next_after"])
    assert [r["arrival_no"] for r in second["rows"]] == [f"AR-{i:05d}" for i in range(50, 100)]
    assert second["has_more"] is True

    third = search(db, "AR-0", after=second["next_after"])
    assert [r["arrival_no"] for r in third["rows"]] == [f"AR-{i:05d}" for i in range(100, 120)]
    assert third["has_more"] is False
    assert third["next_after"] == ""


def test_next_page_uses_records_still_matching_after_cursor(db):
    # All 60 arrivals carry the same barcode; fragment "9000000" (7 chars) hits
    # only through the barcode, so a barcode correction can remove a match.
    body = "编号,采购行号,扫码条码,UTC时刻\n" + "".join(
        f"AR-{i:05d},PO-X,6900000000007,2026-11-01T08:30:00Z\n" for i in range(60)
    )
    up(db, "arrivals", body)
    first = search(db, "9000000")
    assert len(first["rows"]) == 50
    cursor = first["next_after"]
    assert cursor == "AR-00049"

    # Upcoming AR-00055 is corrected to a barcode that no longer matches; a new
    # arrival is added before the cursor (AR-000025) and one after (AR-00060).
    up(db, "arrivals", "编号,采购行号,扫码条码,UTC时刻\n"
                       "AR-00055,PO-X,6111111111111,2026-11-01T08:30:00Z\n"
                       "AR-000025,PO-X,6900000000007,2026-11-01T08:30:00Z\n"
                       "AR-00060,PO-X,6900000000007,2026-11-01T08:30:00Z\n",
       "recode.csv")

    second = search(db, "9000000", after=cursor)
    got = [r["arrival_no"] for r in second["rows"]]
    assert got[0] == "AR-00050"
    assert "AR-000025" not in got   # inserted before the cursor: not replayed
    assert "AR-00055" not in got    # no longer matches at read time
    assert "AR-00060" in got        # still matches after the cursor
    assert len(got) == 10           # 50..59 minus 55, plus 60


def test_recode_with_same_number_moves_index(db):
    seed(db)
    assert [r["arrival_no"] for r in search(db, "1234")["rows"]] == ["AR-001234-XY"]
    # Same arrival number, new line and barcode: the number still matches, but
    # the old line/barcode must stop matching and the new ones must start.
    up(db, "arrivals", "编号,采购行号,扫码条码,UTC时刻\n"
                       "AR-001234-XY,PO-5555,6900000005555,2026-11-01T08:30:00Z\n",
       "recode.csv")
    assert [r["arrival_no"] for r in search(db, "1234")["rows"]] == ["AR-001234-XY"]
    assert search(db, "1001")["total"] == 0
    assert search(db, "6900000000017")["total"] == 0
    assert [r["arrival_no"] for r in search(db, "5555")["rows"]] == ["AR-001234-XY"]
    assert [r["arrival_no"] for r in search(db, "6900000005555")["rows"]] == ["AR-001234-XY"]


def test_recode_trial_valid_fragment_through_index(db):
    """The delivery trial: same number re-coded, 7-char fragments only."""
    up(db, "arrivals", "编号,采购行号,扫码条码,UTC时刻\n"
                       "AR-ABCD12,PO-1,69000OLDCODE,2026-11-01T12:00:00Z\n")
    assert [r["arrival_no"] for r in search(db, "OLDCODE")["rows"]] == ["AR-ABCD12"]
    up(db, "arrivals", "编号,采购行号,扫码条码,UTC时刻\n"
                       "AR-ABCD12,PO-1,69000NEWCODE,2026-11-01T12:00:00Z\n",
       "recode.csv")
    assert search(db, "OLDCODE")["total"] == 0
    assert [r["arrival_no"] for r in search(db, "NEWCODE")["rows"]] == ["AR-ABCD12"]
    # the arrival number itself still matches
    assert [r["arrival_no"] for r in search(db, "ABCD12")["rows"]] == ["AR-ABCD12"]
    with connect(db) as conn:
        indexed = conn.execute(
            "SELECT arrival_no, po_line_no, barcode FROM arrival_search"
        ).fetchall()
    assert [tuple(row) for row in indexed] == [("AR-ABCD12", "PO-1", "69000NEWCODE")]


def test_duplicate_number_in_one_batch_indexes_last_row(db):
    up(db, "arrivals", "编号,采购行号,扫码条码,UTC时刻\n"
                       "AR-DUP-1,PO-OLD,6900000000001,2026-11-01T08:30:00Z\n"
                       "AR-DUP-1,PO-NEW,6900000000002,2026-11-01T08:30:00Z\n")
    assert search(db, "PO-OLD")["total"] == 0
    assert [r["arrival_no"] for r in search(db, "PO-NEW")["rows"]] == ["AR-DUP-1"]


def test_failed_upload_keeps_index_and_results(db):
    seed(db)
    before = search(db, "1234")
    with pytest.raises(Exception):
        up(db, "arrivals", "编号,采购行号,扫码条码,UTC时刻\n"
                           "AR-001234-XY,PO-QQQQ,6900000008888,bad-time\n", "bad.csv")
    after = search(db, "1234")
    assert [r["arrival_no"] for r in after["rows"]] == [r["arrival_no"] for r in before["rows"]]
    assert search(db, "QQQQ")["total"] == 0
    assert search(db, "8888")["total"] == 0


def test_mid_transaction_failure_rolls_back_index(db, monkeypatch):
    """The trigram index is written in the upload transaction, not beside it."""
    seed(db)
    before = search(db, "5555")

    from freshcheck import rules

    def boom(*_args, **_kwargs):
        raise RuntimeError("disk full")

    monkeypatch.setattr(rules, "recheck_arrivals", boom)
    with pytest.raises(RuntimeError):
        up(db, "arrivals", "编号,采购行号,扫码条码,UTC时刻\n"
                           "AR-001234-XY,PO-5555,6900000005555,2026-11-01T08:30:00Z\n",
           "mutate.csv")
    monkeypatch.undo()

    # The upsert (and therefore the trigger-fed index write) was rolled back.
    assert search(db, "5555")["total"] == before["total"] == 0
    assert [r["arrival_no"] for r in search(db, "1234")["rows"]] == ["AR-001234-XY"]


def test_page_status_and_issues_come_from_same_snapshot(db):
    seed(db)
    result = search(db, "1234")
    row = result["rows"][0]
    assert row["released"] is True
    assert row["issue_count"] == len(row["issues"]) == 0

    # Block it, then read page + detail through one read transaction.
    up(db, "scan_records", "编号,条码,商品,UTC起,UTC止\n"
                           "SC-01,6900000000017,红富士苹果,2026-11-02T00:00:00Z,2026-11-03T00:00:00Z\n",
       "shrink.csv")
    with connect(db) as conn:
        conn.execute("BEGIN")
        page = service.search_arrivals(conn, "1234")
        detail = service.arrival_detail(conn, "AR-001234-XY")
        conn.rollback()
    row = page["rows"][0]
    assert row["released"] is False
    assert row["issue_count"] == len(row["issues"]) == len(detail["issues"]) > 0


def test_old_database_file_is_backfilled(tmp_path):
    """A DB created before the index existed still answers fragment lookups."""
    path = str(tmp_path / "old.db")
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE arrivals (arrival_no TEXT PRIMARY KEY, po_line_no TEXT NOT NULL,
            barcode TEXT NOT NULL, arrived_at TEXT NOT NULL, batch_id INTEGER NOT NULL,
            updated_at TEXT NOT NULL);
        CREATE TABLE purchase_lines (po_line_no TEXT PRIMARY KEY, agreed_barcode TEXT NOT NULL,
            batch_id INTEGER NOT NULL, updated_at TEXT NOT NULL);
        CREATE TABLE scan_records (record_no TEXT PRIMARY KEY, barcode TEXT NOT NULL,
            product TEXT NOT NULL, starts_at TEXT NOT NULL, ends_at TEXT NOT NULL,
            batch_id INTEGER NOT NULL, updated_at TEXT NOT NULL);
        CREATE TABLE arrival_issues (arrival_no TEXT NOT NULL, side TEXT NOT NULL,
            reason TEXT NOT NULL, ref TEXT NOT NULL, detail TEXT NOT NULL,
            PRIMARY KEY (arrival_no, reason, ref));
        CREATE TABLE upload_batches (id INTEGER PRIMARY KEY AUTOINCREMENT, kind TEXT NOT NULL,
            filename TEXT NOT NULL, row_count INTEGER NOT NULL, received_at TEXT NOT NULL,
            status TEXT NOT NULL, note TEXT);
        INSERT INTO arrivals VALUES ('LEGACY-ABC-1','PO-LEGACY','6900000000042',
            '2026-01-01T00:00:00.000000Z',1,'x');
        """
    )
    conn.commit()
    conn.close()

    init_db(path)
    with connect(path) as conn:
        page = service.search_arrivals(conn, "abc")
    assert [r["arrival_no"] for r in page["rows"]] == ["LEGACY-ABC-1"]

    # Running init again (every worker start) must not duplicate index rows.
    init_db(path)
    with connect(path) as conn:
        assert service.search_arrivals(conn, "abc")["total"] == 1


def test_web_fragment_search_and_pagination(client_with_data):
    resp = client_with_data.get("/arrivals?q=1234")
    body = resp.data.decode()
    assert resp.status_code == 200
    assert "AR-001234-XY" in body
    assert "AR-009999-ZZ" not in body
    assert "命中 1 笔" in body

    resp = client_with_data.get("/arrivals?q=ab")  # 2 chars -> legacy behaviour
    assert resp.status_code == 200
    assert "没有匹配" in resp.data.decode() or "AR-" not in resp.data.decode()


def test_web_pagination_links_walk_all_pages(client_pages):
    first = client_pages.get("/arrivals?q=PGA00").data.decode()
    assert "命中 120 笔" in first
    assert 'after=PGA00049' in first
    assert "下一页" in first

    second = client_pages.get("/arrivals?q=PGA00&after=PGA00049").data.decode()
    assert 'after=PGA00099' in second
    assert "第一页" in second

    last = client_pages.get("/arrivals?q=PGA00&after=PGA00099").data.decode()
    assert "已到最后一页" in last
    assert "下一页" not in last
    assert "PGA00119" in last


@pytest.fixture()
def client_pages(tmp_path):
    flask_app.config.update(TESTING=True, DB_PATH=str(tmp_path / "pages.db"), SECRET_KEY="t")
    body = "编号,采购行号,扫码条码,UTC时刻\n" + "".join(
        f"PGA{i:05d},POX{i:05d},69{i:011d},2026-01-01T12:00:00Z\n" for i in range(120)
    )
    with flask_app.test_client() as client:
        client.post(
            "/upload",
            data={"kind": "arrivals", "file": (__import__("io").BytesIO(body.encode()), "p.csv")},
            content_type="multipart/form-data",
        )
        yield client


@pytest.fixture()
def client_with_data(tmp_path):
    flask_app.config.update(TESTING=True, DB_PATH=str(tmp_path / "web.db"), SECRET_KEY="t")
    with flask_app.test_client() as client:
        for kind, body in (
            ("purchase_lines", PURCHASE),
            ("scan_records", SCANS),
            ("arrivals", ARRIVALS),
        ):
            client.post(
                "/upload",
                data={"kind": kind, "file": (__import__("io").BytesIO(body.encode()), "f.csv")},
                content_type="multipart/form-data",
            )
        yield client
