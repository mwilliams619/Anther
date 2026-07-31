"""Tests for anther_ml.mpd_sql — the mysqldump → SQLite loader and sampler."""

from __future__ import annotations

import textwrap

import pytest

from anther_ml import mpd_sql


# ──────────────────────────────────────────────────────────────────────────
# Extended-INSERT parsing
# ──────────────────────────────────────────────────────────────────────────
def test_parse_track_line_basic():
    line = (
        "INSERT INTO `track` VALUES "
        "('abc','Song Name',210000,42.5,'0','https://p.scdn.co/x?cid=1',"
        "'spotify:track:abc','alb1'),"
        "('def','Other',NULL,0,'0',NULL,'spotify:track:def','alb2');"
    )
    table, rows = mpd_sql.parse_insert_line(line)
    assert table == "track"
    # kept columns: id, name, popularity, duration, preview_url, album_id
    assert rows[0] == ["abc", "Song Name", "42.5", "210000",
                       "https://p.scdn.co/x?cid=1", "alb1"]
    assert rows[1] == ["def", "Other", "0", None, None, "alb2"]


def test_parse_handles_escaped_quotes_and_commas_in_strings():
    line = (
        "INSERT INTO `track` VALUES "
        "('id1','Lil\\' Ol\\', Lonesome',1,2,'0','u','spotify:track:id1','a');"
    )
    _, rows = mpd_sql.parse_insert_line(line)
    # the comma and the escaped quotes live inside the name, not field separators
    assert rows[0][0] == "id1"
    assert rows[0][1] == "Lil' Ol', Lonesome"


def test_parse_link_table():
    line = "INSERT INTO `track_artist1` VALUES ('t1','a1'),('t1','a2'),('t2','a1');"
    table, rows = mpd_sql.parse_insert_line(line)
    assert table == "track_artist1"
    assert rows == [["t1", "a1"], ["t1", "a2"], ["t2", "a1"]]


def test_parse_ignores_other_lines():
    assert mpd_sql.parse_insert_line("-- a comment") is None
    assert mpd_sql.parse_insert_line("INSERT INTO `album` VALUES ('x','y','z');") is None
    assert mpd_sql.parse_insert_line("CREATE TABLE `track` (...);") is None


# ──────────────────────────────────────────────────────────────────────────
# End-to-end: tiny dump → SQLite → sample
# ──────────────────────────────────────────────────────────────────────────
SYNTHETIC_DUMP = textwrap.dedent(
    """\
    -- MySQL dump
    DROP TABLE IF EXISTS `artist`;
    INSERT INTO `artist` VALUES ('a1','Artist One'),('a2','Artist Two');
    INSERT INTO `playlist` VALUES ('p1','Chill',10,'uri',3),('p2','Party',5,'uri',2);
    INSERT INTO `track` VALUES ('t1','Track One',50,200000,'0','http://prev/t1','uri','al1'),('t2','Track Two',40,180000,'0','http://prev/t2','uri','al2'),('t3','No Preview',30,150000,'0',NULL,'uri','al3'),('t4','Track Four',20,90000,'0','http://prev/t4','uri','al4');
    INSERT INTO `track_artist1` VALUES ('t1','a1'),('t2','a1'),('t3','a2'),('t4','a2');
    INSERT INTO `track_playlist1` VALUES ('t1','p1'),('t1','p2'),('t2','p1');
    """
)


@pytest.fixture()
def loaded_db(tmp_path):
    dump = tmp_path / "dump.sql"
    dump.write_text(SYNTHETIC_DUMP)
    db = tmp_path / "dump.sqlite"
    mpd_sql.load_dump_to_sqlite(dump, db)
    return db


def test_load_marks_done_and_reuses(loaded_db, tmp_path):
    assert mpd_sql.is_loaded(loaded_db)
    # ensure_db reuses a completed load without needing the dump again
    assert mpd_sql.ensure_db(db_path=loaded_db) == loaded_db


def test_sample_excludes_missing_preview_and_attaches_membership(loaded_db):
    rows, membership = mpd_sql.sample_tracks(
        loaded_db, sample_n=None, artist_cap=None, seed=1
    )
    ids = {r["track_id"] for r in rows}
    assert "t3" not in ids            # NULL preview_url → excluded
    assert ids == {"t1", "t2", "t4"}
    # membership joined on playlist names
    names = {pl["name"] for pl in membership["t1"]}
    assert names == {"Chill", "Party"}
    assert membership["t2"] == [{"pid": "p1", "name": "Chill"}]


def test_sample_supports_long_tail_popularity_bound(loaded_db):
    rows, _ = mpd_sql.sample_tracks(
        loaded_db, sample_n=None, artist_cap=None, max_popularity=180000
    )
    assert {r["track_id"] for r in rows} == {"t2", "t4"}
    assert {r["popularity"] for r in rows} == {180000.0, 90000.0}


def test_artist_cap_limits_per_artist(loaded_db):
    # a1 has t1,t2 with previews; cap=1 keeps exactly one of them
    rows, _ = mpd_sql.sample_tracks(loaded_db, sample_n=None, artist_cap=1, seed=7)
    by_artist = {}
    for r in rows:
        by_artist.setdefault(r["artist_name"], 0)
        by_artist[r["artist_name"]] += 1
    assert all(c <= 1 for c in by_artist.values())


def test_sample_is_deterministic(loaded_db):
    a, _ = mpd_sql.sample_tracks(loaded_db, sample_n=2, artist_cap=None, seed=123)
    b, _ = mpd_sql.sample_tracks(loaded_db, sample_n=2, artist_cap=None, seed=123)
    assert [r["track_id"] for r in a] == [r["track_id"] for r in b]


def test_resume_is_idempotent(tmp_path):
    dump = tmp_path / "dump.sql"
    dump.write_text(SYNTHETIC_DUMP)
    db = tmp_path / "dump.sqlite"
    mpd_sql.load_dump_to_sqlite(dump, db)
    # re-running must not duplicate rows (INSERT OR IGNORE + done flag)
    mpd_sql.load_dump_to_sqlite(dump, db)
    import sqlite3

    con = sqlite3.connect(str(db))
    try:
        assert con.execute("SELECT COUNT(*) FROM track").fetchone()[0] == 4
        assert con.execute("SELECT COUNT(*) FROM track_artist1").fetchone()[0] == 4
    finally:
        con.close()
