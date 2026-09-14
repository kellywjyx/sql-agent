import sqlite3

from sql_agent.schema import inspect_schema
from sql_agent.v12_notes import column_notes_v13, sql_identifier
from sql_agent.value_profiles import ValueProfiler


def _database(tmp_path):
    folder = tmp_path / "school"
    (folder / "database_description").mkdir(parents=True)
    database = folder / "school.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute('CREATE TABLE frpm (CDSCode TEXT PRIMARY KEY, "Enrollment (K-12)" REAL, County TEXT, Notes TEXT)')
        connection.executemany("INSERT INTO frpm VALUES (?, ?, ?, ?)",
                               [("01", 100.0, "Alameda", "x"), ("02", 50.0, "Fresno", "y")])
    (folder / "database_description" / "frpm.csv").write_text(
        "original_column_name,column_name,column_description,data_format,value_description\n"
        "Enrollment (K-12),,enrollment for kindergarten to grade 12,real,\n"
        "County,,county name,text,\n"
        "Notes,,internal remarks,text,\n", encoding="utf-8")
    return database


def test_identifiers_are_written_as_sqlite_requires():
    assert sql_identifier("County") == "County"
    assert sql_identifier("Enrollment (K-12)") == '"Enrollment (K-12)"'
    assert sql_identifier('odd"name') == '"odd""name"'


def test_v13_notes_are_relevant_quoted_unqualified_and_use_stored_spelling(tmp_path):
    database = _database(tmp_path)
    text, meta = column_notes_v13(database, inspect_schema(database), "What is the enrollment in Alameda county?",
                                  ValueProfiler(tmp_path / "artifacts"))
    assert '- "Enrollment (K-12)" in table frpm: enrollment for kindergarten to grade 12' in text
    assert "- County in table frpm: county name | stored values matching the question: 'Alameda'" in text
    assert "Notes in table" not in text and "CDSCode" not in text
    assert "frpm." not in text
    assert meta["notes_version"] == "sql-v13-column-notes-v1" and meta["notes_truncated"] == 0


def test_v13_value_matches_require_whole_words(tmp_path):
    database = _database(tmp_path)
    text, _ = column_notes_v13(database, inspect_schema(database), "Which county is Fresnoville in?",
                               ValueProfiler(tmp_path / "artifacts"))
    assert "- County in table frpm: county name | examples:" in text
    assert "stored values matching the question" not in text
