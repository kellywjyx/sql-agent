import sqlite3

from sql_agent.schema import inspect_schema
from sql_agent.v12_notes import column_notes, load_descriptions
from sql_agent.value_profiles import ValueProfiler


def _database(tmp_path):
    folder = tmp_path / "shop"
    (folder / "database_description").mkdir(parents=True)
    database = folder / "shop.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE employee (id INTEGER PRIMARY KEY, name TEXT, salary TEXT, state TEXT)")
        connection.executemany("INSERT INTO employee (name, salary, state) VALUES (?, ?, ?)",
                               [("Ann", "US$57,500.00", "WI"), ("Bo", "US$100,000.00", "CA")])
    (folder / "database_description" / "employee.csv").write_text(
        "original_column_name,column_name,column_description,data_format,value_description\n"
        "salary,salary,yearly salary,text,\"format: US$ with thousands separators\"\n"
        "state,state,state abbreviation,text,\n", encoding="utf-8")
    return database


def test_descriptions_include_value_descriptions(tmp_path):
    notes = load_descriptions(_database(tmp_path))
    assert notes[("employee", "salary")] == "yearly salary format: US$ with thousands separators"
    assert notes[("employee", "state")] == "state abbreviation"


def test_column_notes_ground_stored_values_without_reference_sql(tmp_path):
    database = _database(tmp_path)
    text, meta = column_notes(database, inspect_schema(database), "What is Ann's salary in Wisconsin?",
                              ValueProfiler(tmp_path / "artifacts"))
    assert text.startswith("### Column notes")
    assert "- employee.salary: yearly salary" in text
    assert "'US$57,500.00'" in text
    assert "- employee.state: state abbreviation | examples: 'WI', 'CA'" in text
    assert meta["notes_version"] == "sql-v12-column-notes-v1" and meta["notes_columns"] >= 2


def test_column_notes_respect_the_byte_budget(tmp_path):
    database = _database(tmp_path)
    text, meta = column_notes(database, inspect_schema(database), "salary", ValueProfiler(tmp_path / "artifacts"),
                              byte_budget=80)
    assert meta["notes_bytes"] <= 80
    assert meta["notes_columns"] < meta["notes_candidates"]
    assert all(line == "### Column notes" or line.startswith("- employee.") for line in text.splitlines())
