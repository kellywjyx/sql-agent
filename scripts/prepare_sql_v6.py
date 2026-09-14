from pathlib import Path

from sql_agent.v6_data import prepare


if __name__ == "__main__":
    print(prepare((Path(__file__).resolve().parents[2] / "artifacts").resolve()))
