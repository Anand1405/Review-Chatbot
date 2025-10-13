import argparse
import sqlite3
from pathlib import Path
from typing import Union

import pandas as pd


def ingest_data(file_path: Union[str, Path], db_path: Union[str, Path]) -> None:
    """Read review data from ``file_path`` and write it into ``db_path``."""

    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Data file {file_path} does not exist")

    # Load the data using pandas.  Pandas can transparently read CSV or JSON.
    if file_path.suffix.lower() == ".csv":
        df = pd.read_csv(file_path)
    elif file_path.suffix.lower() in {".json", ".jsonl"}:
        df = pd.read_json(file_path, lines=file_path.suffix.lower() == ".jsonl")
    else:
        raise ValueError(
            "Unsupported file type. Please provide a .csv, .json or .jsonl file."
        )

    expected_cols = {
        "id": "id",
        "rating": "rating",
        "text": "text",
        "date": "date",
        "version": "version",
        "platform": "platform",
    }
    df = df.rename(columns={c: expected_cols[c] for c in df.columns if c in expected_cols})

    missing_cols = [col for col in expected_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Input data is missing required columns: {', '.join(missing_cols)}"
        )

    # Deduplicate based on all identifying fields except the auto‑generated id.
    df = df.drop_duplicates(subset=["rating", "text", "date", "version", "platform"])

    # Ensure correct dtypes.  Pandas will coerce int and string types
    df["rating"] = df["rating"].astype(int)
    df["text"] = df["text"].astype(str)
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df["version"] = df["version"].astype(str)
    df["platform"] = df["platform"].astype(str)

    # Connect to SQLite and write the data
    conn = sqlite3.connect(db_path)
    try:
        # Replace the existing table if present
        df.to_sql("reviews", conn, if_exists="replace", index=False)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_reviews_date ON reviews (date);"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_reviews_rating ON reviews (rating);"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_reviews_platform ON reviews (platform);"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_reviews_version ON reviews (version);"
        )
    finally:
        conn.commit()
        conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest Liquide app review data")
    parser.add_argument(
        "--data",
        required=True,
        help="Path to the CSV/JSON file containing review data",
    )
    parser.add_argument(
        "--db",
        default="reviews.db",
        help="Path to the SQLite database file to create or overwrite",
    )
    args = parser.parse_args()
    ingest_data(args.data, args.db)