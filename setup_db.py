import os
from urllib.parse import urlparse
import pandas as pd
import requests
from sqlalchemy import create_engine, text
from config import DATABASE_URL as url
from config import (
    get_engine,
    select_db_from_args,
)

CSV_URL = (
    "https://raw.githubusercontent.com/onkarkasture/"
    "Retail_Orders_Data_Analysis/main/orders.csv"
)
CSV_PATH = os.path.join(os.path.dirname(__file__), "orders.csv")


def download_csv():
    if os.path.exists(CSV_PATH):
        print(f"[ETL] CSV already exists at {CSV_PATH}, skipping download.")
        return
    print(f"[ETL] Downloading orders.csv from GitHub...")
    resp = requests.get(CSV_URL, timeout=60)
    resp.raise_for_status()
    with open(CSV_PATH, "wb") as f:
        f.write(resp.content)
    print(f"[ETL] Saved {len(resp.content):,} bytes to {CSV_PATH}")


def transform(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = df.columns.str.lower().str.replace(" ", "_")

    df["discount"] = df["list_price"] * df["discount_percent"] / 100
    df["sale_price"] = df["list_price"] - df["discount"]
    df["profit"] = round(df["sale_price"] - df["cost_price"], 2)

    df.drop(columns=["list_price", "cost_price", "discount_percent"], inplace=True)
    df["order_date"] = pd.to_datetime(df["order_date"], format="%Y-%m-%d")

    return df


def _ensure_mysql_database():
    """Create the MySQL database if it doesn't exist."""
    
    if "mysql" not in url:
        return
    parsed = urlparse(url)
    db_name = parsed.path.lstrip("/")
    server_url = url.rsplit("/", 1)[0]
    engine = create_engine(server_url)
    with engine.connect() as conn:
        conn.execute(text(f"CREATE DATABASE IF NOT EXISTS `{db_name}`"))
        conn.commit()
    engine.dispose()
    print(f"[ETL] Ensured MySQL database '{db_name}' exists.")


def load(df: pd.DataFrame, table_name: str = "orders"):
    _ensure_mysql_database()
    engine = get_engine()

    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    print("[ETL] Database connection OK.")

    df.to_sql(
        name=table_name,
        con=engine,
        if_exists="replace",
        index=False,
    )
    print(f"[ETL] Loaded {len(df):,} rows into `{table_name}` table.")


def main():
    select_db_from_args("ETL: load orders data")

    download_csv()

    print("[ETL] Reading CSV...")
    df = pd.read_csv(CSV_PATH, na_values=["Not Available", "unknown"])
    print(f"[ETL] Raw shape: {df.shape}")

    print("[ETL] Transforming...")
    df = transform(df)
    print(f"[ETL] Transformed shape: {df.shape}")
    print(f"[ETL] Columns: {list(df.columns)}")

    print("[ETL] Loading into database...")
    load(df)

    print("[ETL] Done! Database is ready.")


if __name__ == "__main__":
    main()
