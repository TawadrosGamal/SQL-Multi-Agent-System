from __future__ import annotations
import random
import pandas as pd
from sqlalchemy import text
from validators.schema_catalog import SchemaCatalog
from config import (
    get_engine,
    select_db_from_args,
    get_fk_graph,
)
import config

FIRST_NAMES = [
    "James", "Mary", "Robert", "Patricia", "John", "Jennifer", "Michael",
    "Linda", "David", "Elizabeth", "William", "Barbara", "Richard", "Susan",
    "Joseph", "Jessica", "Thomas", "Sarah", "Charles", "Karen", "Daniel",
    "Lisa", "Matthew", "Nancy", "Anthony", "Betty", "Mark", "Margaret",
    "Donald", "Sandra", "Steven", "Ashley", "Andrew", "Dorothy", "Paul",
    "Kimberly", "Joshua", "Emily", "Kenneth", "Donna", "Kevin", "Michelle",
    "Brian", "Carol", "George", "Amanda", "Timothy", "Melissa", "Ronald",
    "Deborah",
]

LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
    "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez",
    "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin",
    "Lee", "Perez", "Thompson", "White", "Harris", "Sanchez", "Clark",
    "Ramirez", "Lewis", "Robinson", "Walker", "Young", "Allen", "King",
    "Wright", "Scott", "Torres", "Nguyen", "Hill", "Flores", "Green",
    "Adams", "Nelson", "Baker", "Hall", "Rivera", "Campbell", "Mitchell",
    "Carter", "Roberts",
]


def build_products(engine) -> pd.DataFrame:
    """Extract distinct products from orders."""
    print("[SETUP] Building products table...")
    df = pd.read_sql("SELECT DISTINCT product_id, category, sub_category FROM orders", engine)

    df["product_name"] = df.apply(
        lambda r: f"{r['sub_category']} - {r['product_id'][-4:]}", axis=1
    )
    df = df[["product_id", "product_name", "category", "sub_category"]]
    print(f"[SETUP]   {len(df)} distinct products found.")
    return df


def build_customers(engine) -> tuple[pd.DataFrame, dict[tuple, int]]:
    """Generate synthetic customers from segment/region combos in orders.

    Returns the customers DataFrame and a mapping of
    (segment, region) -> customer_id for assigning to orders.
    """
    print("[SETUP] Building customers table...")
    combos = pd.read_sql(
        "SELECT DISTINCT segment, region FROM orders ORDER BY segment, region",
        engine,
    )

    rng = random.Random(42)
    rows = []
    combo_to_id: dict[tuple, int] = {}
    cid = 1

    for _, row in combos.iterrows():
        seg, reg = row["segment"], row["region"]
        n_customers = rng.randint(3, 6)
        ids_for_combo = []
        for _ in range(n_customers):
            first = rng.choice(FIRST_NAMES)
            last = rng.choice(LAST_NAMES)
            name = f"{first} {last}"
            email = f"{first.lower()}.{last.lower()}@example.com"
            rows.append({
                "customer_id": cid,
                "customer_name": name,
                "email": email,
                "segment": seg,
                "region": reg,
            })
            ids_for_combo.append(cid)
            cid += 1
        combo_to_id[(seg, reg)] = ids_for_combo

    customers_df = pd.DataFrame(rows)
    print(f"[SETUP]   {len(customers_df)} synthetic customers generated.")
    return customers_df, combo_to_id


def assign_customers_to_orders(engine, combo_to_id: dict[tuple, list[int]]):
    """Add customer_id column to orders, assigning based on segment+region."""
    print("[SETUP] Assigning customer_id to orders...")
    orders_df = pd.read_sql("SELECT * FROM orders", engine)

    rng = random.Random(42)

    def pick_customer(row):
        key = (row["segment"], row["region"])
        candidates = combo_to_id.get(key, [1])
        return rng.choice(candidates)

    orders_df["customer_id"] = orders_df.apply(pick_customer, axis=1)
    print(f"[SETUP]   Assigned customer_id to {len(orders_df)} orders.")
    return orders_df


def create_tables(engine, products_df, customers_df, orders_df):
    """Drop and recreate all three tables with explicit FK constraints."""
    print("[SETUP] Creating tables with FK constraints...")

    is_mysql = config.DB_DIALECT == "mysql"
    str_id = "VARCHAR(50)" if is_mysql else "TEXT"
    str_col = "VARCHAR(255)" if is_mysql else "TEXT"
    real_type = "DOUBLE" if is_mysql else "REAL"
    dt_type = "DATETIME" if is_mysql else "TIMESTAMP"

    with engine.connect() as conn:
        if not is_mysql:
            conn.execute(text("PRAGMA foreign_keys = OFF"))
        conn.execute(text("DROP TABLE IF EXISTS orders"))
        conn.execute(text("DROP TABLE IF EXISTS products"))
        conn.execute(text("DROP TABLE IF EXISTS customers"))

        conn.execute(text(f"""
            CREATE TABLE products (
                product_id {str_id} PRIMARY KEY,
                product_name {str_col},
                category {str_col},
                sub_category {str_col}
            )
        """))

        conn.execute(text(f"""
            CREATE TABLE customers (
                customer_id INTEGER PRIMARY KEY,
                customer_name {str_col},
                email {str_col},
                segment {str_col},
                region {str_col}
            )
        """))

        order_cols = orders_df.columns.tolist()
        col_defs = []
        for col in order_cols:
            if col == "order_id":
                col_defs.append("order_id INTEGER")
            elif col == "customer_id":
                col_defs.append("customer_id INTEGER")
            elif col == "product_id":
                col_defs.append(f"product_id {str_id}")
            elif col == "postal_code":
                col_defs.append("postal_code INTEGER")
            elif col == "quantity":
                col_defs.append("quantity INTEGER")
            elif col in ("discount", "sale_price", "profit"):
                col_defs.append(f"{col} {real_type}")
            elif col == "order_date":
                col_defs.append(f"order_date {dt_type}")
            else:
                col_defs.append(f"{col} {str_col}")

        col_defs.append("FOREIGN KEY (product_id) REFERENCES products(product_id)")
        col_defs.append("FOREIGN KEY (customer_id) REFERENCES customers(customer_id)")

        ddl = f"CREATE TABLE orders ({', '.join(col_defs)})"
        conn.execute(text(ddl))
        conn.commit()

    products_df.to_sql("products", engine, if_exists="append", index=False)
    customers_df.to_sql("customers", engine, if_exists="append", index=False)
    orders_df.to_sql("orders", engine, if_exists="append", index=False)

    with engine.connect() as conn:
        if config.DB_DIALECT == "sqlite":
            conn.execute(text("PRAGMA foreign_keys = ON"))
        conn.commit()

    print("[SETUP]   Tables created: products, customers, orders")

    with engine.connect() as conn:
        for tbl in ["products", "customers", "orders"]:
            row = conn.execute(text(f"SELECT COUNT(*) FROM {tbl}")).fetchone()
            print(f"[SETUP]   {tbl}: {row[0]} rows")


def rebuild_schema_catalog():
    """Force rebuild of the ChromaDB schema catalog."""
    print("[SETUP] Rebuilding schema catalog...")

    catalog = SchemaCatalog()
    catalog.rebuild()
    print("[SETUP]   Schema catalog rebuilt.")


def verify_fk_graph():
    """Print the FK graph to verify relationships are visible."""
    
    fk = get_fk_graph()
    print("[SETUP] FK Graph:")
    for table, edges in fk.items():
        if edges:
            for e in edges:
                print(f"[SETUP]   {table}.{e['column']} -> {e['references']}")
        else:
            print(f"[SETUP]   {table}: (no FKs)")


def main():
    
    select_db_from_args("Create products + customers tables")

    engine = get_engine()

    products_df = build_products(engine)
    customers_df, combo_to_id = build_customers(engine)
    orders_df = assign_customers_to_orders(engine, combo_to_id)

    create_tables(engine, products_df, customers_df, orders_df)
    rebuild_schema_catalog()
    verify_fk_graph()

    print("\n[SETUP] Done! Database now has 3 tables: orders, products, customers")


if __name__ == "__main__":
    main()
