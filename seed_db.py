import sqlite3
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DB_DIR = BASE_DIR / "backend"
DB_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DB_DIR / "transactions.db"

parquet_path = BASE_DIR / "transactions.parquet"
if not parquet_path.exists():
    print(f"Error: {parquet_path} not found.")
    exit(1)

print(f"Loading {parquet_path}...")
df = pd.read_parquet(parquet_path)

# Map column names to lowercase snake_case
col_map = {c: c.lower() for c in df.columns}
df = df.rename(columns=col_map)

# Add is_suspicious boolean/int
if "suspicious_transaction" in df.columns:
    df["is_suspicious"] = df["suspicious_transaction"].apply(lambda x: 1 if str(x).strip().lower() in ["yes", "1", "true"] else 0)
else:
    df["is_suspicious"] = 0

# Add fraud_probability if not present
if "fraud_probability" not in df.columns:
    if "fraud_risk_score" in df.columns:
        df["fraud_probability"] = (df["fraud_risk_score"] / 100.0).round(4)
    else:
        df["fraud_probability"] = 0.0

if "top_reasons" not in df.columns:
    df["top_reasons"] = "[]"

print(f"Writing {len(df)} records to SQLite database: {DB_PATH}...")
conn = sqlite3.connect(DB_PATH)
df.to_sql("transactions", conn, if_exists="replace", index=False)
conn.commit()

# Verify
cursor = conn.cursor()
count = cursor.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]
print(f"Database successfully seeded! Total transactions in DB: {count:,}")
conn.close()
