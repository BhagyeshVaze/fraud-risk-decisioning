"""Load the staged parquet files into typed raw tables in FRAUD.RAW.

Reads s3://fraud-risk-raw-bhagyesh-8471-ca/raw/ through the existing
FRAUD.RAW.s3_fraud_stage and creates FRAUD.RAW.RAW_TRANSACTIONS and
FRAUD.RAW.RAW_IDENTITY with one properly typed column per parquet column,
rather than a single VARIANT blob.

Column types come from Snowflake's INFER_SCHEMA reading the parquet footers,
fed into CREATE TABLE USING TEMPLATE. If INFER_SCHEMA is unavailable the
script falls back to deriving the same column list from the local parquet
schema with pyarrow. Neither path hand-lists columns.

The script is idempotent: tables are CREATE OR REPLACE, so rerunning gives the
same result and resets COPY load history along with the table. FRAUD_WH is
resumed at the start and suspended at the end so credits are not left burning.
The warehouse auto-suspend setting is never modified.

Connection settings come from .env at the repo root. See .env.
"""

import os
import sys
from pathlib import Path

import snowflake.connector
from dotenv import load_dotenv

STAGE = "FRAUD.RAW.s3_fraud_stage"
FILE_FORMAT = "FRAUD.RAW.parquet_ff"
SCHEMA = "FRAUD.RAW"

# Expected values, established from the local parquet in src/prepare_data.py.
# A mismatch means the load is not faithful to the source, so the script stops.
EXPECTED = {
    "RAW_TRANSACTIONS": {"rows": 590_540, "cols": 395},
    "RAW_IDENTITY": {"rows": 144_233, "cols": 41},
    "fraud_rate": 0.034990,
    "txn_day_min": 1,
    "txn_day_max": 182,
}

TARGETS = [
    {"table": "RAW_TRANSACTIONS", "stage_path": "transactions/",
     "local": "data/parquet/transactions.parquet"},
    {"table": "RAW_IDENTITY", "stage_path": "identity/",
     "local": "data/parquet/identity.parquet"},
]

# Arrow to Snowflake type mapping, used only by the fallback path.
ARROW_TO_SNOWFLAKE = {
    "int8": "NUMBER(38,0)", "int16": "NUMBER(38,0)",
    "int32": "NUMBER(38,0)", "int64": "NUMBER(38,0)",
    "float": "FLOAT", "double": "FLOAT", "halffloat": "FLOAT",
    "bool": "BOOLEAN",
    "string": "VARCHAR", "large_string": "VARCHAR",
}


def env(name, default=None):
    value = os.environ.get(name, default)
    if value is None or value == "CHANGE_ME":
        sys.exit(
            f"{name} is not set in .env (or is still CHANGE_ME). "
            "Fill in .env at the repo root before running this script."
        )
    return value


def connect():
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    return snowflake.connector.connect(
        account=env("SNOWFLAKE_ACCOUNT"),
        user=env("SNOWFLAKE_USER"),
        password=env("SNOWFLAKE_PASSWORD"),
        role=env("SNOWFLAKE_ROLE"),
        warehouse=env("SNOWFLAKE_WAREHOUSE"),
        database=env("SNOWFLAKE_DATABASE"),
        schema="RAW",
    )


def run(cur, sql, quiet=False):
    if not quiet:
        print(f"  > {' '.join(sql.split())[:110]}")
    cur.execute(sql)
    return cur


def create_table_from_inferred_schema(cur, table, stage_path):
    """Type the table from parquet metadata via INFER_SCHEMA."""
    run(cur, f"""
        CREATE OR REPLACE TABLE {SCHEMA}.{table}
        USING TEMPLATE (
            SELECT ARRAY_AGG(OBJECT_CONSTRUCT(*)) WITHIN GROUP (ORDER BY ORDER_ID)
            FROM TABLE(
                INFER_SCHEMA(
                    LOCATION => '@{STAGE}/{stage_path}',
                    FILE_FORMAT => '{FILE_FORMAT}'
                )
            )
        )
    """)


def create_table_from_local_parquet(cur, table, local_path):
    """Fallback: derive the same column list from the local parquet schema."""
    import pyarrow.parquet as pq

    schema = pq.ParquetFile(local_path).schema_arrow
    cols = []
    for field in schema:
        sf_type = ARROW_TO_SNOWFLAKE.get(str(field.type), "VARCHAR")
        cols.append(f'"{field.name}" {sf_type}')
    ddl = f"CREATE OR REPLACE TABLE {SCHEMA}.{table} (\n    " + ",\n    ".join(cols) + "\n)"
    print(f"  fallback: building {len(cols)} columns from {local_path}")
    run(cur, ddl, quiet=True)


def load(cur, target):
    table, stage_path = target["table"], target["stage_path"]
    print(f"\n[{table}]")
    try:
        create_table_from_inferred_schema(cur, table, stage_path)
    except snowflake.connector.errors.ProgrammingError as exc:
        print(f"  INFER_SCHEMA path failed ({exc.errno}), using local schema instead")
        create_table_from_local_parquet(cur, table, target["local"])

    run(cur, f"""
        COPY INTO {SCHEMA}.{table}
        FROM '@{STAGE}/{stage_path}'
        FILE_FORMAT = (FORMAT_NAME = '{FILE_FORMAT}')
        MATCH_BY_COLUMN_NAME = CASE_INSENSITIVE
    """)
    for row in cur.fetchall():
        print(f"  copied: {row[0]} status={row[1]} rows_loaded={row[2]}")


def column_map(cur, table):
    """Map upper-cased column name to the identifier as actually stored.

    INFER_SCHEMA preserves the parquet casing, so columns land as
    case-sensitive identifiers like "isFraud" and must be quoted.
    """
    cur.execute(f"""
        SELECT column_name FROM FRAUD.INFORMATION_SCHEMA.COLUMNS
        WHERE table_schema = 'RAW' AND table_name = '{table}'
    """)
    return {r[0].upper(): r[0] for r in cur.fetchall()}


def verify(cur):
    print("\n=== verification ===")
    failures = []

    cols = {}
    for target in TARGETS:
        table = target["table"]
        cur.execute(f"SELECT COUNT(*) FROM {SCHEMA}.{table}")
        rows = cur.fetchone()[0]
        cols[table] = column_map(cur, table)
        ncols = len(cols[table])

        exp = EXPECTED[table]
        ok_rows = rows == exp["rows"]
        ok_cols = ncols == exp["cols"]
        print(f"{table:<18} rows {rows:>9,} (expect {exp['rows']:>9,}) {'OK' if ok_rows else 'MISMATCH'}")
        print(f"{table:<18} cols {ncols:>9}   (expect {exp['cols']:>9})   {'OK' if ok_cols else 'MISMATCH'}")
        if not ok_rows:
            failures.append(f"{table} row count {rows} != {exp['rows']}")
        if not ok_cols:
            failures.append(f"{table} column count {ncols} != {exp['cols']}")

    txn = cols["RAW_TRANSACTIONS"]
    fraud_col, day_col = txn["ISFRAUD"], txn["TXN_DAY"]

    cur.execute(f'SELECT AVG("{fraud_col}") FROM {SCHEMA}.RAW_TRANSACTIONS')
    rate = float(cur.fetchone()[0])
    ok_rate = abs(rate - EXPECTED["fraud_rate"]) < 5e-7
    print(f"{'isFraud rate':<18} {rate:.6f}   (expect {EXPECTED['fraud_rate']:.6f})   "
          f"{'OK' if ok_rate else 'MISMATCH'}")
    if not ok_rate:
        failures.append(f"fraud rate {rate:.6f} != {EXPECTED['fraud_rate']:.6f}")

    cur.execute(f'SELECT MIN("{day_col}"), MAX("{day_col}") FROM {SCHEMA}.RAW_TRANSACTIONS')
    dmin, dmax = (int(v) for v in cur.fetchone())
    ok_day = dmin == EXPECTED["txn_day_min"] and dmax == EXPECTED["txn_day_max"]
    print(f"{'txn_day range':<18} {dmin} to {dmax}   "
          f"(expect {EXPECTED['txn_day_min']} to {EXPECTED['txn_day_max']})   "
          f"{'OK' if ok_day else 'MISMATCH'}")
    if not ok_day:
        failures.append(f"txn_day range {dmin}-{dmax} != "
                        f"{EXPECTED['txn_day_min']}-{EXPECTED['txn_day_max']}")

    return failures


def main():
    conn = connect()
    warehouse = os.environ["SNOWFLAKE_WAREHOUSE"]
    failures = []
    try:
        cur = conn.cursor()
        print(f"Resuming {warehouse} (auto-suspend setting left untouched)")
        run(cur, f"ALTER WAREHOUSE {warehouse} RESUME IF SUSPENDED")
        run(cur, f"USE WAREHOUSE {warehouse}")

        for target in TARGETS:
            load(cur, target)

        failures = verify(cur)
    finally:
        try:
            conn.cursor().execute(f"ALTER WAREHOUSE {warehouse} SUSPEND")
            print(f"\n{warehouse} suspended")
        except snowflake.connector.errors.ProgrammingError as exc:
            print(f"\nCould not suspend {warehouse}: {exc}")
        conn.close()

    if failures:
        print("\nFAILED, loaded data does not match the local source:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
