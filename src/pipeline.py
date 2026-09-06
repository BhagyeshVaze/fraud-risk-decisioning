"""Raw CSV -> parquet -> Snowflake, plus the generated staging SQL.

Usage: python src/pipeline.py {prepare|sql|load}
"""

import json
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
import core

RAW, OUT = Path("data/raw"), core.DATA
CHUNK, SECONDS_PER_DAY = 50_000, 86_400
STAGE, FILE_FORMAT, SCHEMA = "FRAUD.RAW.s3_fraud_stage", "FRAUD.RAW.parquet_ff", "FRAUD.RAW"
MODELS_DIR = Path("fraud_dbt/models/staging")

EXPECTED = {"RAW_TRANSACTIONS": (590_540, 395), "RAW_IDENTITY": (144_233, 41),
            "fraud_rate": 0.034990, "txn_day": (1, 182)}

RENAMES = {"TransactionID": "transaction_id", "isFraud": "is_fraud",
           "TransactionDT": "transaction_dt", "TransactionAmt": "transaction_amt",
           "ProductCD": "product_cd", "P_emaildomain": "p_emaildomain",
           "R_emaildomain": "r_emaildomain", "DeviceType": "device_type",
           "DeviceInfo": "device_info"}
TYPE_OVERRIDES = {"transaction_id": "NUMBER(38,0)", "is_fraud": "NUMBER(1,0)",
                  "transaction_dt": "NUMBER(38,0)", "transaction_amt": "NUMBER(18,3)",
                  "txn_day": "NUMBER(38,0)"}
ARROW_SQL = {"int64": "NUMBER(38,0)", "int32": "NUMBER(38,0)", "double": "FLOAT",
             "float": "FLOAT", "bool": "BOOLEAN"}


# --------------------------------------------------------------------------- #
# prepare: CSV -> parquet
# --------------------------------------------------------------------------- #

def _unified_dtypes(csv_path):
    """One dtype per column valid for every chunk, so the parquet schema is stable."""
    import numpy as np
    resolved = {}
    for chunk in pd.read_csv(csv_path, chunksize=CHUNK, low_memory=False):
        for col, dtype in chunk.dtypes.items():
            prior = resolved.get(col)
            if prior is None or prior == dtype:
                resolved[col] = dtype
            elif prior == object or dtype == object:
                resolved[col] = object
            else:
                resolved[col] = pd.api.types.pandas_dtype(np.promote_types(prior, dtype))
    return resolved


def _csv_to_parquet(csv_path, out_path, add_txn_day):
    dtypes = _unified_dtypes(csv_path)
    writer, schema, rows = None, None, 0
    try:
        for chunk in pd.read_csv(csv_path, chunksize=CHUNK, low_memory=False):
            chunk = chunk.astype(dtypes)
            if add_txn_day:
                chunk["txn_day"] = (chunk["TransactionDT"] // SECONDS_PER_DAY).astype("int64")
            tbl = pa.Table.from_pandas(chunk, schema=schema, preserve_index=False)
            if writer is None:
                schema = tbl.schema
                writer = pq.ParquetWriter(out_path, schema, compression="snappy")
            writer.write_table(tbl)
            rows += len(chunk)
    finally:
        if writer:
            writer.close()
    return rows


def prepare():
    """Chunked because 683 MB across 394 columns is a large share of 8 GB."""
    OUT.mkdir(parents=True, exist_ok=True)
    _csv_to_parquet(RAW / "train_transaction.csv", OUT / "transactions.parquet", True)
    # train_identity.csv has no TransactionDT, so no txn_day can be derived.
    _csv_to_parquet(RAW / "train_identity.csv", OUT / "identity.parquet", False)
    for name in ("transactions", "identity"):
        md = pq.ParquetFile(OUT / f"{name}.parquet").metadata
        print(f"{name:<14} {md.num_rows:>9,} rows x {md.num_columns} cols")


# --------------------------------------------------------------------------- #
# sql: generate the staging models
# --------------------------------------------------------------------------- #

def _snake(name):
    return RENAMES.get(name, name.lower())


def _build(parquet_path, source_table, model_name, header, extra=()):
    schema = pq.ParquetFile(parquet_path).schema_arrow
    lines = []
    for f in schema:
        alias = _snake(f.name)
        sql_type = TYPE_OVERRIDES.get(alias, ARROW_SQL.get(str(f.type), "VARCHAR"))
        # Source columns are mixed-case quoted identifiers in Snowflake.
        lines.append(f'    CAST("{f.name}" AS {sql_type}) AS {alias}')
    lines += [f"    {s}" for s in extra]
    body = (f"{header}\n\nSELECT\n" + ",\n".join(lines)
            + f"\nFROM {{{{ source('raw', '{source_table}') }}}}\n")
    (MODELS_DIR / f"{model_name}.sql").write_text(body)
    print(f"{model_name}.sql: {len(lines)} columns")


def sql():
    """397 and 41 columns are too many to hand-write, so emit them."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    _build(OUT / "transactions.parquet", "RAW_TRANSACTIONS", "stg_transactions",
           "{{ config(materialized='view') }}\n\n"
           "-- Typed, snake_cased passthrough of FRAUD.RAW.RAW_TRANSACTIONS.\n"
           "-- txn_hour and txn_dow are relative: TransactionDT is seconds from an\n"
           "-- unstated origin, so no real calendar date is recoverable.",
           ('CAST(FLOOR(MOD("TransactionDT", 86400) / 3600) AS NUMBER(2,0)) AS txn_hour',
            'CAST(MOD("txn_day", 7) AS NUMBER(1,0)) AS txn_dow'))
    _build(OUT / "identity.parquet", "RAW_IDENTITY", "stg_identity",
           "{{ config(materialized='view') }}\n\n"
           "-- Typed, snake_cased passthrough of FRAUD.RAW.RAW_IDENTITY.")


# --------------------------------------------------------------------------- #
# load: parquet -> Snowflake
# --------------------------------------------------------------------------- #

TARGETS = [("RAW_TRANSACTIONS", "transactions/", OUT / "transactions.parquet"),
           ("RAW_IDENTITY", "identity/", OUT / "identity.parquet")]


def _create_from_local(cur, table, local_path):
    """Fallback if INFER_SCHEMA is unavailable: same column list, from pyarrow."""
    arrow_sql = {"int64": "NUMBER(38,0)", "double": "FLOAT", "bool": "BOOLEAN"}
    cols = [f'"{f.name}" {arrow_sql.get(str(f.type), "VARCHAR")}'
            for f in pq.ParquetFile(local_path).schema_arrow]
    cur.execute(f"CREATE OR REPLACE TABLE {SCHEMA}.{table} (\n  "
                + ",\n  ".join(cols) + "\n)")


def load():
    """Idempotent: CREATE OR REPLACE also resets COPY load history."""
    import snowflake.connector
    conn = core.connect(schema="RAW")
    cur = conn.cursor()
    try:
        for table, path, local in TARGETS:
            print(f"\n[{table}]")
            try:
                cur.execute(f"""
                    CREATE OR REPLACE TABLE {SCHEMA}.{table} USING TEMPLATE (
                        SELECT ARRAY_AGG(OBJECT_CONSTRUCT(*)) WITHIN GROUP (ORDER BY ORDER_ID)
                        FROM TABLE(INFER_SCHEMA(LOCATION => '@{STAGE}/{path}',
                                                FILE_FORMAT => '{FILE_FORMAT}')))""")
            except snowflake.connector.errors.ProgrammingError as e:
                print(f"  INFER_SCHEMA failed ({e.errno}), using the local schema")
                _create_from_local(cur, table, local)
            cur.execute(f"""
                COPY INTO {SCHEMA}.{table} FROM '@{STAGE}/{path}'
                FILE_FORMAT = (FORMAT_NAME = '{FILE_FORMAT}')
                MATCH_BY_COLUMN_NAME = CASE_INSENSITIVE""")
            for row in cur.fetchall():
                print(f"  {row[0]} status={row[1]} rows={row[2]}")

        failures = []
        for table, exp in (("RAW_TRANSACTIONS", EXPECTED["RAW_TRANSACTIONS"]),
                           ("RAW_IDENTITY", EXPECTED["RAW_IDENTITY"])):
            cur.execute(f"SELECT COUNT(*) FROM {SCHEMA}.{table}")
            n = cur.fetchone()[0]
            cur.execute(f"""SELECT COUNT(*) FROM FRAUD.INFORMATION_SCHEMA.COLUMNS
                            WHERE table_schema='RAW' AND table_name='{table}'""")
            k = cur.fetchone()[0]
            print(f"{table}: {n:,} rows x {k} cols (expect {exp[0]:,} x {exp[1]})")
            if (n, k) != exp:
                failures.append(f"{table} {n}x{k} != {exp}")
        if failures:
            sys.exit("FAILED: " + "; ".join(failures))
        print("\nAll checks passed.")
    finally:
        conn.close()
        core.suspend_warehouse()


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "prepare"
    {"prepare": prepare, "sql": sql, "load": load}[cmd]()
