"""Generate the dbt staging model SQL from the parquet schemas.

stg_transactions has 395 columns and stg_identity has 41. Hand-writing those
select lists would be unreadable and would drift from the source, so this
script emits them. It reads the local parquet files, so it needs no warehouse.

Rerun after any schema change:
    python src/generate_staging_models.py
"""

from pathlib import Path

import pyarrow.parquet as pq

MODELS = Path("fraud_dbt/models/staging")

# Names that do not lowercase cleanly into snake_case.
RENAMES = {
    "TransactionID": "transaction_id",
    "isFraud": "is_fraud",
    "TransactionDT": "transaction_dt",
    "TransactionAmt": "transaction_amt",
    "ProductCD": "product_cd",
    "P_emaildomain": "p_emaildomain",
    "R_emaildomain": "r_emaildomain",
    "DeviceType": "device_type",
    "DeviceInfo": "device_info",
}

# Columns needing something other than the default type for their arrow type.
TYPE_OVERRIDES = {
    "transaction_id": "NUMBER(38,0)",
    "is_fraud": "NUMBER(1,0)",
    "transaction_dt": "NUMBER(38,0)",
    "transaction_amt": "NUMBER(18,3)",
    "txn_day": "NUMBER(38,0)",
}

ARROW_DEFAULT = {
    "int64": "NUMBER(38,0)", "int32": "NUMBER(38,0)",
    "double": "FLOAT", "float": "FLOAT",
    "bool": "BOOLEAN",
}


def snake(name):
    return RENAMES.get(name, name.lower())


def sql_type(alias, arrow_type):
    if alias in TYPE_OVERRIDES:
        return TYPE_OVERRIDES[alias]
    return ARROW_DEFAULT.get(str(arrow_type), "VARCHAR")


def build(parquet_path, source_table, model_name, header, extra_selects=()):
    schema = pq.ParquetFile(parquet_path).schema_arrow
    lines = []
    for field in schema:
        alias = snake(field.name)
        # Source columns are mixed-case quoted identifiers in Snowflake
        # (INFER_SCHEMA preserved the parquet casing), so they must be quoted.
        lines.append(f'    CAST("{field.name}" AS {sql_type(alias, field.type)}) AS {alias}')
    lines.extend(f"    {s}" for s in extra_selects)

    body = (
        f"{header}\n\n"
        "SELECT\n"
        + ",\n".join(lines)
        + f"\nFROM {{{{ source('raw', '{source_table}') }}}}\n"
    )
    out = MODELS / f"{model_name}.sql"
    out.write_text(body)
    print(f"{out}: {len(lines)} columns")


TXN_HEADER = """{{ config(materialized='view') }}

-- Typed, snake_cased passthrough of FRAUD.RAW.RAW_TRANSACTIONS.
-- No filtering, no business logic, no joins.
--
-- txn_hour and txn_dow are derived from TransactionDT. TransactionDT is
-- seconds from an unstated reference point rather than a real timestamp, so
-- both are relative: txn_dow groups days seven apart but which one is Monday
-- is unknowable from the data."""

IDY_HEADER = """{{ config(materialized='view') }}

-- Typed, snake_cased passthrough of FRAUD.RAW.RAW_IDENTITY.
-- No filtering, no business logic, no joins."""


def main():
    MODELS.mkdir(parents=True, exist_ok=True)
    build(
        "data/parquet/transactions.parquet", "RAW_TRANSACTIONS", "stg_transactions",
        TXN_HEADER,
        extra_selects=(
            'CAST(FLOOR(MOD("TransactionDT", 86400) / 3600) AS NUMBER(2,0)) AS txn_hour',
            'CAST(MOD("txn_day", 7) AS NUMBER(1,0)) AS txn_dow',
        ),
    )
    build(
        "data/parquet/identity.parquet", "RAW_IDENTITY", "stg_identity",
        IDY_HEADER,
    )


if __name__ == "__main__":
    main()
