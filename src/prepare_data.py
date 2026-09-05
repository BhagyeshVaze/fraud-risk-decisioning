"""Convert the IEEE-CIS fraud CSVs to parquet.

Reads data/raw/train_transaction.csv and data/raw/train_identity.csv, adds a
txn_day column to the transactions table, and writes both to data/parquet/.

The two tables are written separately and are not joined. No columns are
dropped and no features beyond txn_day are created.

The transactions file is 683 MB across 394 columns, which is a large fraction
of an 8 GB machine, so it is read in chunks and streamed to parquet rather
than being materialised in full. Chunking needs a stable schema: pandas infers
dtypes per chunk, so a column that is all integers in chunk 1 and has a null
in chunk 2 would be int64 then float64 and the parquet writer would reject the
second batch. Pass 1 therefore scans the file to settle on one dtype per
column, and pass 2 applies it while writing.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

RAW = Path("data/raw")
OUT = Path("data/parquet")
CHUNK_ROWS = 50_000
SECONDS_PER_DAY = 86_400


def unified_dtypes(csv_path, chunk_rows=CHUNK_ROWS):
    """Return one dtype per column that is valid for every chunk in the file."""
    resolved = {}
    for chunk in pd.read_csv(csv_path, chunksize=chunk_rows, low_memory=False):
        for col, dtype in chunk.dtypes.items():
            prior = resolved.get(col)
            if prior is None:
                resolved[col] = dtype
            elif prior != dtype:
                # object wins over everything; otherwise promote numerically
                # (int64 + float64 -> float64).
                if prior == object or dtype == object:
                    resolved[col] = object
                else:
                    resolved[col] = pd.api.types.pandas_dtype(
                        np.promote_types(prior, dtype)
                    )
    return resolved


def csv_to_parquet(csv_path, out_path, add_txn_day, chunk_rows=CHUNK_ROWS):
    """Stream a CSV to parquet, optionally adding txn_day."""
    dtypes = unified_dtypes(csv_path, chunk_rows)

    writer = None
    schema = None
    rows = 0
    try:
        for chunk in pd.read_csv(csv_path, chunksize=chunk_rows, low_memory=False):
            chunk = chunk.astype(dtypes)
            if add_txn_day:
                chunk["txn_day"] = (chunk["TransactionDT"] // SECONDS_PER_DAY).astype("int64")

            table = pa.Table.from_pandas(chunk, schema=schema, preserve_index=False)
            if writer is None:
                schema = table.schema
                writer = pq.ParquetWriter(out_path, schema, compression="snappy")
            writer.write_table(table)
            rows += len(chunk)
    finally:
        if writer is not None:
            writer.close()
    return rows


def null_fractions(parquet_path, top_n=10):
    """Highest-null columns, read from parquet footer metadata, not the data."""
    md = pq.ParquetFile(parquet_path).metadata
    nulls = {}
    for rg in range(md.num_row_groups):
        for c in range(md.num_columns):
            col = md.row_group(rg).column(c)
            name = col.path_in_schema
            nulls[name] = nulls.get(name, 0) + col.statistics.null_count
    total = md.num_rows
    ranked = sorted(nulls.items(), key=lambda kv: kv[1], reverse=True)
    return [(n, v / total) for n, v in ranked[:top_n]], total


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    txn_out = OUT / "transactions.parquet"
    idy_out = OUT / "identity.parquet"

    print("Writing parquet...")
    csv_to_parquet(RAW / "train_transaction.csv", txn_out, add_txn_day=True)
    # train_identity.csv has no TransactionDT column, so txn_day cannot be
    # derived for it. It is left off rather than filled with a placeholder.
    csv_to_parquet(RAW / "train_identity.csv", idy_out, add_txn_day=False)

    txn_md = pq.ParquetFile(txn_out).metadata
    idy_md = pq.ParquetFile(idy_out).metadata

    print("\n=== shape ===")
    print(f"transactions : {txn_md.num_rows:,} rows x {txn_md.num_columns} cols")
    print(f"identity     : {idy_md.num_rows:,} rows x {idy_md.num_columns} cols")

    fraud = pq.read_table(txn_out, columns=["isFraud"]).column("isFraud").to_pandas()
    print("\n=== isFraud ===")
    print(f"rate      : {fraud.mean():.6f}  ({int(fraud.sum()):,} of {len(fraud):,})")

    day = pq.read_table(txn_out, columns=["txn_day"]).column("txn_day").to_pandas()
    print("\n=== txn_day ===")
    print(f"min       : {day.min()}")
    print(f"max       : {day.max()}")
    print(f"span      : {day.max() - day.min() + 1} distinct days possible, {day.nunique()} present")

    amt = pq.read_table(txn_out, columns=["TransactionAmt"]).column("TransactionAmt").to_pandas()
    print("\n=== TransactionAmt.describe() ===")
    print(amt.describe().to_string())

    txn_ids = pq.read_table(txn_out, columns=["TransactionID"]).column("TransactionID").to_pandas()
    idy_ids = pq.read_table(idy_out, columns=["TransactionID"]).column("TransactionID").to_pandas()
    matched = txn_ids.isin(set(idy_ids)).sum()
    print("\n=== identity coverage ===")
    print(f"transaction rows with a matching identity row : {matched:,} "
          f"({matched / len(txn_ids):.4%})")
    print(f"identity rows total                           : {len(idy_ids):,}")
    orphans = len(set(idy_ids) - set(txn_ids))
    print(f"identity rows with no matching transaction    : {orphans:,}")

    for label, path in (("transactions", txn_out), ("identity", idy_out)):
        ranked, total = null_fractions(path)
        print(f"\n=== top 10 null fraction, {label} ===")
        for name, frac in ranked:
            print(f"{name:<20} {frac:.4%}")


if __name__ == "__main__":
    main()
