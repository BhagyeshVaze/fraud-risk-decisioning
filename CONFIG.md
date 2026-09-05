# Configuration

Shared infrastructure settings for the Fraud Risk Decisioning System.

## AWS

| Setting | Value |
| --- | --- |
| S3 bucket | `fraud-risk-raw-bhagyesh-8471-ca` |
| Region | `ca-central-1` |

The bucket is in `ca-central-1` to match the region of the Snowflake account
and avoid cross-region reads. Note that the AWS CLI default region is
`us-east-1`, so pass `--region ca-central-1` explicitly when working with this
bucket.

The AWS account ID is deliberately not recorded here, because this repository
is public. It is kept locally in `SETUP_NOTES.md`, which is untracked.

## Bucket layout

```
s3://fraud-risk-raw-bhagyesh-8471-ca/
  raw/
    transactions/transactions.parquet
    identity/identity.parquet
```

The bucket is private. All four S3 Block Public Access settings are enabled,
object ownership is BucketOwnerEnforced, and the bucket ACL grants full
control to the owner only. These are the defaults applied at creation and
were not modified.

## Data

Source is the IEEE-CIS Fraud Detection competition on Kaggle. The raw CSVs and
the generated parquet files are excluded from version control by `.gitignore`.
Regenerate the parquet locally with:

```
python src/prepare_data.py
```

| Table | Rows | Columns |
| --- | --- | --- |
| transactions | 590,540 | 395 |
| identity | 144,233 | 41 |

The transactions table carries one derived column, `txn_day`, defined as
`TransactionDT // 86400` and stored as int64. The identity table has no
`TransactionDT` column, so it has no `txn_day`. The two tables are not joined.

## Credentials

No credentials are stored in this repository. Local locations, all gitignored
or outside the repo:

| Credential | Location |
| --- | --- |
| AWS | `~/.aws/credentials` |
| Kaggle | `~/.kaggle/access_token` |
