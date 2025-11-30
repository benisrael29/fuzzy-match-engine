# Configuration Reference

This document provides a comprehensive reference for all configuration options available in the Fuzzy Matching Engine.

## Table of Contents

- [Basic Configuration](#basic-configuration)
- [Data Sources](#data-sources)
- [Output Options](#output-options)
- [MySQL Credentials](#mysql-credentials)
- [S3 Credentials](#s3-credentials)
- [Environment Variables](#environment-variables)
- [Match Configuration](#match-configuration)
- [Clustering Configuration](#clustering-configuration)
- [Complete Examples](#complete-examples)

## Basic Configuration

Every configuration file requires at minimum `source1` and `output` fields. The `mode` field determines whether to perform matching (between two sources) or clustering (within a single source).

### Operation Modes

The engine supports two operation modes:

- **matching** (default): Matches rows between two data sources (`source1` and `source2`)
- **clustering**: Finds duplicates and groups similar records within a single data source (`source1` only)

### Required Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `mode` | string | No | Operation mode: `"matching"` (default) or `"clustering"` |
| `source1` | string or object | Yes | First data source (CSV file, S3 URL, or MySQL table) |
| `source2` | string or object | Yes* | Second data source (required for matching mode, not used in clustering) |
| `output` | string or object | Yes | Output destination (CSV file, S3 URL, or MySQL table) |

*`source2` is required when `mode` is `"matching"` (or omitted, as matching is the default). It is not used and should be omitted when `mode` is `"clustering"`.

## Data Sources

### CSV Files

Specify a local CSV file path:

```json
{
  "source1": "data/master.csv",
  "source2": "data/new_records.csv"
}
```

### S3 Files

Specify an S3 URL (requires `s3_credentials` or IAM role):

```json
{
  "source1": "s3://my-bucket/data/master.csv",
  "source2": "s3://my-bucket/data/new_records.csv"
}
```

### MySQL Tables

#### Option 1: Table Name with Global Credentials

```json
{
  "source1": "customers",
  "source2": "leads",
  "mysql_credentials": {
    "host": "localhost",
    "user": "myuser",
    "password": "mypassword",
    "database": "mydatabase"
  }
}
```

#### Option 2: Inline Credentials

```json
{
  "source1": {
    "table": "customers",
    "host": "localhost",
    "user": "myuser",
    "password": "mypassword",
    "database": "mydatabase"
  },
  "source2": {
    "table": "leads",
    "host": "localhost",
    "user": "myuser",
    "password": "mypassword",
    "database": "mydatabase"
  }
}
```

## Output Options

### CSV File

Write results to a local CSV file:

```json
{
  "output": "results/matches.csv"
}
```

### S3

Write results to an S3 bucket:

```json
{
  "output": "s3://my-bucket/results/matches.csv"
}
```

**Note:** If `s3_credentials` are not provided, the system will automatically use IAM role credentials.

### MySQL Table

#### Option 1: Table Name with Global Credentials

```json
{
  "output": "match_results",
  "mysql_credentials": {
    "host": "localhost",
    "user": "myuser",
    "password": "mypassword",
    "database": "mydatabase"
  }
}
```

#### Option 2: Inline Credentials

```json
{
  "output": {
    "table": "match_results",
    "host": "localhost",
    "user": "myuser",
    "password": "mypassword",
    "database": "mydatabase"
  }
}
```

**Note:** The table will be created automatically if it doesn't exist. Existing tables will be replaced by default.

## MySQL Credentials

MySQL credentials can be specified globally (for all sources and output) or inline (per source/output).

### Global MySQL Credentials

```json
{
  "mysql_credentials": {
    "host": "localhost",
    "user": "myuser",
    "password": "mypassword",
    "database": "mydatabase"
  }
}
```

### Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `host` | string | Yes | MySQL server hostname or IP address |
| `user` | string | Yes | MySQL username |
| `password` | string | Yes | MySQL password |
| `database` | string | Yes | MySQL database name |

## S3 Credentials

S3 credentials are optional. If not provided, the system will use:
1. IAM role credentials (if running on AWS)
2. Environment variables (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`)
3. AWS credentials file (`~/.aws/credentials`)

### S3 Credentials Configuration

```json
{
  "s3_credentials": {
    "aws_access_key_id": "your_access_key_id",
    "aws_secret_access_key": "your_secret_access_key",
    "region_name": "us-east-1"
  }
}
```

### Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `aws_access_key_id` | string | No* | AWS access key ID |
| `aws_secret_access_key` | string | No* | AWS secret access key |
| `region_name` | string | No | AWS region (defaults to `us-east-1`) |

\* Required only if not using IAM role or environment variables

## Environment Variables

You can reference environment variables in your configuration using the `${VAR_NAME}` syntax. This is especially useful for sensitive credentials.

### Syntax

- `${VAR_NAME}` - Required variable (will error if not set)
- `${VAR_NAME:default_value}` - Optional variable with default value

### Example

```json
{
  "mysql_credentials": {
    "host": "${MYSQL_HOST:localhost}",
    "user": "${MYSQL_USER}",
    "password": "${MYSQL_PASSWORD}",
    "database": "${MYSQL_DATABASE}"
  },
  "s3_credentials": {
    "aws_access_key_id": "${AWS_ACCESS_KEY_ID}",
    "aws_secret_access_key": "${AWS_SECRET_ACCESS_KEY}",
    "region_name": "${AWS_REGION:us-east-1}"
  }
}
```

### Loading from .env File

Create a `.env` file in your project root:

```bash
# MySQL Credentials
MYSQL_HOST=localhost
MYSQL_USER=myuser
MYSQL_PASSWORD=mypassword
MYSQL_DATABASE=mydatabase

# AWS S3 Credentials (optional - will use IAM role if not provided)
AWS_ACCESS_KEY_ID=your_access_key_id
AWS_SECRET_ACCESS_KEY=your_secret_access_key
AWS_REGION=us-east-1
```

The system will automatically load variables from `.env` if the file exists.

## Match Configuration

The `match_config` section controls how matching is performed.

### Basic Match Config

```json
{
  "match_config": {
    "threshold": 0.85,
    "undecided_range": 0.05
  }
}
```

### Full Match Config

```json
{
  "match_config": {
    "columns": [
      {
        "source1": "name",
        "source2": "full_name",
        "weight": 0.4
      },
      {
        "source1": "email",
        "source2": "email_address",
        "weight": 0.6
      }
    ],
    "threshold": 0.85,
    "undecided_range": 0.05,
    "return_all_matches": false
  }
}
```

### Match Config Fields

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `columns` | array | No | Auto-detect | Explicit column mappings with weights |
| `threshold` | number | No | 0.85 | Match score threshold (0-1) |
| `undecided_range` | number | No | 0.05 | Range around threshold for undecided matches (0-1) |
| `return_all_matches` | boolean | No | false | Return all matches above threshold instead of best match only |

### Performance Tuning

Use these optional flags inside `match_config` to control throughput when working with large datasets:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `use_multiprocessing` | boolean | true | Enables multi-process execution for chunked matching. Disable if the environment has high process start overhead. |
| `num_workers` | integer | min(cpu_count, 8) | Limits the number of worker processes used when multiprocessing is enabled. |
| `chunk_size` | integer | 10000 | Number of source1 rows per chunk submitted to workers. Larger chunks reduce scheduling overhead at the cost of higher peak memory. |
| `load_chunk_size` | integer | 100000 | Chunk size used while loading CSV/MySQL data. Reduce if you hit memory limits while loading. |
| `early_termination` | boolean | true | Stops scoring remaining columns once a match is already above the `perfect_match_threshold`. |
| `perfect_match_threshold` | number | 0.99 | Score required to trigger early termination when enabled. |
| `blocking_strategies` | array | `["first_char","three_gram","last_three"]` | Controls which blocking keys are generated. Fewer strategies build smaller indexes and reduce serialization time. Valid tokens: `first_char`, `two_gram`, `three_gram`, `last_three`, `word_prefix`, `word_suffix`. |
| `max_block_size` | integer | null | Maximum number of source2 rows allowed per blocking key. Larger buckets are either trimmed or skipped depending on `skip_high_cardinality`. |
| `skip_high_cardinality` | boolean | true | When true, blocking keys exceeding `max_block_size` are dropped entirely; when false, they are truncated to the cap. |
| `max_candidates` | integer | null | Caps the number of candidate rows evaluated per source1 record. Helpful when blocking still yields thousands of matches. |
| `candidate_trim_strategy` | string | `"truncate"` | Strategy to enforce `max_candidates`. `"truncate"` keeps the first N candidates, `"fallback"` retries using only higher-priority keys before truncating. |

Example:

```json
{
  "match_config": {
    "threshold": 0.9,
    "use_multiprocessing": true,
    "num_workers": 6,
    "chunk_size": 8000,
    "blocking_strategies": ["first_char", "three_gram"],
    "max_block_size": 2500,
    "skip_high_cardinality": false,
    "max_candidates": 500,
    "candidate_trim_strategy": "fallback"
  }
}
```

**Note:** When multiprocessing is enabled, the matcher automatically mirrors source data and normalized columns into OS-level shared memory blocks so workers avoid reloading or copying large arrays.

### Column Mapping

Each column mapping specifies:

```json
{
  "source1": "column_name_in_source1",
  "source2": "column_name_in_source2",
  "weight": 0.5
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `source1` | string | Yes | Column name in the first data source |
| `source2` | string | Yes | Column name in the second data source |
| `weight` | number | No | Weight for this column (default: 1.0) |

**Note:** If `columns` is not specified, the engine will automatically:
- Match columns by name similarity
- Detect column types and select appropriate algorithms
- Assign equal weights to all matched columns

### Match Classification

Results are classified based on the overall score:

- **Accept**: `score >= threshold + undecided_range`
- **Undecided**: `threshold - undecided_range < score < threshold + undecided_range`
- **Reject**: `score <= threshold - undecided_range`

**Example:** With `threshold: 0.85` and `undecided_range: 0.05`:
- Accept: scores >= 0.90
- Undecided: scores between 0.80 and 0.90
- Reject: scores <= 0.80

### Return All Matches

By default, the engine returns only the best match for each row in source1. Set `return_all_matches: true` to return all matches above the threshold:

```json
{
  "match_config": {
    "threshold": 0.75,
    "return_all_matches": true
  }
}
```

## Clustering Configuration

The `cluster_config` section controls how clustering/duplicate finding is performed. Clustering mode finds duplicates and groups similar records within a single dataset.

### Basic Clustering Config

```json
{
  "mode": "clustering",
  "source1": "data/customers.csv",
  "output": "results/clusters.csv",
  "cluster_config": {
    "threshold": 0.85,
    "generate_summary": false
  }
}
```

### Full Clustering Config

```json
{
  "mode": "clustering",
  "source1": "data/customers.csv",
  "output": "results/clusters.csv",
  "cluster_config": {
    "columns": [
      {
        "source1": "name",
        "weight": 0.4
      },
      {
        "source1": "email",
        "weight": 0.6
      }
    ],
    "threshold": 0.85,
    "generate_summary": true,
    "use_blocking": true,
    "use_multiprocessing": true,
    "num_workers": 4
  }
}
```

### Cluster Config Fields

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `columns` | array | No | Auto-detect | Explicit column mappings with weights (only `source1` needed) |
| `threshold` | number | No | 0.85 | Similarity threshold for clustering records (0-1) |
| `generate_summary` | boolean | No | false | Generate summary report with cluster statistics |
| `use_blocking` | boolean | No | true | Use blocking/indexing for performance optimization |
| `use_multiprocessing` | boolean | No | true | Enable multi-process execution for parallel clustering |
| `num_workers` | integer | No | min(cpu_count, 8) | Number of worker processes for parallel processing |
| `chunk_size` | integer | No | 10000 | Number of record pairs per chunk for workers |
| `load_chunk_size` | integer | No | 100000 | Chunk size for loading CSV/MySQL data |
| `blocking_strategies` | array | No | `["first_char","three_gram","last_three"]` | Blocking strategies to use |
| `max_block_size` | integer | No | null | Maximum records per blocking key |
| `skip_high_cardinality` | boolean | No | true | Skip blocking keys exceeding max_block_size |

### Clustering Column Mapping

For clustering, column mappings only need `source1` (since we're working with a single dataset):

```json
{
  "source1": "column_name",
  "weight": 0.5
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `source1` | string | Yes | Column name in the data source |
| `weight` | number | No | Weight for this column (default: 1.0) |

**Note:** If `columns` is not specified, the engine will automatically:
- Use all columns in the dataset
- Detect column types and select appropriate algorithms
- Assign equal weights to all columns

### Clustering Output

The clustering output includes:
- All original columns from the source data
- `cluster_id`: Unique identifier for each cluster (sequential integers starting from 0)
- `cluster_size`: Number of records in the cluster

Records with the same `cluster_id` are considered duplicates or similar records.

### Summary Report

When `generate_summary: true`, a summary report is generated (saved as `{output}_summary.txt`) containing:
- Total records and total clusters
- Number of singleton clusters (unique records)
- Number of multi-record clusters (duplicates)
- Cluster size distribution
- Largest clusters (top 10)

## Complete Examples

### Minimal Configuration

```json
{
  "source1": "data/master.csv",
  "source2": "data/new_records.csv",
  "output": "results/matches.csv"
}
```

### CSV to MySQL with Environment Variables

```json
{
  "source1": "data/master.csv",
  "source2": "data/new_records.csv",
  "output": "match_results",
  "mysql_credentials": {
    "host": "${MYSQL_HOST:localhost}",
    "user": "${MYSQL_USER}",
    "password": "${MYSQL_PASSWORD}",
    "database": "${MYSQL_DATABASE}"
  },
  "match_config": {
    "threshold": 0.85,
    "undecided_range": 0.05
  }
}
```

### MySQL to S3 with Custom Column Mapping

```json
{
  "source1": "customers",
  "source2": "leads",
  "output": "s3://my-bucket/results/matches.csv",
  "mysql_credentials": {
    "host": "localhost",
    "user": "myuser",
    "password": "mypassword",
    "database": "mydatabase"
  },
  "match_config": {
    "columns": [
      {
        "source1": "name",
        "source2": "full_name",
        "weight": 0.3
      },
      {
        "source1": "email",
        "source2": "email_address",
        "weight": 0.4
      },
      {
        "source1": "phone",
        "source2": "phone_number",
        "weight": 0.3
      }
    ],
    "threshold": 0.80,
    "undecided_range": 0.08,
    "return_all_matches": false
  }
}
```

### S3 to MySQL with IAM Role

```json
{
  "source1": "s3://my-bucket/data/master.csv",
  "source2": "s3://my-bucket/data/new_records.csv",
  "output": {
    "table": "match_results",
    "host": "${MYSQL_HOST}",
    "user": "${MYSQL_USER}",
    "password": "${MYSQL_PASSWORD}",
    "database": "${MYSQL_DATABASE}"
  }
}
```

**Note:** No `s3_credentials` needed - will use IAM role automatically.

### Hybrid Sources with Environment Variables

```json
{
  "source1": "data/master.csv",
  "source2": {
    "table": "new_records",
    "host": "${MYSQL_HOST}",
    "user": "${MYSQL_USER}",
    "password": "${MYSQL_PASSWORD}",
    "database": "${MYSQL_DATABASE}"
  },
  "output": "s3://my-bucket/results/matches.csv",
  "s3_credentials": {
    "aws_access_key_id": "${AWS_ACCESS_KEY_ID}",
    "aws_secret_access_key": "${AWS_SECRET_ACCESS_KEY}",
    "region_name": "${AWS_REGION:us-east-1}"
  },
  "match_config": {
    "columns": [
      {
        "source1": "name",
        "source2": "full_name",
        "weight": 0.2
      },
      {
        "source1": "address",
        "source2": "street_address",
        "weight": 0.4
      },
      {
        "source1": "city",
        "source2": "city_name",
        "weight": 0.2
      },
      {
        "source1": "zip",
        "source2": "postal_code",
        "weight": 0.2
      }
    ],
    "threshold": 0.75,
    "undecided_range": 0.08,
    "return_all_matches": true
  }
}
```

## Configuration Examples Directory

See the `config-examples/` directory for more complete examples:

- `minimal_matching.json` - Minimal configuration
- `mysql_output_example.json` - MySQL output example
- `s3_output_example.json` - S3 output with credentials
- `s3_iam_role_example.json` - S3 output using IAM role
- `env_vars_example.json` - Using environment variables
- `address_heavy_matching.json` - Address-focused matching
- `full_example.json` - Complete configuration example

## Validation

The configuration is validated against a JSON schema. Common validation errors:

- **Missing required fields**: Ensure `source1`, `source2`, and `output` are present
- **Invalid types**: Check that field types match the expected types
- **Missing credentials**: If using MySQL/S3, ensure credentials are provided or environment variables are set
- **File not found**: Verify CSV file paths are correct
- **Invalid threshold**: Threshold must be between 0 and 1

## Tips

1. **Use environment variables** for sensitive credentials - never commit passwords to version control
2. **Start with minimal config** - let the engine auto-detect columns and types
3. **Adjust threshold** based on your data quality - lower for noisy data, higher for clean data
4. **Use weights** to emphasize important columns (e.g., email > name > address)
5. **Test with small datasets** before running on large production data
6. **Use S3 IAM roles** in AWS environments for better security
7. **Monitor undecided matches** - they may need manual review

