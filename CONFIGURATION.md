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
- [Complete Examples](#complete-examples)

## Basic Configuration

Every configuration file requires three fields:

```json
{
  "source1": "...",
  "source2": "...",
  "output": "..."
}
```

### Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `source1` | string or object | First data source (CSV file, S3 URL, or MySQL table) |
| `source2` | string or object | Second data source (CSV file, S3 URL, or MySQL table) |
| `output` | string or object | Output destination (CSV file, S3 URL, or MySQL table) |

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

