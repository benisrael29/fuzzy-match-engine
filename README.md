# Fuzzy Matching Engine

A configurable, high-performance fuzzy matching engine that can match rows between two data sources (CSV files or MySQL tables) with automatic algorithm selection based on column data types.

## Features

- **Automatic Algorithm Selection**: Detects column types (string, numeric, date, email, phone) and selects optimal matching algorithms
- **Multiple Data Sources**: Supports CSV files, MySQL tables, and S3 buckets
- **Flexible Output**: Write results to CSV files, MySQL tables, or S3 buckets
- **Environment Variables**: Secure credential management with `.env` file support
- **IAM Role Support**: Automatic AWS IAM role detection for S3 operations
- **Performance Optimized**: Uses `rapidfuzz` for fast string matching, blocking/indexing for large datasets
- **Flexible Configuration**: Minimal config with auto-detection or full control with custom mappings
- **Multi-column Matching**: Weighted scoring across multiple column pairs
- **Match Classification**: Results classified as accept, reject, or undecided based on threshold

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Generate Configuration

```bash
python main.py --setup
```

This will interactively prompt you for:
- Source 1 (CSV file or MySQL table)
- Source 2 (CSV file or MySQL table)
- Output file path
- Optional: threshold and undecided range

### 2. Run Matching

```bash
python main.py --config config/example_config.json
```

## Configuration

For a complete configuration reference, see [CONFIGURATION.md](CONFIGURATION.md).

### Quick Examples

**Minimal Configuration (Auto-detection):**

```json
{
  "source1": "data/master.csv",
  "source2": "data/new_records.csv",
  "output": "results/matches.csv"
}
```

**With MySQL:**

```json
{
  "source1": "master_data.csv",
  "source2": "customers",
  "mysql_credentials": {
    "host": "localhost",
    "user": "user",
    "password": "pass",
    "database": "dbname"
  },
  "output": "results/matches.csv"
}
```

**With Environment Variables:**

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
  }
}
```

**S3 Output with IAM Role:**

```json
{
  "source1": "data/master.csv",
  "source2": "data/new_records.csv",
  "output": "s3://my-bucket/results/matches.csv"
}
```

**Full Configuration:**

```json
{
  "source1": "data/master.csv",
  "source2": "data/new_records.csv",
  "output": "results/matches.csv",
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

### Column Mapping Between Datasets

The engine builds a column map between `source1` and `source2` in two phases:

- **Auto-analysis**: If `match_config.columns` is omitted, `column_analyzer.py` inspects both tables, infers the data type for every column, and auto-pairs columns that share the same header name (e.g., `email` ↔ `email`). Each inferred pair stores its detected type so the matcher can pick the right similarity algorithm.
- **Explicit mappings**: When you provide `match_config.columns`, each entry pins an exact pair using the `source1` and `source2` keys and can include a `weight` (default `1.0`). Only the listed pairs are evaluated, which lets you map differently named columns such as `name` ↔ `full_name` or control how much each column influences the final score.

Regardless of how pairs are defined, the output file keeps both originals prefixed with `source1_` / `source2_` (e.g., `source1_email`, `source2_email_address`) so you can audit the mapping that produced a score.

### Supported Data Sources

- **CSV Files**: Local file paths
- **S3 Files**: `s3://bucket/key` URLs
- **MySQL Tables**: Table names with credentials

### Supported Output Destinations

- **CSV Files**: Local file paths
- **S3**: `s3://bucket/key` URLs (uses IAM role if credentials not provided)
- **MySQL Tables**: Table names with credentials

### Environment Variables

Use `${VAR_NAME}` or `${VAR_NAME:default}` syntax to reference environment variables. See [CONFIGURATION.md](CONFIGURATION.md) for details.

## Algorithm Selection

The engine automatically selects algorithms based on detected column types:

- **String Names**: Jaro-Winkler similarity (good for person names)
- **General Strings**: Levenshtein distance
- **Numeric**: Ratio-based similarity
- **Dates**: Temporal distance (normalized by days)
- **Email/Phone**: Token-based matching after normalization

## Output Format

The output CSV includes:
- All original columns from both sources (prefixed with `source1_` and `source2_`)
- Individual column match scores (`score_<column_name>`)
- Overall match score (`overall_score`)
- Match result (`match_result`: accept/reject/undecided)
- Row identifiers (`source1_index`, `source2_index`)

## Match Classification

- **Accept**: Score >= threshold + undecided_range
- **Reject**: Score <= threshold - undecided_range
- **Undecided**: Score in the range between accept and reject

Example: With threshold 0.85 and undecided_range 0.05:
- Accept: >= 0.90
- Undecided: 0.80 - 0.90
- Reject: <= 0.80

## Performance Considerations

- **Blocking/Indexing**: Automatically enabled for datasets >10K rows
- **Chunking**: Large CSV files are processed in chunks
- **Vectorized Operations**: Uses pandas/numpy for efficient processing
- **rapidfuzz**: C++ backend for fast string matching

## Testing

### Core Suite

```bash
python -m pytest tests/test_integration.py -v
```

### Large Dataset Benchmarks

```bash
python -m pytest tests/test_large_scale_accuracy.py::TestLargeDatasetPerformance -v
```

To exercise the optional 100K streaming and 500K stress tests, set `RUN_HEAVY_DATASET_TESTS=1` before running pytest.

### Accuracy Metrics

```bash
python -m pytest tests/test_accuracy_metrics.py -v
```

Tests cover:
- End-to-end matching workflow plus result export validation
- Large dataset performance (10K, 50K, 100K rows and uneven set sizes)
- Detailed accuracy metrics (precision/recall/F1, score distributions, false positives)
- Name, address, phone, email, and date variations
- Noise tolerance (typos, missing data) and partial matches

## Dependencies

- `pandas>=2.0.0` - Data manipulation
- `rapidfuzz>=3.0.0` - High-performance fuzzy matching
- `sqlalchemy>=2.0.0` - MySQL connection abstraction
- `pymysql>=1.1.0` - MySQL driver
- `numpy>=1.24.0` - Numerical operations
- `python-dateutil>=2.8.0` - Date parsing
- `jsonschema>=4.0.0` - JSON config validation
- `boto3>=1.28.0` - AWS S3 support
- `python-dotenv>=1.0.0` - Environment variable support

## License

MIT

