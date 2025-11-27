import os
import pandas as pd
from typing import Optional, List, Dict, Iterator, Tuple
import io
try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
    S3_AVAILABLE = True
except ImportError:
    S3_AVAILABLE = False


def _is_s3_path(path: str) -> bool:
    """Check if path is an S3 URL."""
    return path.startswith('s3://')


def _parse_s3_path(s3_path: str) -> Tuple[str, str]:
    """Parse S3 path into bucket and key."""
    if not s3_path.startswith('s3://'):
        raise ValueError(f"Invalid S3 path: {s3_path}")
    path_without_prefix = s3_path[5:]
    parts = path_without_prefix.split('/', 1)
    bucket = parts[0]
    key = parts[1] if len(parts) > 1 else ''
    return bucket, key


def _get_s3_client(config: Optional[Dict] = None):
    """Get S3 client with credentials from config, environment variables, or IAM role."""
    if not S3_AVAILABLE:
        raise ImportError("boto3 is not installed. Install it with: pip install boto3")
    
    if config and 's3_credentials' in config:
        creds = config['s3_credentials']
        aws_access_key_id = creds.get('aws_access_key_id')
        aws_secret_access_key = creds.get('aws_secret_access_key')
        region_name = creds.get('region_name', 'us-east-1')
        
        if aws_access_key_id and aws_secret_access_key:
            return boto3.client(
                's3',
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                region_name=region_name
            )
    
    return boto3.client('s3')


def _upload_to_s3(data: bytes, bucket: str, key: str, config: Optional[Dict] = None):
    """Upload data to S3."""
    s3_client = _get_s3_client(config)
    s3_client.put_object(Bucket=bucket, Key=key, Body=data)


def _read_from_s3(s3_path: str, config: Optional[Dict] = None) -> pd.DataFrame:
    """Read CSV from S3."""
    bucket, key = _parse_s3_path(s3_path)
    s3_client = _get_s3_client(config)
    
    response = s3_client.get_object(Bucket=bucket, Key=key)
    csv_data = response['Body'].read().decode('utf-8')
    
    return pd.read_csv(io.StringIO(csv_data))


def write_results(df: pd.DataFrame, output_path: str, stream: bool = False, config: Optional[Dict] = None):
    """
    Write match results to CSV file or S3.
    
    Args:
        df: DataFrame with match results
        output_path: Path to output CSV file or S3 URL (s3://bucket/key)
        stream: If True, stream results in chunks for large datasets
        config: Optional configuration dict containing s3_credentials
    """
    if _is_s3_path(output_path):
        bucket, key = _parse_s3_path(output_path)
        
        if stream and len(df) > 50000:
            buffer = io.StringIO()
            df.to_csv(buffer, index=False, chunksize=10000, mode='w')
            data = buffer.getvalue().encode('utf-8')
        else:
            buffer = io.StringIO()
            df.to_csv(buffer, index=False)
            data = buffer.getvalue().encode('utf-8')
        
        _upload_to_s3(data, bucket, key, config)
    else:
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        if stream and len(df) > 50000:
            df.to_csv(output_path, index=False, chunksize=10000, mode='w')
        else:
            df.to_csv(output_path, index=False)


def write_results_streaming(results: Iterator[Dict], output_path: str, columns: List[str], config: Optional[Dict] = None):
    """
    Stream results directly to CSV file or S3 for very large datasets.
    
    Args:
        results: Iterator of result dictionaries
        output_path: Path to output CSV file or S3 URL (s3://bucket/key)
        columns: List of column names for the output
        config: Optional configuration dict containing s3_credentials
    """
    import csv
    
    if _is_s3_path(output_path):
        bucket, key = _parse_s3_path(output_path)
        buffer = io.StringIO()
        
        first_result = True
        writer = None
        for result in results:
            if first_result:
                if not columns:
                    columns = list(result.keys())
                writer = csv.DictWriter(buffer, fieldnames=columns)
                writer.writeheader()
                first_result = False
            writer.writerow(result)
        
        data = buffer.getvalue().encode('utf-8')
        _upload_to_s3(data, bucket, key, config)
    else:
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        first_result = True
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = None
            for result in results:
                if first_result:
                    if not columns:
                        columns = list(result.keys())
                    writer = csv.DictWriter(f, fieldnames=columns)
                    writer.writeheader()
                    first_result = False
                writer.writerow(result)

