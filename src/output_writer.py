import os
import pandas as pd
from typing import Optional, List, Dict, Iterator, Tuple, Union
import io
try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
    S3_AVAILABLE = True
except ImportError:
    S3_AVAILABLE = False

try:
    from sqlalchemy import create_engine, text
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False


def _is_s3_path(path: str) -> bool:
    """Check if path is an S3 URL."""
    return path.startswith('s3://')


def _is_mysql_table(output: Union[str, Dict], mysql_credentials: Optional[Dict] = None) -> bool:
    """Check if output is a MySQL table."""
    if isinstance(output, dict):
        return 'table' in output
    if isinstance(output, str):
        if output.startswith('s3://') or output.endswith('.csv') or os.path.exists(output):
            return False
        return mysql_credentials is not None
    return False


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


def _get_mysql_engine(mysql_credentials: Dict):
    """Create SQLAlchemy engine for MySQL connection."""
    if not SQLALCHEMY_AVAILABLE:
        raise ImportError("sqlalchemy is not installed. Install it with: pip install sqlalchemy")
    
    required_keys = ['host', 'user', 'password', 'database']
    missing_keys = [key for key in required_keys if key not in mysql_credentials]
    
    if missing_keys:
        raise ValueError(f"Missing MySQL credentials: {', '.join(missing_keys)}")
    
    connection_string = (
        f"mysql+pymysql://{mysql_credentials['user']}:"
        f"{mysql_credentials['password']}@{mysql_credentials['host']}/"
        f"{mysql_credentials['database']}"
    )
    
    return create_engine(connection_string)


def _write_to_mysql(df: pd.DataFrame, table_name: str, mysql_credentials: Dict, if_exists: str = 'replace'):
    """Write DataFrame to MySQL table."""
    engine = _get_mysql_engine(mysql_credentials)
    
    try:
        df.to_sql(
            name=table_name,
            con=engine,
            if_exists=if_exists,
            index=False,
            method='multi',
            chunksize=10000
        )
    except Exception as e:
        raise ValueError(f"Error writing to MySQL table {table_name}: {str(e)}")


def write_results(df: pd.DataFrame, output_path: Union[str, Dict], stream: bool = False, config: Optional[Dict] = None):
    """
    Write match results to CSV file, S3, or MySQL table.
    
    Args:
        df: DataFrame with match results
        output_path: Path to output CSV file, S3 URL (s3://bucket/key), MySQL table name (str), or dict with table and credentials
        stream: If True, stream results in chunks for large datasets
        config: Optional configuration dict containing s3_credentials or mysql_credentials
    """
    mysql_credentials = config.get('mysql_credentials') if config else None
    
    if isinstance(output_path, dict):
        table_name = output_path.get('table')
        mysql_creds = output_path.copy()
        mysql_creds.pop('table', None)
        if not mysql_creds and mysql_credentials:
            mysql_creds = mysql_credentials
        _write_to_mysql(df, table_name, mysql_creds)
    elif isinstance(output_path, str) and _is_s3_path(output_path):
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
    elif isinstance(output_path, str) and _is_mysql_table(output_path, mysql_credentials):
        _write_to_mysql(df, output_path, mysql_credentials)
    else:
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        if stream and len(df) > 50000:
            df.to_csv(output_path, index=False, chunksize=10000, mode='w')
        else:
            df.to_csv(output_path, index=False)


def write_results_streaming(results: Iterator[Dict], output_path: Union[str, Dict], columns: List[str], config: Optional[Dict] = None):
    """
    Stream results directly to CSV file, S3, or MySQL table for very large datasets.
    
    Args:
        results: Iterator of result dictionaries
        output_path: Path to output CSV file, S3 URL (s3://bucket/key), MySQL table name (str), or dict with table and credentials
        columns: List of column names for the output
        config: Optional configuration dict containing s3_credentials or mysql_credentials
    """
    import csv
    
    mysql_credentials = config.get('mysql_credentials') if config else None
    
    if isinstance(output_path, dict):
        table_name = output_path.get('table')
        mysql_creds = output_path.copy()
        mysql_creds.pop('table', None)
        if not mysql_creds and mysql_credentials:
            mysql_creds = mysql_credentials
        
        results_list = []
        first_chunk = True
        for result in results:
            results_list.append(result)
            if len(results_list) >= 10000:
                df_chunk = pd.DataFrame(results_list)
                _write_to_mysql(df_chunk, table_name, mysql_creds, if_exists='replace' if first_chunk else 'append')
                results_list = []
                first_chunk = False
        
        if results_list:
            df_chunk = pd.DataFrame(results_list)
            _write_to_mysql(df_chunk, table_name, mysql_creds, if_exists='replace' if first_chunk else 'append')
    elif isinstance(output_path, str) and _is_s3_path(output_path):
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
    elif isinstance(output_path, str) and _is_mysql_table(output_path, mysql_credentials):
        results_list = []
        first_chunk = True
        for result in results:
            results_list.append(result)
            if len(results_list) >= 10000:
                df_chunk = pd.DataFrame(results_list)
                _write_to_mysql(df_chunk, output_path, mysql_credentials, if_exists='replace' if first_chunk else 'append')
                results_list = []
                first_chunk = False
        
        if results_list:
            df_chunk = pd.DataFrame(results_list)
            _write_to_mysql(df_chunk, output_path, mysql_credentials, if_exists='replace' if first_chunk else 'append')
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

