import os
import tempfile
import pandas as pd
from sqlalchemy import create_engine, text
from typing import Union, Dict, Optional
from urllib.parse import urlparse
from .normalizers import normalize_string

try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False


def load_source(
    source: Union[str, Dict],
    mysql_credentials: Optional[Dict] = None,
    s3_credentials: Optional[Dict] = None,
    chunk_size: Optional[int] = None
) -> pd.DataFrame:
    """
    Load data from CSV file, MySQL table, or S3 bucket.
    
    Args:
        source: File path (str), S3 URL (s3://bucket/key), table name (str), or dict with connection details
        mysql_credentials: MySQL connection details (host, user, password, database)
        s3_credentials: S3 credentials (aws_access_key_id, aws_secret_access_key, region_name)
        chunk_size: Optional chunk size for large files
    
    Returns:
        pandas DataFrame with normalized string columns
    """
    if isinstance(source, dict):
        return _load_from_mysql_dict(source)
    
    if isinstance(source, str):
        if _is_s3_url(source):
            return _load_from_s3(source, s3_credentials, chunk_size)
        elif _is_mysql_table(source, mysql_credentials):
            return _load_from_mysql(source, mysql_credentials)
        else:
            return _load_from_csv(source, chunk_size)
    
    raise ValueError(f"Invalid source type: {type(source)}")


def _is_s3_url(source: str) -> bool:
    """Check if source is an S3 URL."""
    return source.startswith('s3://')


def _is_mysql_table(source: str, mysql_credentials: Optional[Dict]) -> bool:
    """Check if source is a MySQL table name."""
    if mysql_credentials is None:
        return False
    
    if _is_s3_url(source):
        return False
    
    if os.path.exists(source) or source.endswith('.csv'):
        return False
    
    return True


def _load_from_s3(
    s3_url: str,
    s3_credentials: Optional[Dict] = None,
    chunk_size: Optional[int] = None
) -> pd.DataFrame:
    """Load data from S3 bucket."""
    if not BOTO3_AVAILABLE:
        raise ImportError(
            "boto3 is required for S3 support. Install it with: pip install boto3"
        )
    
    try:
        parsed = urlparse(s3_url)
        bucket_name = parsed.netloc
        key = parsed.path.lstrip('/')
        
        if not bucket_name or not key:
            raise ValueError(f"Invalid S3 URL format: {s3_url}. Expected format: s3://bucket-name/path/to/file.csv")
        
        s3_client = _get_s3_client(s3_credentials)
        
        with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.csv') as tmp_file:
            tmp_path = tmp_file.name
            
            try:
                s3_client.download_fileobj(bucket_name, key, tmp_file)
            except ClientError as e:
                os.unlink(tmp_path)
                error_code = e.response.get('Error', {}).get('Code', 'Unknown')
                if error_code == 'NoSuchBucket':
                    raise ValueError(f"S3 bucket not found: {bucket_name}")
                elif error_code == 'NoSuchKey':
                    raise ValueError(f"S3 object not found: s3://{bucket_name}/{key}")
                elif error_code == 'AccessDenied':
                    raise ValueError(f"Access denied to S3 object: s3://{bucket_name}/{key}. Check your credentials.")
                else:
                    raise ValueError(f"Error downloading from S3: {str(e)}")
            except NoCredentialsError:
                os.unlink(tmp_path)
                raise ValueError(
                    "AWS credentials not found. Provide s3_credentials in config or set AWS credentials "
                    "via environment variables, AWS credentials file, or IAM role."
                )
        
        try:
            df = _load_from_csv(tmp_path, chunk_size)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        
        return df
    except Exception as e:
        if isinstance(e, (ValueError, ImportError)):
            raise
        raise ValueError(f"Error loading from S3 {s3_url}: {str(e)}")


def _get_s3_client(s3_credentials: Optional[Dict] = None):
    """Create and return an S3 client with optional credentials, or use IAM role."""
    if s3_credentials:
        aws_access_key_id = s3_credentials.get('aws_access_key_id')
        aws_secret_access_key = s3_credentials.get('aws_secret_access_key')
        region_name = s3_credentials.get('region_name', 'us-east-1')
        
        if aws_access_key_id and aws_secret_access_key:
            return boto3.client(
                's3',
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                region_name=region_name
            )
    
    return boto3.client('s3')


def _load_from_csv(file_path: str, chunk_size: Optional[int] = None) -> pd.DataFrame:
    """Load data from CSV file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"CSV file not found: {file_path}")
    
    try:
        if chunk_size:
            chunks = []
            for chunk in pd.read_csv(file_path, chunksize=chunk_size):
                chunks.append(_normalize_dataframe(chunk))
            df = pd.concat(chunks, ignore_index=True)
        else:
            df = pd.read_csv(file_path)
            df = _normalize_dataframe(df)
        
        return df
    except Exception as e:
        raise ValueError(f"Error reading CSV file {file_path}: {str(e)}")


def _load_from_mysql(table_name: str, mysql_credentials: Dict) -> pd.DataFrame:
    """Load data from MySQL table using SQLAlchemy."""
    required_keys = ['host', 'user', 'password', 'database']
    missing_keys = [key for key in required_keys if key not in mysql_credentials]
    
    if missing_keys:
        raise ValueError(f"Missing MySQL credentials: {', '.join(missing_keys)}")
    
    try:
        connection_string = (
            f"mysql+pymysql://{mysql_credentials['user']}:"
            f"{mysql_credentials['password']}@{mysql_credentials['host']}/"
            f"{mysql_credentials['database']}"
        )
        
        engine = create_engine(connection_string)
        
        query = text(f"SELECT * FROM `{table_name}`")
        df = pd.read_sql(query, engine)
        
        df = _normalize_dataframe(df)
        
        return df
    except Exception as e:
        raise ValueError(f"Error connecting to MySQL table {table_name}: {str(e)}")


def _load_from_mysql_dict(source: Dict) -> pd.DataFrame:
    """Load data from MySQL using dict with connection details."""
    if 'table' not in source:
        raise ValueError("MySQL source dict must contain 'table' key")
    
    mysql_creds = {k: v for k, v in source.items() if k != 'table'}
    return _load_from_mysql(source['table'], mysql_creds)


def _normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply basic normalization to string columns using vectorized operations."""
    df = df.copy()
    
    for col in df.columns:
        if df[col].dtype == 'object':
            mask = df[col].notna() & (df[col].astype(str) != 'nan')
            df.loc[mask, col] = df.loc[mask, col].astype(str).str.lower().str.strip()
            df.loc[mask, col] = df.loc[mask, col].str.replace(r'\s+', ' ', regex=True)
            df.loc[~mask, col] = ''
    
    return df

