import json
import os
import re
from typing import Dict, Any, Optional
from jsonschema import validate, ValidationError, SchemaError

try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False


CONFIG_SCHEMA = {
    "type": "object",
    "required": ["output"],
    "properties": {
        "mode": {
            "type": "string",
            "enum": ["matching", "clustering", "search"],
            "default": "matching"
        },
        "source1": {
            "oneOf": [
                {"type": "string"},
                {
                    "type": "object",
                    "properties": {
                        "table": {"type": "string"},
                        "host": {"type": "string"},
                        "user": {"type": "string"},
                        "password": {"type": "string"},
                        "database": {"type": "string"}
                    },
                    "required": ["table"]
                }
            ]
        },
        "source2": {
            "oneOf": [
                {"type": "string"},
                {
                    "type": "object",
                    "properties": {
                        "table": {"type": "string"},
                        "host": {"type": "string"},
                        "user": {"type": "string"},
                        "password": {"type": "string"},
                        "database": {"type": "string"}
                    },
                    "required": ["table"]
                }
            ]
        },
        "output": {
            "oneOf": [
                {"type": "string"},
                {
                    "type": "object",
                    "properties": {
                        "table": {"type": "string"},
                        "host": {"type": "string"},
                        "user": {"type": "string"},
                        "password": {"type": "string"},
                        "database": {"type": "string"}
                    },
                    "required": ["table"]
                }
            ]
        },
        "mysql_credentials": {
            "type": "object",
            "properties": {
                "host": {"type": "string"},
                "user": {"type": "string"},
                "password": {"type": "string"},
                "database": {"type": "string"}
            },
            "required": ["host", "user", "password", "database"]
        },
        "s3_credentials": {
            "type": "object",
            "properties": {
                "aws_access_key_id": {"type": "string"},
                "aws_secret_access_key": {"type": "string"},
                "region_name": {"type": "string"}
            }
        },
        "match_config": {
            "type": "object",
            "properties": {
                "columns": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "source1": {"type": "string"},
                            "source2": {"type": "string"},
                            "weight": {"type": "number", "minimum": 0}
                        },
                        "required": ["source1", "source2"]
                    }
                },
                "threshold": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1
                },
                "undecided_range": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1
                },
                "return_all_matches": {
                    "type": "boolean"
                }
            }
        },
        "cluster_config": {
            "type": "object",
            "properties": {
                "columns": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "source1": {"type": "string"},
                            "weight": {"type": "number", "minimum": 0}
                        },
                        "required": ["source1"]
                    }
                },
                "threshold": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1
                },
                "generate_summary": {
                    "type": "boolean"
                },
                "use_blocking": {
                    "type": "boolean"
                },
                "use_multiprocessing": {
                    "type": "boolean"
                },
                "num_workers": {
                    "type": "integer",
                    "minimum": 1
                },
                "chunk_size": {
                    "type": "integer",
                    "minimum": 1
                },
                "load_chunk_size": {
                    "type": "integer",
                    "minimum": 1
                },
                "blocking_strategies": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "max_block_size": {
                    "type": "integer",
                    "minimum": 1
                },
                "skip_high_cardinality": {
                    "type": "boolean"
                }
            }
        }
    }
}


def validate_config(config_path: str, env_file: Optional[str] = None) -> Dict[str, Any]:
    """
    Validate JSON configuration file and resolve environment variable references.
    
    Args:
        config_path: Path to JSON configuration file
        env_file: Optional path to .env file (defaults to .env in current directory)
    
    Returns:
        Validated configuration dictionary with environment variables resolved
    
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config is invalid
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    if DOTENV_AVAILABLE:
        if env_file is None:
            env_file = '.env'
        if os.path.exists(env_file):
            load_dotenv(env_file)
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in configuration file: {str(e)}")
    
    config = _resolve_env_vars(config)
    
    mode = config.get('mode', 'matching')
    
    if mode == 'clustering':
        if 'source2' in config:
            raise ValueError("clustering mode does not require source2. Remove source2 from config.")
        if 'source1' not in config:
            raise ValueError("clustering mode requires source1. Add source1 to config.")
    elif mode == 'search':
        if 'source2' not in config:
            raise ValueError("search mode requires source2 (master dataset). Add source2 to config.")
        if 'source1' in config:
            raise ValueError("search mode does not require source1. Remove source1 from config or set mode to 'matching'.")
    else:
        if 'source1' not in config:
            raise ValueError("matching mode requires source1. Add source1 to config.")
        if 'source2' not in config:
            raise ValueError("matching mode requires source2. Add source2 to config or set mode to 'clustering' or 'search'.")
    
    try:
        validate(instance=config, schema=CONFIG_SCHEMA)
    except ValidationError as e:
        error_path = '.'.join(str(p) for p in e.path)
        raise ValueError(
            f"Configuration validation error at '{error_path}': {e.message}\n"
            f"Suggested fix: {_get_validation_suggestion(e)}"
        )
    except SchemaError as e:
        raise ValueError(f"Configuration schema error: {str(e)}")
    
    _validate_file_paths(config, mode)
    
    return config


def _validate_file_paths(config: Dict[str, Any], mode: str = 'matching'):
    """Validate that CSV file paths exist (skip S3 URLs and MySQL tables)."""
    source_keys = []
    if mode == 'matching':
        source_keys = ['source1', 'source2']
    elif mode == 'clustering':
        source_keys = ['source1']
    elif mode == 'search':
        source_keys = ['source2']
    
    for source_key in source_keys:
        source = config.get(source_key)
        
        if isinstance(source, str):
            if source.startswith('s3://'):
                continue
            
            if not source.endswith('.csv'):
                continue
            
            if not os.path.exists(source):
                raise ValueError(
                    f"CSV file not found: {source}\n"
                    f"Please check the path, use S3 URL (s3://bucket/key), or use MySQL table name with mysql_credentials."
                )
    
    output = config.get('output')
    if isinstance(output, str):
        if output.startswith('s3://'):
            return
        if not output.endswith('.csv'):
            return
        output_dir = os.path.dirname(output)
        if output_dir and not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir, exist_ok=True)
            except Exception:
                pass


def _resolve_env_vars(obj: Any) -> Any:
    """
    Recursively resolve environment variable references in config.
    Supports ${VAR_NAME} or ${VAR_NAME:default_value} syntax.
    """
    if isinstance(obj, dict):
        return {key: _resolve_env_vars(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [_resolve_env_vars(item) for item in obj]
    elif isinstance(obj, str):
        pattern = r'\$\{([^}:]+)(?::([^}]*))?\}'
        
        def replace_env_var(match):
            var_name = match.group(1)
            default_value = match.group(2) if match.group(2) is not None else None
            env_value = os.getenv(var_name)
            
            if env_value is not None:
                return env_value
            elif default_value is not None:
                return default_value
            else:
                raise ValueError(
                    f"Environment variable '{var_name}' not found and no default value provided. "
                    f"Set it in your .env file or environment."
                )
        
        if re.search(pattern, obj):
            return re.sub(pattern, replace_env_var, obj)
        return obj
    else:
        return obj


def _get_validation_suggestion(error: ValidationError) -> str:
    """Generate helpful suggestion based on validation error."""
    error_path = '.'.join(str(p) for p in error.path)
    
    if 'required' in error.message.lower():
        missing_field = error_path.split('.')[-1] if error_path else 'field'
        return f"Add missing required field: {missing_field}"
    
    if 'type' in error.message.lower():
        return f"Check that {error_path} has the correct data type"
    
    return "Review the configuration structure and ensure all required fields are present"

