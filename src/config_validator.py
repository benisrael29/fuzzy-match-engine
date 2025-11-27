import json
import os
from typing import Dict, Any
from jsonschema import validate, ValidationError, SchemaError


CONFIG_SCHEMA = {
    "type": "object",
    "required": ["source1", "source2", "output"],
    "properties": {
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
        "output": {"type": "string"},
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
            },
            "required": ["aws_access_key_id", "aws_secret_access_key"]
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
        }
    }
}


def validate_config(config_path: str) -> Dict[str, Any]:
    """
    Validate JSON configuration file.
    
    Args:
        config_path: Path to JSON configuration file
    
    Returns:
        Validated configuration dictionary
    
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config is invalid
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in configuration file: {str(e)}")
    
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
    
    _validate_file_paths(config)
    
    return config


def _validate_file_paths(config: Dict[str, Any]):
    """Validate that CSV file paths exist (skip S3 URLs and MySQL tables)."""
    for source_key in ['source1', 'source2']:
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


def _get_validation_suggestion(error: ValidationError) -> str:
    """Generate helpful suggestion based on validation error."""
    error_path = '.'.join(str(p) for p in error.path)
    
    if 'required' in error.message.lower():
        missing_field = error_path.split('.')[-1] if error_path else 'field'
        return f"Add missing required field: {missing_field}"
    
    if 'type' in error.message.lower():
        return f"Check that {error_path} has the correct data type"
    
    return "Review the configuration structure and ensure all required fields are present"

