import sys
import time
from typing import Dict, Any, Optional
from .config_validator import validate_config
from .matcher import FuzzyMatcher
from .output_writer import write_results


class JobRunner:
    """Executes matching jobs with progress display."""
    
    def __init__(self):
        """Initialize JobRunner."""
        pass
    
    def run_job(self, config: Dict[str, Any], job_name: str = "") -> bool:
        """
        Execute a matching job.
        
        Args:
            config: Job configuration dictionary
            job_name: Optional job name for display
        
        Returns:
            True if successful, False otherwise
        """
        try:
            print("\n" + "=" * 60)
            if job_name:
                print(f"RUNNING JOB: {job_name}")
            else:
                print("RUNNING MATCHING JOB")
            print("=" * 60)
            
            print("\n[1/4] Validating configuration...")
            try:
                validated_config = validate_config_dict(config)
            except ValueError as e:
                print(f"✗ Configuration error: {e}")
                return False
            print("✓ Configuration valid")
            
            print("\n[2/4] Loading data sources...")
            try:
                matcher = FuzzyMatcher(validated_config)
            except Exception as e:
                print(f"✗ Error loading data: {e}")
                return False
            
            size1 = len(matcher.source1)
            size2 = len(matcher.source2)
            total_comparisons = size1 * size2
            
            print(f"✓ Source 1: {size1:,} rows")
            print(f"✓ Source 2: {size2:,} rows")
            print(f"✓ Column pairs to match: {len(matcher.column_analyses)}")
            print(f"✓ Estimated comparisons: {total_comparisons:,}")
            
            if matcher.use_blocking:
                print(f"ℹ Using blocking/indexing with {len(matcher.blocking_index):,} keys")
            
            print(f"ℹ Using {matcher.num_workers} worker(s) for processing")
            
            print("\n[3/4] Executing matching...")
            match_start_time = time.time()
            
            try:
                use_streaming = total_comparisons > 10000000
                if use_streaming:
                    print("ℹ Using streaming mode for very large dataset")
                    results = matcher.match(stream_to_file=validated_config['output'])
                else:
                    results = matcher.match()
            except Exception as e:
                print(f"✗ Error during matching: {e}")
                import traceback
                traceback.print_exc()
                return False
            
            match_time = time.time() - match_start_time
            print(f"✓ Found {len(results):,} matches in {match_time:.2f} seconds")
            
            if 'match_result' in results.columns:
                print("\nMatch distribution:")
                distribution = results['match_result'].value_counts()
                for result_type, count in distribution.items():
                    print(f"  {result_type}: {count:,}")
            
            if not use_streaming:
                print(f"\n[4/4] Writing results to {validated_config['output']}...")
                write_start_time = time.time()
                try:
                    write_results(results, validated_config['output'], stream=len(results) > 50000)
                    write_time = time.time() - write_start_time
                    print(f"✓ Results written successfully in {write_time:.2f} seconds")
                except Exception as e:
                    print(f"✗ Error writing results: {e}")
                    return False
            else:
                print(f"✓ Results already streamed to {validated_config['output']}")
            
            print("\n" + "=" * 60)
            print("JOB COMPLETED SUCCESSFULLY")
            print("=" * 60)
            print(f"\nResults saved to: {validated_config['output']}")
            print(f"Total matches: {len(results)}")
            
            return True
            
        except KeyboardInterrupt:
            print("\n\n✗ Job cancelled by user")
            return False
        except Exception as e:
            print(f"\n✗ Unexpected error: {e}")
            import traceback
            traceback.print_exc()
            return False


def validate_config_dict(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate configuration dictionary (without file path).
    
    Args:
        config: Configuration dictionary
    
    Returns:
        Validated configuration dictionary
    
    Raises:
        ValueError: If configuration is invalid
    """
    from jsonschema import validate, ValidationError, SchemaError
    from .config_validator import CONFIG_SCHEMA
    
    try:
        validate(instance=config, schema=CONFIG_SCHEMA)
    except ValidationError as e:
        error_path = '.'.join(str(p) for p in e.path)
        raise ValueError(f"Configuration validation error at '{error_path}': {e.message}")
    except SchemaError as e:
        raise ValueError(f"Configuration schema error: {str(e)}")
    
    import os
    for source_key in ['source1', 'source2']:
        source = config.get(source_key)
        
        if isinstance(source, str):
            if not source.endswith('.csv'):
                continue
            
            if not os.path.exists(source):
                raise ValueError(
                    f"CSV file not found: {source}\n"
                    f"Please check the path or use MySQL table name with mysql_credentials."
                )
    
    return config

