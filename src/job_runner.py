import sys
import os
import time
import threading
from typing import Dict, Any, Optional
from pathlib import Path
from .config_validator import validate_config
from .matcher import FuzzyMatcher
from .output_writer import write_results, write_cluster_results, write_cluster_summary


class JobRunner:
    """Executes matching jobs with progress display."""
    
    def __init__(self):
        """Initialize JobRunner."""
        pass
    
    def run_job(
        self,
        config: Dict[str, Any],
        job_name: str = "",
        cancel_event: Optional[threading.Event] = None
    ) -> bool:
        """
        Execute a matching job.
        
        Args:
            config: Job configuration dictionary
            job_name: Optional job name for display
            cancel_event: Optional threading.Event to check for cancellation
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check for cancellation before starting
            if cancel_event and cancel_event.is_set():
                print("\n✗ Job cancelled before execution")
                return False
            print("\n" + "=" * 60)
            if job_name:
                print(f"RUNNING JOB: {job_name}")
            else:
                mode = config.get('mode', 'matching')
                if mode == 'clustering':
                    print("RUNNING CLUSTERING JOB")
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
            
            mode = validated_config.get('mode', 'matching')
            
            if mode == 'clustering':
                return self._run_clustering_job(validated_config, cancel_event)
            else:
                return self._run_matching_job(validated_config, cancel_event)
            
        except KeyboardInterrupt:
            print("\n\n✗ Job cancelled by user")
            return False
        except Exception as e:
            print(f"\n✗ Unexpected error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _run_clustering_job(
        self,
        validated_config: Dict[str, Any],
        cancel_event: Optional[threading.Event] = None
    ) -> bool:
        """Execute a clustering job."""
        try:
            from .clusterer import Clusterer
            
            print("\n[2/4] Loading data source...")
            try:
                clusterer = Clusterer(validated_config)
            except Exception as e:
                print(f"✗ Error loading data: {e}")
                return False
            
            size = len(clusterer.source)
            print(f"✓ Source: {size:,} rows")
            print(f"✓ Columns to cluster: {len(clusterer.column_analyses)}")
            
            if clusterer.use_blocking:
                print(f"ℹ Using blocking/indexing with {len(clusterer.blocking_index):,} keys")
            
            print(f"ℹ Using {clusterer.num_workers} worker(s) for processing")
            
            print("\n[3/4] Executing clustering...")
            cluster_start_time = time.time()
            
            if cancel_event and cancel_event.is_set():
                print("\n✗ Job cancelled before clustering")
                return False
            
            try:
                results = clusterer.cluster()
                
                if cancel_event and cancel_event.is_set():
                    print("\n✗ Job cancelled during clustering")
                    return False
            except KeyboardInterrupt:
                print("\n✗ Job cancelled by user")
                return False
            except Exception as e:
                print(f"✗ Error during clustering: {e}")
                import traceback
                traceback.print_exc()
                return False
            
            cluster_time = time.time() - cluster_start_time
            print(f"✓ Found {len(results):,} clusters in {cluster_time:.2f} seconds")
            
            if cancel_event and cancel_event.is_set():
                print("\n✗ Job cancelled before writing results")
                return False
            
            print(f"\n[4/4] Writing results to {validated_config['output']}...")
            write_start_time = time.time()
            try:
                write_cluster_results(results, validated_config['output'], config=validated_config)
                write_time = time.time() - write_start_time
                print(f"✓ Results written successfully in {write_time:.2f} seconds")
            except Exception as e:
                print(f"✗ Error writing results: {e}")
                return False
            
            if validated_config.get('cluster_config', {}).get('generate_summary', False):
                output_path = validated_config['output']
                if isinstance(output_path, str) and output_path.endswith('.csv'):
                    summary_path = output_path.replace('.csv', '_summary.txt')
                elif isinstance(output_path, str):
                    summary_path = output_path + '_summary.txt'
                else:
                    summary_path = 'results/cluster_summary.txt'
                print(f"\nGenerating summary report to {summary_path}...")
                write_cluster_summary(results, summary_path)
            
            if cancel_event and cancel_event.is_set():
                print("\n✗ Job cancelled")
                return False
            
            print("\n" + "=" * 60)
            print("JOB COMPLETED SUCCESSFULLY")
            print("=" * 60)
            print(f"\nResults saved to: {validated_config['output']}")
            print(f"Total clusters: {len(results)}")
            
            return True
            
        except Exception as e:
            print(f"\n✗ Unexpected error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _run_matching_job(
        self,
        validated_config: Dict[str, Any],
        cancel_event: Optional[threading.Event] = None
    ) -> bool:
        """Execute a matching job."""
        try:
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
            
            if cancel_event and cancel_event.is_set():
                print("\n✗ Job cancelled before matching")
                return False
            
            try:
                use_streaming = total_comparisons > 10000000
                if use_streaming:
                    print("ℹ Using streaming mode for very large dataset")
                    results = matcher.match(stream_to_file=validated_config['output'])
                else:
                    results = matcher.match()
                
                if cancel_event and cancel_event.is_set():
                    print("\n✗ Job cancelled during matching")
                    return False
            except KeyboardInterrupt:
                print("\n✗ Job cancelled by user")
                return False
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
            
            if cancel_event and cancel_event.is_set():
                print("\n✗ Job cancelled before writing results")
                return False
            
            if not use_streaming:
                print(f"\n[4/4] Writing results to {validated_config['output']}...")
                write_start_time = time.time()
                try:
                    write_results(results, validated_config['output'], stream=len(results) > 50000, config=validated_config)
                    write_time = time.time() - write_start_time
                    print(f"✓ Results written successfully in {write_time:.2f} seconds")
                except Exception as e:
                    print(f"✗ Error writing results: {e}")
                    return False
            else:
                print(f"✓ Results already streamed to {validated_config['output']}")
            
            if cancel_event and cancel_event.is_set():
                print("\n✗ Job cancelled")
                return False
            
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


def _get_project_root() -> Path:
    """Find the project root directory (where start-backend.py is located)."""
    # __file__ is src/job_runner.py, so go up to src/, then up to project root
    src_dir = Path(__file__).resolve().parent  # src/ directory
    project_root = src_dir.parent  # Project root (parent of src/)
    
    # Verify by checking for start-backend.py
    if (project_root / 'start-backend.py').exists():
        return project_root
    
    # Fallback: look for start-backend.py by going up the tree
    current = src_dir
    while current.parent != current:
        if (current / 'start-backend.py').exists():
            return current
        current = current.parent
    
    # Final fallback: use current working directory
    return Path.cwd()


def _resolve_path(path: str, project_root: Path) -> str:
    """
    Resolve a file path relative to project root if it's a relative path.
    
    Args:
        path: File path (can be absolute, relative, S3 URL, etc.)
        project_root: Project root directory
    
    Returns:
        Resolved absolute path (or original if S3 URL, MySQL table, etc.)
    """
    # Don't resolve S3 URLs, absolute paths, or non-CSV paths
    if path.startswith('s3://') or os.path.isabs(path) or not path.endswith('.csv'):
        return path
    
    # Resolve relative paths relative to project root
    resolved = (project_root / path).resolve()
    return str(resolved)


def validate_config_dict(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate configuration dictionary (without file path) and resolve environment variables.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        Validated configuration dictionary with environment variables resolved
    
    Raises:
        ValueError: If configuration is invalid
    """
    from jsonschema import validate, ValidationError, SchemaError
    from .config_validator import CONFIG_SCHEMA, _resolve_env_vars
    
    try:
        from dotenv import load_dotenv
        project_root = _get_project_root()
        env_file = project_root / '.env'
        if env_file.exists():
            load_dotenv(str(env_file))
    except ImportError:
        pass
    
    config = _resolve_env_vars(config)
    
    mode = config.get('mode', 'matching')
    
    # Mode-specific validation
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
        raise ValueError(f"Configuration validation error at '{error_path}': {e.message}")
    except SchemaError as e:
        raise ValueError(f"Configuration schema error: {str(e)}")
    
    project_root = _get_project_root()
    
    # Resolve and validate source paths based on mode
    if mode == 'clustering':
        source_keys = ['source1']
    elif mode == 'search':
        source_keys = ['source2']
    else:
        source_keys = ['source1', 'source2']
    
    for source_key in source_keys:
        source = config.get(source_key)
        
        if isinstance(source, str):
            if not source.endswith('.csv'):
                continue
            
            resolved_path = _resolve_path(source, project_root)
            config[source_key] = resolved_path
            
            if not os.path.exists(resolved_path):
                raise ValueError(
                    f"CSV file not found: {source}\n"
                    f"Resolved to: {resolved_path}\n"
                    f"Please check the path or use MySQL table name with mysql_credentials."
                )
    
    # Resolve output path
    output = config.get('output')
    if isinstance(output, str) and output.endswith('.csv'):
        resolved_output = _resolve_path(output, project_root)
        config['output'] = resolved_output
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(resolved_output)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
    
    return config

