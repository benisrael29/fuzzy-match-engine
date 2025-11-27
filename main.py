#!/usr/bin/env python3
import argparse
import json
import os
import sys
from src.config_validator import validate_config
from src.matcher import FuzzyMatcher
from src.output_writer import write_results


def setup_config():
    """Interactive setup to generate configuration template."""
    print("Fuzzy Matching Engine - Configuration Setup")
    print("=" * 50)
    
    config = {}
    
    print("\nSource 1:")
    source1_type = input("Is source1 a CSV file or MySQL table? (csv/mysql) [csv]: ").strip().lower()
    if source1_type == 'mysql':
        config['source1'] = input("Enter table name: ").strip()
        config['mysql_credentials'] = {
            'host': input("MySQL host [localhost]: ").strip() or 'localhost',
            'user': input("MySQL user: ").strip(),
            'password': input("MySQL password: ").strip(),
            'database': input("MySQL database: ").strip()
        }
    else:
        config['source1'] = input("Enter CSV file path: ").strip()
    
    print("\nSource 2:")
    source2_type = input("Is source2 a CSV file or MySQL table? (csv/mysql) [csv]: ").strip().lower()
    if source2_type == 'mysql':
        if 'mysql_credentials' not in config:
            config['mysql_credentials'] = {
                'host': input("MySQL host [localhost]: ").strip() or 'localhost',
                'user': input("MySQL user: ").strip(),
                'password': input("MySQL password: ").strip(),
                'database': input("MySQL database: ").strip()
            }
        config['source2'] = input("Enter table name: ").strip()
    else:
        config['source2'] = input("Enter CSV file path: ").strip()
    
    config['output'] = input("\nOutput CSV file path [results/matches.csv]: ").strip() or 'results/matches.csv'
    
    advanced = input("\nConfigure advanced options? (y/n) [n]: ").strip().lower()
    if advanced == 'y':
        threshold = input("Match threshold (0-1) [0.85]: ").strip()
        if threshold:
            if 'match_config' not in config:
                config['match_config'] = {}
            config['match_config']['threshold'] = float(threshold)
        
        undecided = input("Undecided range (0-1) [0.05]: ").strip()
        if undecided:
            if 'match_config' not in config:
                config['match_config'] = {}
            config['match_config']['undecided_range'] = float(undecided)
    
    os.makedirs('config', exist_ok=True)
    config_path = 'config/example_config.json'
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\nConfiguration saved to {config_path}")
    return config_path


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Fuzzy Matching Engine - Match rows between CSV files or MySQL tables'
    )
    parser.add_argument(
        '--config',
        type=str,
        help='Path to JSON configuration file'
    )
    parser.add_argument(
        '--setup',
        action='store_true',
        help='Generate configuration template interactively'
    )
    parser.add_argument(
        '--ui',
        action='store_true',
        help='Launch interactive CLI job management UI'
    )
    parser.add_argument(
        '--gui',
        action='store_true',
        help='Launch visual GUI application'
    )
    
    args = parser.parse_args()
    
    if args.gui or (not args.config and not args.setup and not args.ui):
        from src.gui import run_gui
        run_gui()
        return
    
    if args.ui:
        from src.cli_ui import CLIUI
        ui = CLIUI()
        ui.run()
        return
    
    if args.setup:
        config_path = setup_config()
        print(f"\nRun with: python main.py --config {config_path}")
        return
    
    if not args.config:
        parser.print_help()
        print("\nOptions:")
        print("  --config FILE    : Run with configuration file")
        print("  --setup          : Generate configuration template")
        print("  --ui             : Launch CLI job management UI")
        print("  --gui            : Launch visual GUI application")
        print("\nRunning without arguments launches GUI by default")
        sys.exit(1)
    
    try:
        print(f"Loading configuration from {args.config}...")
        config = validate_config(args.config)
        
        print("Loading data sources...")
        matcher = FuzzyMatcher(config)
        
        print(f"Source 1: {len(matcher.source1)} rows")
        print(f"Source 2: {len(matcher.source2)} rows")
        print(f"Column pairs to match: {len(matcher.column_analyses)}")
        
        if matcher.use_blocking:
            print("Using blocking/indexing for performance optimization...")
        
        print("\nExecuting matching...")
        results = matcher.match()
        
        print(f"Found {len(results)} matches")
        print(f"\nMatch distribution:")
        if 'match_result' in results.columns:
            print(results['match_result'].value_counts().to_string())
        
        print(f"\nWriting results to {config['output']}...")
        write_results(results, config['output'])
        
        print("Done!")
        
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

