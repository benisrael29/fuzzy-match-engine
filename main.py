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
        '--search',
        action='store_true',
        help='Enable search mode (search for a person in master dataset)'
    )
    parser.add_argument(
        '--master',
        type=str,
        help='Path to master dataset (CSV, MySQL table, or S3 URL) - required for search mode'
    )
    parser.add_argument(
        '--query',
        type=str,
        help='JSON string with person details to search for (e.g., \'{"fname":"John","lname":"Smith"}\')'
    )
    parser.add_argument(
        '--query-file',
        type=str,
        help='Path to JSON file containing query record'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        help='Match threshold (0-1) for search mode'
    )
    parser.add_argument(
        '--max-results',
        type=int,
        help='Maximum number of results to return (search mode only)'
    )
    
    args = parser.parse_args()
    
    if args.ui:
        from src.cli_ui import CLIUI
        ui = CLIUI()
        ui.run()
        return
    
    if args.setup:
        config_path = setup_config()
        print(f"\nRun with: python main.py --config {config_path}")
        return
    
    if args.search:
        if not args.master:
            print("Error: --master is required for search mode", file=sys.stderr)
            sys.exit(1)
        
        if not args.query and not args.query_file:
            print("Error: Either --query or --query-file is required for search mode", file=sys.stderr)
            sys.exit(1)
        
        try:
            if args.query_file:
                with open(args.query_file, 'r') as f:
                    query_record = json.load(f)
            else:
                query_record = json.loads(args.query)
            
            print(f"Searching in master dataset: {args.master}")
            print(f"Query: {json.dumps(query_record, indent=2)}")
            
            if args.config:
                config = validate_config(args.config)
                if config.get('mode') != 'search':
                    config['mode'] = 'search'
                if 'source2' not in config:
                    config['source2'] = args.master
                if args.threshold is not None:
                    if 'match_config' not in config:
                        config['match_config'] = {}
                    config['match_config']['threshold'] = args.threshold
                    config['match_config']['return_all_matches'] = True
                
                matcher = FuzzyMatcher(config)
            else:
                threshold = args.threshold if args.threshold is not None else 0.85
                matcher = FuzzyMatcher.create_search_matcher(
                    master_source=args.master,
                    query_record=query_record,
                    threshold=threshold
                )
            
            print(f"\nMaster dataset: {len(matcher.source2)} rows")
            print(f"Column mappings: {len(matcher.column_analyses)}")
            print("\nExecuting search...")
            
            results = matcher.search(
                query_record=query_record,
                threshold=args.threshold,
                max_results=args.max_results
            )
            
            print(f"\nFound {len(results)} matches")
            if results:
                print("\nTop matches:")
                for i, result in enumerate(results[:10], 1):
                    print(f"\n{i}. Score: {result['overall_score']:.3f} ({result['match_result']})")
                    for key, val in result.items():
                        if key.startswith('master_') and val:
                            print(f"   {key[7:]}: {val}")
            
            output_file = 'results/search_results.json'
            os.makedirs('results', exist_ok=True)
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            print(f"\nResults saved to {output_file}")
            print("Done!")
            
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON in query - {e}", file=sys.stderr)
            sys.exit(1)
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
        return
    
    if not args.config:
        parser.print_help()
        print("\nOptions:")
        print("  --config FILE    : Run with configuration file")
        print("  --setup          : Generate configuration template")
        print("  --ui             : Launch CLI job management UI")
        print("  --search         : Enable search mode")
        print("  --master PATH    : Master dataset path (for search mode)")
        print("  --query JSON     : Query record as JSON string (for search mode)")
        print("  --query-file PATH: Query record from JSON file (for search mode)")
        print("\nFor web service, use: python web_server.py")
        sys.exit(1)
    
    try:
        print(f"Loading configuration from {args.config}...")
        config = validate_config(args.config)
        
        mode = config.get('mode', 'matching')
        
        if mode == 'clustering':
            from src.clusterer import Clusterer
            from src.output_writer import write_cluster_results, write_cluster_summary
            
            print("Loading data source...")
            clusterer = Clusterer(config)
            
            print(f"Source: {len(clusterer.source)} rows")
            print(f"Columns to cluster: {len(clusterer.column_analyses)}")
            
            if clusterer.use_blocking:
                print("Using blocking/indexing for performance optimization...")
            
            print("\nExecuting clustering...")
            results = clusterer.cluster()
            
            print(f"\nWriting results to {config['output']}...")
            write_cluster_results(results, config['output'], config=config)
            
            if config.get('cluster_config', {}).get('generate_summary', False):
                output_path = config['output']
                if isinstance(output_path, str) and output_path.endswith('.csv'):
                    summary_path = output_path.replace('.csv', '_summary.txt')
                elif isinstance(output_path, str):
                    summary_path = output_path + '_summary.txt'
                else:
                    summary_path = 'results/cluster_summary.txt'
                print(f"\nGenerating summary report to {summary_path}...")
                write_cluster_summary(results, summary_path)
            
            print("Done!")
        else:
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
            write_results(results, config['output'], config=config)
            
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

