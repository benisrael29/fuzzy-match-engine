import os
import pandas as pd
from typing import Optional, List, Dict, Iterator


def write_results(df: pd.DataFrame, output_path: str, stream: bool = False):
    """
    Write match results to CSV file.
    
    Args:
        df: DataFrame with match results
        output_path: Path to output CSV file
        stream: If True, stream results in chunks for large datasets
    """
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    if stream and len(df) > 50000:
        df.to_csv(output_path, index=False, chunksize=10000, mode='w')
    else:
        df.to_csv(output_path, index=False)


def write_results_streaming(results: Iterator[Dict], output_path: str, columns: List[str]):
    """
    Stream results directly to CSV file for very large datasets.
    
    Args:
        results: Iterator of result dictionaries
        output_path: Path to output CSV file
        columns: List of column names for the output
    """
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    import csv
    
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

