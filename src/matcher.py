import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from multiprocessing import cpu_count
from functools import partial
import time
from tqdm import tqdm
from .data_loader import load_source
from .column_analyzer import analyze_columns
from .normalizers import (
    normalize_phone,
    normalize_email,
    normalize_address,
    normalize_name,
    normalize_string
)


class FuzzyMatcher:
    """Main fuzzy matching engine with performance optimizations."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize matcher with configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.source1 = None
        self.source2 = None
        self.source1_normalized = {}
        self.source2_normalized = {}
        self.column_analyses = {}
        self.match_config = config.get('match_config', {})
        self.threshold = self.match_config.get('threshold', 0.85)
        self.undecided_range = self.match_config.get('undecided_range', 0.05)
        self.return_all_matches = self.match_config.get('return_all_matches', False)
        self.use_blocking = True
        self.blocking_index = {}
        self.num_workers = self.match_config.get('num_workers', min(cpu_count(), 8))
        self.chunk_size = self.match_config.get('chunk_size', 10000)
        self.use_multiprocessing = self.match_config.get('use_multiprocessing', True)
        
        self._load_data()
        self._analyze_columns()
        self._pre_normalize_data()
        self._create_blocking_index()
    
    def _load_data(self):
        """Load data from both sources."""
        mysql_creds = self.config.get('mysql_credentials')
        s3_creds = self.config.get('s3_credentials')
        
        self.source1 = load_source(self.config['source1'], mysql_creds, s3_creds)
        self.source2 = load_source(self.config['source2'], mysql_creds, s3_creds)
    
    def _analyze_columns(self):
        """Analyze columns and select algorithms."""
        column_mappings = self.match_config.get('columns')
        self.column_analyses = analyze_columns(
            self.source1,
            self.source2,
            column_mappings
        )
    
    def _pre_normalize_data(self):
        """Pre-normalize all data columns based on their types."""
        for (col1, col2), analysis in self.column_analyses.items():
            type1 = analysis['type1']
            type2 = analysis['type2']
            
            if col1 not in self.source1_normalized:
                self.source1_normalized[col1] = self._normalize_column(
                    self.source1[col1], type1
                )
            
            if col2 not in self.source2_normalized:
                self.source2_normalized[col2] = self._normalize_column(
                    self.source2[col2], type2
                )
    
    def _normalize_column(self, series: pd.Series, column_type: str) -> pd.Series:
        """Normalize an entire column vectorized."""
        if column_type == 'phone':
            return series.astype(str).apply(normalize_phone)
        elif column_type == 'email':
            return series.astype(str).apply(normalize_email)
        elif column_type == 'string_name':
            return series.astype(str).apply(normalize_name)
        elif 'address' in column_type.lower() or column_type == 'string_general':
            return series.astype(str).apply(normalize_string)
        else:
            return series
    
    def _create_blocking_index(self):
        """Create enhanced blocking index with multiple strategies."""
        self.blocking_index = {}
        
        for idx in tqdm(range(len(self.source2)), desc="Building blocking index", disable=len(self.source2) < 1000):
            row = self.source2.iloc[idx]
            blocking_keys = self._get_blocking_keys(row, idx)
            for key in blocking_keys:
                if key not in self.blocking_index:
                    self.blocking_index[key] = []
                self.blocking_index[key].append(idx)
    
    def _get_blocking_keys(self, row: pd.Series, idx: int) -> List[str]:
        """Generate multiple blocking keys for a row using various strategies."""
        keys = []
        
        for (col1, col2), analysis in self.column_analyses.items():
            if col2 not in row.index:
                continue
            
            value = str(row[col2])
            if not value or value == 'nan':
                continue
            
            value_lower = value.lower()
            
            first_char = value_lower[0] if value_lower[0].isalnum() else '#'
            keys.append(f"{col2}:first:{first_char}")
            
            if len(value_lower) >= 2:
                first_two = value_lower[:2]
                keys.append(f"{col2}:2gram:{first_two}")
            
            if len(value_lower) >= 3:
                first_three = value_lower[:3]
                keys.append(f"{col2}:3gram:{first_three}")
                last_three = value_lower[-3:] if len(value_lower) > 3 else first_three
                keys.append(f"{col2}:last3:{last_three}")
            
            if len(value_lower) >= 4:
                first_four = value_lower[:4]
                keys.append(f"{col2}:4gram:{first_four}")
            
            words = value_lower.split()
            if words:
                first_word = words[0]
                if len(first_word) >= 2:
                    keys.append(f"{col2}:word1:{first_word[:2]}")
                if len(first_word) >= 3:
                    keys.append(f"{col2}:word1:{first_word[:3]}")
            
            if len(words) > 1:
                last_word = words[-1]
                if len(last_word) >= 2:
                    keys.append(f"{col2}:wordN:{last_word[:2]}")
        
        return keys if keys else [f'default:{idx % 100}']
    
    def _get_candidate_indices(self, row: pd.Series, idx1: int) -> List[int]:
        """Get candidate indices for matching using blocking."""
        blocking_keys = self._get_blocking_keys(row, idx1)
        candidate_indices = set()
        
        for key in blocking_keys:
            if key in self.blocking_index:
                candidate_indices.update(self.blocking_index[key])
        
        if not candidate_indices:
            size2 = len(self.source2)
            if size2 > 10000:
                return []
            return list(range(size2))
        
        return list(candidate_indices)
    
    def _match_single_row(
        self,
        row1_tuple: Tuple,
        row1_idx: int,
        source1_cols: List[str],
        source2_cols: List[str]
    ) -> List[Dict]:
        """Match a single row from source1 against source2."""
        results = []
        lower_bound = self.threshold - self.undecided_range
        
        row1_dict = dict(zip(source1_cols, row1_tuple))
        row1_series = pd.Series(row1_dict)
        
        candidate_indices = self._get_candidate_indices(row1_series, row1_idx)
        
        if not candidate_indices:
            return results
        
        matches = []
        best_match = None
        best_score = 0.0
        best_column_scores = {}
        
        for idx2 in candidate_indices:
            row2 = self.source2.iloc[idx2]
            total_score = 0.0
            total_weight = 0.0
            match_column_scores = {}
            
            for (col1, col2), analysis in self.column_analyses.items():
                if col1 not in row1_dict or col2 not in row2.index:
                    continue
                
                val1 = row1_dict[col1]
                val2 = row2[col2]
                
                normalized_val1 = self.source1_normalized[col1].iloc[row1_idx] if col1 in self.source1_normalized else val1
                normalized_val2 = self.source2_normalized[col2].iloc[idx2] if col2 in self.source2_normalized else val2
                
                if pd.isna(normalized_val1) or pd.isna(normalized_val2) or normalized_val1 == '' or normalized_val2 == '':
                    score = 0.0
                else:
                    algorithm = analysis['algorithm']
                    score = algorithm(str(normalized_val1), str(normalized_val2))
                
                weight = analysis.get('weight', 1.0)
                total_score += score * weight
                total_weight += weight
                match_column_scores[col1] = score
            
            if total_weight > 0:
                overall_score = total_score / total_weight
                
                if self.return_all_matches:
                    if overall_score >= lower_bound:
                        matches.append({
                            'idx2': idx2,
                            'score': overall_score,
                            'column_scores': match_column_scores
                        })
                else:
                    if overall_score > best_score:
                        best_score = overall_score
                        best_match = idx2
                        best_column_scores = match_column_scores
        
        if self.return_all_matches:
            for match in matches:
                row2 = self.source2.iloc[match['idx2']]
                result = self._create_result(
                    row1_series, row2, match['score'], 
                    match['column_scores'], row1_idx, match['idx2']
                )
                results.append(result)
        elif best_match is not None:
            row2 = self.source2.iloc[best_match]
            result = self._create_result(
                row1_series, row2, best_score, 
                best_column_scores, row1_idx, best_match
            )
            results.append(result)
        
        return results
    
    def _match_rows_parallel(self) -> List[Dict]:
        """Match rows using parallel processing with threading."""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        size1 = len(self.source1)
        chunk_size = min(self.chunk_size, max(100, size1 // (self.num_workers * 4)))
        
        chunks = []
        for i in range(0, size1, chunk_size):
            chunk = list(range(i, min(i + chunk_size, size1)))
            chunks.append(chunk)
        
        results = []
        
        if self.use_multiprocessing and len(chunks) > 1 and self.num_workers > 1:
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                future_to_chunk = {
                    executor.submit(self._match_chunk, chunk): chunk 
                    for chunk in chunks
                }
                
                completed = 0
                with tqdm(total=len(chunks), desc="Matching chunks") as pbar:
                    for future in as_completed(future_to_chunk):
                        chunk_result = future.result()
                        results.extend(chunk_result)
                        completed += 1
                        pbar.update(1)
        else:
            chunk_results = [
                self._match_chunk(chunk) 
                for chunk in tqdm(chunks, desc="Matching chunks")
            ]
            for chunk_result in chunk_results:
                results.extend(chunk_result)
        
        return results
    
    def _match_chunk(self, chunk_indices: List[int]) -> List[Dict]:
        """Match a chunk of rows from source1."""
        results = []
        source1_cols = list(self.source1.columns)
        source2_cols = list(self.source2.columns)
        source1_array = self.source1.values
        source2_array = self.source2.values
        lower_bound = self.threshold - self.undecided_range
        
        for idx1 in chunk_indices:
            row1_values = source1_array[idx1]
            row1_dict = dict(zip(source1_cols, row1_values))
            row1_series = pd.Series(row1_dict)
            
            candidate_indices = self._get_candidate_indices(row1_series, idx1)
            
            if not candidate_indices:
                continue
            
            matches = []
            best_match = None
            best_score = 0.0
            best_column_scores = {}
            
            for idx2 in candidate_indices:
                row2_values = source2_array[idx2]
                row2_dict = dict(zip(source2_cols, row2_values))
                row2_series = pd.Series(row2_dict)
                
                total_score = 0.0
                total_weight = 0.0
                match_column_scores = {}
                
                for (col1, col2), analysis in self.column_analyses.items():
                    if col1 not in row1_dict or col2 not in row2_dict:
                        continue
                    
                    val1 = row1_dict[col1]
                    val2 = row2_dict[col2]
                    
                    normalized_val1 = self.source1_normalized[col1].iloc[idx1] if col1 in self.source1_normalized else val1
                    normalized_val2 = self.source2_normalized[col2].iloc[idx2] if col2 in self.source2_normalized else val2
                    
                    if pd.isna(normalized_val1) or pd.isna(normalized_val2) or normalized_val1 == '' or normalized_val2 == '':
                        score = 0.0
                    else:
                        algorithm = analysis['algorithm']
                        score = algorithm(str(normalized_val1), str(normalized_val2))
                    
                    weight = analysis.get('weight', 1.0)
                    total_score += score * weight
                    total_weight += weight
                    match_column_scores[col1] = score
                
                if total_weight > 0:
                    overall_score = total_score / total_weight
                    
                    if self.return_all_matches:
                        if overall_score >= lower_bound:
                            matches.append({
                                'idx2': idx2,
                                'score': overall_score,
                                'column_scores': match_column_scores
                            })
                    else:
                        if overall_score > best_score:
                            best_score = overall_score
                            best_match = idx2
                            best_column_scores = match_column_scores
            
            if self.return_all_matches:
                for match in matches:
                    row2_values = source2_array[match['idx2']]
                    row2_dict = dict(zip(source2_cols, row2_values))
                    row2_series = pd.Series(row2_dict)
                    result = self._create_result(
                        row1_series, row2_series, match['score'], 
                        match['column_scores'], idx1, match['idx2']
                    )
                    results.append(result)
            elif best_match is not None:
                row2_values = source2_array[best_match]
                row2_dict = dict(zip(source2_cols, row2_values))
                row2_series = pd.Series(row2_dict)
                result = self._create_result(
                    row1_series, row2_series, best_score, 
                    best_column_scores, idx1, best_match
                )
                results.append(result)
        
        return results
    
    def _match_rows_sequential(self) -> List[Dict]:
        """Match rows sequentially using itertuples for speed."""
        results = []
        lower_bound = self.threshold - self.undecided_range
        source1_cols = list(self.source1.columns)
        source2_cols = list(self.source2.columns)
        
        source1_array = self.source1.values
        source2_array = self.source2.values
        
        for idx1 in tqdm(range(len(self.source1)), desc="Matching rows"):
            row1_values = source1_array[idx1]
            row1_dict = dict(zip(source1_cols, row1_values))
            row1_series = pd.Series(row1_dict)
            
            candidate_indices = self._get_candidate_indices(row1_series, idx1)
            
            if not candidate_indices:
                continue
            
            matches = []
            best_match = None
            best_score = 0.0
            best_column_scores = {}
            
            for idx2 in candidate_indices:
                row2_values = source2_array[idx2]
                row2_dict = dict(zip(source2_cols, row2_values))
                row2_series = pd.Series(row2_dict)
                
                total_score = 0.0
                total_weight = 0.0
                match_column_scores = {}
                
                for (col1, col2), analysis in self.column_analyses.items():
                    if col1 not in row1_dict or col2 not in row2_dict:
                        continue
                    
                    val1 = row1_dict[col1]
                    val2 = row2_dict[col2]
                    
                    normalized_val1 = self.source1_normalized[col1].iloc[idx1] if col1 in self.source1_normalized else val1
                    normalized_val2 = self.source2_normalized[col2].iloc[idx2] if col2 in self.source2_normalized else val2
                    
                    if pd.isna(normalized_val1) or pd.isna(normalized_val2) or normalized_val1 == '' or normalized_val2 == '':
                        score = 0.0
                    else:
                        algorithm = analysis['algorithm']
                        score = algorithm(str(normalized_val1), str(normalized_val2))
                    
                    weight = analysis.get('weight', 1.0)
                    total_score += score * weight
                    total_weight += weight
                    match_column_scores[col1] = score
                
                if total_weight > 0:
                    overall_score = total_score / total_weight
                    
                    if self.return_all_matches:
                        if overall_score >= lower_bound:
                            matches.append({
                                'idx2': idx2,
                                'score': overall_score,
                                'column_scores': match_column_scores
                            })
                    else:
                        if overall_score > best_score:
                            best_score = overall_score
                            best_match = idx2
                            best_column_scores = match_column_scores
            
            if self.return_all_matches:
                for match in matches:
                    row2_values = source2_array[match['idx2']]
                    row2_dict = dict(zip(source2_cols, row2_values))
                    row2_series = pd.Series(row2_dict)
                    result = self._create_result(
                        row1_series, row2_series, match['score'], 
                        match['column_scores'], idx1, match['idx2']
                    )
                    results.append(result)
            elif best_match is not None:
                row2_values = source2_array[best_match]
                row2_dict = dict(zip(source2_cols, row2_values))
                row2_series = pd.Series(row2_dict)
                result = self._create_result(
                    row1_series, row2_series, best_score, 
                    best_column_scores, idx1, best_match
                )
                results.append(result)
        
        return results
    
    def _create_result(
        self,
        row1: pd.Series,
        row2: pd.Series,
        overall_score: float,
        column_scores: Dict[str, float],
        idx1: int,
        idx2: int
    ) -> Dict:
        """Create result dictionary for a match."""
        result = {}
        
        for col in self.source1.columns:
            result[f"source1_{col}"] = row1[col]
        
        for col in self.source2.columns:
            result[f"source2_{col}"] = row2[col]
        
        for col, score in column_scores.items():
            result[f"score_{col}"] = score
        
        result['overall_score'] = overall_score
        result['match_result'] = self._classify_match(overall_score)
        result['source1_index'] = idx1
        result['source2_index'] = idx2
        
        return result
    
    def _classify_match(self, score: float) -> str:
        """Classify match as accept, reject, or undecided."""
        lower_bound = self.threshold - self.undecided_range
        upper_bound = self.threshold + self.undecided_range
        
        if score >= upper_bound:
            return 'accept'
        elif score <= lower_bound:
            return 'reject'
        else:
            return 'undecided'
    
    def match(self, stream_to_file: Optional[str] = None) -> pd.DataFrame:
        """
        Execute matching process and return results.
        
        Args:
            stream_to_file: Optional file path to stream results directly (for very large datasets)
        
        Returns:
            DataFrame with match results
        """
        start_time = time.time()
        
        size1 = len(self.source1)
        size2 = len(self.source2)
        
        print(f"Matching {size1} rows from source1 against {size2} rows from source2")
        print(f"Using {self.num_workers} worker(s) for parallel processing")
        print(f"Blocking index contains {len(self.blocking_index)} keys")
        
        if stream_to_file and size1 * size2 > 10000000:
            return self._match_with_streaming(stream_to_file, start_time)
        
        if self.use_multiprocessing and size1 > 1000 and self.num_workers > 1:
            results = self._match_rows_parallel()
        else:
            results = self._match_rows_sequential()
        
        elapsed_time = time.time() - start_time
        print(f"Matching completed in {elapsed_time:.2f} seconds")
        print(f"Found {len(results)} matches")
        
        return pd.DataFrame(results)
    
    def _match_with_streaming(self, output_path: str, start_time: float) -> pd.DataFrame:
        """Match with streaming output for very large datasets."""
        from .output_writer import write_results_streaming
        
        size1 = len(self.source1)
        results_written = 0
        
        def result_generator():
            nonlocal results_written
            if self.use_multiprocessing and size1 > 1000 and self.num_workers > 1:
                chunk_results = self._match_rows_parallel()
                for result in chunk_results:
                    results_written += 1
                    yield result
            else:
                for idx1 in tqdm(range(len(self.source1)), desc="Matching rows"):
                    row1_values = self.source1.values[idx1]
                    row1_dict = dict(zip(self.source1.columns, row1_values))
                    row1_series = pd.Series(row1_dict)
                    
                    candidate_indices = self._get_candidate_indices(row1_series, idx1)
                    if not candidate_indices:
                        continue
                    
                    best_match = None
                    best_score = 0.0
                    best_column_scores = {}
                    lower_bound = self.threshold - self.undecided_range
                    source2_array = self.source2.values
                    source2_cols = list(self.source2.columns)
                    
                    for idx2 in candidate_indices:
                        row2_values = source2_array[idx2]
                        row2_dict = dict(zip(source2_cols, row2_values))
                        row2_series = pd.Series(row2_dict)
                        
                        total_score = 0.0
                        total_weight = 0.0
                        match_column_scores = {}
                        
                        for (col1, col2), analysis in self.column_analyses.items():
                            if col1 not in row1_dict or col2 not in row2_dict:
                                continue
                            
                            val1 = row1_dict[col1]
                            val2 = row2_dict[col2]
                            
                            normalized_val1 = self.source1_normalized[col1].iloc[idx1] if col1 in self.source1_normalized else val1
                            normalized_val2 = self.source2_normalized[col2].iloc[idx2] if col2 in self.source2_normalized else val2
                            
                            if pd.isna(normalized_val1) or pd.isna(normalized_val2) or normalized_val1 == '' or normalized_val2 == '':
                                score = 0.0
                            else:
                                algorithm = analysis['algorithm']
                                score = algorithm(str(normalized_val1), str(normalized_val2))
                            
                            weight = analysis.get('weight', 1.0)
                            total_score += score * weight
                            total_weight += weight
                            match_column_scores[col1] = score
                        
                        if total_weight > 0:
                            overall_score = total_score / total_weight
                            if overall_score > best_score:
                                best_score = overall_score
                                best_match = idx2
                                best_column_scores = match_column_scores
                    
                    if best_match is not None:
                        row2_values = source2_array[best_match]
                        row2_dict = dict(zip(source2_cols, row2_values))
                        row2_series = pd.Series(row2_dict)
                        result = self._create_result(
                            row1_series, row2_series, best_score, 
                            best_column_scores, idx1, best_match
                        )
                        results_written += 1
                        yield result
        
        columns = None
        if len(self.source1) > 0:
            columns = [f"source1_{col}" for col in self.source1.columns]
            columns.extend([f"source2_{col}" for col in self.source2.columns])
            columns.extend([f"score_{col1}" for col1, _ in self.column_analyses.keys()])
            columns.extend(['overall_score', 'match_result', 'source1_index', 'source2_index'])
        
        write_results_streaming(result_generator(), output_path, columns, config=self.config)
        
        elapsed_time = time.time() - start_time
        print(f"Matching completed in {elapsed_time:.2f} seconds")
        print(f"Found {results_written} matches (streamed to {output_path})")
        
        from .output_writer import _is_s3_path, _read_from_s3, _is_mysql_table, _get_mysql_engine
        from sqlalchemy import text
        
        if _is_s3_path(output_path):
            return _read_from_s3(output_path, config=self.config)
        elif _is_mysql_table(output_path, self.config.get('mysql_credentials')):
            mysql_credentials = self.config.get('mysql_credentials')
            engine = _get_mysql_engine(mysql_credentials)
            return pd.read_sql(text(f"SELECT * FROM `{output_path}`"), engine)
        else:
            return pd.read_csv(output_path)
