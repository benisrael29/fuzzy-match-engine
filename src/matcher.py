import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from multiprocessing import cpu_count, Manager, shared_memory, Value
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

_WORKER_SHARED_CACHE: Dict[str, Tuple[shared_memory.SharedMemory, np.ndarray]] = {}


def _get_shared_array(meta: Optional[Dict]) -> Optional[np.ndarray]:
    """Attach to a shared memory block and return a cached numpy view."""
    if not meta:
        return None
    name = meta['name']
    cached = _WORKER_SHARED_CACHE.get(name)
    if cached:
        return cached[1]
    shm = shared_memory.SharedMemory(name=name)
    arr = np.ndarray(meta['shape'], dtype=np.dtype(meta['dtype']), buffer=shm.buf)
    _WORKER_SHARED_CACHE[name] = (shm, arr)
    return arr


def _match_chunk_worker(chunk_indices: List[int], match_data: Dict) -> List[Dict]:
    """Worker function for parallel processing (must be at module level for pickling)."""
    import pandas as pd
    import numpy as np
    
    results = []
    source1_array = _get_shared_array(match_data.get('source1_shared'))
    source1_cols = match_data['source1_cols']
    source2_array = _get_shared_array(match_data.get('source2_shared'))
    source2_cols = match_data['source2_cols']
    source1_normalized_meta = match_data.get('source1_normalized_meta', {})
    source2_normalized_meta = match_data.get('source2_normalized_meta', {})
    column_analyses = match_data['column_analyses']
    blocking_index = match_data['blocking_index']
    threshold = match_data['threshold']
    undecided_range = match_data['undecided_range']
    return_all_matches = match_data['return_all_matches']
    early_termination = match_data.get('early_termination', True)
    perfect_match_threshold = match_data.get('perfect_match_threshold', 0.99)
    max_candidates = match_data.get('max_candidates')
    candidate_trim_strategy = match_data.get('candidate_trim_strategy', 'truncate')
    candidate_counter = match_data.get('candidate_counter')
    
    lower_bound = threshold - undecided_range
    
    def _get_blocking_keys_worker(row_dict: Dict, col_analyses: Dict) -> List[str]:
        keys = []
        for (col1, col2), _ in col_analyses.items():
            if col2 not in row_dict:
                continue
            value = str(row_dict[col2])
            if not value or value == 'nan':
                continue
            value_lower = value.lower()
            first_char = value_lower[0] if value_lower[0].isalnum() else '#'
            keys.append(f"{col2}:first:{first_char}")
            if len(value_lower) >= 2:
                keys.append(f"{col2}:2gram:{value_lower[:2]}")
            if len(value_lower) >= 3:
                keys.append(f"{col2}:3gram:{value_lower[:3]}")
                keys.append(f"{col2}:last3:{value_lower[-3:]}")
        return keys if keys else []
    
    def _get_candidate_indices_worker(row_dict: Dict, idx1: int, blocking_idx: Dict) -> np.ndarray:
        blocking_keys = _get_blocking_keys_worker(row_dict, column_analyses)
        candidate_sets = []
        key_groups = []
        for key in blocking_keys:
            if key in blocking_idx:
                arr = np.array(blocking_idx[key], dtype=np.int32)
                candidate_sets.append(arr)
                key_groups.append(_parse_block_group_worker(key))
        if not candidate_sets:
            return np.array([], dtype=np.int32)
        if len(candidate_sets) == 1 and not max_candidates:
            return candidate_sets[0]
        return _combine_candidates_worker(candidate_sets, key_groups)

    def _parse_block_group_worker(key: str) -> str:
        parts = key.split(':')
        return parts[1] if len(parts) > 1 else 'unknown'

    def _combine_candidates_worker(candidate_sets: List[np.ndarray], key_groups: List[str]) -> np.ndarray:
        combined = np.unique(np.concatenate(candidate_sets))
        if not max_candidates or len(combined) <= max_candidates:
            return combined
        _increment_candidate_counter_worker()
        if candidate_trim_strategy == 'fallback':
            filtered = _limit_candidates_by_priority_worker(candidate_sets, key_groups)
            if filtered is not None:
                return filtered
        return combined[:max_candidates]

    def _limit_candidates_by_priority_worker(candidate_sets: List[np.ndarray], key_groups: List[str]) -> Optional[np.ndarray]:
        priority_order = ['3gram', 'last3', 'word1', 'wordN', '2gram', 'first']
        for cutoff in range(len(priority_order)):
            allowed = set(priority_order[:cutoff + 1])
            filtered = [candidate_sets[idx] for idx, group in enumerate(key_groups) if group in allowed]
            if not filtered:
                continue
            merged = np.unique(np.concatenate(filtered))
            if len(merged) <= max_candidates:
                return merged
        return None

    def _increment_candidate_counter_worker():
        if candidate_counter is None:
            return
        with candidate_counter.get_lock():
            candidate_counter.value += 1
    
    if source1_array is None or source2_array is None:
        return results
    
    for idx1 in chunk_indices:
        row1_values = source1_array[idx1]
        row1_dict = {col: row1_values[pos] for pos, col in enumerate(source1_cols)}
        row1_series = pd.Series(row1_dict)
        
        candidate_indices = _get_candidate_indices_worker(row1_dict, idx1, blocking_index)
        
        if len(candidate_indices) == 0:
            continue
        
        matches = []
        best_match = None
        best_score = 0.0
        best_column_scores = {}
        
        for idx2 in candidate_indices:
            row2_values = source2_array[idx2]
            row2_dict = {col: row2_values[pos] for pos, col in enumerate(source2_cols)}
            
            total_score = 0.0
            total_weight = 0.0
            match_column_scores = {}
            
            for (col1, col2), analysis in column_analyses.items():
                if col1 not in row1_dict or col2 not in row2_dict:
                    continue
                
                val1 = row1_dict[col1]
                val2 = row2_dict[col2]
                
                norm1_arr = _get_shared_array(source1_normalized_meta.get(col1))
                norm2_arr = _get_shared_array(source2_normalized_meta.get(col2))
                
                if norm1_arr is not None:
                    normalized_val1 = norm1_arr[idx1]
                else:
                    normalized_val1 = val1
                
                if norm2_arr is not None:
                    normalized_val2 = norm2_arr[idx2]
                else:
                    normalized_val2 = val2
                
                if pd.isna(normalized_val1) or pd.isna(normalized_val2) or normalized_val1 == '' or normalized_val2 == '':
                    score = 0.0
                else:
                    algorithm = analysis['algorithm']
                    score = algorithm(str(normalized_val1), str(normalized_val2))
                
                weight = analysis.get('weight', 1.0)
                total_score += score * weight
                total_weight += weight
                match_column_scores[col1] = score
                
                if early_termination and not return_all_matches:
                    if total_weight > 0 and (total_score / total_weight) >= perfect_match_threshold:
                        break
            
            if total_weight > 0:
                overall_score = total_score / total_weight
                
                if early_termination and overall_score >= perfect_match_threshold:
                    best_score = overall_score
                    best_match = int(idx2)
                    best_column_scores = match_column_scores
                    break
                
                if return_all_matches:
                    if overall_score >= lower_bound:
                        matches.append({
                            'idx2': int(idx2),
                            'score': overall_score,
                            'column_scores': match_column_scores.copy()
                        })
                else:
                    if overall_score > best_score:
                        best_score = overall_score
                        best_match = int(idx2)
                        best_column_scores = match_column_scores.copy()
        
        if return_all_matches:
            for match in matches:
                row2_values = source2_array[match['idx2']]
                row2_dict = dict(zip(source2_cols, row2_values))
                row2_series = pd.Series(row2_dict)
                result = _create_result_worker(
                    row1_series, row2_series, match['score'],
                    match['column_scores'], idx1, match['idx2'],
                    source1_cols, source2_cols, threshold, undecided_range
                )
                results.append(result)
        elif best_match is not None:
            row2_values = source2_array[best_match]
            row2_dict = dict(zip(source2_cols, row2_values))
            row2_series = pd.Series(row2_dict)
            result = _create_result_worker(
                row1_series, row2_series, best_score,
                best_column_scores, idx1, best_match,
                source1_cols, source2_cols, threshold, undecided_range
            )
            results.append(result)
    
    return results


def _create_result_worker(
    row1: pd.Series,
    row2: pd.Series,
    overall_score: float,
    column_scores: Dict[str, float],
    idx1: int,
    idx2: int,
    source1_cols: List[str],
    source2_cols: List[str],
    threshold: float,
    undecided_range: float
) -> Dict:
    """Create result dictionary for a match (worker version)."""
    result = {}
    
    for col in source1_cols:
        result[f"source1_{col}"] = row1[col]
    
    for col in source2_cols:
        result[f"source2_{col}"] = row2[col]
    
    for col, score in column_scores.items():
        result[f"score_{col}"] = score
    
    result['overall_score'] = overall_score
    
    lower_bound = threshold - undecided_range
    upper_bound = threshold + undecided_range
    
    if overall_score >= upper_bound:
        result['match_result'] = 'accept'
    elif overall_score <= lower_bound:
        result['match_result'] = 'reject'
    else:
        result['match_result'] = 'undecided'
    
    result['source1_index'] = idx1
    result['source2_index'] = idx2
    
    return result


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
        self.early_termination = self.match_config.get('early_termination', True)
        self.perfect_match_threshold = self.match_config.get('perfect_match_threshold', 0.99)
        self.blocking_strategies = self.match_config.get(
            'blocking_strategies',
            ['first_char', 'three_gram', 'last_three']
        )
        self.max_block_size = self.match_config.get('max_block_size')
        self.skip_high_cardinality = self.match_config.get('skip_high_cardinality', True)
        self.blocking_stats: Dict[str, Any] = {'skipped_keys': 0, 'trimmed_keys': 0, 'largest_block': 0}
        self.max_candidates = self.match_config.get('max_candidates')
        self.candidate_trim_strategy = self.match_config.get('candidate_trim_strategy', 'truncate')
        self.candidate_stats: Dict[str, Any] = {'capped_rows': 0}
        self._candidate_counter = None
        self._shared_memory_blocks: List[shared_memory.SharedMemory] = []
        self._shared_meta: Dict[str, Any] = {}
        
        self._load_data()
        self._analyze_columns()
        self._pre_normalize_data()
        self._initialize_shared_memory()
        self._create_blocking_index()
    
    def _load_data(self):
        """Load data from both sources with optimized chunking for large datasets."""
        mysql_creds = self.config.get('mysql_credentials')
        s3_creds = self.config.get('s3_credentials')
        
        load_chunk_size = self.match_config.get('load_chunk_size', 100000)
        
        self.source1 = load_source(self.config['source1'], mysql_creds, s3_creds, chunk_size=load_chunk_size)
        self.source2 = load_source(self.config['source2'], mysql_creds, s3_creds, chunk_size=load_chunk_size)
    
    def _analyze_columns(self):
        """Analyze columns and select algorithms."""
        column_mappings = self.match_config.get('columns')
        self.column_analyses = analyze_columns(
            self.source1,
            self.source2,
            column_mappings
        )
    
    def _pre_normalize_data(self):
        """Pre-normalize all data columns based on their types using vectorized operations."""
        for (col1, col2), analysis in self.column_analyses.items():
            type1 = analysis['type1']
            type2 = analysis['type2']
            
            if col1 not in self.source1_normalized:
                normalized = self._normalize_column(self.source1[col1], type1)
                self.source1_normalized[col1] = normalized.values if isinstance(normalized, pd.Series) else normalized
            
            if col2 not in self.source2_normalized:
                normalized = self._normalize_column(self.source2[col2], type2)
                self.source2_normalized[col2] = normalized.values if isinstance(normalized, pd.Series) else normalized

    def _initialize_shared_memory(self):
        """Create shared memory blocks for heavy arrays to avoid per-worker serialization."""
        if not self.use_multiprocessing:
            self._shared_meta = {}
            return
        self._shared_meta = {
            'source1': self._df_to_shared_matrix(self.source1, 'source1'),
            'source2': self._df_to_shared_matrix(self.source2, 'source2'),
            'source1_normalized': {},
            'source2_normalized': {}
        }
        
        for col, values in self.source1_normalized.items():
            meta = self._series_to_shared_array(values, f"s1norm_{col}")
            if meta:
                self._shared_meta['source1_normalized'][col] = meta
        
        for col, values in self.source2_normalized.items():
            meta = self._series_to_shared_array(values, f"s2norm_{col}")
            if meta:
                self._shared_meta['source2_normalized'][col] = meta

    def _df_to_shared_matrix(self, df: pd.DataFrame, label: str) -> Optional[Dict[str, Any]]:
        if df is None or df.empty:
            return None
        safe_df = df.fillna('').astype(str)
        max_len = int(safe_df.apply(lambda col: col.str.len().max() or 1).max())
        max_len = max(1, max_len)
        dtype = f"<U{max_len}"
        array = safe_df.to_numpy(dtype=dtype)
        return self._create_shared_block(array, f"{label}_all")

    def _series_to_shared_array(self, series: Any, label: str) -> Optional[Dict[str, Any]]:
        if series is None:
            return None
        if isinstance(series, pd.Series):
            values = series
        else:
            values = pd.Series(series)
        if values.empty:
            return None
        if pd.api.types.is_numeric_dtype(values):
            array = values.to_numpy()
        else:
            safe = values.fillna('').astype(str)
            max_len = int(safe.str.len().max() or 1)
            max_len = max(1, max_len)
            dtype = f"<U{max_len}"
            array = safe.to_numpy(dtype=dtype)
        return self._create_shared_block(array, label)

    def _create_shared_block(self, array: np.ndarray, label: str) -> Dict[str, Any]:
        shm = shared_memory.SharedMemory(create=True, size=array.nbytes)
        shared_array = np.ndarray(array.shape, dtype=array.dtype, buffer=shm.buf)
        shared_array[:] = array[:]
        self._shared_memory_blocks.append(shm)
        return {
            'name': shm.name,
            'shape': array.shape,
            'dtype': array.dtype.str,
            'label': label
        }

    def _cleanup_shared_memory(self):
        for shm in getattr(self, '_shared_memory_blocks', []):
            try:
                shm.close()
            except FileNotFoundError:
                pass
            try:
                shm.unlink()
            except FileNotFoundError:
                pass
        self._shared_memory_blocks = []

    def _init_candidate_counter(self):
        if not self.max_candidates:
            self._candidate_counter = None
            return None
        counter = Value('i', 0)
        self._candidate_counter = counter
        return counter
    
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
        """Create enhanced blocking index with multiple strategies using vectorized operations."""
        self.blocking_index = {}
        self.blocking_stats = {'skipped_keys': 0, 'trimmed_keys': 0, 'largest_block': 0}
        
        size2 = len(self.source2)
        if size2 == 0:
            return
        
        blocking_columns = {col2 for _, col2 in self.column_analyses.keys() if col2 in self.source2.columns}
        if not blocking_columns:
            return
        
        lower_cache = {
            col: self.source2[col].astype(str).str.lower().fillna('') for col in blocking_columns
        }
        
        for idx in tqdm(range(size2), desc="Building blocking index", disable=size2 < 2000):
            for col in blocking_columns:
                value_lower = lower_cache[col].iat[idx]
                if not value_lower or value_lower == 'nan':
                    continue
                keys = self._generate_blocking_keys_for_value(col, value_lower)
                for key in keys:
                    bucket = self.blocking_index.setdefault(key, [])
                    bucket.append(idx)
        
        for key in list(self.blocking_index.keys()):
            bucket = self.blocking_index[key]
            block_size = len(bucket)
            if block_size > self.blocking_stats['largest_block']:
                self.blocking_stats['largest_block'] = block_size
            if self.max_block_size and block_size > self.max_block_size:
                if self.skip_high_cardinality:
                    del self.blocking_index[key]
                    self.blocking_stats['skipped_keys'] += 1
                    continue
                bucket = bucket[:self.max_block_size]
                block_size = len(bucket)
                self.blocking_stats['trimmed_keys'] += 1
            self.blocking_index[key] = np.array(bucket, dtype=np.int32)
    
    def _get_blocking_keys(self, row: pd.Series, idx: int) -> List[str]:
        """Generate multiple blocking keys for a row using various strategies."""
        keys = []
        
        for (col1, col2), analysis in self.column_analyses.items():
            if col2 not in row.index:
                continue
            
            value = str(row[col2])
            if not value or value == 'nan':
                continue
            
            keys.extend(self._generate_blocking_keys_for_value(col2, value.lower()))
        
        return keys if keys else [f'default:{idx % 100}']

    def _generate_blocking_keys_for_value(self, column: str, value_lower: str) -> List[str]:
        """Generate blocking keys for a single column value."""
        if not value_lower:
            return []
        
        keys = []
        length = len(value_lower)
        words = value_lower.split()
        
        if 'first_char' in self.blocking_strategies:
            first_char = value_lower[0]
            if not first_char.isalnum():
                first_char = '#'
            keys.append(f"{column}:first:{first_char}")
        
        if 'two_gram' in self.blocking_strategies and length >= 2:
            keys.append(f"{column}:2gram:{value_lower[:2]}")
        
        if 'three_gram' in self.blocking_strategies and length >= 3:
            keys.append(f"{column}:3gram:{value_lower[:3]}")
        
        if 'last_three' in self.blocking_strategies and length >= 3:
            keys.append(f"{column}:last3:{value_lower[-3:]}")
        
        if 'word_prefix' in self.blocking_strategies and words:
            first_word = words[0]
            if len(first_word) >= 2:
                keys.append(f"{column}:word1:{first_word[:2]}")
            if len(first_word) >= 3:
                keys.append(f"{column}:word1:{first_word[:3]}")
        
        if 'word_suffix' in self.blocking_strategies and len(words) > 1:
            last_word = words[-1]
            if len(last_word) >= 2:
                keys.append(f"{column}:wordN:{last_word[:2]}")
        
        if not keys:
            keys.append(f"{column}:fallback:{value_lower[:1]}")
        return keys
    
    def _get_candidate_indices(self, row: pd.Series, idx1: int) -> np.ndarray:
        """Get candidate indices for matching using blocking (returns numpy array for efficiency)."""
        blocking_keys = self._get_blocking_keys(row, idx1)
        candidate_sets = []
        key_groups = []
        
        for key in blocking_keys:
            if key in self.blocking_index:
                candidate_sets.append(self.blocking_index[key])
                key_groups.append(self._parse_block_group(key))
        
        if not candidate_sets:
            size2 = len(self.source2)
            if size2 > 10000:
                return np.array([], dtype=np.int32)
            return np.arange(size2, dtype=np.int32)
        
        if len(candidate_sets) == 1 and not self.max_candidates:
            return candidate_sets[0]
        
        return self._combine_candidate_sets(candidate_sets, key_groups)

    def _combine_candidate_sets(self, candidate_sets: List[np.ndarray], key_groups: List[str]) -> np.ndarray:
        combined = np.unique(np.concatenate(candidate_sets))
        if not self.max_candidates or len(combined) <= self.max_candidates:
            return combined
        self.candidate_stats['capped_rows'] = self.candidate_stats.get('capped_rows', 0) + 1
        if self.candidate_trim_strategy == 'fallback':
            filtered = self._limit_candidates_by_priority(candidate_sets, key_groups)
            if filtered is not None:
                return filtered
        return combined[:self.max_candidates]

    def _limit_candidates_by_priority(self, candidate_sets: List[np.ndarray], key_groups: List[str]) -> Optional[np.ndarray]:
        priority_order = ['3gram', 'last3', 'word1', 'wordN', '2gram', 'first']
        for cutoff in range(len(priority_order)):
            allowed = set(priority_order[:cutoff + 1])
            filtered = [candidate_sets[idx] for idx, group in enumerate(key_groups) if group in allowed]
            if not filtered:
                continue
            merged = np.unique(np.concatenate(filtered))
            if len(merged) <= self.max_candidates:
                return merged
        return None

    def _parse_block_group(self, key: str) -> str:
        parts = key.split(':')
        return parts[1] if len(parts) > 1 else 'unknown'
    
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
        """Match rows using parallel processing with multiprocessing for true parallelism."""
        from concurrent.futures import ProcessPoolExecutor, as_completed
        import pickle
        
        size1 = len(self.source1)
        chunk_size = min(self.chunk_size, max(1000, size1 // (self.num_workers * 2)))
        
        chunks = []
        for i in range(0, size1, chunk_size):
            chunk = list(range(i, min(i + chunk_size, size1)))
            chunks.append(chunk)
        
        results = []
        
        if self.use_multiprocessing and len(chunks) > 1 and self.num_workers > 1:
            try:
                match_data = self._prepare_match_data_for_workers()
                
                with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                    future_to_chunk = {}
                    for chunk in chunks:
                        future = executor.submit(_match_chunk_worker, chunk, match_data)
                        future_to_chunk[future] = chunk
                    
                    with tqdm(total=len(chunks), desc="Matching chunks") as pbar:
                        for future in as_completed(future_to_chunk):
                            try:
                                chunk_result = future.result()
                                results.extend(chunk_result)
                            except Exception as e:
                                print(f"Error in worker: {e}")
                            pbar.update(1)
            except Exception as e:
                print(f"Multiprocessing failed, falling back to sequential: {e}")
                return self._match_rows_sequential()
        else:
            chunk_results = [
                self._match_chunk(chunk) 
                for chunk in tqdm(chunks, desc="Matching chunks")
            ]
            for chunk_result in chunk_results:
                results.extend(chunk_result)

        if self._candidate_counter is not None:
            self.candidate_stats['capped_rows'] = self.candidate_stats.get('capped_rows', 0) + self._candidate_counter.value
            self._candidate_counter = None
        
        return results
    
    def _prepare_match_data_for_workers(self) -> Dict:
        """Prepare data structures for worker processes."""
        return {
            'source1_shared': self._shared_meta.get('source1'),
            'source1_cols': list(self.source1.columns),
            'source2_shared': self._shared_meta.get('source2'),
            'source2_cols': list(self.source2.columns),
            'source1_normalized_meta': self._shared_meta.get('source1_normalized', {}),
            'source2_normalized_meta': self._shared_meta.get('source2_normalized', {}),
            'column_analyses': self.column_analyses,
            'blocking_index': {k: (v.tolist() if isinstance(v, np.ndarray) else v) 
                             for k, v in self.blocking_index.items()},
            'threshold': self.threshold,
            'undecided_range': self.undecided_range,
            'return_all_matches': self.return_all_matches,
            'early_termination': self.early_termination,
            'perfect_match_threshold': self.perfect_match_threshold,
            'max_candidates': self.max_candidates,
            'candidate_trim_strategy': self.candidate_trim_strategy,
            'candidate_counter': self._init_candidate_counter()
        }
    
    def _match_chunk(self, chunk_indices: List[int], match_data: Optional[Dict] = None) -> List[Dict]:
        """Match a chunk of rows from source1 with optimized data access."""
        if match_data:
            return _match_chunk_worker(chunk_indices, match_data)
        
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
            
            if len(candidate_indices) == 0:
                continue
            
            matches = []
            best_match = None
            best_score = 0.0
            best_column_scores = {}
            
            for idx2 in candidate_indices:
                row2_values = source2_array[idx2]
                row2_dict = dict(zip(source2_cols, row2_values))
                
                total_score = 0.0
                total_weight = 0.0
                match_column_scores = {}
                
                for (col1, col2), analysis in self.column_analyses.items():
                    if col1 not in row1_dict or col2 not in row2_dict:
                        continue
                    
                    val1 = row1_dict[col1]
                    val2 = row2_dict[col2]
                    
                    norm1_arr = self.source1_normalized.get(col1)
                    norm2_arr = self.source2_normalized.get(col2)
                    
                    if norm1_arr is not None:
                        normalized_val1 = norm1_arr[idx1] if isinstance(norm1_arr, np.ndarray) else norm1_arr.iloc[idx1]
                    else:
                        normalized_val1 = val1
                    
                    if norm2_arr is not None:
                        normalized_val2 = norm2_arr[idx2] if isinstance(norm2_arr, np.ndarray) else norm2_arr.iloc[idx2]
                    else:
                        normalized_val2 = val2
                    
                    if pd.isna(normalized_val1) or pd.isna(normalized_val2) or normalized_val1 == '' or normalized_val2 == '':
                        score = 0.0
                    else:
                        algorithm = analysis['algorithm']
                        score = algorithm(str(normalized_val1), str(normalized_val2))
                    
                    weight = analysis.get('weight', 1.0)
                    total_score += score * weight
                    total_weight += weight
                    match_column_scores[col1] = score
                    
                    if self.early_termination and not self.return_all_matches:
                        if total_weight > 0 and (total_score / total_weight) >= self.perfect_match_threshold:
                            break
                
                if total_weight > 0:
                    overall_score = total_score / total_weight
                    
                    if self.early_termination and overall_score >= self.perfect_match_threshold:
                        best_score = overall_score
                        best_match = idx2
                        best_column_scores = match_column_scores
                        break
                    
                    if self.return_all_matches:
                        if overall_score >= lower_bound:
                            matches.append({
                                'idx2': int(idx2),
                                'score': overall_score,
                                'column_scores': match_column_scores.copy()
                            })
                    else:
                        if overall_score > best_score:
                            best_score = overall_score
                            best_match = idx2
                            best_column_scores = match_column_scores.copy()
            
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
                    best_column_scores, idx1, int(best_match)
                )
                results.append(result)
        
        return results
    
    def _match_rows_sequential(self) -> List[Dict]:
        """Match rows sequentially with optimized data access."""
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
            
            if len(candidate_indices) == 0:
                continue
            
            matches = []
            best_match = None
            best_score = 0.0
            best_column_scores = {}
            
            for idx2 in candidate_indices:
                row2_values = source2_array[idx2]
                row2_dict = dict(zip(source2_cols, row2_values))
                
                total_score = 0.0
                total_weight = 0.0
                match_column_scores = {}
                
                for (col1, col2), analysis in self.column_analyses.items():
                    if col1 not in row1_dict or col2 not in row2_dict:
                        continue
                    
                    val1 = row1_dict[col1]
                    val2 = row2_dict[col2]
                    
                    norm1_arr = self.source1_normalized.get(col1)
                    norm2_arr = self.source2_normalized.get(col2)
                    
                    if norm1_arr is not None:
                        normalized_val1 = norm1_arr[idx1] if isinstance(norm1_arr, np.ndarray) else norm1_arr.iloc[idx1]
                    else:
                        normalized_val1 = val1
                    
                    if norm2_arr is not None:
                        normalized_val2 = norm2_arr[idx2] if isinstance(norm2_arr, np.ndarray) else norm2_arr.iloc[idx2]
                    else:
                        normalized_val2 = val2
                    
                    if pd.isna(normalized_val1) or pd.isna(normalized_val2) or normalized_val1 == '' or normalized_val2 == '':
                        score = 0.0
                    else:
                        algorithm = analysis['algorithm']
                        score = algorithm(str(normalized_val1), str(normalized_val2))
                    
                    weight = analysis.get('weight', 1.0)
                    total_score += score * weight
                    total_weight += weight
                    match_column_scores[col1] = score
                    
                    if self.early_termination and not self.return_all_matches:
                        if total_weight > 0 and (total_score / total_weight) >= self.perfect_match_threshold:
                            break
                
                if total_weight > 0:
                    overall_score = total_score / total_weight
                    
                    if self.early_termination and overall_score >= self.perfect_match_threshold:
                        best_score = overall_score
                        best_match = idx2
                        best_column_scores = match_column_scores
                        break
                    
                    if self.return_all_matches:
                        if overall_score >= lower_bound:
                            matches.append({
                                'idx2': int(idx2),
                                'score': overall_score,
                                'column_scores': match_column_scores.copy()
                            })
                    else:
                        if overall_score > best_score:
                            best_score = overall_score
                            best_match = idx2
                            best_column_scores = match_column_scores.copy()
            
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
                    best_column_scores, idx1, int(best_match)
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
        self.candidate_stats['capped_rows'] = 0
        
        size1 = len(self.source1)
        size2 = len(self.source2)
        
        print(f"Matching {size1} rows from source1 against {size2} rows from source2")
        print(f"Using {self.num_workers} worker(s) for parallel processing")
        print(f"Blocking index contains {len(self.blocking_index)} keys")
        if self.max_block_size:
            skipped = self.blocking_stats.get('skipped_keys', 0)
            trimmed = self.blocking_stats.get('trimmed_keys', 0)
            largest = self.blocking_stats.get('largest_block', 0)
            if skipped or trimmed:
                print(f"Blocking stats: largest block={largest}, trimmed={trimmed}, skipped={skipped} (limit={self.max_block_size})")
        capped = self.candidate_stats.get('capped_rows', 0)
        if self.max_candidates and capped:
            print(f"Candidate limit reached for {capped} rows (limit={self.max_candidates}, strategy={self.candidate_trim_strategy})")
        
        if stream_to_file and size1 * size2 > 10000000:
            return self._match_with_streaming(stream_to_file, start_time)
        
        try:
            if self.use_multiprocessing and size1 > 1000 and self.num_workers > 1:
                results = self._match_rows_parallel()
            else:
                results = self._match_rows_sequential()
        finally:
            self._cleanup_shared_memory()
        
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
        
        try:
            write_results_streaming(result_generator(), output_path, columns, config=self.config)
        finally:
            self._cleanup_shared_memory()
        
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
