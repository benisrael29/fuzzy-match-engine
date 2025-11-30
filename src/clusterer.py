import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from multiprocessing import cpu_count, Manager, shared_memory, Value
from functools import partial
import time
from tqdm import tqdm
from .data_loader import load_source
from .column_analyzer import detect_column_type, select_algorithm
from .normalizers import (
    normalize_phone,
    normalize_email,
    normalize_address,
    normalize_name,
    normalize_string
)


class UnionFind:
    """Union-Find (Disjoint Set) data structure for efficient cluster building."""
    
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n
    
    def find(self, x: int) -> int:
        """Find root of x with path compression."""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x: int, y: int):
        """Union two sets by rank."""
        root_x = self.find(x)
        root_y = self.find(y)
        
        if root_x == root_y:
            return
        
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1
    
    def get_clusters(self) -> Dict[int, List[int]]:
        """Get all clusters as a dictionary mapping root to list of indices."""
        clusters = {}
        for i in range(len(self.parent)):
            root = self.find(i)
            if root not in clusters:
                clusters[root] = []
            clusters[root].append(i)
        return clusters


def _cluster_chunk_worker(chunk_pairs: List[Tuple[int, int]], match_data: Dict) -> List[Tuple[int, int, float]]:
    """Worker function for parallel clustering."""
    import pandas as pd
    import numpy as np
    
    results = []
    source_array = _get_shared_array(match_data.get('source_shared'))
    source_cols = match_data['source_cols']
    source_normalized_meta = match_data.get('source_normalized_meta', {})
    column_analyses = match_data['column_analyses']
    threshold = match_data['threshold']
    
    if source_array is None:
        return results
    
    for idx1, idx2 in chunk_pairs:
        row1_values = source_array[idx1]
        row1_dict = {col: row1_values[pos] for pos, col in enumerate(source_cols)}
        
        row2_values = source_array[idx2]
        row2_dict = {col: row2_values[pos] for pos, col in enumerate(source_cols)}
        
        total_score = 0.0
        total_weight = 0.0
        
        for col, analysis in column_analyses.items():
            if col not in row1_dict or col not in row2_dict:
                continue
            
            val1 = row1_dict[col]
            val2 = row2_dict[col]
            
            norm1_arr = _get_shared_array(source_normalized_meta.get(col))
            norm2_arr = _get_shared_array(source_normalized_meta.get(col))
            
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
        
        if total_weight > 0:
            overall_score = total_score / total_weight
            if overall_score >= threshold:
                results.append((idx1, idx2, overall_score))
    
    return results


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


class Clusterer:
    """Clustering engine for finding duplicates and grouping similar records."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize clusterer with configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.source = None
        self.source_normalized = {}
        self.column_analyses = {}
        self.cluster_config = config.get('cluster_config', {})
        self.threshold = self.cluster_config.get('threshold', 0.85)
        self.num_workers = self.cluster_config.get('num_workers', min(cpu_count(), 8))
        self.chunk_size = self.cluster_config.get('chunk_size', 10000)
        self.use_multiprocessing = self.cluster_config.get('use_multiprocessing', True)
        self.use_blocking = self.cluster_config.get('use_blocking', True)
        self.blocking_index = {}
        self.blocking_strategies = self.cluster_config.get(
            'blocking_strategies',
            ['first_char', 'three_gram', 'last_three']
        )
        self.max_block_size = self.cluster_config.get('max_block_size')
        self.skip_high_cardinality = self.cluster_config.get('skip_high_cardinality', True)
        self._shared_memory_blocks: List[shared_memory.SharedMemory] = []
        self._shared_meta: Dict[str, Any] = {}
        
        self._load_data()
        self._analyze_columns()
        self._pre_normalize_data()
        self._initialize_shared_memory()
        if self.use_blocking:
            self._create_blocking_index()
    
    def _load_data(self):
        """Load data from source."""
        mysql_creds = self.config.get('mysql_credentials')
        s3_creds = self.config.get('s3_credentials')
        
        load_chunk_size = self.cluster_config.get('load_chunk_size', 100000)
        
        self.source = load_source(self.config['source1'], mysql_creds, s3_creds, chunk_size=load_chunk_size)
    
    def _analyze_columns(self):
        """Analyze columns and select algorithms."""
        column_mappings = self.cluster_config.get('columns')
        
        if column_mappings:
            self.column_analyses = {}
            for mapping in column_mappings:
                col = mapping.get('source1')
                if col not in self.source.columns:
                    raise ValueError(f"Column '{col}' not found in source")
                
                col_type = detect_column_type(self.source[col], col)
                algorithm = select_algorithm(col_type)
                
                self.column_analyses[col] = {
                    'type': col_type,
                    'algorithm': algorithm,
                    'weight': mapping.get('weight', 1.0)
                }
        else:
            self.column_analyses = {}
            for col in self.source.columns:
                col_type = detect_column_type(self.source[col], col)
                algorithm = select_algorithm(col_type)
                
                self.column_analyses[col] = {
                    'type': col_type,
                    'algorithm': algorithm,
                    'weight': 1.0
                }
    
    def _pre_normalize_data(self):
        """Pre-normalize all data columns based on their types."""
        for col, analysis in self.column_analyses.items():
            if col not in self.source_normalized:
                normalized = self._normalize_column(self.source[col], analysis['type'])
                self.source_normalized[col] = normalized.values if isinstance(normalized, pd.Series) else normalized
    
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
    
    def _initialize_shared_memory(self):
        """Create shared memory blocks for heavy arrays."""
        if not self.use_multiprocessing:
            self._shared_meta = {}
            return
        
        self._shared_meta = {
            'source': self._df_to_shared_matrix(self.source, 'source'),
            'source_normalized': {}
        }
        
        for col, values in self.source_normalized.items():
            meta = self._series_to_shared_array(values, f"norm_{col}")
            if meta:
                self._shared_meta['source_normalized'][col] = meta
    
    def _df_to_shared_matrix(self, df: pd.DataFrame, label: str) -> Optional[Dict[str, Any]]:
        """Convert DataFrame to shared memory matrix."""
        if df is None or df.empty:
            return None
        safe_df = df.fillna('').astype(str)
        max_len = int(safe_df.apply(lambda col: col.str.len().max() or 1).max())
        max_len = max(1, max_len)
        dtype = f"<U{max_len}"
        array = safe_df.to_numpy(dtype=dtype)
        return self._create_shared_block(array, f"{label}_all")
    
    def _series_to_shared_array(self, series: Any, label: str) -> Optional[Dict[str, Any]]:
        """Convert Series to shared memory array."""
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
        """Create shared memory block."""
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
        """Clean up shared memory blocks."""
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
    
    def _create_blocking_index(self):
        """Create blocking index for performance optimization."""
        self.blocking_index = {}
        
        size = len(self.source)
        if size == 0:
            return
        
        blocking_columns = set(self.column_analyses.keys())
        if not blocking_columns:
            return
        
        lower_cache = {
            col: self.source[col].astype(str).str.lower().fillna('') for col in blocking_columns
        }
        
        for idx in tqdm(range(size), desc="Building blocking index", disable=size < 2000):
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
            if self.max_block_size and block_size > self.max_block_size:
                if self.skip_high_cardinality:
                    del self.blocking_index[key]
                    continue
                bucket = bucket[:self.max_block_size]
            self.blocking_index[key] = np.array(bucket, dtype=np.int32)
    
    def _generate_blocking_keys_for_value(self, column: str, value_lower: str) -> List[str]:
        """Generate blocking keys for a single column value."""
        if not value_lower:
            return []
        
        keys = []
        length = len(value_lower)
        
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
        
        if not keys:
            keys.append(f"{column}:fallback:{value_lower[:1]}")
        return keys
    
    def _get_candidate_indices(self, row: pd.Series, idx: int) -> np.ndarray:
        """Get candidate indices for clustering using blocking."""
        if not self.use_blocking:
            size = len(self.source)
            all_indices = np.arange(idx + 1, size, dtype=np.int32)
            return all_indices
        
        blocking_keys = []
        for col in self.column_analyses.keys():
            if col not in row.index:
                continue
            value = str(row[col])
            if not value or value == 'nan':
                continue
            blocking_keys.extend(self._generate_blocking_keys_for_value(col, value.lower()))
        
        if not blocking_keys:
            size = len(self.source)
            return np.arange(idx + 1, size, dtype=np.int32)
        
        candidate_sets = []
        for key in blocking_keys:
            if key in self.blocking_index:
                candidate_sets.append(self.blocking_index[key])
        
        if not candidate_sets:
            size = len(self.source)
            return np.arange(idx + 1, size, dtype=np.int32)
        
        combined = np.unique(np.concatenate(candidate_sets))
        return combined[combined > idx]
    
    def _cluster_rows_parallel(self) -> List[Tuple[int, int, float]]:
        """Cluster rows using parallel processing."""
        from concurrent.futures import ProcessPoolExecutor, as_completed
        
        size = len(self.source)
        
        if self.use_blocking:
            pairs = []
            for idx1 in range(size):
                row1 = self.source.iloc[idx1]
                candidate_indices = self._get_candidate_indices(row1, idx1)
                for idx2 in candidate_indices:
                    pairs.append((idx1, int(idx2)))
        else:
            pairs = [(i, j) for i in range(size) for j in range(i + 1, size)]
        
        if not pairs:
            return []
        
        chunk_size = min(self.chunk_size, max(1000, len(pairs) // (self.num_workers * 2)))
        chunks = []
        for i in range(0, len(pairs), chunk_size):
            chunk = pairs[i:i + chunk_size]
            chunks.append(chunk)
        
        results = []
        
        if self.use_multiprocessing and len(chunks) > 1 and self.num_workers > 1:
            try:
                match_data = self._prepare_cluster_data_for_workers()
                
                with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                    future_to_chunk = {}
                    for chunk in chunks:
                        future = executor.submit(_cluster_chunk_worker, chunk, match_data)
                        future_to_chunk[future] = chunk
                    
                    with tqdm(total=len(chunks), desc="Clustering chunks") as pbar:
                        for future in as_completed(future_to_chunk):
                            try:
                                chunk_result = future.result()
                                results.extend(chunk_result)
                            except Exception as e:
                                print(f"Error in worker: {e}")
                            pbar.update(1)
            except Exception as e:
                print(f"Multiprocessing failed, falling back to sequential: {e}")
                return self._cluster_rows_sequential()
        else:
            chunk_results = [
                _cluster_chunk_worker(chunk, self._prepare_cluster_data_for_workers())
                for chunk in tqdm(chunks, desc="Clustering chunks")
            ]
            for chunk_result in chunk_results:
                results.extend(chunk_result)
        
        return results
    
    def _cluster_rows_sequential(self) -> List[Tuple[int, int, float]]:
        """Cluster rows sequentially."""
        results = []
        size = len(self.source)
        source_cols = list(self.source.columns)
        source_array = self.source.values
        
        for idx1 in tqdm(range(size), desc="Clustering rows"):
            row1_values = source_array[idx1]
            row1_dict = dict(zip(source_cols, row1_values))
            row1_series = pd.Series(row1_dict)
            
            candidate_indices = self._get_candidate_indices(row1_series, idx1)
            
            if len(candidate_indices) == 0:
                continue
            
            for idx2 in candidate_indices:
                row2_values = source_array[idx2]
                row2_dict = dict(zip(source_cols, row2_values))
                
                total_score = 0.0
                total_weight = 0.0
                
                for col, analysis in self.column_analyses.items():
                    if col not in row1_dict or col not in row2_dict:
                        continue
                    
                    val1 = row1_dict[col]
                    val2 = row2_dict[col]
                    
                    norm1_arr = self.source_normalized.get(col)
                    norm2_arr = self.source_normalized.get(col)
                    
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
                
                if total_weight > 0:
                    overall_score = total_score / total_weight
                    if overall_score >= self.threshold:
                        results.append((idx1, int(idx2), overall_score))
        
        return results
    
    def _prepare_cluster_data_for_workers(self) -> Dict:
        """Prepare data structures for worker processes."""
        return {
            'source_shared': self._shared_meta.get('source'),
            'source_cols': list(self.source.columns),
            'source_normalized_meta': self._shared_meta.get('source_normalized', {}),
            'column_analyses': self.column_analyses,
            'threshold': self.threshold
        }
    
    def cluster(self) -> pd.DataFrame:
        """
        Execute clustering process and return results with cluster IDs.
        
        Returns:
            DataFrame with original data plus cluster_id and cluster_size columns
        """
        start_time = time.time()
        
        size = len(self.source)
        print(f"Clustering {size} records")
        print(f"Using {self.num_workers} worker(s) for parallel processing")
        if self.use_blocking:
            print(f"Blocking index contains {len(self.blocking_index)} keys")
        
        try:
            if self.use_multiprocessing and size > 1000 and self.num_workers > 1:
                matches = self._cluster_rows_parallel()
            else:
                matches = self._cluster_rows_sequential()
        finally:
            self._cleanup_shared_memory()
        
        print(f"Found {len(matches)} matching pairs")
        
        uf = UnionFind(size)
        for idx1, idx2, score in matches:
            uf.union(idx1, idx2)
        
        clusters = uf.get_clusters()
        cluster_map = {}
        cluster_id = 0
        for root, indices in clusters.items():
            for idx in indices:
                cluster_map[idx] = cluster_id
            cluster_id += 1
        
        result_df = self.source.copy()
        result_df['cluster_id'] = result_df.index.map(cluster_map)
        
        cluster_sizes = {idx: len(indices) for idx, indices in clusters.items()}
        result_df['cluster_size'] = result_df.index.map(lambda x: cluster_sizes.get(uf.find(x), 1))
        
        elapsed_time = time.time() - start_time
        print(f"Clustering completed in {elapsed_time:.2f} seconds")
        print(f"Found {len(clusters)} clusters")
        
        return result_df

