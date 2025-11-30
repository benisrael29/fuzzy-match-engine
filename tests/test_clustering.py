import pytest
import pandas as pd
import os
import json
import tempfile
from src.clusterer import Clusterer, UnionFind
from src.config_validator import validate_config


class TestUnionFind:
    """Test Union-Find data structure."""
    
    def test_union_find_basic(self):
        uf = UnionFind(5)
        assert uf.find(0) == 0
        assert uf.find(1) == 1
        
        uf.union(0, 1)
        assert uf.find(0) == uf.find(1)
        
        uf.union(2, 3)
        uf.union(3, 4)
        assert uf.find(2) == uf.find(4)
        assert uf.find(0) != uf.find(2)
    
    def test_union_find_clusters(self):
        uf = UnionFind(6)
        uf.union(0, 1)
        uf.union(1, 2)
        uf.union(3, 4)
        
        clusters = uf.get_clusters()
        assert len(clusters) == 3
        assert len(clusters[uf.find(0)]) == 3
        assert len(clusters[uf.find(3)]) == 2
        assert len(clusters[uf.find(5)]) == 1


class TestClusterer:
    """Test clustering functionality."""
    
    def test_clusterer_basic(self):
        data = {
            'name': ['John Doe', 'John Doe', 'Jane Smith', 'Jane Smith', 'Bob Johnson'],
            'email': ['john@example.com', 'john@example.com', 'jane@example.com', 'jane@example.com', 'bob@example.com']
        }
        df = pd.DataFrame(data)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.to_csv(f.name, index=False)
            temp_path = f.name
        
        try:
            config = {
                'mode': 'clustering',
                'source1': temp_path,
                'output': 'results/test_clusters.csv',
                'cluster_config': {
                    'threshold': 0.85
                }
            }
            
            clusterer = Clusterer(config)
            results = clusterer.cluster()
            
            assert 'cluster_id' in results.columns
            assert 'cluster_size' in results.columns
            assert len(results) == 5
            
            cluster_ids = results['cluster_id'].values
            assert cluster_ids[0] == cluster_ids[1]
            assert cluster_ids[2] == cluster_ids[3]
            assert cluster_ids[0] != cluster_ids[2]
            assert cluster_ids[4] != cluster_ids[0]
        finally:
            os.unlink(temp_path)
            if os.path.exists('results/test_clusters.csv'):
                os.unlink('results/test_clusters.csv')
    
    def test_clusterer_all_unique(self):
        data = {
            'name': ['John Doe', 'Jane Smith', 'Bob Johnson', 'Alice Brown'],
            'email': ['john@example.com', 'jane@example.com', 'bob@example.com', 'alice@example.com']
        }
        df = pd.DataFrame(data)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.to_csv(f.name, index=False)
            temp_path = f.name
        
        try:
            config = {
                'mode': 'clustering',
                'source1': temp_path,
                'output': 'results/test_clusters_unique.csv',
                'cluster_config': {
                    'threshold': 0.85
                }
            }
            
            clusterer = Clusterer(config)
            results = clusterer.cluster()
            
            unique_clusters = results['cluster_id'].nunique()
            assert unique_clusters == len(results)
        finally:
            os.unlink(temp_path)
            if os.path.exists('results/test_clusters_unique.csv'):
                os.unlink('results/test_clusters_unique.csv')
    
    def test_clusterer_all_duplicates(self):
        data = {
            'name': ['John Doe'] * 5,
            'email': ['john@example.com'] * 5
        }
        df = pd.DataFrame(data)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.to_csv(f.name, index=False)
            temp_path = f.name
        
        try:
            config = {
                'mode': 'clustering',
                'source1': temp_path,
                'output': 'results/test_clusters_dups.csv',
                'cluster_config': {
                    'threshold': 0.85
                }
            }
            
            clusterer = Clusterer(config)
            results = clusterer.cluster()
            
            unique_clusters = results['cluster_id'].nunique()
            assert unique_clusters == 1
            assert all(results['cluster_size'] == 5)
        finally:
            os.unlink(temp_path)
            if os.path.exists('results/test_clusters_dups.csv'):
                os.unlink('results/test_clusters_dups.csv')
    
    def test_clusterer_with_column_mapping(self):
        data = {
            'name': ['John Doe', 'John Doe', 'Jane Smith'],
            'email': ['john@example.com', 'john@example.com', 'jane@example.com'],
            'phone': ['123-456-7890', '123-456-7890', '987-654-3210']
        }
        df = pd.DataFrame(data)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.to_csv(f.name, index=False)
            temp_path = f.name
        
        try:
            config = {
                'mode': 'clustering',
                'source1': temp_path,
                'output': 'results/test_clusters_mapped.csv',
                'cluster_config': {
                    'threshold': 0.85,
                    'columns': [
                        {'source1': 'name', 'weight': 0.5},
                        {'source1': 'email', 'weight': 0.5}
                    ]
                }
            }
            
            clusterer = Clusterer(config)
            results = clusterer.cluster()
            
            assert len(clusterer.column_analyses) == 2
            assert 'name' in clusterer.column_analyses
            assert 'email' in clusterer.column_analyses
        finally:
            os.unlink(temp_path)
            if os.path.exists('results/test_clusters_mapped.csv'):
                os.unlink('results/test_clusters_mapped.csv')


class TestClusteringConfig:
    """Test clustering configuration validation."""
    
    def test_clustering_config_valid(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config = {
                'mode': 'clustering',
                'source1': 'data/test.csv',
                'output': 'results/clusters.csv',
                'cluster_config': {
                    'threshold': 0.85,
                    'generate_summary': True
                }
            }
            json.dump(config, f)
            temp_path = f.name
        
        try:
            result = validate_config(temp_path)
            assert result['mode'] == 'clustering'
            assert 'source2' not in result
        finally:
            os.unlink(temp_path)
    
    def test_clustering_config_with_source2_error(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config = {
                'mode': 'clustering',
                'source1': 'data/test.csv',
                'source2': 'data/test2.csv',
                'output': 'results/clusters.csv'
            }
            json.dump(config, f)
            temp_path = f.name
        
        try:
            with pytest.raises(ValueError, match='clustering mode does not require source2'):
                validate_config(temp_path)
        finally:
            os.unlink(temp_path)
    
    def test_matching_config_requires_source2(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config = {
                'mode': 'matching',
                'source1': 'data/test.csv',
                'output': 'results/matches.csv'
            }
            json.dump(config, f)
            temp_path = f.name
        
        try:
            with pytest.raises(ValueError, match='matching mode requires source2'):
                validate_config(temp_path)
        finally:
            os.unlink(temp_path)

