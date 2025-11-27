import unittest
import os
import pandas as pd
from src.config_validator import validate_config
from src.matcher import FuzzyMatcher
from src.output_writer import write_results


class TestIntegration(unittest.TestCase):
    """Integration tests for fuzzy matching engine."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config_path = 'tests/fixtures/test_config.json'
        self.output_path = 'tests/fixtures/test_results.csv'
        
        if os.path.exists(self.output_path):
            os.remove(self.output_path)
    
    def tearDown(self):
        """Clean up test files."""
        if os.path.exists(self.output_path):
            os.remove(self.output_path)
    
    def test_config_validation(self):
        """Test configuration validation."""
        config = validate_config(self.config_path)
        
        self.assertIn('source1', config)
        self.assertIn('source2', config)
        self.assertIn('output', config)
        self.assertIn('match_config', config)
    
    def test_end_to_end_matching(self):
        """Test end-to-end matching workflow."""
        config = validate_config(self.config_path)
        matcher = FuzzyMatcher(config)
        results = matcher.match()
        
        self.assertIsInstance(results, pd.DataFrame)
        self.assertGreater(len(results), 0)
        
        write_results(results, self.output_path)
        self.assertTrue(os.path.exists(self.output_path))
    
    def test_output_structure(self):
        """Test output CSV structure."""
        config = validate_config(self.config_path)
        matcher = FuzzyMatcher(config)
        results = matcher.match()
        write_results(results, self.output_path)
        
        df = pd.read_csv(self.output_path)
        
        required_columns = ['overall_score', 'match_result', 'source1_index', 'source2_index']
        for col in required_columns:
            self.assertIn(col, df.columns, f"Missing required column: {col}")
        
        self.assertIn('match_result', df.columns)
        self.assertTrue(df['match_result'].isin(['accept', 'reject', 'undecided']).all())
    
    def test_name_variations(self):
        """Test matching with name variations (Bob/Robert)."""
        config = validate_config(self.config_path)
        matcher = FuzzyMatcher(config)
        results = matcher.match()
        
        bob_robert_match = results[
            (results['source1_name'].str.contains('Robert', case=False, na=False)) &
            (results['source2_full_name'].str.contains('Bob', case=False, na=False))
        ]
        
        if len(bob_robert_match) > 0:
            score = bob_robert_match.iloc[0]['overall_score']
            self.assertGreater(score, 0.5, "Bob/Robert should have reasonable similarity")
    
    def test_address_variations(self):
        """Test matching with address variations (St vs Street)."""
        config = validate_config(self.config_path)
        matcher = FuzzyMatcher(config)
        results = matcher.match()
        
        address_matches = results[
            (results['source1_address'].str.contains('St', case=False, na=False)) &
            (results['source2_street_address'].str.contains('Street', case=False, na=False))
        ]
        
        if len(address_matches) > 0:
            score = address_matches.iloc[0]['overall_score']
            self.assertGreater(score, 0.6, "Address variations should match well")
    
    def test_phone_formats(self):
        """Test matching with different phone number formats."""
        config = validate_config(self.config_path)
        matcher = FuzzyMatcher(config)
        results = matcher.match()
        
        phone_matches = results[
            (results['source1_phone'].notna()) &
            (results['source2_phone_number'].notna())
        ]
        
        if len(phone_matches) > 0:
            for _, row in phone_matches.head(3).iterrows():
                score = row['overall_score']
                self.assertGreater(score, 0.0, "Phone matches should have positive scores")
    
    def test_dob_variations(self):
        """Test matching with different date formats."""
        config = validate_config(self.config_path)
        matcher = FuzzyMatcher(config)
        results = matcher.match()
        
        dob_matches = results[
            (results['source1_dob'].notna()) &
            (results['source2_date_of_birth'].notna())
        ]
        
        if len(dob_matches) > 0:
            for _, row in dob_matches.head(3).iterrows():
                score = row['overall_score']
                self.assertGreater(score, 0.0, "Date matches should have positive scores")
    
    def test_performance_large_dataset(self):
        """Test performance with larger dataset (1K+ rows)."""
        import tempfile
        import shutil
        
        temp_dir = tempfile.mkdtemp()
        try:
            large_file1 = os.path.join(temp_dir, 'large1.csv')
            large_file2 = os.path.join(temp_dir, 'large2.csv')
            
            rows = []
            for i in range(1000):
                rows.append({
                    'name': f'Person {i}',
                    'email': f'person{i}@test.com',
                    'phone': f'555{i:07d}'
                })
            
            df1 = pd.DataFrame(rows)
            df2 = df1.copy()
            df2['name'] = df2['name'].str.replace('Person', 'P')
            
            df1.to_csv(large_file1, index=False)
            df2.to_csv(large_file2, index=False)
            
            config = {
                'source1': large_file1,
                'source2': large_file2,
                'output': os.path.join(temp_dir, 'results.csv')
            }
            
            import time
            start = time.time()
            matcher = FuzzyMatcher(config)
            results = matcher.match()
            elapsed = time.time() - start
            
            self.assertGreater(len(results), 0)
            self.assertLess(elapsed, 60, "Should complete 1K rows in under 60 seconds")
            
        finally:
            shutil.rmtree(temp_dir)


if __name__ == '__main__':
    unittest.main()

