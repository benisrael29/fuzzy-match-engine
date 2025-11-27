import unittest
import os
import pandas as pd
import tempfile
import shutil
from src.config_validator import validate_config
from src.matcher import FuzzyMatcher
from src.output_writer import write_results
from src.algorithms import (
    levenshtein_similarity,
    jaro_winkler_similarity,
    token_set_ratio,
    token_sort_ratio,
    numeric_similarity,
    date_similarity
)
from src.normalizers import (
    normalize_phone,
    normalize_email,
    normalize_address,
    normalize_name,
    normalize_string
)


class TestAlgorithms(unittest.TestCase):
    """Test individual matching algorithms."""
    
    def test_levenshtein_exact_match(self):
        """Test Levenshtein with exact matches."""
        self.assertEqual(levenshtein_similarity("hello", "hello"), 1.0)
        self.assertEqual(levenshtein_similarity("", ""), 1.0)
    
    def test_levenshtein_similar_strings(self):
        """Test Levenshtein with similar strings."""
        score = levenshtein_similarity("hello", "hallo")
        self.assertGreater(score, 0.7)
        self.assertLess(score, 1.0)
    
    def test_levenshtein_different_strings(self):
        """Test Levenshtein with different strings."""
        score = levenshtein_similarity("hello", "world")
        self.assertLess(score, 0.5)
    
    def test_jaro_winkler_names(self):
        """Test Jaro-Winkler with names."""
        score1 = jaro_winkler_similarity("Robert", "Bob")
        score2 = jaro_winkler_similarity("Robert", "Robert")
        self.assertGreater(score2, score1)
        self.assertEqual(score2, 1.0)
    
    def test_jaro_winkler_prefix_bonus(self):
        """Test Jaro-Winkler prefix bonus."""
        score1 = jaro_winkler_similarity("John", "Johnny")
        score2 = jaro_winkler_similarity("John", "Jonathan")
        self.assertGreater(score1, 0.5)
        self.assertGreater(score2, 0.5)
    
    def test_token_set_ratio(self):
        """Test token set ratio."""
        score = token_set_ratio("John Smith", "Smith John")
        self.assertGreater(score, 0.9)
    
    def test_token_sort_ratio(self):
        """Test token sort ratio."""
        score = token_sort_ratio("apple banana cherry", "cherry banana apple")
        self.assertEqual(score, 1.0)
    
    def test_numeric_similarity_exact(self):
        """Test numeric similarity with exact match."""
        self.assertEqual(numeric_similarity(100, 100), 1.0)
        self.assertEqual(numeric_similarity(0, 0), 1.0)
    
    def test_numeric_similarity_close(self):
        """Test numeric similarity with close values."""
        score = numeric_similarity(100, 105)
        self.assertGreater(score, 0.9)
    
    def test_numeric_similarity_different(self):
        """Test numeric similarity with different values."""
        score = numeric_similarity(10, 1000)
        self.assertLess(score, 0.5)
    
    def test_date_similarity_exact(self):
        """Test date similarity with exact match."""
        score = date_similarity("2020-01-15", "2020-01-15")
        self.assertEqual(score, 1.0)
    
    def test_date_similarity_same_year(self):
        """Test date similarity within same year."""
        score = date_similarity("2020-01-15", "2020-06-15")
        self.assertGreater(score, 0.5)
    
    def test_date_similarity_different_years(self):
        """Test date similarity with different years."""
        score = date_similarity("2020-01-15", "2010-01-15")
        self.assertLess(score, 0.5)


class TestNormalizers(unittest.TestCase):
    """Test data normalization functions."""
    
    def test_normalize_phone_various_formats(self):
        """Test phone normalization with various formats."""
        self.assertEqual(normalize_phone("555-123-4567"), "5551234567")
        self.assertEqual(normalize_phone("(555) 987-6543"), "5559876543")
        self.assertEqual(normalize_phone("5551112222"), "5551112222")
        self.assertEqual(normalize_phone("1-555-123-4567"), "5551234567")
    
    def test_normalize_email(self):
        """Test email normalization."""
        self.assertEqual(normalize_email("John@Example.COM"), "john@example.com")
        self.assertEqual(normalize_email("  test@test.com  "), "test@test.com")
    
    def test_normalize_address_abbreviations(self):
        """Test address normalization with abbreviations."""
        self.assertIn("street", normalize_address("123 Main St"))
        self.assertIn("avenue", normalize_address("456 Oak Ave"))
        self.assertIn("boulevard", normalize_address("789 Park Blvd"))
    
    def test_normalize_name_prefixes(self):
        """Test name normalization with prefixes."""
        result1 = normalize_name("Mr John Smith")
        result2 = normalize_name("Dr Jane Doe")
        self.assertNotIn("mr", result1.lower())
        self.assertNotIn("dr", result2.lower())
        self.assertIn("john", result1.lower())
        self.assertIn("jane", result2.lower())
    
    def test_normalize_name_suffixes(self):
        """Test name normalization with suffixes."""
        result = normalize_name("John Smith Jr")
        self.assertNotIn("jr", result.lower())
    
    def test_normalize_string_general(self):
        """Test general string normalization."""
        self.assertEqual(normalize_string("  HELLO   WORLD  "), "hello world")
        self.assertEqual(normalize_string("Test\n\nTest"), "test test")


class TestMatchingScenarios(unittest.TestCase):
    """Test various matching scenarios with sample data."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.output_path = os.path.join(self.temp_dir, 'test_results.csv')
    
    def tearDown(self):
        """Clean up test files."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_exact_matches(self):
        """Test matching with exact duplicate records."""
        source1_data = {
            'name': ['John Smith', 'Jane Doe', 'Bob Johnson'],
            'email': ['john@test.com', 'jane@test.com', 'bob@test.com'],
            'phone': ['5551234567', '5559876543', '5551112222']
        }
        source2_data = {
            'name': ['John Smith', 'Jane Doe', 'Bob Johnson'],
            'email': ['john@test.com', 'jane@test.com', 'bob@test.com'],
            'phone': ['5551234567', '5559876543', '5551112222']
        }
        
        df1 = pd.DataFrame(source1_data)
        df2 = pd.DataFrame(source2_data)
        
        file1 = os.path.join(self.temp_dir, 'source1.csv')
        file2 = os.path.join(self.temp_dir, 'source2.csv')
        df1.to_csv(file1, index=False)
        df2.to_csv(file2, index=False)
        
        config = {
            'source1': file1,
            'source2': file2,
            'output': self.output_path,
            'match_config': {
                'columns': [
                    {'source1': 'name', 'source2': 'name', 'weight': 0.4},
                    {'source1': 'email', 'source2': 'email', 'weight': 0.4},
                    {'source1': 'phone', 'source2': 'phone', 'weight': 0.2}
                ],
                'threshold': 0.85
            }
        }
        
        matcher = FuzzyMatcher(config)
        results = matcher.match()
        
        self.assertEqual(len(results), 3)
        self.assertTrue(all(results['overall_score'] >= 0.95))
        self.assertTrue(all(results['match_result'] == 'accept'))
    
    def test_name_variations(self):
        """Test matching with name variations."""
        source1_data = {
            'name': ['Robert Johnson', 'William Smith', 'Michael Brown'],
            'email': ['robert@test.com', 'william@test.com', 'michael@test.com']
        }
        source2_data = {
            'name': ['Bob Johnson', 'Bill Smith', 'Mike Brown'],
            'email': ['robert@test.com', 'william@test.com', 'michael@test.com']
        }
        
        df1 = pd.DataFrame(source1_data)
        df2 = pd.DataFrame(source2_data)
        
        file1 = os.path.join(self.temp_dir, 'source1.csv')
        file2 = os.path.join(self.temp_dir, 'source2.csv')
        df1.to_csv(file1, index=False)
        df2.to_csv(file2, index=False)
        
        config = {
            'source1': file1,
            'source2': file2,
            'output': self.output_path,
            'match_config': {
                'columns': [
                    {'source1': 'name', 'source2': 'name', 'weight': 0.5},
                    {'source1': 'email', 'source2': 'email', 'weight': 0.5}
                ],
                'threshold': 0.75
            }
        }
        
        matcher = FuzzyMatcher(config)
        results = matcher.match()
        
        self.assertEqual(len(results), 3)
        self.assertTrue(all(results['overall_score'] > 0.7))
    
    def test_phone_format_variations(self):
        """Test matching with different phone formats."""
        source1_data = {
            'name': ['Alice', 'Bob', 'Charlie'],
            'phone': ['555-123-4567', '(555) 987-6543', '5551112222']
        }
        source2_data = {
            'name': ['Alice', 'Bob', 'Charlie'],
            'phone': ['5551234567', '5559876543', '1-555-111-2222']
        }
        
        df1 = pd.DataFrame(source1_data)
        df2 = pd.DataFrame(source2_data)
        
        file1 = os.path.join(self.temp_dir, 'source1.csv')
        file2 = os.path.join(self.temp_dir, 'source2.csv')
        df1.to_csv(file1, index=False)
        df2.to_csv(file2, index=False)
        
        config = {
            'source1': file1,
            'source2': file2,
            'output': self.output_path,
            'match_config': {
                'columns': [
                    {'source1': 'name', 'source2': 'name', 'weight': 0.3},
                    {'source1': 'phone', 'source2': 'phone', 'weight': 0.7}
                ],
                'threshold': 0.8
            }
        }
        
        matcher = FuzzyMatcher(config)
        results = matcher.match()
        
        self.assertEqual(len(results), 3)
        self.assertTrue(all(results['overall_score'] > 0.85))
    
    def test_address_variations(self):
        """Test matching with address variations."""
        source1_data = {
            'name': ['John', 'Jane', 'Bob'],
            'address': ['123 Main St', '456 Oak Ave', '789 Park Blvd']
        }
        source2_data = {
            'name': ['John', 'Jane', 'Bob'],
            'address': ['123 Main Street', '456 Oak Avenue', '789 Park Boulevard']
        }
        
        df1 = pd.DataFrame(source1_data)
        df2 = pd.DataFrame(source2_data)
        
        file1 = os.path.join(self.temp_dir, 'source1.csv')
        file2 = os.path.join(self.temp_dir, 'source2.csv')
        df1.to_csv(file1, index=False)
        df2.to_csv(file2, index=False)
        
        config = {
            'source1': file1,
            'source2': file2,
            'output': self.output_path,
            'match_config': {
                'columns': [
                    {'source1': 'name', 'source2': 'name', 'weight': 0.4},
                    {'source1': 'address', 'source2': 'address', 'weight': 0.6}
                ],
                'threshold': 0.75
            }
        }
        
        matcher = FuzzyMatcher(config)
        results = matcher.match()
        
        self.assertEqual(len(results), 3)
        self.assertTrue(all(results['overall_score'] > 0.8))
    
    def test_email_case_variations(self):
        """Test matching with email case variations."""
        source1_data = {
            'name': ['Alice', 'Bob'],
            'email': ['Alice@Test.COM', 'Bob@Example.ORG']
        }
        source2_data = {
            'name': ['Alice', 'Bob'],
            'email': ['alice@test.com', 'bob@example.org']
        }
        
        df1 = pd.DataFrame(source1_data)
        df2 = pd.DataFrame(source2_data)
        
        file1 = os.path.join(self.temp_dir, 'source1.csv')
        file2 = os.path.join(self.temp_dir, 'source2.csv')
        df1.to_csv(file1, index=False)
        df2.to_csv(file2, index=False)
        
        config = {
            'source1': file1,
            'source2': file2,
            'output': self.output_path,
            'match_config': {
                'columns': [
                    {'source1': 'name', 'source2': 'name', 'weight': 0.3},
                    {'source1': 'email', 'source2': 'email', 'weight': 0.7}
                ],
                'threshold': 0.85
            }
        }
        
        matcher = FuzzyMatcher(config)
        results = matcher.match()
        
        self.assertEqual(len(results), 2)
        self.assertTrue(all(results['overall_score'] >= 1.0))
    
    def test_date_variations(self):
        """Test matching with date variations."""
        source1_data = {
            'name': ['John', 'Jane'],
            'dob': ['1980-01-15', '1985-03-20']
        }
        source2_data = {
            'name': ['John', 'Jane'],
            'dob': ['1980-01-15', '1985-03-21']
        }
        
        df1 = pd.DataFrame(source1_data)
        df2 = pd.DataFrame(source2_data)
        
        file1 = os.path.join(self.temp_dir, 'source1.csv')
        file2 = os.path.join(self.temp_dir, 'source2.csv')
        df1.to_csv(file1, index=False)
        df2.to_csv(file2, index=False)
        
        config = {
            'source1': file1,
            'source2': file2,
            'output': self.output_path,
            'match_config': {
                'columns': [
                    {'source1': 'name', 'source2': 'name', 'weight': 0.5},
                    {'source1': 'dob', 'source2': 'dob', 'weight': 0.5}
                ],
                'threshold': 0.8
            }
        }
        
        matcher = FuzzyMatcher(config)
        results = matcher.match()
        
        self.assertEqual(len(results), 2)
        self.assertTrue(all(results['overall_score'] > 0.9))
    
    def test_no_matches(self):
        """Test with completely different records."""
        source1_data = {
            'name': ['Alice', 'Bob'],
            'email': ['alice@test.com', 'bob@test.com']
        }
        source2_data = {
            'name': ['Charlie', 'Diana'],
            'email': ['charlie@test.com', 'diana@test.com']
        }
        
        df1 = pd.DataFrame(source1_data)
        df2 = pd.DataFrame(source2_data)
        
        file1 = os.path.join(self.temp_dir, 'source1.csv')
        file2 = os.path.join(self.temp_dir, 'source2.csv')
        df1.to_csv(file1, index=False)
        df2.to_csv(file2, index=False)
        
        config = {
            'source1': file1,
            'source2': file2,
            'output': self.output_path,
            'match_config': {
                'columns': [
                    {'source1': 'name', 'source2': 'name', 'weight': 0.5},
                    {'source1': 'email', 'source2': 'email', 'weight': 0.5}
                ],
                'threshold': 0.85
            }
        }
        
        matcher = FuzzyMatcher(config)
        results = matcher.match()
        
        self.assertGreater(len(results), 0)
        self.assertTrue(all(results['overall_score'] < 0.85))
        self.assertTrue(all(results['match_result'].isin(['reject', 'undecided'])))
    
    def test_partial_matches(self):
        """Test with partially matching records."""
        source1_data = {
            'name': ['John Smith', 'Jane Doe'],
            'email': ['john@test.com', 'jane@test.com'],
            'phone': ['5551234567', '5559876543']
        }
        source2_data = {
            'name': ['John Smith', 'Jane Doe'],
            'email': ['john@test.com', 'jane.doe@test.com'],
            'phone': ['5551234567', '5559876544']
        }
        
        df1 = pd.DataFrame(source1_data)
        df2 = pd.DataFrame(source2_data)
        
        file1 = os.path.join(self.temp_dir, 'source1.csv')
        file2 = os.path.join(self.temp_dir, 'source2.csv')
        df1.to_csv(file1, index=False)
        df2.to_csv(file2, index=False)
        
        config = {
            'source1': file1,
            'source2': file2,
            'output': self.output_path,
            'match_config': {
                'columns': [
                    {'source1': 'name', 'source2': 'name', 'weight': 0.4},
                    {'source1': 'email', 'source2': 'email', 'weight': 0.3},
                    {'source1': 'phone', 'source2': 'phone', 'weight': 0.3}
                ],
                'threshold': 0.85,
                'undecided_range': 0.1
            }
        }
        
        matcher = FuzzyMatcher(config)
        results = matcher.match()
        
        self.assertEqual(len(results), 2)
        self.assertGreater(results.iloc[0]['overall_score'], 0.9)
        self.assertGreater(results.iloc[1]['overall_score'], 0.7)
    
    def test_missing_data(self):
        """Test matching with missing/null values."""
        source1_data = {
            'name': ['John', 'Jane', 'Bob'],
            'email': ['john@test.com', None, 'bob@test.com'],
            'phone': ['5551234567', '5559876543', None]
        }
        source2_data = {
            'name': ['John', 'Jane', 'Bob'],
            'email': ['john@test.com', 'jane@test.com', None],
            'phone': [None, '5559876543', '5551112222']
        }
        
        df1 = pd.DataFrame(source1_data)
        df2 = pd.DataFrame(source2_data)
        
        file1 = os.path.join(self.temp_dir, 'source1.csv')
        file2 = os.path.join(self.temp_dir, 'source2.csv')
        df1.to_csv(file1, index=False)
        df2.to_csv(file2, index=False)
        
        config = {
            'source1': file1,
            'source2': file2,
            'output': self.output_path,
            'match_config': {
                'columns': [
                    {'source1': 'name', 'source2': 'name', 'weight': 0.5},
                    {'source1': 'email', 'source2': 'email', 'weight': 0.25},
                    {'source1': 'phone', 'source2': 'phone', 'weight': 0.25}
                ],
                'threshold': 0.75
            }
        }
        
        matcher = FuzzyMatcher(config)
        results = matcher.match()
        
        self.assertEqual(len(results), 3)
    
    def test_numeric_matching(self):
        """Test matching with numeric fields."""
        source1_data = {
            'id': [100, 200, 300],
            'name': ['John', 'Jane', 'Bob'],
            'age': [30, 25, 40]
        }
        source2_data = {
            'id': [100, 200, 301],
            'name': ['John', 'Jane', 'Bob'],
            'age': [30, 25, 40]
        }
        
        df1 = pd.DataFrame(source1_data)
        df2 = pd.DataFrame(source2_data)
        
        file1 = os.path.join(self.temp_dir, 'source1.csv')
        file2 = os.path.join(self.temp_dir, 'source2.csv')
        df1.to_csv(file1, index=False)
        df2.to_csv(file2, index=False)
        
        config = {
            'source1': file1,
            'source2': file2,
            'output': self.output_path,
            'match_config': {
                'columns': [
                    {'source1': 'name', 'source2': 'name', 'weight': 0.4},
                    {'source1': 'id', 'source2': 'id', 'weight': 0.3},
                    {'source1': 'age', 'source2': 'age', 'weight': 0.3}
                ],
                'threshold': 0.85
            }
        }
        
        matcher = FuzzyMatcher(config)
        results = matcher.match()
        
        self.assertEqual(len(results), 3)
        self.assertGreater(results.iloc[0]['overall_score'], 0.95)
        self.assertGreater(results.iloc[1]['overall_score'], 0.95)
    
    def test_return_all_matches(self):
        """Test return_all_matches option."""
        source1_data = {
            'name': ['John Smith'],
            'email': ['john@test.com']
        }
        source2_data = {
            'name': ['John Smith', 'John A. Smith', 'Johnny Smith'],
            'email': ['john@test.com', 'john@test.com', 'johnny@test.com']
        }
        
        df1 = pd.DataFrame(source1_data)
        df2 = pd.DataFrame(source2_data)
        
        file1 = os.path.join(self.temp_dir, 'source1.csv')
        file2 = os.path.join(self.temp_dir, 'source2.csv')
        df1.to_csv(file1, index=False)
        df2.to_csv(file2, index=False)
        
        config = {
            'source1': file1,
            'source2': file2,
            'output': self.output_path,
            'match_config': {
                'columns': [
                    {'source1': 'name', 'source2': 'name', 'weight': 0.5},
                    {'source1': 'email', 'source2': 'email', 'weight': 0.5}
                ],
                'threshold': 0.7,
                'undecided_range': 0.1,
                'return_all_matches': True
            }
        }
        
        matcher = FuzzyMatcher(config)
        results = matcher.match()
        
        self.assertGreaterEqual(len(results), 2)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.output_path = os.path.join(self.temp_dir, 'test_results.csv')
    
    def tearDown(self):
        """Clean up test files."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_empty_strings(self):
        """Test matching with empty strings."""
        source1_data = {'name': ['John', ''], 'email': ['john@test.com', '']}
        source2_data = {'name': ['John', 'Jane'], 'email': ['john@test.com', 'jane@test.com']}
        
        df1 = pd.DataFrame(source1_data)
        df2 = pd.DataFrame(source2_data)
        
        file1 = os.path.join(self.temp_dir, 'source1.csv')
        file2 = os.path.join(self.temp_dir, 'source2.csv')
        df1.to_csv(file1, index=False)
        df2.to_csv(file2, index=False)
        
        config = {
            'source1': file1,
            'source2': file2,
            'output': self.output_path,
            'match_config': {
                'columns': [
                    {'source1': 'name', 'source2': 'name', 'weight': 0.5},
                    {'source1': 'email', 'source2': 'email', 'weight': 0.5}
                ]
            }
        }
        
        matcher = FuzzyMatcher(config)
        results = matcher.match()
        
        self.assertGreater(len(results), 0)
    
    def test_special_characters(self):
        """Test matching with special characters."""
        source1_data = {
            'name': ["O'Brien", "Smith-Jones", "José García"],
            'email': ["obrien@test.com", "smith-jones@test.com", "jose@test.com"]
        }
        source2_data = {
            'name': ["O'Brien", "Smith-Jones", "Jose Garcia"],
            'email': ["obrien@test.com", "smith-jones@test.com", "jose@test.com"]
        }
        
        df1 = pd.DataFrame(source1_data)
        df2 = pd.DataFrame(source2_data)
        
        file1 = os.path.join(self.temp_dir, 'source1.csv')
        file2 = os.path.join(self.temp_dir, 'source2.csv')
        df1.to_csv(file1, index=False)
        df2.to_csv(file2, index=False)
        
        config = {
            'source1': file1,
            'source2': file2,
            'output': self.output_path,
            'match_config': {
                'columns': [
                    {'source1': 'name', 'source2': 'name', 'weight': 0.5},
                    {'source1': 'email', 'source2': 'email', 'weight': 0.5}
                ],
                'threshold': 0.75
            }
        }
        
        matcher = FuzzyMatcher(config)
        results = matcher.match()
        
        self.assertEqual(len(results), 3)
    
    def test_very_long_strings(self):
        """Test matching with very long strings."""
        long_name1 = "A" * 1000
        long_name2 = "A" * 1000
        
        source1_data = {'name': [long_name1], 'email': ['test@test.com']}
        source2_data = {'name': [long_name2], 'email': ['test@test.com']}
        
        df1 = pd.DataFrame(source1_data)
        df2 = pd.DataFrame(source2_data)
        
        file1 = os.path.join(self.temp_dir, 'source1.csv')
        file2 = os.path.join(self.temp_dir, 'source2.csv')
        df1.to_csv(file1, index=False)
        df2.to_csv(file2, index=False)
        
        config = {
            'source1': file1,
            'source2': file2,
            'output': self.output_path,
            'match_config': {
                'columns': [
                    {'source1': 'name', 'source2': 'name', 'weight': 0.5},
                    {'source1': 'email', 'source2': 'email', 'weight': 0.5}
                ]
            }
        }
        
        matcher = FuzzyMatcher(config)
        results = matcher.match()
        
        self.assertEqual(len(results), 1)
        self.assertEqual(results.iloc[0]['overall_score'], 1.0)


if __name__ == '__main__':
    unittest.main(verbosity=2)

