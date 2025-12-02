import csv
import os
import random
import string
import tempfile
import time
import unittest

import pandas as pd
import shutil
import numpy as np

from src.matcher import FuzzyMatcher
from src.algorithms import (
    levenshtein_similarity,
    jaro_winkler_similarity,
    token_set_ratio
)


HEAVY_DATASET_TESTS = os.getenv("RUN_HEAVY_DATASET_TESTS", "").lower() in ("1", "true", "yes")

DATAFRAME_CACHE = {}


FIRST_NAMES = ['John', 'Jane', 'Robert', 'Mary', 'William', 'Patricia',
               'Michael', 'Jennifer', 'David', 'Linda', 'Richard', 'Elizabeth',
               'Joseph', 'Susan', 'Thomas', 'Jessica', 'Charles', 'Sarah',
               'Christopher', 'Karen', 'Daniel', 'Nancy', 'Matthew', 'Lisa',
               'Anthony', 'Betty', 'Mark', 'Margaret', 'Donald', 'Sandra']
LAST_NAMES = ['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia',
              'Miller', 'Davis', 'Rodriguez', 'Martinez', 'Hernandez', 'Lopez',
              'Wilson', 'Anderson', 'Thomas', 'Taylor', 'Moore', 'Jackson',
              'Martin', 'Lee', 'Thompson', 'White', 'Harris', 'Sanchez',
              'Clark', 'Ramirez', 'Lewis', 'Robinson', 'Walker', 'Young']
EMAIL_DOMAINS = ['gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com', 'test.com']


def generate_random_address():
    """Generate a random address."""
    numbers = random.randint(100, 9999)
    streets = ['Main St', 'Oak Ave', 'Park Blvd', 'Elm St', 'Maple Dr',
               'Cedar Ln', 'Pine Rd', 'First St', 'Second Ave', 'Third Blvd']
    return f"{numbers} {random.choice(streets)}"


def _get_cached_dataframe(n_records: int) -> pd.DataFrame:
    """Return cached DataFrame for heavy synthetic datasets."""
    if n_records not in DATAFRAME_CACHE:
        rng = np.random.default_rng(seed=n_records)
        first = rng.choice(FIRST_NAMES, size=n_records)
        last = rng.choice(LAST_NAMES, size=n_records)
        names = [f"{f} {l}" for f, l in zip(first, last)]
        domain_choices = rng.choice(EMAIL_DOMAINS, size=n_records)
        emails = []
        for name, domain in zip(names, domain_choices):
            if rng.random() > 0.2:
                base = name.lower().replace(' ', '.')
            else:
                letters = rng.choice(list(string.ascii_lowercase), size=8)
                base = ''.join(letters)
            emails.append(f"{base}@{domain}")
        area_codes = rng.integers(200, 1000, size=n_records)
        exchanges = rng.integers(200, 1000, size=n_records)
        line_nums = rng.integers(1000, 10000, size=n_records)
        phones = [f"{a}{e}{l}" for a, e, l in zip(area_codes, exchanges, line_nums)]
        DATAFRAME_CACHE[n_records] = pd.DataFrame({
            'name': names,
            'email': emails,
            'phone': phones
        })
    return DATAFRAME_CACHE[n_records].copy(deep=True)


class TestLargeDatasetPerformance(unittest.TestCase):
    """Test performance with large datasets."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.output_path = os.path.join(self.temp_dir, 'results.csv')
    
    def tearDown(self):
        """Clean up test files."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def _write_simple_csv(self, path: str, n_records: int, prefix: str = "Person"):
        """Write a simple single-column CSV for heavy dataset tests."""
        with open(path, 'w', newline='', encoding='utf-8') as handle:
            writer = csv.writer(handle)
            writer.writerow(['name'])
            for i in range(n_records):
                writer.writerow([f"{prefix} {i}"])

    def _write_cached_pair(self, file1: str, file2: str, n_records: int):
        """Write cached synthetic dataset to both files to avoid regeneration."""
        df = _get_cached_dataframe(n_records)
        df.to_csv(file1, index=False)
        df.to_csv(file2, index=False)
    
    def test_10k_records(self):
        """Test matching with 10,000 records."""
        n_records = 10000
        
        file1 = os.path.join(self.temp_dir, 'source1_10k.csv')
        file2 = os.path.join(self.temp_dir, 'source2_10k.csv')
        self._write_cached_pair(file1, file2, n_records)
        
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
                'threshold': 0.85,
                'use_multiprocessing': True,
                'num_workers': 4,
                'blocking_strategies': ['first_char', 'three_gram']
            }
        }
        
        start_time = time.time()
        matcher = FuzzyMatcher(config)
        results = matcher.match()
        elapsed = time.time() - start_time
        
        self.assertGreater(len(results), 0)
        self.assertLess(elapsed, 300, "10K records should complete in under 5 minutes")
        print(f"10K records matched in {elapsed:.2f} seconds")
    
    def test_50k_records(self):
        """Test matching with 50,000 records."""
        n_records = 50000
        file1 = os.path.join(self.temp_dir, 'source1_50k.csv')
        file2 = os.path.join(self.temp_dir, 'source2_50k.csv')
        self._write_cached_pair(file1, file2, n_records)
        
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
                'threshold': 0.85,
                'use_multiprocessing': True,
                'num_workers': 4,
                'chunk_size': 5000,
                'blocking_strategies': ['first_char', 'three_gram']
            }
        }
        
        start_time = time.time()
        matcher = FuzzyMatcher(config)
        results = matcher.match()
        elapsed = time.time() - start_time
        
        self.assertGreater(len(results), 0)
        self.assertLess(elapsed, 1800, "50K records should complete in under 30 minutes")
        print(f"50K records matched in {elapsed:.2f} seconds")
    
    def test_100k_records(self):
        """Test matching with 100,000 records."""
        n_records = 100000
        file1 = os.path.join(self.temp_dir, 'source1_100k.csv')
        file2 = os.path.join(self.temp_dir, 'source2_100k.csv')
        self._write_cached_pair(file1, file2, n_records)
        
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
                'threshold': 0.85,
                'use_multiprocessing': True,
                'num_workers': 4,
                'chunk_size': 10000,
                'early_termination': True,
                'blocking_strategies': ['first_char', 'three_gram']
            }
        }
        
        start_time = time.time()
        matcher = FuzzyMatcher(config)
        results = matcher.match()
        elapsed = time.time() - start_time
        
        self.assertGreater(len(results), 0)
        print(f"100K records matched in {elapsed:.2f} seconds")
    
    @unittest.skipUnless(HEAVY_DATASET_TESTS, "Set RUN_HEAVY_DATASET_TESTS=1 to run heavy 100K streaming test")
    def test_100k_records_streaming(self):
        """Test streaming workflow with 100,000 minimalist records."""
        n_records = 100000
        file1 = os.path.join(self.temp_dir, 'heavy_source1_100k.csv')
        file2 = os.path.join(self.temp_dir, 'heavy_source2_100k.csv')
        stream_output = os.path.join(self.temp_dir, 'stream_100k.csv')
        
        self._write_simple_csv(file1, n_records, prefix="Alpha")
        self._write_simple_csv(file2, n_records, prefix="Alpha")
        
        config = {
            'source1': file1,
            'source2': file2,
            'output': stream_output,
            'match_config': {
                'columns': [
                    {'source1': 'name', 'source2': 'name', 'weight': 1.0}
                ],
                'threshold': 0.95,
                'use_multiprocessing': True,
                'num_workers': 8,
                'chunk_size': 20000,
                'early_termination': True
            }
        }
        
        matcher = FuzzyMatcher(config)
        results = matcher.match(stream_to_file=stream_output)
        
        self.assertTrue(os.path.exists(stream_output), "Streaming output not created")
        self.assertEqual(len(results), n_records)
    
    def test_unequal_sized_datasets(self):
        """Test matching with unequal dataset sizes."""
        source1_size = 1000
        source2_size = 10000
        
        file1 = os.path.join(self.temp_dir, 'source1_1k.csv')
        file2 = os.path.join(self.temp_dir, 'source2_10k.csv')
        df1 = _get_cached_dataframe(source1_size)
        df2 = _get_cached_dataframe(source2_size)
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
        
        start_time = time.time()
        matcher = FuzzyMatcher(config)
        results = matcher.match()
        elapsed = time.time() - start_time
        
        self.assertGreater(len(results), 0)
        self.assertLess(elapsed, 120, "1K vs 10K should complete in under 2 minutes")
        print(f"1K vs 10K matched in {elapsed:.2f} seconds")

    def test_blocking_and_candidate_limits(self):
        """Ensure blocking and candidate capping settings take effect."""
        n_records = 500
        duplicated_name = "Alpha Echo"
        emails = [f"user{i}@example.com" for i in range(n_records)]
        phones = [f"555000{i:04d}" for i in range(n_records)]
        df = pd.DataFrame({
            'name': [duplicated_name] * n_records,
            'email': emails,
            'phone': phones
        })
        file1 = os.path.join(self.temp_dir, 'limit_source1.csv')
        file2 = os.path.join(self.temp_dir, 'limit_source2.csv')
        df.to_csv(file1, index=False)
        df.to_csv(file2, index=False)
        
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
                'threshold': 0.85,
                'use_multiprocessing': False,
                'max_block_size': 50,
                'skip_high_cardinality': False,
                'max_candidates': 10,
                'candidate_trim_strategy': 'truncate',
                'blocking_strategies': ['first_char']
            }
        }
        
        matcher = FuzzyMatcher(config)
        matcher.match()
        
        self.assertGreaterEqual(matcher.blocking_stats.get('trimmed_keys', 0), 1)
        self.assertGreaterEqual(matcher.candidate_stats.get('capped_rows', 0), 1)
    
    @unittest.skipUnless(HEAVY_DATASET_TESTS, "Set RUN_HEAVY_DATASET_TESTS=1 to run 500K row streaming test")
    def test_500k_records_streaming(self):
        """Test streaming workflow with 500,000 minimalist records."""
        n_records = 500000
        file1 = os.path.join(self.temp_dir, 'heavy_source1_500k.csv')
        file2 = os.path.join(self.temp_dir, 'heavy_source2_500k.csv')
        stream_output = os.path.join(self.temp_dir, 'stream_500k.csv')
        
        self._write_simple_csv(file1, n_records, prefix="Omega")
        self._write_simple_csv(file2, n_records, prefix="Omega")
        
        config = {
            'source1': file1,
            'source2': file2,
            'output': stream_output,
            'match_config': {
                'columns': [
                    {'source1': 'name', 'source2': 'name', 'weight': 1.0}
                ],
                'threshold': 0.95,
                'use_multiprocessing': True,
                'num_workers': 8,
                'chunk_size': 50000,
                'early_termination': True,
                'perfect_match_threshold': 0.999
            }
        }
        
        matcher = FuzzyMatcher(config)
        results = matcher.match(stream_to_file=stream_output)
        
        self.assertTrue(os.path.exists(stream_output), "Streaming output not created for 500K rows")
        self.assertEqual(len(results), n_records)
        self.assertTrue((results['overall_score'] >= 0.95).all())


class TestMatchingAccuracy(unittest.TestCase):
    """Test matching accuracy with known ground truth."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.output_path = os.path.join(self.temp_dir, 'results.csv')
    
    def tearDown(self):
        """Clean up test files."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def calculate_metrics(self, results, ground_truth):
        """Calculate precision, recall, and F1 score."""
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        
        accepted_matches = set()
        for _, row in results.iterrows():
            if row['match_result'] == 'accept':
                idx1 = row['source1_index']
                idx2 = row['source2_index']
                accepted_matches.add((idx1, idx2))
        
        for idx1, idx2 in ground_truth:
            if (idx1, idx2) in accepted_matches:
                true_positives += 1
            else:
                false_negatives += 1
        
        for idx1, idx2 in accepted_matches:
            if (idx1, idx2) not in ground_truth:
                false_positives += 1
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives
        }
    
    def test_exact_match_accuracy(self):
        """Test accuracy with exact matches."""
        n_records = 1000
        
        source1_data = {
            'name': [f"Person {i}" for i in range(n_records)],
            'email': [f"person{i}@test.com" for i in range(n_records)],
            'phone': [f"555{i:07d}" for i in range(n_records)]
        }
        
        source2_data = {
            'name': [f"Person {i}" for i in range(n_records)],
            'email': [f"person{i}@test.com" for i in range(n_records)],
            'phone': [f"555{i:07d}" for i in range(n_records)]
        }
        
        ground_truth = {(i, i) for i in range(n_records)}
        
        df1 = pd.DataFrame(source1_data)
        df2 = pd.DataFrame(source2_data)
        
        file1 = os.path.join(self.temp_dir, 'source1_exact.csv')
        file2 = os.path.join(self.temp_dir, 'source2_exact.csv')
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
        
        metrics = self.calculate_metrics(results, ground_truth)
        
        self.assertGreater(metrics['precision'], 0.95, "Precision should be > 95% for exact matches")
        self.assertGreater(metrics['recall'], 0.95, "Recall should be > 95% for exact matches")
        self.assertGreater(metrics['f1'], 0.95, "F1 should be > 95% for exact matches")
        print(f"Exact match metrics: Precision={metrics['precision']:.3f}, Recall={metrics['recall']:.3f}, F1={metrics['f1']:.3f}")
    
    def test_name_variation_accuracy(self):
        """Test accuracy with name variations."""
        name_variations = [
            ('Robert Johnson', 'Bob Johnson'),
            ('William Smith', 'Bill Smith'),
            ('Michael Brown', 'Mike Brown'),
            ('Richard Davis', 'Rick Davis'),
            ('Christopher Wilson', 'Chris Wilson'),
            ('Jennifer Martinez', 'Jen Martinez'),
            ('Patricia Anderson', 'Pat Anderson'),
            ('Elizabeth Taylor', 'Liz Taylor'),
            ('Catherine White', 'Cathy White'),
            ('Margaret Harris', 'Maggie Harris')
        ]
        
        source1_data = {
            'name': [var[0] for var in name_variations],
            'email': [f"email{i}@test.com" for i in range(len(name_variations))],
            'phone': [f"555{i:07d}" for i in range(len(name_variations))]
        }
        
        source2_data = {
            'name': [var[1] for var in name_variations],
            'email': [f"email{i}@test.com" for i in range(len(name_variations))],
            'phone': [f"555{i:07d}" for i in range(len(name_variations))]
        }
        
        ground_truth = {(i, i) for i in range(len(name_variations))}
        
        df1 = pd.DataFrame(source1_data)
        df2 = pd.DataFrame(source2_data)
        
        file1 = os.path.join(self.temp_dir, 'source1_variations.csv')
        file2 = os.path.join(self.temp_dir, 'source2_variations.csv')
        df1.to_csv(file1, index=False)
        df2.to_csv(file2, index=False)
        
        config = {
            'source1': file1,
            'source2': file2,
            'output': self.output_path,
            'match_config': {
                'columns': [
                    {'source1': 'name', 'source2': 'name', 'weight': 0.3},
                    {'source1': 'email', 'source2': 'email', 'weight': 0.4},
                    {'source1': 'phone', 'source2': 'phone', 'weight': 0.3}
                ],
                'threshold': 0.75
            }
        }
        
        matcher = FuzzyMatcher(config)
        results = matcher.match()
        
        metrics = self.calculate_metrics(results, ground_truth)
        
        self.assertGreater(metrics['precision'], 0.8, "Precision should be > 80% for name variations")
        self.assertGreater(metrics['recall'], 0.8, "Recall should be > 80% for name variations")
        print(f"Name variation metrics: Precision={metrics['precision']:.3f}, Recall={metrics['recall']:.3f}, F1={metrics['f1']:.3f}")
    
    def test_phone_format_accuracy(self):
        """Test accuracy with different phone formats."""
        base_phones = [f"555{i:07d}" for i in range(100)]
        
        source1_data = {
            'name': [f"Person {i}" for i in range(100)],
            'email': [f"person{i}@test.com" for i in range(100)],
            'phone': [f"{phone[:3]}-{phone[3:6]}-{phone[6:]}" for phone in base_phones]
        }
        
        source2_data = {
            'name': [f"Person {i}" for i in range(100)],
            'email': [f"person{i}@test.com" for i in range(100)],
            'phone': [f"({phone[:3]}) {phone[3:6]}-{phone[6:]}" for phone in base_phones]
        }
        
        ground_truth = {(i, i) for i in range(100)}
        
        df1 = pd.DataFrame(source1_data)
        df2 = pd.DataFrame(source2_data)
        
        file1 = os.path.join(self.temp_dir, 'source1_phone.csv')
        file2 = os.path.join(self.temp_dir, 'source2_phone.csv')
        df1.to_csv(file1, index=False)
        df2.to_csv(file2, index=False)
        
        config = {
            'source1': file1,
            'source2': file2,
            'output': self.output_path,
            'match_config': {
                'columns': [
                    {'source1': 'name', 'source2': 'name', 'weight': 0.3},
                    {'source1': 'email', 'source2': 'email', 'weight': 0.3},
                    {'source1': 'phone', 'source2': 'phone', 'weight': 0.4}
                ],
                'threshold': 0.85
            }
        }
        
        matcher = FuzzyMatcher(config)
        results = matcher.match()
        
        metrics = self.calculate_metrics(results, ground_truth)
        
        self.assertGreater(metrics['precision'], 0.9, "Precision should be > 90% for phone format variations")
        self.assertGreater(metrics['recall'], 0.9, "Recall should be > 90% for phone format variations")
        print(f"Phone format metrics: Precision={metrics['precision']:.3f}, Recall={metrics['recall']:.3f}, F1={metrics['f1']:.3f}")
    
    def test_email_case_accuracy(self):
        """Test accuracy with email case variations."""
        base_emails = [f"person{i}@test.com" for i in range(100)]
        
        source1_data = {
            'name': [f"Person {i}" for i in range(100)],
            'email': [email.upper() for email in base_emails],
            'phone': [f"555{i:07d}" for i in range(100)]
        }
        
        source2_data = {
            'name': [f"Person {i}" for i in range(100)],
            'email': [email.lower() for email in base_emails],
            'phone': [f"555{i:07d}" for i in range(100)]
        }
        
        ground_truth = {(i, i) for i in range(100)}
        
        df1 = pd.DataFrame(source1_data)
        df2 = pd.DataFrame(source2_data)
        
        file1 = os.path.join(self.temp_dir, 'source1_email.csv')
        file2 = os.path.join(self.temp_dir, 'source2_email.csv')
        df1.to_csv(file1, index=False)
        df2.to_csv(file2, index=False)
        
        config = {
            'source1': file1,
            'source2': file2,
            'output': self.output_path,
            'match_config': {
                'columns': [
                    {'source1': 'name', 'source2': 'name', 'weight': 0.3},
                    {'source1': 'email', 'source2': 'email', 'weight': 0.5},
                    {'source1': 'phone', 'source2': 'phone', 'weight': 0.2}
                ],
                'threshold': 0.85
            }
        }
        
        matcher = FuzzyMatcher(config)
        results = matcher.match()
        
        metrics = self.calculate_metrics(results, ground_truth)
        
        self.assertEqual(metrics['precision'], 1.0, "Precision should be 100% for email case variations")
        self.assertEqual(metrics['recall'], 1.0, "Recall should be 100% for email case variations")
        print(f"Email case metrics: Precision={metrics['precision']:.3f}, Recall={metrics['recall']:.3f}, F1={metrics['f1']:.3f}")
    
    def test_partial_match_accuracy(self):
        """Test accuracy with partial matches (some fields match, some don't)."""
        n_matches = 50
        n_non_matches = 50
        
        source1_data = {
            'name': (
                [f"Person {i}" for i in range(n_matches)] +
                [f"XYZNonMatch {i + 2000}" for i in range(n_non_matches)]
            ),
            'email': (
                [f"person{i}@test.com" for i in range(n_matches)] +
                [f"xyznonmatch{i + 2000}@otherdomain.org" for i in range(n_non_matches)]
            ),
            'phone': (
                [f"555{i:07d}" for i in range(n_matches)] +
                [f"777{i + 2000:07d}" for i in range(n_non_matches)]
            )
        }
        
        source2_data = {
            'name': (
                [f"Person {i}" for i in range(n_matches)] +
                [f"NonMatchPerson {i + 1000}" for i in range(n_non_matches)]
            ),
            'email': (
                [f"person{i}@test.com" for i in range(n_matches)] +
                [f"nonmatchperson{i + 1000}@differentdomain.net" for i in range(n_non_matches)]
            ),
            'phone': (
                [f"555{i:07d}" for i in range(n_matches)] +
                [f"999{i + 1000:07d}" for i in range(n_non_matches)]
            )
        }
        
        ground_truth = {(i, i) for i in range(n_matches)}
        
        df1 = pd.DataFrame(source1_data)
        df2 = pd.DataFrame(source2_data)
        
        file1 = os.path.join(self.temp_dir, 'source1_partial.csv')
        file2 = os.path.join(self.temp_dir, 'source2_partial.csv')
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
                'threshold': 0.85,
                'undecided_range': 0.05
            }
        }
        
        matcher = FuzzyMatcher(config)
        results = matcher.match()
        
        metrics = self.calculate_metrics(results, ground_truth)
        
        self.assertGreater(metrics['precision'], 0.9, "Precision should be > 90% for partial matches")
        self.assertGreater(metrics['recall'], 0.9, "Recall should be > 90% for partial matches")
        print(f"Partial match metrics: Precision={metrics['precision']:.3f}, Recall={metrics['recall']:.3f}, F1={metrics['f1']:.3f}")
    
    def test_large_scale_accuracy(self):
        """Test accuracy with large dataset (5K records with known matches)."""
        n_records = 5000
        
        source1_data = {
            'name': [f"Person {i}" for i in range(n_records)],
            'email': [f"person{i}@test.com" for i in range(n_records)],
            'phone': [f"555{i:07d}" for i in range(n_records)]
        }
        
        source2_data = {
            'name': [f"Person {i}" for i in range(n_records)],
            'email': [f"person{i}@test.com" for i in range(n_records)],
            'phone': [f"555{i:07d}" for i in range(n_records)]
        }
        
        ground_truth = {(i, i) for i in range(n_records)}
        
        df1 = pd.DataFrame(source1_data)
        df2 = pd.DataFrame(source2_data)
        
        file1 = os.path.join(self.temp_dir, 'source1_large.csv')
        file2 = os.path.join(self.temp_dir, 'source2_large.csv')
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
                'threshold': 0.85,
                'use_multiprocessing': True,
                'num_workers': 4
            }
        }
        
        matcher = FuzzyMatcher(config)
        results = matcher.match()
        
        metrics = self.calculate_metrics(results, ground_truth)
        
        self.assertGreater(metrics['precision'], 0.95, "Precision should be > 95% for large exact matches")
        self.assertGreater(metrics['recall'], 0.95, "Recall should be > 95% for large exact matches")
        print(f"Large scale metrics: Precision={metrics['precision']:.3f}, Recall={metrics['recall']:.3f}, F1={metrics['f1']:.3f}")
    
    def test_noise_accuracy(self):
        """Test accuracy with noise (typos, missing data, etc.)."""
        base_names = [f"Person {i}" for i in range(100)]
        base_emails = [f"person{i}@test.com" for i in range(100)]
        base_phones = [f"555{i:07d}" for i in range(100)]
        
        source1_data = {
            'name': base_names,
            'email': base_emails,
            'phone': base_phones
        }
        
        source2_data = {
            'name': [name.replace('Person', 'Persn') if i % 10 == 0 else name for i, name in enumerate(base_names)],
            'email': [email.replace('@test.com', '@test.co') if i % 10 == 1 else email for i, email in enumerate(base_emails)],
            'phone': [phone[:-1] + '0' if i % 10 == 2 else phone for i, phone in enumerate(base_phones)]
        }
        
        ground_truth = {(i, i) for i in range(100)}
        
        df1 = pd.DataFrame(source1_data)
        df2 = pd.DataFrame(source2_data)
        
        file1 = os.path.join(self.temp_dir, 'source1_noise.csv')
        file2 = os.path.join(self.temp_dir, 'source2_noise.csv')
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
                'threshold': 0.75
            }
        }
        
        matcher = FuzzyMatcher(config)
        results = matcher.match()
        
        metrics = self.calculate_metrics(results, ground_truth)
        
        self.assertGreater(metrics['precision'], 0.85, "Precision should be > 85% with noise")
        self.assertGreater(metrics['recall'], 0.85, "Recall should be > 85% with noise")
        print(f"Noise metrics: Precision={metrics['precision']:.3f}, Recall={metrics['recall']:.3f}, F1={metrics['f1']:.3f}")
    
    def test_address_variation_accuracy(self):
        """Test accuracy with address variations."""
        base_addresses = [f"{100+i} Main St" for i in range(100)]
        
        source1_data = {
            'name': [f"Person {i}" for i in range(100)],
            'email': [f"person{i}@test.com" for i in range(100)],
            'address': base_addresses
        }
        
        source2_data = {
            'name': [f"Person {i}" for i in range(100)],
            'email': [f"person{i}@test.com" for i in range(100)],
            'address': [addr.replace('St', 'Street') for addr in base_addresses]
        }
        
        ground_truth = {(i, i) for i in range(100)}
        
        df1 = pd.DataFrame(source1_data)
        df2 = pd.DataFrame(source2_data)
        
        file1 = os.path.join(self.temp_dir, 'source1_addr.csv')
        file2 = os.path.join(self.temp_dir, 'source2_addr.csv')
        df1.to_csv(file1, index=False)
        df2.to_csv(file2, index=False)
        
        config = {
            'source1': file1,
            'source2': file2,
            'output': self.output_path,
            'match_config': {
                'columns': [
                    {'source1': 'name', 'source2': 'name', 'weight': 0.3},
                    {'source1': 'email', 'source2': 'email', 'weight': 0.3},
                    {'source1': 'address', 'source2': 'address', 'weight': 0.4}
                ],
                'threshold': 0.85
            }
        }
        
        matcher = FuzzyMatcher(config)
        results = matcher.match()
        
        metrics = self.calculate_metrics(results, ground_truth)
        
        self.assertGreater(metrics['precision'], 0.9, "Precision should be > 90% for address variations")
        self.assertGreater(metrics['recall'], 0.9, "Recall should be > 90% for address variations")
        print(f"Address variation metrics: Precision={metrics['precision']:.3f}, Recall={metrics['recall']:.3f}, F1={metrics['f1']:.3f}")


if __name__ == '__main__':
    unittest.main(verbosity=2)

