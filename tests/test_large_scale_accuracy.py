import unittest
import os
import pandas as pd
import tempfile
import shutil
import time
import random
import string
from src.matcher import FuzzyMatcher
from src.algorithms import (
    levenshtein_similarity,
    jaro_winkler_similarity,
    token_set_ratio
)


def generate_random_name():
    """Generate a random name."""
    first_names = ['John', 'Jane', 'Robert', 'Mary', 'William', 'Patricia', 
                   'Michael', 'Jennifer', 'David', 'Linda', 'Richard', 'Elizabeth',
                   'Joseph', 'Susan', 'Thomas', 'Jessica', 'Charles', 'Sarah',
                   'Christopher', 'Karen', 'Daniel', 'Nancy', 'Matthew', 'Lisa',
                   'Anthony', 'Betty', 'Mark', 'Margaret', 'Donald', 'Sandra']
    last_names = ['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia',
                  'Miller', 'Davis', 'Rodriguez', 'Martinez', 'Hernandez', 'Lopez',
                  'Wilson', 'Anderson', 'Thomas', 'Taylor', 'Moore', 'Jackson',
                  'Martin', 'Lee', 'Thompson', 'White', 'Harris', 'Sanchez',
                  'Clark', 'Ramirez', 'Lewis', 'Robinson', 'Walker', 'Young']
    return f"{random.choice(first_names)} {random.choice(last_names)}"


def generate_random_email(name=None):
    """Generate a random email."""
    if name:
        base = name.lower().replace(' ', '.')
    else:
        base = ''.join(random.choices(string.ascii_lowercase, k=8))
    domains = ['gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com', 'test.com']
    return f"{base}@{random.choice(domains)}"


def generate_random_phone():
    """Generate a random phone number."""
    area = random.randint(200, 999)
    exchange = random.randint(200, 999)
    number = random.randint(1000, 9999)
    return f"{area}{exchange}{number}"


def generate_random_address():
    """Generate a random address."""
    numbers = random.randint(100, 9999)
    streets = ['Main St', 'Oak Ave', 'Park Blvd', 'Elm St', 'Maple Dr',
               'Cedar Ln', 'Pine Rd', 'First St', 'Second Ave', 'Third Blvd']
    return f"{numbers} {random.choice(streets)}"


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
    
    def test_10k_records(self):
        """Test matching with 10,000 records."""
        n_records = 10000
        
        source1_data = {
            'name': [generate_random_name() for _ in range(n_records)],
            'email': [generate_random_email() for _ in range(n_records)],
            'phone': [generate_random_phone() for _ in range(n_records)]
        }
        
        source2_data = {
            'name': [name for name in source1_data['name']],
            'email': [email for email in source1_data['email']],
            'phone': [phone for phone in source1_data['phone']]
        }
        
        df1 = pd.DataFrame(source1_data)
        df2 = pd.DataFrame(source2_data)
        
        file1 = os.path.join(self.temp_dir, 'source1_10k.csv')
        file2 = os.path.join(self.temp_dir, 'source2_10k.csv')
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
        
        source1_data = {
            'name': [generate_random_name() for _ in range(n_records)],
            'email': [generate_random_email() for _ in range(n_records)],
            'phone': [generate_random_phone() for _ in range(n_records)]
        }
        
        source2_data = {
            'name': [name for name in source1_data['name']],
            'email': [email for email in source1_data['email']],
            'phone': [phone for phone in source1_data['phone']]
        }
        
        df1 = pd.DataFrame(source1_data)
        df2 = pd.DataFrame(source2_data)
        
        file1 = os.path.join(self.temp_dir, 'source1_50k.csv')
        file2 = os.path.join(self.temp_dir, 'source2_50k.csv')
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
                'num_workers': 4,
                'chunk_size': 5000
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
        
        source1_data = {
            'name': [generate_random_name() for _ in range(n_records)],
            'email': [generate_random_email() for _ in range(n_records)],
            'phone': [generate_random_phone() for _ in range(n_records)]
        }
        
        source2_data = {
            'name': [name for name in source1_data['name']],
            'email': [email for email in source1_data['email']],
            'phone': [phone for phone in source1_data['phone']]
        }
        
        df1 = pd.DataFrame(source1_data)
        df2 = pd.DataFrame(source2_data)
        
        file1 = os.path.join(self.temp_dir, 'source1_100k.csv')
        file2 = os.path.join(self.temp_dir, 'source2_100k.csv')
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
                'num_workers': 4,
                'chunk_size': 10000,
                'early_termination': True
            }
        }
        
        start_time = time.time()
        matcher = FuzzyMatcher(config)
        results = matcher.match()
        elapsed = time.time() - start_time
        
        self.assertGreater(len(results), 0)
        print(f"100K records matched in {elapsed:.2f} seconds")
    
    def test_unequal_sized_datasets(self):
        """Test matching with unequal dataset sizes."""
        source1_size = 1000
        source2_size = 10000
        
        source1_data = {
            'name': [generate_random_name() for _ in range(source1_size)],
            'email': [generate_random_email() for _ in range(source1_size)],
            'phone': [generate_random_phone() for _ in range(source1_size)]
        }
        
        source2_data = {
            'name': [generate_random_name() for _ in range(source2_size)],
            'email': [generate_random_email() for _ in range(source2_size)],
            'phone': [generate_random_phone() for _ in range(source2_size)]
        }
        
        df1 = pd.DataFrame(source1_data)
        df2 = pd.DataFrame(source2_data)
        
        file1 = os.path.join(self.temp_dir, 'source1_1k.csv')
        file2 = os.path.join(self.temp_dir, 'source2_10k.csv')
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
            'name': [f"Person {i}" for i in range(n_matches + n_non_matches)],
            'email': [f"person{i}@test.com" for i in range(n_matches + n_non_matches)],
            'phone': [f"555{i:07d}" for i in range(n_matches + n_non_matches)]
        }
        
        source2_data = {
            'name': [f"Person {i}" for i in range(n_matches)] + [f"Different {i}" for i in range(n_non_matches)],
            'email': [f"person{i}@test.com" for i in range(n_matches)] + [f"different{i}@test.com" for i in range(n_non_matches)],
            'phone': [f"555{i:07d}" for i in range(n_matches)] + [f"999{i:07d}" for i in range(n_non_matches)]
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
                'threshold': 0.85
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

