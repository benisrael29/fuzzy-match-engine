import unittest
import os
import pandas as pd
import tempfile
import shutil
from src.matcher import FuzzyMatcher


class TestAccuracyMetrics(unittest.TestCase):
    """Test accuracy metrics and evaluation methods."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.output_path = os.path.join(self.temp_dir, 'results.csv')
    
    def tearDown(self):
        """Clean up test files."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def calculate_confusion_matrix(self, results, ground_truth, threshold=0.85):
        """Calculate confusion matrix metrics."""
        true_positives = 0
        false_positives = 0
        true_negatives = 0
        false_negatives = 0
        
        accepted_matches = set()
        for _, row in results.iterrows():
            if row['match_result'] == 'accept' and row['overall_score'] >= threshold:
                idx1 = row['source1_index']
                idx2 = row['source2_index']
                accepted_matches.add((idx1, idx2))
        
        all_pairs = set()
        for idx1, idx2 in ground_truth:
            all_pairs.add((idx1, idx2))
        
        for idx1, idx2 in ground_truth:
            if (idx1, idx2) in accepted_matches:
                true_positives += 1
            else:
                false_negatives += 1
        
        for _, row in results.iterrows():
            idx1 = row['source1_index']
            idx2 = row['source2_index']
            if (idx1, idx2) not in ground_truth and (idx1, idx2) in accepted_matches:
                false_positives += 1
        
        return {
            'tp': true_positives,
            'fp': false_positives,
            'tn': true_negatives,
            'fn': false_negatives
        }
    
    def test_precision_recall_at_different_thresholds(self):
        """Test precision and recall at different threshold values."""
        n_records = 200
        
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
        
        file1 = os.path.join(self.temp_dir, 'source1_thresh.csv')
        file2 = os.path.join(self.temp_dir, 'source2_thresh.csv')
        df1.to_csv(file1, index=False)
        df2.to_csv(file2, index=False)
        
        thresholds = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        results_by_threshold = {}
        
        for threshold in thresholds:
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
                    'threshold': threshold
                }
            }
            
            matcher = FuzzyMatcher(config)
            results = matcher.match()
            
            cm = self.calculate_confusion_matrix(results, ground_truth, threshold)
            precision = cm['tp'] / (cm['tp'] + cm['fp']) if (cm['tp'] + cm['fp']) > 0 else 0.0
            recall = cm['tp'] / (cm['tp'] + cm['fn']) if (cm['tp'] + cm['fn']) > 0 else 0.0
            
            results_by_threshold[threshold] = {
                'precision': precision,
                'recall': recall,
                'f1': 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            }
        
        for threshold, metrics in results_by_threshold.items():
            print(f"Threshold {threshold}: Precision={metrics['precision']:.3f}, Recall={metrics['recall']:.3f}, F1={metrics['f1']:.3f}")
        
        self.assertGreater(results_by_threshold[0.85]['precision'], 0.9)
        self.assertGreater(results_by_threshold[0.85]['recall'], 0.9)
    
    def test_score_distribution(self):
        """Test that match scores are properly distributed."""
        n_records = 500
        
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
        
        df1 = pd.DataFrame(source1_data)
        df2 = pd.DataFrame(source2_data)
        
        file1 = os.path.join(self.temp_dir, 'source1_dist.csv')
        file2 = os.path.join(self.temp_dir, 'source2_dist.csv')
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
        
        scores = results['overall_score'].values
        mean_score = scores.mean()
        min_score = scores.min()
        max_score = scores.max()
        
        self.assertGreater(mean_score, 0.8, "Mean score should be high for exact matches")
        self.assertGreater(min_score, 0.7, "Minimum score should be reasonable")
        self.assertEqual(max_score, 1.0, "Maximum score should be 1.0 for exact matches")
        
        print(f"Score distribution: Mean={mean_score:.3f}, Min={min_score:.3f}, Max={max_score:.3f}")
    
    def test_false_positive_rate(self):
        """Test false positive rate with non-matching records."""
        n_records = 200
        
        source1_data = {
            'name': [f"Person {i}" for i in range(n_records)],
            'email': [f"person{i}@test.com" for i in range(n_records)],
            'phone': [f"555{i:07d}" for i in range(n_records)]
        }
        
        source2_data = {
            'name': [f"Different {i}" for i in range(n_records)],
            'email': [f"different{i}@test.com" for i in range(n_records)],
            'phone': [f"999{i:07d}" for i in range(n_records)]
        }
        
        ground_truth = set()
        
        df1 = pd.DataFrame(source1_data)
        df2 = pd.DataFrame(source2_data)
        
        file1 = os.path.join(self.temp_dir, 'source1_fp.csv')
        file2 = os.path.join(self.temp_dir, 'source2_fp.csv')
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
        
        false_positives = len(results[results['match_result'] == 'accept'])
        false_positive_rate = false_positives / len(results) if len(results) > 0 else 0.0
        
        self.assertLess(false_positive_rate, 0.1, "False positive rate should be < 10% for non-matching records")
        print(f"False positive rate: {false_positive_rate:.3f}")
    
    def test_matching_coverage(self):
        """Test that all expected matches are found."""
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
        
        file1 = os.path.join(self.temp_dir, 'source1_coverage.csv')
        file2 = os.path.join(self.temp_dir, 'source2_coverage.csv')
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
        
        found_matches = set()
        for _, row in results.iterrows():
            if row['match_result'] == 'accept':
                found_matches.add((row['source1_index'], row['source2_index']))
        
        coverage = len(found_matches & ground_truth) / len(ground_truth) if len(ground_truth) > 0 else 0.0
        
        self.assertGreater(coverage, 0.95, "Coverage should be > 95%")
        print(f"Matching coverage: {coverage:.3f}")
    
    def test_score_consistency(self):
        """Test that scores are consistent across multiple runs."""
        n_records = 100
        
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
        
        df1 = pd.DataFrame(source1_data)
        df2 = pd.DataFrame(source2_data)
        
        file1 = os.path.join(self.temp_dir, 'source1_consistency.csv')
        file2 = os.path.join(self.temp_dir, 'source2_consistency.csv')
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
        
        scores_run1 = None
        scores_run2 = None
        
        for run in range(2):
            matcher = FuzzyMatcher(config)
            results = matcher.match()
            scores = results.sort_values('source1_index')['overall_score'].values
            
            if run == 0:
                scores_run1 = scores
            else:
                scores_run2 = scores
        
        if scores_run1 is not None and scores_run2 is not None:
            import numpy as np
            diff = np.abs(scores_run1 - scores_run2)
            max_diff = diff.max()
            
            self.assertLess(max_diff, 0.001, "Scores should be consistent across runs")
            print(f"Maximum score difference between runs: {max_diff:.6f}")


if __name__ == '__main__':
    unittest.main(verbosity=2)

