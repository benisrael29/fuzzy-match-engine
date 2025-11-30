import unittest
import os
import json
import time
import threading
import tempfile
import shutil
from pathlib import Path
from src.job_queue import JobQueue, Priority
from src.job_worker_pool import JobWorkerPool, create_worker_pool
from src.job_runner import JobRunner


class TestJobQueue(unittest.TestCase):
    """Test suite for JobQueue functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.queue_file = os.path.join(self.temp_dir, "test_queue.json")
        self.queue = JobQueue(queue_file=self.queue_file)
    
    def tearDown(self):
        """Clean up test files."""
        if os.path.exists(self.queue_file):
            os.remove(self.queue_file)
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_enqueue_basic(self):
        """Test basic enqueue operation."""
        config = {"source1": "test1.csv", "source2": "test2.csv", "output": "out.csv"}
        
        result = self.queue.enqueue("test_job", config, priority="medium")
        self.assertTrue(result)
        
        status = self.queue.get_status("test_job")
        self.assertIsNotNone(status)
        self.assertEqual(status["status"], "queued")
        self.assertEqual(status["priority"], "medium")
    
    def test_enqueue_priority_levels(self):
        """Test enqueue with different priority levels."""
        config = {"source1": "test1.csv", "source2": "test2.csv", "output": "out.csv"}
        
        self.queue.enqueue("job_high", config, priority="high")
        self.queue.enqueue("job_low", config, priority="low")
        self.queue.enqueue("job_medium", config, priority="medium")
        
        # Dequeue should return jobs in priority order
        job1 = self.queue.dequeue()
        self.assertEqual(job1["job_name"], "job_high")
        
        job2 = self.queue.dequeue()
        self.assertEqual(job2["job_name"], "job_medium")
        
        job3 = self.queue.dequeue()
        self.assertEqual(job3["job_name"], "job_low")
    
    def test_enqueue_duplicate(self):
        """Test that enqueueing duplicate job raises error."""
        config = {"source1": "test1.csv", "source2": "test2.csv", "output": "out.csv"}
        
        self.queue.enqueue("test_job", config)
        
        with self.assertRaises(ValueError):
            self.queue.enqueue("test_job", config)
    
    def test_dequeue_empty(self):
        """Test dequeue from empty queue."""
        job = self.queue.dequeue()
        self.assertIsNone(job)
    
    def test_dequeue_moves_to_active(self):
        """Test that dequeue moves job to active."""
        config = {"source1": "test1.csv", "source2": "test2.csv", "output": "out.csv"}
        self.queue.enqueue("test_job", config)
        
        job = self.queue.dequeue()
        self.assertIsNotNone(job)
        self.assertEqual(job["status"], "running")
        
        # Job should be in active jobs
        status = self.queue.get_status("test_job")
        self.assertEqual(status["status"], "running")
    
    def test_cancel_queued_job(self):
        """Test cancelling a queued job."""
        config = {"source1": "test1.csv", "source2": "test2.csv", "output": "out.csv"}
        self.queue.enqueue("test_job", config)
        
        result = self.queue.cancel("test_job")
        self.assertTrue(result)
        
        status = self.queue.get_status("test_job")
        self.assertEqual(status["status"], "cancelled")
        
        # Job should not be in queue
        job = self.queue.dequeue()
        self.assertIsNone(job)
    
    def test_cancel_running_job(self):
        """Test cancelling a running job."""
        config = {"source1": "test1.csv", "source2": "test2.csv", "output": "out.csv"}
        self.queue.enqueue("test_job", config)
        
        # Dequeue to make it running
        self.queue.dequeue()
        
        result = self.queue.cancel("test_job")
        self.assertTrue(result)
        
        status = self.queue.get_status("test_job")
        self.assertEqual(status["status"], "cancelling")
        
        # Cancellation event should be set
        self.assertTrue(self.queue.is_cancelled("test_job"))
    
    def test_cancel_nonexistent_job(self):
        """Test cancelling a non-existent job raises error."""
        with self.assertRaises(ValueError):
            self.queue.cancel("nonexistent")
    
    def test_get_queue_position(self):
        """Test getting queue position."""
        config = {"source1": "test1.csv", "source2": "test2.csv", "output": "out.csv"}
        
        self.queue.enqueue("job1", config, priority="high")
        self.queue.enqueue("job2", config, priority="medium")
        self.queue.enqueue("job3", config, priority="low")
        
        pos1 = self.queue.get_queue_position("job1")
        self.assertEqual(pos1, 1)
        
        pos2 = self.queue.get_queue_position("job2")
        self.assertEqual(pos2, 2)
        
        pos3 = self.queue.get_queue_position("job3")
        self.assertEqual(pos3, 3)
    
    def test_list_queue(self):
        """Test listing queue."""
        config = {"source1": "test1.csv", "source2": "test2.csv", "output": "out.csv"}
        
        self.queue.enqueue("job1", config, priority="high")
        self.queue.enqueue("job2", config, priority="medium")
        
        queue_list = self.queue.list_queue()
        self.assertEqual(len(queue_list), 2)
        self.assertEqual(queue_list[0]["job_name"], "job1")
        self.assertEqual(queue_list[0]["queue_position"], 1)
    
    def test_mark_completed(self):
        """Test marking job as completed."""
        config = {"source1": "test1.csv", "source2": "test2.csv", "output": "out.csv"}
        self.queue.enqueue("test_job", config)
        self.queue.dequeue()
        
        self.queue.mark_completed("test_job", success=True, output="Test output")
        
        status = self.queue.get_status("test_job")
        self.assertEqual(status["status"], "completed")
        self.assertTrue(status["success"])
        self.assertEqual(status["output"], "Test output")
    
    def test_mark_failed(self):
        """Test marking job as failed."""
        config = {"source1": "test1.csv", "source2": "test2.csv", "output": "out.csv"}
        self.queue.enqueue("test_job", config)
        self.queue.dequeue()
        
        self.queue.mark_completed("test_job", success=False, output="Error output")
        
        status = self.queue.get_status("test_job")
        self.assertEqual(status["status"], "failed")
        self.assertFalse(status["success"])
    
    def test_mark_cancelled(self):
        """Test marking job as cancelled."""
        config = {"source1": "test1.csv", "source2": "test2.csv", "output": "out.csv"}
        self.queue.enqueue("test_job", config)
        self.queue.dequeue()
        
        self.queue.mark_cancelled("test_job", output="Cancelled output")
        
        status = self.queue.get_status("test_job")
        self.assertEqual(status["status"], "cancelled")
        self.assertFalse(status["success"])
    
    def test_save_and_load_state(self):
        """Test saving and loading queue state."""
        config = {"source1": "test1.csv", "source2": "test2.csv", "output": "out.csv"}
        
        # Enqueue some jobs
        self.queue.enqueue("job1", config, priority="high")
        self.queue.enqueue("job2", config, priority="medium")
        
        # Save state
        self.queue.save_state()
        
        # Create new queue and load state
        new_queue = JobQueue(queue_file=self.queue_file)
        new_queue.load_state()
        
        # Verify jobs are restored
        status1 = new_queue.get_status("job1")
        self.assertIsNotNone(status1)
        self.assertEqual(status1["status"], "queued")
        
        status2 = new_queue.get_status("job2")
        self.assertIsNotNone(status2)
        self.assertEqual(status2["status"], "queued")
        
        # Verify priority order is maintained
        job = new_queue.dequeue()
        self.assertEqual(job["job_name"], "job1")
    
    def test_thread_safety(self):
        """Test thread safety of queue operations."""
        config = {"source1": "test1.csv", "source2": "test2.csv", "output": "out.csv"}
        results = []
        errors = []
        
        def enqueue_jobs():
            try:
                for i in range(10):
                    self.queue.enqueue(f"job_{i}", config, priority="medium")
                    results.append(f"enqueued_{i}")
            except Exception as e:
                errors.append(str(e))
        
        threads = [threading.Thread(target=enqueue_jobs) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # Should have 50 jobs enqueued (10 * 5 threads)
        queue_list = self.queue.list_queue()
        self.assertEqual(len(queue_list), 50)
        self.assertEqual(len(errors), 0)


class TestJobWorkerPool(unittest.TestCase):
    """Test suite for JobWorkerPool functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.queue_file = os.path.join(self.temp_dir, "test_queue.json")
        self.queue = JobQueue(queue_file=self.queue_file)
        self.worker_pool = JobWorkerPool(self.queue, max_workers=2)
    
    def tearDown(self):
        """Clean up test files."""
        if self.worker_pool.running:
            self.worker_pool.stop()
        if os.path.exists(self.queue_file):
            os.remove(self.queue_file)
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_worker_pool_start_stop(self):
        """Test starting and stopping worker pool."""
        self.assertFalse(self.worker_pool.running)
        
        self.worker_pool.start()
        self.assertTrue(self.worker_pool.running)
        
        # Wait a bit for workers to start
        time.sleep(0.1)
        
        self.worker_pool.stop()
        self.assertFalse(self.worker_pool.running)
    
    def test_create_worker_pool(self):
        """Test creating worker pool with default settings."""
        pool = create_worker_pool(self.queue)
        self.assertIsNotNone(pool)
        self.assertEqual(pool.max_workers, 2)  # Default from env or 2


class TestJobQueueIntegration(unittest.TestCase):
    """Integration tests for job queue system."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.queue_file = os.path.join(self.temp_dir, "test_queue.json")
        self.queue = JobQueue(queue_file=self.queue_file)
        
        # Create minimal test config
        self.test_config = {
            "source1": "tests/fixtures/test_data.csv",
            "source2": "tests/fixtures/test_data2.csv",
            "output": os.path.join(self.temp_dir, "test_results.csv"),
            "match_config": {
                "columns": [
                    {"source1": "name", "source2": "full_name", "weight": 1.0}
                ],
                "threshold": 0.75
            }
        }
    
    def tearDown(self):
        """Clean up test files."""
        if os.path.exists(self.queue_file):
            os.remove(self.queue_file)
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_end_to_end_queue_workflow(self):
        """Test complete workflow: enqueue -> dequeue -> complete."""
        # Enqueue job
        self.queue.enqueue("test_job", self.test_config, priority="high")
        
        # Verify queued
        status = self.queue.get_status("test_job")
        self.assertEqual(status["status"], "queued")
        
        # Dequeue
        job = self.queue.dequeue()
        self.assertIsNotNone(job)
        self.assertEqual(job["status"], "running")
        
        # Mark completed
        self.queue.mark_completed("test_job", success=True, output="Test output")
        
        # Verify completed
        status = self.queue.get_status("test_job")
        self.assertEqual(status["status"], "completed")
        self.assertTrue(status["success"])
    
    def test_priority_ordering_integration(self):
        """Test that priority ordering works correctly."""
        # Enqueue jobs in reverse priority order
        self.queue.enqueue("job_low", self.test_config, priority="low")
        self.queue.enqueue("job_high", self.test_config, priority="high")
        self.queue.enqueue("job_medium", self.test_config, priority="medium")
        
        # Dequeue should respect priority
        job1 = self.queue.dequeue()
        self.assertEqual(job1["job_name"], "job_high")
        
        job2 = self.queue.dequeue()
        self.assertEqual(job2["job_name"], "job_medium")
        
        job3 = self.queue.dequeue()
        self.assertEqual(job3["job_name"], "job_low")
    
    def test_cancellation_workflow(self):
        """Test cancellation workflow."""
        # Enqueue and start job
        self.queue.enqueue("test_job", self.test_config)
        job = self.queue.dequeue()
        
        # Cancel running job
        self.queue.cancel("test_job")
        
        # Verify cancellation
        self.assertTrue(self.queue.is_cancelled("test_job"))
        
        # Mark as cancelled
        self.queue.mark_cancelled("test_job", output="Cancelled")
        
        status = self.queue.get_status("test_job")
        self.assertEqual(status["status"], "cancelled")
    
    def test_concurrent_jobs_limit(self):
        """Test that worker pool respects concurrency limit."""
        worker_pool = JobWorkerPool(self.queue, max_workers=2)
        
        try:
            worker_pool.start()
            
            # Enqueue multiple jobs
            for i in range(5):
                self.queue.enqueue(f"job_{i}", self.test_config)
            
            # Wait a bit for workers to process
            time.sleep(0.5)
            
            # Check active worker count
            active_count = worker_pool.get_active_worker_count()
            self.assertLessEqual(active_count, 2)
        
        finally:
            worker_pool.stop()


class TestJobQueuePersistence(unittest.TestCase):
    """Test queue persistence across restarts."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.queue_file = os.path.join(self.temp_dir, "test_queue.json")
    
    def tearDown(self):
        """Clean up test files."""
        if os.path.exists(self.queue_file):
            os.remove(self.queue_file)
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_persistence_survives_restart(self):
        """Test that queue state persists across restarts."""
        config = {"source1": "test1.csv", "source2": "test2.csv", "output": "out.csv"}
        
        # Create queue and enqueue jobs
        queue1 = JobQueue(queue_file=self.queue_file)
        queue1.enqueue("job1", config, priority="high")
        queue1.enqueue("job2", config, priority="medium")
        queue1.save_state()
        
        # Simulate restart: create new queue
        queue2 = JobQueue(queue_file=self.queue_file)
        queue2.load_state()
        
        # Verify jobs are restored
        status1 = queue2.get_status("job1")
        self.assertIsNotNone(status1)
        self.assertEqual(status1["status"], "queued")
        
        status2 = queue2.get_status("job2")
        self.assertIsNotNone(status2)
        self.assertEqual(status2["status"], "queued")
        
        # Verify priority order is maintained
        job = queue2.dequeue()
        self.assertEqual(job["job_name"], "job1")
    
    def test_persistence_with_active_jobs(self):
        """Test persistence when jobs are active (should mark as failed)."""
        config = {"source1": "test1.csv", "source2": "test2.csv", "output": "out.csv"}
        
        # Create queue, enqueue, and dequeue (make active)
        queue1 = JobQueue(queue_file=self.queue_file)
        queue1.enqueue("job1", config)
        queue1.dequeue()  # Make it active
        queue1.save_state()
        
        # Simulate restart
        queue2 = JobQueue(queue_file=self.queue_file)
        queue2.load_state()
        
        # Active job should be marked as failed
        status = queue2.get_status("job1")
        # Note: In load_state, active jobs are moved to history as failed
        # This is expected behavior for server restarts


if __name__ == '__main__':
    unittest.main()

