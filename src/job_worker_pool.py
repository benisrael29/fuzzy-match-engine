import os
import sys
import io
import threading
import time
from typing import Optional
from .job_queue import JobQueue
from .job_runner import JobRunner


class JobWorkerPool:
    """Manages worker threads to process jobs from the queue."""
    
    def __init__(self, job_queue: JobQueue, max_workers: int = 2):
        """
        Initialize JobWorkerPool.
        
        Args:
            job_queue: JobQueue instance to process jobs from
            max_workers: Maximum number of concurrent workers
        """
        self.job_queue = job_queue
        self.max_workers = max_workers
        self.job_runner = JobRunner()
        
        self.workers: list[threading.Thread] = []
        self.running = False
        self.shutdown_event = threading.Event()
        self.worker_lock = threading.Lock()
        
        # Output capture: job_name -> StringIO buffer
        self.output_buffers: dict[str, io.StringIO] = {}
        self.output_lock = threading.Lock()
    
    def start(self):
        """Start the worker pool."""
        if self.running:
            return
        
        self.running = True
        self.shutdown_event.clear()
        
        # Start worker threads
        for i in range(self.max_workers):
            worker = threading.Thread(
                target=self._worker_loop,
                name=f"JobWorker-{i+1}",
                daemon=True
            )
            worker.start()
            self.workers.append(worker)
    
    def stop(self, timeout: float = 30.0):
        """
        Stop the worker pool gracefully.
        
        Args:
            timeout: Maximum time to wait for workers to finish (seconds)
        """
        if not self.running:
            return
        
        self.running = False
        self.shutdown_event.set()
        
        # Wait for workers to finish
        for worker in self.workers:
            worker.join(timeout=timeout)
        
        self.workers.clear()
    
    def _worker_loop(self):
        """Main loop for worker thread."""
        while self.running and not self.shutdown_event.is_set():
            try:
                # Get next job from queue
                job_data = self.job_queue.dequeue()
                
                if job_data is None:
                    # No jobs available, wait a bit
                    time.sleep(1)
                    continue
                
                job_name = job_data["job_name"]
                config = job_data["config"]
                
                # Check if job was cancelled before starting
                if self.job_queue.is_cancelled(job_name):
                    self.job_queue.mark_cancelled(job_name, "Job was cancelled before execution started")
                    continue
                
                # Execute job
                self._execute_job(job_name, config)
                
            except Exception as e:
                # Handle unexpected errors
                print(f"Worker error: {e}", file=sys.stderr)
                import traceback
                traceback.print_exc()
                time.sleep(1)
    
    def _execute_job(self, job_name: str, config: dict):
        """
        Execute a job and update queue status.
        
        Args:
            job_name: Name of the job
            config: Job configuration dictionary
        """
        # Set up output capture
        old_stdout = sys.stdout
        buffer = io.StringIO()
        
        with self.output_lock:
            self.output_buffers[job_name] = buffer
        
        sys.stdout = buffer
        
        try:
            # Get cancellation event
            cancel_event = self.job_queue.get_cancel_event(job_name)
            
            # Execute job (pass cancellation event if needed)
            # For now, JobRunner will check cancellation via the queue
            success = self.job_runner.run_job(config, job_name, cancel_event)
            
            # Get output
            output = buffer.getvalue()
            
            # Check if job was cancelled
            if self.job_queue.is_cancelled(job_name):
                self.job_queue.mark_cancelled(job_name, output)
            else:
                self.job_queue.mark_completed(job_name, success, output)
        
        except KeyboardInterrupt:
            output = buffer.getvalue() + "\nJob cancelled by user"
            self.job_queue.mark_cancelled(job_name, output)
        
        except Exception as e:
            output = buffer.getvalue() + f"\nError: {str(e)}"
            import traceback
            output += "\n" + traceback.format_exc()
            self.job_queue.mark_completed(job_name, False, output)
        
        finally:
            # Restore stdout
            sys.stdout = old_stdout
            
            # Clean up output buffer
            with self.output_lock:
                if job_name in self.output_buffers:
                    del self.output_buffers[job_name]
    
    def get_output(self, job_name: str) -> str:
        """
        Get output for a job (from buffer if running, or from queue if completed).
        
        Args:
            job_name: Name of the job
        
        Returns:
            Job output text
        """
        # Check if job is currently running (output in buffer)
        with self.output_lock:
            if job_name in self.output_buffers:
                return self.output_buffers[job_name].getvalue()
        
        # Check queue status for completed jobs
        status = self.job_queue.get_status(job_name)
        if status:
            return status.get("output", "")
        
        return ""
    
    def get_active_worker_count(self) -> int:
        """
        Get number of currently active workers.
        
        Returns:
            Number of active workers
        """
        with self.worker_lock:
            return len([w for w in self.workers if w.is_alive()])


def create_worker_pool(job_queue: JobQueue, max_workers: Optional[int] = None) -> JobWorkerPool:
    """
    Create and configure a JobWorkerPool.
    
    Args:
        job_queue: JobQueue instance
        max_workers: Maximum workers (defaults to MAX_CONCURRENT_JOBS env var or 2)
    
    Returns:
        Configured JobWorkerPool instance
    """
    if max_workers is None:
        max_workers = int(os.getenv("MAX_CONCURRENT_JOBS", "2"))
    
    return JobWorkerPool(job_queue, max_workers=max_workers)

