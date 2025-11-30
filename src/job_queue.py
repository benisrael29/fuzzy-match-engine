import json
import os
import threading
from typing import Dict, Any, Optional, List
from datetime import datetime
from enum import IntEnum


class Priority(IntEnum):
    HIGH = 0
    MEDIUM = 1
    LOW = 2


class JobQueue:
    """Thread-safe job queue with priority support and file persistence."""
    
    def __init__(self, queue_file: str = "jobs/queue_state.json"):
        """
        Initialize JobQueue.
        
        Args:
            queue_file: Path to file for persisting queue state
        """
        # Normalize path to handle both relative and absolute paths
        if not os.path.isabs(queue_file):
            # Relative path - ensure it's relative to current working directory
            self.queue_file = os.path.normpath(queue_file)
        else:
            self.queue_file = queue_file
        self.lock = threading.Lock()
        
        # Priority queue: list of (priority, queued_at, job_name, job_data)
        # Lower priority number = higher priority
        self._queue: List[tuple] = []
        
        # Active jobs: job_name -> job_data
        self._active_jobs: Dict[str, Dict[str, Any]] = {}
        
        # Job history: job_name -> job_data (completed/failed/cancelled)
        self._history: Dict[str, Dict[str, Any]] = {}
        
        # Cancellation events: job_name -> threading.Event
        self._cancel_events: Dict[str, threading.Event] = {}
        
        # Load state from file if it exists
        # Wrap in try-except to prevent initialization failures
        try:
            self.load_state()
        except Exception as e:
            print(f"Warning: Failed to load queue state during initialization: {e}")
            # Continue with empty queue
    
    def _get_priority_value(self, priority: str) -> int:
        """Convert priority string to integer value."""
        priority_map = {
            "high": Priority.HIGH,
            "medium": Priority.MEDIUM,
            "low": Priority.LOW
        }
        return priority_map.get(priority.lower(), Priority.MEDIUM)
    
    def enqueue(
        self,
        job_name: str,
        config: Dict[str, Any],
        priority: str = "medium"
    ) -> bool:
        """
        Add a job to the queue.
        
        Args:
            job_name: Name of the job
            config: Job configuration dictionary
            priority: Priority level ("high", "medium", "low")
        
        Returns:
            True if enqueued successfully
        
        Raises:
            ValueError: If job is already queued or running
        """
        with self.lock:
            # Check if job is already queued
            for _, _, name, _ in self._queue:
                if name == job_name:
                    raise ValueError(f"Job '{job_name}' is already queued")
            
            # Check if job is already running
            if job_name in self._active_jobs:
                raise ValueError(f"Job '{job_name}' is already running")
            
            # Check if job was recently completed/failed (prevent immediate re-queue)
            if job_name in self._history:
                history_status = self._history[job_name].get("status")
                if history_status in ["completed", "failed", "cancelled"]:
                    # Allow re-queueing after previous completion
                    pass
            
            # Create job entry
            priority_value = self._get_priority_value(priority)
            job_data = {
                "job_name": job_name,
                "config": config,
                "priority": priority,
                "priority_value": priority_value,
                "status": "queued",
                "queued_at": datetime.now().isoformat(),
                "started_at": None,
                "completed_at": None,
                "message": "Job queued"
            }
            
            # Insert into priority queue (sorted by priority, then by queued_at)
            queue_entry = (priority_value, job_data["queued_at"], job_name, job_data)
            self._queue.append(queue_entry)
            self._queue.sort(key=lambda x: (x[0], x[1]))
            
            # Create cancellation event
            self._cancel_events[job_name] = threading.Event()
        
        # Save state outside of lock to avoid blocking
        try:
            self.save_state()
        except Exception as e:
            print(f"Warning: Failed to save queue state after enqueue: {e}")
        
        return True
    
    def dequeue(self) -> Optional[Dict[str, Any]]:
        """
        Remove and return the next job from the queue.
        
        Returns:
            Job data dictionary or None if queue is empty
        """
        with self.lock:
            if not self._queue:
                return None
            
            # Get highest priority job (first in sorted queue)
            priority_value, queued_at, job_name, job_data = self._queue.pop(0)
            
            # Move to active jobs
            job_data["status"] = "running"
            job_data["started_at"] = datetime.now().isoformat()
            job_data["message"] = "Job execution started"
            self._active_jobs[job_name] = job_data
        
        # Save state outside of lock to avoid blocking
        try:
            self.save_state()
        except Exception as e:
            print(f"Warning: Failed to save queue state after dequeue: {e}")
        
        return job_data
    
    def cancel(self, job_name: str) -> bool:
        """
        Cancel a queued or running job.
        
        Args:
            job_name: Name of the job to cancel
        
        Returns:
            True if cancelled successfully
        
        Raises:
            ValueError: If job is not found or cannot be cancelled
        """
        cancelled_from_queue = False
        with self.lock:
            # Check if job is in queue
            for i, (_, _, name, job_data) in enumerate(self._queue):
                if name == job_name:
                    # Remove from queue
                    self._queue.pop(i)
                    
                    # Move to history
                    job_data["status"] = "cancelled"
                    job_data["completed_at"] = datetime.now().isoformat()
                    job_data["message"] = "Job cancelled (was queued)"
                    self._history[job_name] = job_data
                    
                    # Set cancellation event
                    if job_name in self._cancel_events:
                        self._cancel_events[job_name].set()
                    
                    cancelled_from_queue = True
                    break
            
            # Check if job is running
            if not cancelled_from_queue and job_name in self._active_jobs:
                # Set cancellation event
                if job_name in self._cancel_events:
                    self._cancel_events[job_name].set()
                
                # Update status (will be moved to history when worker finishes)
                self._active_jobs[job_name]["status"] = "cancelling"
                self._active_jobs[job_name]["message"] = "Job cancellation requested"
                cancelled_from_queue = True
            
            if not cancelled_from_queue:
                raise ValueError(f"Job '{job_name}' not found in queue or active jobs")
        
        # Save state outside of lock to avoid blocking
        try:
            self.save_state()
        except Exception as e:
            print(f"Warning: Failed to save queue state after cancel: {e}")
        
        return True
    
    def get_status(self, job_name: str) -> Optional[Dict[str, Any]]:
        """
        Get status of a job.
        
        Args:
            job_name: Name of the job
        
        Returns:
            Job status dictionary or None if not found
        """
        with self.lock:
            # Check queue
            for _, _, name, job_data in self._queue:
                if name == job_name:
                    return job_data.copy()
            
            # Check active jobs
            if job_name in self._active_jobs:
                return self._active_jobs[job_name].copy()
            
            # Check history
            if job_name in self._history:
                return self._history[job_name].copy()
            
            return None
    
    def get_queue_position(self, job_name: str) -> Optional[int]:
        """
        Get position of a job in the queue (1-indexed).
        
        Args:
            job_name: Name of the job
        
        Returns:
            Position in queue (1 = next to run) or None if not queued
        """
        with self.lock:
            for i, (_, _, name, _) in enumerate(self._queue):
                if name == job_name:
                    return i + 1
            return None
    
    def list_queue(self) -> List[Dict[str, Any]]:
        """
        List all jobs in the queue with their positions.
        
        Returns:
            List of job dictionaries with queue positions
        """
        with self.lock:
            result = []
            for i, (_, _, name, job_data) in enumerate(self._queue):
                job_info = job_data.copy()
                job_info["queue_position"] = i + 1
                result.append(job_info)
            return result
    
    def list_active(self) -> List[Dict[str, Any]]:
        """
        List all currently running jobs.
        
        Returns:
            List of active job dictionaries
        """
        with self.lock:
            return [job_data.copy() for job_data in self._active_jobs.values()]
    
    def mark_completed(self, job_name: str, success: bool, output: str = ""):
        """
        Mark a job as completed or failed.
        
        Args:
            job_name: Name of the job
            success: True if successful, False if failed
            output: Job output text
        """
        with self.lock:
            if job_name not in self._active_jobs:
                return
            
            job_data = self._active_jobs.pop(job_name)
            job_data["status"] = "completed" if success else "failed"
            job_data["completed_at"] = datetime.now().isoformat()
            job_data["message"] = "Job completed successfully" if success else "Job execution failed"
            job_data["success"] = success
            job_data["output"] = output
            
            # Move to history
            self._history[job_name] = job_data
            
            # Clean up cancellation event
            if job_name in self._cancel_events:
                del self._cancel_events[job_name]
        
        # Save state outside of lock to avoid blocking
        try:
            self.save_state()
        except Exception as e:
            print(f"Warning: Failed to save queue state after mark_completed: {e}")
    
    def mark_cancelled(self, job_name: str, output: str = ""):
        """
        Mark a job as cancelled.
        
        Args:
            job_name: Name of the job
            output: Job output text
        """
        with self.lock:
            if job_name not in self._active_jobs:
                return
            
            job_data = self._active_jobs.pop(job_name)
            job_data["status"] = "cancelled"
            job_data["completed_at"] = datetime.now().isoformat()
            job_data["message"] = "Job cancelled"
            job_data["success"] = False
            job_data["output"] = output
            
            # Move to history
            self._history[job_name] = job_data
            
            # Clean up cancellation event
            if job_name in self._cancel_events:
                del self._cancel_events[job_name]
        
        # Save state outside of lock to avoid blocking
        try:
            self.save_state()
        except Exception as e:
            print(f"Warning: Failed to save queue state after mark_cancelled: {e}")
    
    def is_cancelled(self, job_name: str) -> bool:
        """
        Check if a job has been cancelled.
        
        Args:
            job_name: Name of the job
        
        Returns:
            True if cancelled, False otherwise
        """
        with self.lock:
            if job_name in self._cancel_events:
                return self._cancel_events[job_name].is_set()
            return False
    
    def get_cancel_event(self, job_name: str) -> Optional[threading.Event]:
        """
        Get the cancellation event for a job.
        
        Args:
            job_name: Name of the job
        
        Returns:
            threading.Event or None if not found
        """
        with self.lock:
            return self._cancel_events.get(job_name)
    
    def save_state(self):
        """Save queue state to file."""
        try:
            # Create directory if needed
            queue_dir = os.path.dirname(self.queue_file)
            if queue_dir:  # Only create directory if path has a directory component
                os.makedirs(queue_dir, exist_ok=True)
            
            # Acquire lock only to read state, release before file I/O
            with self.lock:
                state = {
                    "queue": [
                        {
                            "job_name": name,
                            "config": job_data["config"],
                            "priority": job_data["priority"],
                            "queued_at": job_data["queued_at"]
                        }
                        for _, _, name, job_data in self._queue
                    ],
                    "active_jobs": {
                        name: {
                            "config": job_data["config"],
                            "priority": job_data["priority"],
                            "status": job_data["status"],
                            "queued_at": job_data["queued_at"],
                            "started_at": job_data.get("started_at")
                        }
                        for name, job_data in self._active_jobs.items()
                    },
                    "saved_at": datetime.now().isoformat()
                }
            
            # Write file outside of lock to avoid blocking
            with open(self.queue_file, 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            # Log error but don't fail
            print(f"Warning: Failed to save queue state: {e}")
    
    def load_state(self):
        """Load queue state from file."""
        if not os.path.exists(self.queue_file):
            return
        
        try:
            # Read file first, then acquire lock to minimize lock time
            with open(self.queue_file, 'r') as f:
                state = json.load(f)
            
            # Only acquire lock when actually modifying state
            with self.lock:
                # Restore queue
                self._queue = []
                for job_info in state.get("queue", []):
                    priority_value = self._get_priority_value(job_info.get("priority", "medium"))
                    queued_at = job_info.get("queued_at", datetime.now().isoformat())
                    job_name = job_info["job_name"]
                    
                    job_data = {
                        "job_name": job_name,
                        "config": job_info["config"],
                        "priority": job_info.get("priority", "medium"),
                        "priority_value": priority_value,
                        "status": "queued",
                        "queued_at": queued_at,
                        "started_at": None,
                        "completed_at": None,
                        "message": "Job queued (restored from state)"
                    }
                    
                    queue_entry = (priority_value, queued_at, job_name, job_data)
                    self._queue.append(queue_entry)
                    self._cancel_events[job_name] = threading.Event()
                
                # Sort queue
                self._queue.sort(key=lambda x: (x[0], x[1]))
                
                # Restore active jobs (they will need to be re-queued or handled)
                # For now, we'll mark them as failed since we don't know their state
                for job_name, job_info in state.get("active_jobs", {}).items():
                    priority_value = self._get_priority_value(job_info.get("priority", "medium"))
                    job_data = {
                        "job_name": job_name,
                        "config": job_info["config"],
                        "priority": job_info.get("priority", "medium"),
                        "priority_value": priority_value,
                        "status": "failed",
                        "queued_at": job_info.get("queued_at", datetime.now().isoformat()),
                        "started_at": job_info.get("started_at"),
                        "completed_at": datetime.now().isoformat(),
                        "message": "Job failed (server restart during execution)",
                        "success": False
                    }
                    self._history[job_name] = job_data
        except Exception as e:
            # Log error but don't fail
            print(f"Warning: Failed to load queue state: {e}")
    
    def clear_history(self, older_than_days: Optional[int] = None):
        """
        Clear job history.
        
        Args:
            older_than_days: If provided, only clear jobs older than this many days
        """
        with self.lock:
            if older_than_days is None:
                self._history.clear()
            else:
                cutoff = datetime.now().timestamp() - (older_than_days * 24 * 60 * 60)
                to_remove = []
                for job_name, job_data in self._history.items():
                    completed_at = job_data.get("completed_at")
                    if completed_at:
                        try:
                            completed_ts = datetime.fromisoformat(completed_at).timestamp()
                            if completed_ts < cutoff:
                                to_remove.append(job_name)
                        except (ValueError, TypeError):
                            pass
                
                for job_name in to_remove:
                    del self._history[job_name]
        
        # Save state outside of lock to avoid blocking
        try:
            self.save_state()
        except Exception as e:
            print(f"Warning: Failed to save queue state after clear_history: {e}")
