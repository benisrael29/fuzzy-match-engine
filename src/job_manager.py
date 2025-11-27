import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Any


class JobManager:
    """Manages job configurations storage and retrieval."""
    
    def __init__(self, jobs_dir: str = "jobs"):
        """
        Initialize JobManager.
        
        Args:
            jobs_dir: Directory to store job configurations
        """
        self.jobs_dir = jobs_dir
        os.makedirs(jobs_dir, exist_ok=True)
    
    def list_jobs(self) -> List[Dict[str, Any]]:
        """
        List all saved jobs.
        
        Returns:
            List of job metadata dictionaries
        """
        jobs = []
        
        if not os.path.exists(self.jobs_dir):
            return jobs
        
        for filename in os.listdir(self.jobs_dir):
            if filename.endswith('.json'):
                job_name = filename[:-5]
                try:
                    job = self.get_job(job_name)
                    jobs.append({
                        'name': job['name'],
                        'description': job.get('description', ''),
                        'created': job.get('created', ''),
                        'modified': job.get('modified', '')
                    })
                except Exception:
                    continue
        
        return sorted(jobs, key=lambda x: x.get('modified', ''), reverse=True)
    
    def get_job(self, name: str) -> Dict[str, Any]:
        """
        Load job configuration by name.
        
        Args:
            name: Job name
        
        Returns:
            Job dictionary with name, description, timestamps, and config
        
        Raises:
            FileNotFoundError: If job doesn't exist
        """
        filepath = os.path.join(self.jobs_dir, f"{name}.json")
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Job '{name}' not found")
        
        with open(filepath, 'r') as f:
            job = json.load(f)
        
        return job
    
    def save_job(
        self,
        name: str,
        description: str,
        config: Dict[str, Any],
        update_existing: bool = False
    ) -> bool:
        """
        Save job configuration.
        
        Args:
            name: Job name
            description: Job description
            config: Configuration dictionary
            update_existing: If True, update existing job; if False, raise error if exists
        
        Returns:
            True if saved successfully
        
        Raises:
            ValueError: If job exists and update_existing is False
        """
        filepath = os.path.join(self.jobs_dir, f"{name}.json")
        
        if os.path.exists(filepath) and not update_existing:
            raise ValueError(f"Job '{name}' already exists. Use update_existing=True to overwrite.")
        
        now = datetime.now().isoformat()
        
        if os.path.exists(filepath):
            existing_job = self.get_job(name)
            created = existing_job.get('created', now)
        else:
            created = now
        
        job = {
            'name': name,
            'description': description,
            'created': created,
            'modified': now,
            'config': config
        }
        
        with open(filepath, 'w') as f:
            json.dump(job, f, indent=2)
        
        return True
    
    def delete_job(self, name: str) -> bool:
        """
        Delete job configuration.
        
        Args:
            name: Job name
        
        Returns:
            True if deleted successfully
        
        Raises:
            FileNotFoundError: If job doesn't exist
        """
        filepath = os.path.join(self.jobs_dir, f"{name}.json")
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Job '{name}' not found")
        
        os.remove(filepath)
        return True
    
    def job_exists(self, name: str) -> bool:
        """
        Check if job exists.
        
        Args:
            name: Job name
        
        Returns:
            True if job exists, False otherwise
        """
        filepath = os.path.join(self.jobs_dir, f"{name}.json")
        return os.path.exists(filepath)

