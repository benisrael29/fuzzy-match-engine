from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
from datetime import datetime
from .job_manager import JobManager
from .job_queue import JobQueue
from .job_worker_pool import create_worker_pool, JobWorkerPool

job_manager = JobManager()
job_queue: Optional[JobQueue] = None
worker_pool: Optional[JobWorkerPool] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan - startup and shutdown."""
    global job_queue, worker_pool
    
    # Startup: Initialize queue and worker pool
    try:
        job_queue = JobQueue()
        worker_pool = create_worker_pool(job_queue)
        worker_pool.start()
    except Exception as e:
        print(f"Error initializing job queue: {e}")
        import traceback
        traceback.print_exc()
        # Continue without queue - endpoints will return 503
        job_queue = None
        worker_pool = None
    
    yield
    
    # Shutdown: Stop worker pool gracefully
    if worker_pool:
        try:
            worker_pool.stop()
        except Exception as e:
            print(f"Error stopping worker pool: {e}")


app = FastAPI(
    title="Fuzzy Matching Engine API",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class JobCreateRequest(BaseModel):
    name: str
    description: Optional[str] = ""
    config: Dict[str, Any]


class JobUpdateRequest(BaseModel):
    description: Optional[str] = None
    config: Optional[Dict[str, Any]] = None


class JobResponse(BaseModel):
    name: str
    description: str
    created: str
    modified: str
    config: Dict[str, Any]


class JobListResponse(BaseModel):
    name: str
    description: str
    created: str
    modified: str


class JobStatusResponse(BaseModel):
    status: str
    message: Optional[str] = None
    output: Optional[str] = None
    queue_position: Optional[int] = None
    priority: Optional[str] = None
    queued_at: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None


class QueueJobRequest(BaseModel):
    priority: Optional[str] = "medium"


class QueueResponse(BaseModel):
    job_name: str
    status: str
    priority: str
    queue_position: Optional[int]
    queued_at: str


class SearchRequest(BaseModel):
    master: str
    query: Dict[str, Any]
    threshold: Optional[float] = None
    max_results: Optional[int] = None
    config: Optional[str] = None
    mysql_credentials: Optional[Dict[str, str]] = None
    s3_credentials: Optional[Dict[str, str]] = None


@app.get("/")
async def root():
    return {
        "name": "Fuzzy Matching Engine API",
        "version": "1.0.0",
        "endpoints": {
            "jobs": "/api/jobs",
            "job_detail": "/api/jobs/{name}",
            "run_job": "/api/jobs/{name}/run",
            "job_status": "/api/jobs/{name}/status",
            "queue": "/api/jobs/queue",
            "cancel_job": "/api/jobs/{name}/cancel",
            "queue_status": "/api/jobs/{name}/queue-status",
            "search": "/api/search"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "queue_initialized": job_queue is not None,
        "worker_pool_running": worker_pool is not None and worker_pool.running if worker_pool else False
    }


@app.get("/api/jobs", response_model=List[JobListResponse])
async def list_jobs():
    """List all jobs."""
    try:
        jobs = job_manager.list_jobs()
        return jobs
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing jobs: {str(e)}")


@app.get("/api/jobs/queue", response_model=List[QueueResponse])
async def list_queue():
    """List all jobs in the queue."""
    if not job_queue:
        raise HTTPException(status_code=503, detail="Job queue not initialized")
    
    queued_jobs = job_queue.list_queue()
    active_jobs = job_queue.list_active()
    
    result = []
    
    # Add queued jobs
    for job in queued_jobs:
        result.append({
            "job_name": job["job_name"],
            "status": job["status"],
            "priority": job["priority"],
            "queue_position": job["queue_position"],
            "queued_at": job["queued_at"]
        })
    
    # Add active jobs (no queue position)
    for job in active_jobs:
        result.append({
            "job_name": job["job_name"],
            "status": job["status"],
            "priority": job["priority"],
            "queue_position": None,
            "queued_at": job["queued_at"]
        })
    
    return result


@app.get("/api/jobs/{name}", response_model=JobResponse)
async def get_job(name: str):
    """Get job details by name."""
    try:
        job = job_manager.get_job(name)
        return job
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Job '{name}' not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting job: {str(e)}")


@app.post("/api/jobs", response_model=JobResponse)
async def create_job(request: JobCreateRequest):
    """Create a new job."""
    try:
        if not request.name.strip():
            raise HTTPException(status_code=400, detail="Job name cannot be empty")
        
        update_existing = job_manager.job_exists(request.name)
        
        job_manager.save_job(
            request.name,
            request.description or "",
            request.config,
            update_existing=update_existing
        )
        
        job = job_manager.get_job(request.name)
        return job
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating job: {str(e)}")


@app.put("/api/jobs/{name}", response_model=JobResponse)
async def update_job(name: str, request: JobUpdateRequest):
    """Update an existing job."""
    try:
        existing_job = job_manager.get_job(name)
        
        description = request.description if request.description is not None else existing_job.get('description', '')
        config = request.config if request.config is not None else existing_job['config']
        
        job_manager.save_job(
            name,
            description,
            config,
            update_existing=True
        )
        
        job = job_manager.get_job(name)
        return job
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Job '{name}' not found")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating job: {str(e)}")


@app.delete("/api/jobs/{name}")
async def delete_job(name: str):
    """Delete a job."""
    try:
        # Cancel job if it's queued or running
        if job_queue:
            try:
                job_queue.cancel(name)
            except ValueError:
                pass  # Job not in queue, continue with deletion
        
        job_manager.delete_job(name)
        
        return {"message": f"Job '{name}' deleted successfully"}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Job '{name}' not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting job: {str(e)}")


@app.post("/api/jobs/{name}/run")
async def run_job(name: str, request: Optional[QueueJobRequest] = None):
    """Queue a job for execution."""
    if not job_queue:
        raise HTTPException(status_code=503, detail="Job queue not initialized")
    
    try:
        job = job_manager.get_job(name)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Job '{name}' not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading job: {str(e)}")
    
    # Check if job is already queued or running
    status = job_queue.get_status(name)
    if status:
        current_status = status.get("status")
        if current_status in ["queued", "running"]:
            raise HTTPException(
                status_code=400,
                detail=f"Job '{name}' is already {current_status}"
            )
    
    # Get priority from request or default to medium
    priority = "medium"
    if request and request.priority:
        priority = request.priority.lower()
        if priority not in ["high", "medium", "low"]:
            raise HTTPException(
                status_code=400,
                detail="Priority must be 'high', 'medium', or 'low'"
            )
    
    try:
        job_queue.enqueue(name, job['config'], priority=priority)
        queue_position = job_queue.get_queue_position(name)
        
        return {
            "message": f"Job '{name}' queued successfully",
            "status": "queued",
            "priority": priority,
            "queue_position": queue_position
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error queuing job: {str(e)}")


@app.get("/api/jobs/{name}/status", response_model=JobStatusResponse)
async def get_job_status(name: str):
    """Get job execution status."""
    if not job_queue:
        raise HTTPException(status_code=503, detail="Job queue not initialized")
    
    status_info = job_queue.get_status(name)
    
    if status_info is None:
        raise HTTPException(
            status_code=404,
            detail=f"No status found for job '{name}'. Job may not have been run yet."
        )
    
    # Get output from worker pool if running, or from status if completed
    output = ""
    if status_info.get("status") == "running" and worker_pool:
        output = worker_pool.get_output(name)
    else:
        output = status_info.get("output", "")
    
    # Get queue position if queued
    queue_position = None
    if status_info.get("status") == "queued":
        queue_position = job_queue.get_queue_position(name)
    
    return {
        "status": status_info.get("status", "unknown"),
        "message": status_info.get("message"),
        "output": output,
        "queue_position": queue_position,
        "priority": status_info.get("priority"),
        "queued_at": status_info.get("queued_at"),
        "started_at": status_info.get("started_at"),
        "completed_at": status_info.get("completed_at")
    }


@app.post("/api/jobs/{name}/cancel")
async def cancel_job(name: str):
    """Cancel a queued or running job."""
    if not job_queue:
        raise HTTPException(status_code=503, detail="Job queue not initialized")
    
    try:
        job_queue.cancel(name)
        return {
            "message": f"Job '{name}' cancellation requested",
            "status": "cancelling"
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error cancelling job: {str(e)}")


@app.get("/api/jobs/{name}/queue-status")
async def get_queue_status(name: str):
    """Get queue status for a specific job."""
    if not job_queue:
        raise HTTPException(status_code=503, detail="Job queue not initialized")
    
    status_info = job_queue.get_status(name)
    
    if status_info is None:
        raise HTTPException(
            status_code=404,
            detail=f"Job '{name}' not found in queue"
        )
    
    queue_position = None
    if status_info.get("status") == "queued":
        queue_position = job_queue.get_queue_position(name)
    
    return {
        "job_name": name,
        "status": status_info.get("status"),
        "priority": status_info.get("priority"),
        "queue_position": queue_position,
        "queued_at": status_info.get("queued_at"),
        "started_at": status_info.get("started_at"),
        "message": status_info.get("message")
    }


@app.post("/api/search")
async def search(request: SearchRequest):
    """
    Search for matching records in the master dataset.
    
    Args:
        request: Search request with master dataset path, query record, and optional parameters
    
    Returns:
        List of matching records with scores
    """
    try:
        from src.matcher import FuzzyMatcher
        from src.config_validator import validate_config
        
        if request.config:
            try:
                job = job_manager.get_job(request.config)
                config = job['config']
            except FileNotFoundError:
                config = validate_config(request.config)
            
            if config.get('mode') != 'search':
                config['mode'] = 'search'
            if 'source2' not in config:
                config['source2'] = request.master
            if request.threshold is not None:
                if 'match_config' not in config:
                    config['match_config'] = {}
                config['match_config']['threshold'] = request.threshold
                config['match_config']['return_all_matches'] = True
            
            if request.mysql_credentials:
                config['mysql_credentials'] = request.mysql_credentials
            if request.s3_credentials:
                config['s3_credentials'] = request.s3_credentials
            
            matcher = FuzzyMatcher(config)
        else:
            threshold = request.threshold if request.threshold is not None else 0.85
            matcher = FuzzyMatcher.create_search_matcher(
                master_source=request.master,
                query_record=request.query,
                threshold=threshold,
                mysql_credentials=request.mysql_credentials,
                s3_credentials=request.s3_credentials
            )
        
        results = matcher.search(
            query_record=request.query,
            threshold=request.threshold,
            max_results=request.max_results
        )
        
        return {
            "matches": results,
            "count": len(results),
            "master_dataset": request.master
        }
        
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"File not found: {str(e)}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")

