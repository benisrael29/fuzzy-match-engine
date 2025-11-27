from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import threading
import io
import sys
from datetime import datetime
from .job_manager import JobManager
from .job_runner import JobRunner

app = FastAPI(title="Fuzzy Matching Engine API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

job_manager = JobManager()
job_runner = JobRunner()

job_statuses: Dict[str, Dict[str, Any]] = {}
job_outputs: Dict[str, str] = {}
status_lock = threading.Lock()


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


@app.get("/")
async def root():
    return {
        "name": "Fuzzy Matching Engine API",
        "version": "1.0.0",
        "endpoints": {
            "jobs": "/api/jobs",
            "job_detail": "/api/jobs/{name}",
            "run_job": "/api/jobs/{name}/run",
            "job_status": "/api/jobs/{name}/status"
        }
    }


@app.get("/api/jobs", response_model=List[JobListResponse])
async def list_jobs():
    """List all jobs."""
    try:
        jobs = job_manager.list_jobs()
        return jobs
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing jobs: {str(e)}")


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
        job_manager.delete_job(name)
        
        with status_lock:
            if name in job_statuses:
                del job_statuses[name]
            if name in job_outputs:
                del job_outputs[name]
        
        return {"message": f"Job '{name}' deleted successfully"}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Job '{name}' not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting job: {str(e)}")


def run_job_background(job_name: str, config: Dict[str, Any]):
    """Run job in background thread and capture output."""
    with status_lock:
        job_statuses[job_name] = {
            "status": "running",
            "message": "Job execution started",
            "started_at": datetime.now().isoformat()
        }
        job_outputs[job_name] = ""
    
    old_stdout = sys.stdout
    sys.stdout = buffer = io.StringIO()
    
    try:
        success = job_runner.run_job(config, job_name)
        output = buffer.getvalue()
        
        with status_lock:
            job_statuses[job_name] = {
                "status": "completed" if success else "failed",
                "message": "Job completed successfully" if success else "Job execution failed",
                "completed_at": datetime.now().isoformat(),
                "success": success
            }
            job_outputs[job_name] = output
    except Exception as e:
        output = buffer.getvalue() + f"\nError: {str(e)}"
        with status_lock:
            job_statuses[job_name] = {
                "status": "failed",
                "message": f"Job execution error: {str(e)}",
                "completed_at": datetime.now().isoformat(),
                "success": False
            }
            job_outputs[job_name] = output
    finally:
        sys.stdout = old_stdout


@app.post("/api/jobs/{name}/run")
async def run_job(name: str, background_tasks: BackgroundTasks):
    """Execute a job."""
    try:
        job = job_manager.get_job(name)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Job '{name}' not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading job: {str(e)}")
    
    with status_lock:
        if job_statuses.get(name, {}).get("status") == "running":
            raise HTTPException(status_code=400, detail=f"Job '{name}' is already running")
    
    background_tasks.add_task(run_job_background, name, job['config'])
    
    return {
        "message": f"Job '{name}' execution started",
        "status": "running"
    }


@app.get("/api/jobs/{name}/status", response_model=JobStatusResponse)
async def get_job_status(name: str):
    """Get job execution status."""
    with status_lock:
        status_info = job_statuses.get(name)
        output = job_outputs.get(name)
    
    if status_info is None:
        raise HTTPException(status_code=404, detail=f"No status found for job '{name}'. Job may not have been run yet.")
    
    return {
        "status": status_info["status"],
        "message": status_info.get("message"),
        "output": output
    }

