#!/usr/bin/env python3
"""
Start the FastAPI web service for the Fuzzy Matching Engine.
"""
import uvicorn
from src.web_service import app

if __name__ == "__main__":
    print("Starting Fuzzy Matching Engine API server...")
    print("API will be available at http://localhost:8000")
    print("API documentation at http://localhost:8000/docs")
    print("\nPress Ctrl+C to stop the server\n")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )

