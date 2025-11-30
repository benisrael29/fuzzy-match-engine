# Setup Guide - Fuzzy Matching Engine UI

## Quick Start

### 1. Start the Backend API

In the project root directory:

```bash
python start-backend.py
```

Or using uvicorn directly:

```bash
uvicorn src.web_service:app --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`
- API Documentation: http://localhost:8000/docs
- API Root: http://localhost:8000/

### 2. Start the Frontend

In the `frontend` directory:

```bash
cd frontend
npm install  # First time only
npm run dev
```

The UI will be available at `http://localhost:3000`

### 3. Verify Connection

The UI shows a connection status indicator in the header:
- **Connected** (green) - Backend is reachable
- **Disconnected** (red) - Backend is not reachable

## Features

### Job Management
- **Create Jobs**: Use templates or create custom configurations
- **Edit Jobs**: Update job configurations with real-time JSON validation
- **Run Jobs**: Execute matching jobs and monitor progress
- **View Status**: Real-time status updates with output logs
- **Delete Jobs**: Remove jobs you no longer need

### Search
- Search for matching records in master datasets
- Configurable threshold and max results
- Support for CSV, MySQL, and S3 sources

### Configuration Templates
The UI includes pre-configured templates:
- **Minimal**: Basic auto-detection
- **With Column Mapping**: Custom column mappings with weights
- **Clustering**: Find duplicates within a dataset
- **MySQL Sources**: Match between MySQL tables
- **S3 Sources**: Match between S3 files

## Troubleshooting

### Backend Not Connecting
1. Ensure the backend is running on port 8000
2. Check that CORS is enabled (it should be by default)
3. Verify the API URL in `.env.local` matches your backend URL

### Frontend Build Issues
1. Run `npm install` in the frontend directory
2. Clear `.next` cache: `rm -rf .next` (or `rmdir /s .next` on Windows)
3. Rebuild: `npm run build`

### API Errors
- Check the backend logs for detailed error messages
- Verify your configuration JSON is valid
- Ensure data files exist at the specified paths

