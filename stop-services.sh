#!/bin/bash

set -e

echo "Stopping all Match Engine services..."

# Stop standalone Python backend (uvicorn) processes on port 8000
echo "Checking for Python backend processes on port 8000..."
if command -v lsof &> /dev/null; then
    # Unix-like systems (including Git Bash on Windows)
    BACKEND_PIDS=$(lsof -ti:8000 2>/dev/null || true)
    if [ -n "$BACKEND_PIDS" ]; then
        echo "Stopping Python backend processes..."
        echo "$BACKEND_PIDS" | xargs kill -9 2>/dev/null || true
        echo "Python backend stopped."
    fi
elif command -v netstat &> /dev/null; then
    # Windows (Git Bash)
    BACKEND_PIDS=$(netstat -ano | grep :8000 | grep LISTENING | awk '{print $5}' | sort -u)
    if [ -n "$BACKEND_PIDS" ]; then
        echo "Stopping Python backend processes..."
        for pid in $BACKEND_PIDS; do
            taskkill //F //PID "$pid" 2>/dev/null || true
        done
        echo "Python backend stopped."
    fi
fi

# Stop standalone Next.js frontend processes on port 3000
echo "Checking for Next.js frontend processes on port 3000..."
if command -v lsof &> /dev/null; then
    FRONTEND_PIDS=$(lsof -ti:3000 2>/dev/null || true)
    if [ -n "$FRONTEND_PIDS" ]; then
        echo "Stopping Next.js frontend processes..."
        echo "$FRONTEND_PIDS" | xargs kill -9 2>/dev/null || true
        echo "Next.js frontend stopped."
    fi
elif command -v netstat &> /dev/null; then
    FRONTEND_PIDS=$(netstat -ano | grep :3000 | grep LISTENING | awk '{print $5}' | sort -u)
    if [ -n "$FRONTEND_PIDS" ]; then
        echo "Stopping Next.js frontend processes..."
        for pid in $FRONTEND_PIDS; do
            taskkill //F //PID "$pid" 2>/dev/null || true
        done
        echo "Next.js frontend stopped."
    fi
fi

# Stop standalone MySQL processes on port 3306 (if not in Docker)
echo "Checking for standalone MySQL processes on port 3306..."
if command -v lsof &> /dev/null; then
    MYSQL_PIDS=$(lsof -ti:3306 2>/dev/null || true)
    if [ -n "$MYSQL_PIDS" ]; then
        # Check if it's a Docker container first
        if ! docker ps --format '{{.Names}}' 2>/dev/null | grep -q match-engine-mysql; then
            echo "Stopping standalone MySQL processes..."
            echo "$MYSQL_PIDS" | xargs kill -9 2>/dev/null || true
            echo "Standalone MySQL stopped."
        fi
    fi
elif command -v netstat &> /dev/null; then
    MYSQL_PIDS=$(netstat -ano | grep :3306 | grep LISTENING | awk '{print $5}' | sort -u)
    if [ -n "$MYSQL_PIDS" ]; then
        # Check if it's a Docker container first
        if ! docker ps --format '{{.Names}}' 2>/dev/null | grep -q match-engine-mysql; then
            echo "Stopping standalone MySQL processes..."
            for pid in $MYSQL_PIDS; do
                taskkill //F //PID "$pid" 2>/dev/null || true
            done
            echo "Standalone MySQL stopped."
        fi
    fi
fi

echo ""
echo "All services stopped successfully!"

