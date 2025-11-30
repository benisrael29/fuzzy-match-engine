#!/bin/bash

set -e

echo "Starting Match Engine services..."
echo ""

# Function to check if a port is in use
check_port() {
    local port=$1
    if command -v lsof &> /dev/null; then
        lsof -ti:$port > /dev/null 2>&1
    elif command -v netstat &> /dev/null; then
        netstat -ano | grep ":$port" | grep LISTENING > /dev/null 2>&1
    else
        return 1
    fi
}

# Start Python backend on port 8000
echo "Starting Python backend on port 8000..."
if check_port 8000; then
    echo "  ⚠️  Port 8000 is already in use. Backend may already be running."
else
    if [ -f "start-backend.py" ]; then
        if command -v python3 &> /dev/null; then
            python3 start-backend.py > backend.log 2>&1 &
            BACKEND_PID=$!
            echo "  ✓ Backend started (PID: $BACKEND_PID)"
            echo "  ✓ Logs: backend.log"
        elif command -v python &> /dev/null; then
            python start-backend.py > backend.log 2>&1 &
            BACKEND_PID=$!
            echo "  ✓ Backend started (PID: $BACKEND_PID)"
            echo "  ✓ Logs: backend.log"
        else
            echo "  ✗ Python not found. Please install Python 3."
            exit 1
        fi
    else
        echo "  ✗ start-backend.py not found in current directory."
        exit 1
    fi
fi

# Wait a moment for backend to start
sleep 2

# Start Next.js frontend on port 3000
echo ""
echo "Starting Next.js frontend on port 3000..."
if check_port 3000; then
    echo "  ⚠️  Port 3000 is already in use. Frontend may already be running."
else
    if [ -d "frontend" ]; then
        cd frontend
        
        # Check if node_modules exists, if not, suggest npm install
        if [ ! -d "node_modules" ]; then
            echo "  ⚠️  node_modules not found. Running npm install..."
            if command -v npm &> /dev/null; then
                npm install
            else
                echo "  ✗ npm not found. Please install Node.js and npm."
                cd ..
                exit 1
            fi
        fi
        
        if command -v npm &> /dev/null; then
            npm run dev > ../frontend.log 2>&1 &
            FRONTEND_PID=$!
            cd ..
            echo "  ✓ Frontend started (PID: $FRONTEND_PID)"
            echo "  ✓ Logs: frontend.log"
        else
            echo "  ✗ npm not found. Please install Node.js and npm."
            cd ..
            exit 1
        fi
    else
        echo "  ✗ frontend directory not found."
        exit 1
    fi
fi

echo ""
echo "=========================================="
echo "Services started successfully!"
echo "=========================================="
echo ""
echo "Backend API:  http://localhost:8000"
echo "  API Docs:   http://localhost:8000/docs"
echo ""
echo "Frontend UI:  http://localhost:3000"
echo ""
echo "Logs:"
echo "  Backend:    backend.log"
echo "  Frontend:   frontend.log"
echo ""
echo "To stop services, run: ./stop-services.sh"
echo ""

