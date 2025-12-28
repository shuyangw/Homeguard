#!/bin/bash

# Function to handle cleanup on exit
cleanup() {
    echo "Stopping servers..."
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    exit
}

# Trap signals
trap cleanup SIGINT SIGTERM EXIT

# Get project root (one level up from scripts)
PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
cd "$PROJECT_ROOT"

echo "Starting Backend on port 8000..."
# Start Backend in background
python3 -m uvicorn src.web.backend.main:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!

echo "Starting Frontend on port 5173..."
# Start Frontend in background
cd src/web/frontend
npm run dev &
FRONTEND_PID=$!

echo "Homeguard Web UI is running!"
echo "Backend: http://localhost:8000/docs"
echo "Frontend: http://localhost:5173"
echo "Press Ctrl+C to stop."

# Wait for both processes
wait
