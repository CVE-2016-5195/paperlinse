#!/bin/bash
# Paperlinse startup script

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install backend dependencies
echo "Installing backend dependencies..."
pip install -q -r backend/requirements.txt

# Build frontend if needed
if [ ! -d "backend/frontend/dist" ]; then
    echo "Building frontend..."
    cd frontend
    npm install
    npm run build
    cd ..
    
    # Copy built frontend to backend
    mkdir -p backend/frontend
    cp -r frontend/dist backend/frontend/
fi

# Start the server
echo "Starting Paperlinse server on http://localhost:8000"
cd backend
python main.py
