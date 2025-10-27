#!/bin/bash

# AI Video Transformer - Startup Script
# This script sets up and starts the application

echo "========================================="
echo "AI Video Transformer - Starting Up"
echo "========================================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install/Update dependencies
echo "Installing dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

# Check for .env file
if [ ! -f ".env" ]; then
    echo ""
    echo "WARNING: .env file not found!"
    echo "Please copy .env.example to .env and add your API keys."
    echo "cp .env.example .env"
    echo ""
    exit 1
fi

# Create necessary directories
echo "Creating necessary directories..."
mkdir -p uploads jobs temp logs

# Start the Flask application
echo ""
echo "========================================="
echo "Starting Flask application on http://localhost:5001"
echo "Press Ctrl+C to stop the server"
echo "========================================="
echo ""

python src/api/app.py