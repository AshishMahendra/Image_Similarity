#!/bin/bash
echo "Starting Image Similarity Platform..."
echo "Backend will be available at: http://127.0.0.1:8000"
echo "Frontend will be available at: http://localhost:8501"
echo ""

# Start backend in background
cd backend
python main.py &
BACKEND_PID=$!

# Wait a moment for backend to start
sleep 3

# Start frontend
cd ../frontend
streamlit run app.py &
FRONTEND_PID=$!

# Wait for both processes
wait $BACKEND_PID $FRONTEND_PID