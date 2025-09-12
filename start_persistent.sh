#!/bin/bash
# Script to start persistent model server and webapp

echo "🚀 Starting InternVideo2.5 with Persistent Model..."

# Kill any existing processes
echo "🔄 Cleaning up existing processes..."
pkill -f "python.*model_server.py" 2>/dev/null || true
pkill -f "python.*app.py" 2>/dev/null || true
sleep 2

# Start model server in background
echo "🤖 Starting model server on port 8089..."
cd /workspace/surveillance
python model_server.py --host localhost --port 8089 &
MODEL_PID=$!
echo "Model server PID: $MODEL_PID"

# Wait for model server to start
echo "⏳ Waiting for model server to initialize..."
sleep 20

# Test model server connection
echo "🔍 Testing model server connection..."
python -c "
import sys
sys.path.insert(0, '.')
from model_server import ModelClient
import time

client = ModelClient(host='localhost', port=8089)
for i in range(30):  # Try for 30 seconds
    try:
        result = client.ping()
        if result and result.get('status') == 'ok':
            print('✅ Model server is ready!')
            break
    except:
        pass
    time.sleep(1)
else:
    print('❌ Model server failed to start')
    exit(1)
"

if [ $? -ne 0 ]; then
    echo "❌ Model server failed to start. Check logs."
    kill $MODEL_PID 2>/dev/null || true
    exit 1
fi

# Start webapp with model server enabled
echo "🌐 Starting webapp with model server connection..."
USE_MODEL_SERVER=true python webapp/app.py &
WEBAPP_PID=$!
echo "Webapp PID: $WEBAPP_PID"

echo ""
echo "🎉 Persistent model setup complete!"
echo "📍 Webapp: http://localhost:8088"
echo "📍 Model Server: localhost:8089"
echo ""
echo "📊 To check model status: curl http://localhost:8088/api/model-status"
echo ""
echo "Press Ctrl+C to stop all services"

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Stopping services..."
    kill $MODEL_PID 2>/dev/null || true
    kill $WEBAPP_PID 2>/dev/null || true
    echo "✅ All services stopped"
    exit 0
}

# Set trap for cleanup
trap cleanup SIGINT SIGTERM

# Wait for processes
wait