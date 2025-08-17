#!/bin/bash

PID_FILE="server.pid"

if [ ! -f "$PID_FILE" ]; then
    echo "Server is not running (PID file not found)."
    exit 0
fi

PID=$(cat "$PID_FILE")

# check if a process with that PID is running
if ! ps -p $PID > /dev/null; then
    echo "Server is not running (process with PID $PID not found)."
    echo "Removing stale PID file."
    rm "$PID_FILE"
    exit 0
fi

echo "Stopping nntimer server with PID $PID..."
kill $PID

# wait a moment and check if the process is gone
sleep 1
if ps -p $PID > /dev/null; then
    echo "Process $PID did not stop gracefully. Sending SIGKILL..."
    kill -9 $PID
fi

rm "$PID_FILE"
echo "Server stopped."