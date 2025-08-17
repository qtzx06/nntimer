#!/bin/bash

PID_FILE="server.pid"
LOG_FILE="nntimer.log"
VENV_PATH=".venv/bin/activate"

if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    # check if the process is still running
    if ps -p $PID > /dev/null; then
        echo "Server is already running with PID $PID."
        exit 1
    else
        echo "Warning: Stale PID file found. Removing it."
        rm "$PID_FILE"
    fi
fi

# activate virtual environment
source "$VENV_PATH"
if [ $? -ne 0 ]; then
    echo "Error: Could not activate virtual environment. Make sure it exists at '$VENV_PATH'."
    exit 1
fi

echo "Starting nntimer server in the background..."
# start the server, redirecting stdout/stderr to a log file
uvicorn main:app --host 0.0.0.0 --port 8000 > "$LOG_FILE" 2>&1 &

# save the PID of the background process
echo $! > "$PID_FILE"

# deactivate the virtual environment
deactivate

echo "Server started with PID $(cat "$PID_FILE")."
echo "Logs are being written to '$LOG_FILE'."