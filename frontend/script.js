
const timerDisplay = document.getElementById('timer');
const statusDisplay = document.getElementById('status');

let startTime = 0;
let timerInterval;
let lastServerStatus = "connecting...";

// timer logic
function startTimer() {
    if (timerInterval) return;
    startTime = Date.now();
    timerInterval = requestAnimationFrame(updateTimer);
}

function stopTimer(finalTime) {
    if (timerInterval) {
        cancelAnimationFrame(timerInterval);
        timerInterval = null;
    }
    timerDisplay.textContent = finalTime;
}

function updateTimer() {
    const elapsedTime = (Date.now() - startTime) / 1000;
    timerDisplay.textContent = elapsedTime.toFixed(2);
    timerInterval = requestAnimationFrame(updateTimer);
}

// server status polling
async function getStatus() {
    try {
        const response = await fetch('/status');
        if (!response.ok) throw new Error(`http error! status: ${response.status}`);
        
        const data = await response.json();
        
        if (data.status !== lastServerStatus) {
            statusDisplay.textContent = data.status;
            lastServerStatus = data.status;

            if (data.status === "SOLVING") {
                startTimer();
            } else {
                stopTimer(data.time);
            }
        }
    } catch (error) {
        console.error("could not fetch status:", error);
        statusDisplay.textContent = "disconnected";
        stopTimer("0.00");
    }
}

setInterval(getStatus, 100);
