#!/bin/sh

set -e

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
LOG_DIR="${SCRIPT_DIR}/logs"
PID_FILE="${SCRIPT_DIR}/.daemon.pid"
LOG_FILE="${LOG_DIR}/service.log"
HEALTH_CHECK_URL="http://127.0.0.1:3000/health"
HEALTH_CHECK_INTERVAL=10
RESTART_DELAY=2
MAX_RETRIES=3

mkdir -p "$LOG_DIR"

log_msg() {
  local timestamp
  timestamp=$(date '+%Y-%m-%d %H:%M:%S')
  echo "[$timestamp] $*" | tee -a "$LOG_FILE"
}

health_check() {
  if ! command -v curl >/dev/null 2>&1; then
    return 0
  fi

  if curl -sf "$HEALTH_CHECK_URL" >/dev/null 2>&1; then
    return 0
  else
    return 1
  fi
}

cleanup() {
  log_msg "Received shutdown signal, cleaning up..."
  if [ -f "$PID_FILE" ]; then
    kill "$(cat "$PID_FILE")" 2>/dev/null || true
    rm "$PID_FILE"
  fi
  exit 0
}

trap cleanup SIGTERM SIGINT

start_service() {
  cd "$SCRIPT_DIR"
  log_msg "Starting service..."
  npm start >> "$LOG_FILE" 2>&1 &
  local pid=$!
  echo "$pid" > "$PID_FILE"
  log_msg "Service started with PID $pid"
  return 0
}

monitor_service() {
  local consecutive_failures=0

  while true; do
    if ! kill -0 "$(cat "$PID_FILE" 2>/dev/null)" 2>/dev/null; then
      log_msg "Service process is dead, will restart..."
      consecutive_failures=$((consecutive_failures + 1))
    else
      if health_check; then
        consecutive_failures=0
      else
        consecutive_failures=$((consecutive_failures + 1))
        log_msg "Health check failed ($consecutive_failures/$MAX_RETRIES)"

        if [ "$consecutive_failures" -ge "$MAX_RETRIES" ]; then
          log_msg "Max retries reached, killing service for restart..."
          kill "$(cat "$PID_FILE")" 2>/dev/null || true
          consecutive_failures=0
        fi
      fi
    fi

    if ! kill -0 "$(cat "$PID_FILE" 2>/dev/null)" 2>/dev/null; then
      log_msg "Waiting ${RESTART_DELAY}s before restart..."
      sleep "$RESTART_DELAY"
      start_service
    fi

    sleep "$HEALTH_CHECK_INTERVAL"
  done
}

if [ -f "$PID_FILE" ]; then
  if kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    log_msg "Service already running with PID $(cat "$PID_FILE")"
    exit 0
  else
    rm "$PID_FILE"
  fi
fi

start_service
monitor_service
