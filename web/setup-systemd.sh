#!/bin/sh

set -e

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
SERVICE_NAME="rigorous-human"
SERVICE_FILE="${SCRIPT_DIR}/${SERVICE_NAME}.service"
SYSTEMD_DIR="/etc/systemd/system"

log_msg() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >&2
}

error_msg() {
  echo "ERROR: $*" >&2
  exit 1
}

require_sudo() {
  if [ "$(id -u)" -ne 0 ]; then
    error_msg "This script requires root privileges. Please run with sudo."
  fi
}

check_service_file() {
  if [ ! -f "$SERVICE_FILE" ]; then
    error_msg "Service file not found: $SERVICE_FILE"
  fi
  log_msg "Found service file: $SERVICE_FILE"
}

install_service() {
  log_msg "Installing systemd service..."
  cp rigorous-human.service /etc/systemd/system/
  log_msg "Service file installed to ${SYSTEMD_DIR}/${SERVICE_NAME}.service"
}

enable_and_start_service() {
  log_msg "Reloading systemd daemon..."
  systemctl daemon-reload
  
  log_msg "Enabling ${SERVICE_NAME}..."
  systemctl enable "${SERVICE_NAME}"
  
  log_msg "Starting ${SERVICE_NAME}..."
  systemctl start "${SERVICE_NAME}"
  
  log_msg "Service started successfully!"
}

show_status() {
  log_msg "Current service status:"
  systemctl status "${SERVICE_NAME}"
  
  log_msg ""
  log_msg "Useful commands:"
  log_msg "  Check status:     sudo systemctl status ${SERVICE_NAME}"
  log_msg "  View logs:        sudo journalctl -u ${SERVICE_NAME} -f"
  log_msg "  Stop service:     sudo systemctl stop ${SERVICE_NAME}"
  log_msg "  Restart service:  sudo systemctl restart ${SERVICE_NAME}"
  log_msg "  Disable service:  sudo systemctl disable ${SERVICE_NAME}"
}

main() {
  log_msg "Automating systemd service setup for ${SERVICE_NAME}..."
  log_msg ""
  
  require_sudo
  check_service_file
  
  local temp_file
  temp_file=$(update_service_file)
  
  install_service "$temp_file"
  enable_and_start_service
  
  log_msg ""
  show_status
}

main "$@"
