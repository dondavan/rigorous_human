#!/bin/sh

set -eu

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

command_exists() {
	command -v "$1" >/dev/null 2>&1
}

require_sudo() {
	if [ "$(id -u)" -ne 0 ]; then
		sudo "$@"
	else
		"$@"
	fi
}

install_node_macos() {
	if ! command_exists brew; then
		echo "Homebrew is required on macOS to install Node.js automatically."
		echo "Install Homebrew from https://brew.sh and rerun this script."
		exit 1
	fi

	echo "Installing Node.js with Homebrew..."
	brew install node
}

install_node_linux() {
	if command_exists apt-get; then
		echo "Installing Node.js and npm with apt-get..."
		require_sudo apt-get update
		require_sudo apt-get install -y nodejs npm
		return
	fi

	if command_exists dnf; then
		echo "Installing Node.js and npm with dnf..."
		require_sudo dnf install -y nodejs npm
		return
	fi

	if command_exists yum; then
		echo "Installing Node.js and npm with yum..."
		require_sudo yum install -y nodejs npm
		return
	fi

	if command_exists pacman; then
		echo "Installing Node.js and npm with pacman..."
		require_sudo pacman -Sy --noconfirm nodejs npm
		return
	fi

	echo "Unsupported Linux package manager. Please install Node.js and npm manually."
	exit 1
}

ensure_node_toolchain() {
	if command_exists node && command_exists npm; then
		echo "Node.js and npm are already installed."
		return
	fi

	os_name=$(uname -s)

	case "$os_name" in
		Darwin)
			echo "Detected macOS."
			install_node_macos
			;;
		Linux)
			echo "Detected Linux."
			install_node_linux
			;;
		*)
			echo "Unsupported operating system: $os_name"
			exit 1
			;;
	esac
}

main() {
	cd "$SCRIPT_DIR"

	echo "Starting application..."
	npm start
}

main "$@"