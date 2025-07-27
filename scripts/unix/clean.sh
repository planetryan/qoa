#!/bin/bash

# QOA Unix Clean Script
# This script removes compiled build artifacts for the QOA project.
# Works on Linux, macOS, FreeBSD, OpenBSD, NetBSD, Solaris, and Windows (WSL/Cygwin/MSYS)

set -e  # exit immediately if a command exits with a non-zero status

# function to detect the operating system
detect_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        echo "linux"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        echo "macos"
    elif [[ "$OSTYPE" == "freebsd"* ]]; then
        echo "freebsd"
    elif [[ "$OSTYPE" == "openbsd"* ]]; then
        echo "openbsd"
    elif [[ "$OSTYPE" == "netbsd"* ]]; then
        echo "netbsd"
    elif [[ "$OSTYPE" == "solaris"* ]]; then
        echo "solaris"
    elif [[ "$OSTYPE" == "cygwin" ]] || [[ "$OSTYPE" == "msys" ]]; then
        echo "windows"
    else
        echo "unknown"
    fi
}

# function to check if cargo is installed
check_cargo() {
    if ! command -v cargo >/dev/null 2>&1; then
        echo -e "\033[1;31merror: cargo is not installed or not in path.\033[0m"
        echo "please install rust and cargo from: https://rustup.rs/"
        exit 1
    fi
}

# function to clean previous builds
clean_build() {
    echo -e "\033[1;33m--- removing all compiled builds --- \033[0m"
    echo ""

    # try cargo clean first
    if cargo clean; then
        echo -e "\033[1;32mcargo clean completed successfully.\033[0m"
    else
        echo -e "\033[1;31mwarning: cargo clean failed. attempting to remove 'target' directory manually.\033[0m"
        # if cargo clean fails, try to remove the target directory directly
        if rm -rf target; then
            echo -e "\033[1;32m'target' directory removed successfully.\033[0m"
        else
            echo -e "\033[1;31merror: failed to remove 'target' directory. please check permissions or remove it manually.\033[0m"
            exit 1
        fi
    fi
    echo ""
}

# main execution function
main() {
    echo "=== QOA unix clean script ==="
    echo ""
    
    # check prerequisites
    check_cargo
    
    # perform the clean operation
    clean_build
    
    echo "=== clean script completed ==="
}

# run main function
main
