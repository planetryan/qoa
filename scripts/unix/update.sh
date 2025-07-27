#!/bin/bash

# QOA Unix Update Script
# This script updates the Rust toolchain and all installed Cargo crates.
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

# function to display system information
show_system_info() {
    local os=$(detect_os)
    echo "=== build environment information ==="
    echo "operating system: $os"
    echo "architecture: $(uname -m 2>/dev/null || echo "unknown")"
    
    if command -v rustc >/dev/null 2>&1; then
        echo "rust version: $(rustc --version)"
    fi
    
    if command -v cargo >/dev/null 2>&1; then
        echo "cargo version: $(cargo --version)"
    fi
    
    echo "build date: $(date)"
    echo "=================================="
    echo ""
}

# function to check available disk space (if possible)
check_disk_space() {
    if command -v df >/dev/null 2>&1; then
        local available
        case $(detect_os) in
            "macos"|"freebsd"|"openbsd"|"netbsd")
                available=$(df -h . | tail -1 | awk '{print $4}')
                ;;
            *)
                available=$(df -h . | tail -1 | awk '{print $4}')
                ;;
        esac
        echo "available disk space: $available"
    fi
}

# function to update rust toolchain
update_rust_toolchain() {
    echo -e "\033[1;33m--- updating rust toolchain (rustup update) --- \033[0m"
    echo ""
    if rustup update; then
        echo -e "\033[1;32mrust toolchain updated successfully.\033[0m"
    else
        echo -e "\033[1;31merror: failed to update rust toolchain.\033[0m"
        exit 1
    fi
    echo ""
}

# function to update cargo crates
update_cargo_crates() {
    echo -e "\033[1;33m--- updating cargo crates (cargo update) --- \033[0m"
    echo ""
    if cargo update; then
        echo -e "\033[1;32mcargo crates updated successfully.\033[0m"
    else
        echo -e "\033[1;31merror: failed to update cargo crates.\033[0m"
        exit 1
    fi
    echo ""
}

# main execution function
main() {
    echo "=== qoa unix update script ==="
    echo ""
    
    # check prerequisites
    check_cargo
    
    # show system information
    show_system_info
    
    # check disk space
    check_disk_space
    echo ""
    
    # update rust toolchain
    update_rust_toolchain
    
    # update cargo crates
    update_cargo_crates
    
    echo "=== update script completed ==="
}

# run main function
main
