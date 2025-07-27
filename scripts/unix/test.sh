#!/bin/bash

# QOA Unix Test Script
# This script is specifically configured to run tests for QOA.
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

# function to install intel mkl full if needed
install_mkl_if_needed() {
    local os=$(detect_os)
    
    echo "=== intel mkl full installation check ==="
    echo "detected os: $os"
    echo "checking for intel mkl full..."
    
    # check if mkl is already installed
    if pkg-config --exists mkl 2>/dev/null || [ -d "/opt/intel/mkl" ] || [ -d "/usr/include/mkl" ]; then
        echo "intel mkl appears to be already installed. skipping installation."
        echo ""
        return 0
    fi
    
    echo "intel mkl full not found. proceeding with installation..."
    echo ""
    
    case $os in
        "linux")
            # check if we have apt (debian/ubuntu)
            if command -v apt >/dev/null 2>&1; then
                echo "installing intel mkl full using apt..."
                sudo apt update && sudo apt install -y intel-mkl-full libmkl-dev libmkl-rt intel-oneapi-mkl
            # check if we have yum (rhel/centos/fedora older)
            elif command -v yum >/dev/null 2>&1; then
                echo "installing intel mkl full using yum..."
                sudo yum install -y intel-mkl-full intel-mkl-devel intel-oneapi-mkl
            # check if we have dnf (fedora newer)
            elif command -v dnf >/dev/null 2>&1; then
                echo "installing intel mkl full using dnf..."
                sudo dnf install -y intel-mkl-full intel-mkl-devel intel-oneapi-mkl
            # check if we have pacman (arch linux)
            elif command -v pacman >/dev/null 2>&1; then
                echo "installing intel mkl full using pacman..."
                sudo pacman -S --noconfirm intel-mkl intel-mkl-static
            # check if we have zypper (opensuse)
            elif command -v zypper >/dev/null 2>&1; then
                echo "installing intel mkl full using zypper..."
                sudo zypper install -y intel-mkl-full intel-mkl-devel
            else
                echo "warning: unknown linux package manager. please install intel mkl full manually."
                echo "download from: https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html"
                echo "required packages: intel-mkl-full, intel-mkl-devel, libmkl-rt"
            fi
            ;;
        "macos")
            # check if we have brew
            if command -v brew >/dev/null 2>&1; then
                echo "installing intel mkl full using homebrew..."
                brew install intel-mkl intel-oneapi-mkl
            else
                echo "warning: homebrew not found. please install intel mkl full manually."
                echo "install homebrew from: https://brew.sh/"
                echo "or download intel mkl full from: https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html"
            fi
            ;;
        "freebsd")
            if command -v pkg >/dev/null 2>&1; then
                echo "installing intel mkl full using pkg..."
                sudo pkg install -y intel-mkl math/intel-mkl
            else
                echo "warning: pkg not found. please install intel mkl full manually."
                echo "download from: https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html"
            fi
            ;;
        "openbsd"|"netbsd")
            echo "warning: intel mkl full may not be available for $os."
            echo "please install intel mkl full manually from:"
            echo "https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html"
            echo ""
            echo "alternative: installing openblas as fallback..."
            if command -v pkg_add >/dev/null 2>&1; then
                sudo pkg_add openblas
            fi
            ;;
        "solaris")
            echo "please install intel mkl full manually for solaris."
            echo "download the complete intel mkl package from:"
            echo "https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html"
            ;;
        "windows")
            echo "running in windows environment (cygwin/msys)."
            echo "please install intel mkl full manually from:"
            echo "https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html"
            echo "make sure to install the complete intel oneapi math kernel library package."
            ;;
        *)
            echo "warning: unknown operating system. please install intel mkl full manually."
            echo "download the complete intel mkl package from:"
            echo "https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html"
            ;;
    esac
    
    echo ""
    echo "note: intel mkl full includes all necessary libraries for optimal performance:"
    echo "- blas, lapack, scalapack"
    echo "- sparse solvers and fast fourier transforms" 
    echo "- vector math library and statistical functions"
    echo "- deep neural network primitives"
    echo "======================================="
    echo ""
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

# function to set optimal build flags based on os
set_build_flags() {
    local os=$(detect_os)
    
    case $os in
        "linux"|"freebsd"|"openbsd"|"netbsd")
            # use native target for optimal performance on these systems
            export RUSTFLAGS="${RUSTFLAGS:-} -C target-cpu=native"
            echo "rustflags set for $os: $RUSTFLAGS"
            ;;
        "macos")
            # macos-specific optimizations, using native target
            export RUSTFLAGS="${RUSTFLAGS:-} -C target-cpu=native"
            echo "rustflags set for macos: $RUSTFLAGS"
            ;;
        "solaris")
            # solaris-specific settings, using native target
            export RUSTFLAGS="${RUSTFLAGS:-} -C target-cpu=native -C link-arg=-lsocket -C link-arg=-lnsl"
            echo "rustflags set for solaris: $RUSTFLAGS"
            ;;
        "windows")
            # windows (wsl/cygwin/msys) settings, using native target
            export RUSTFLAGS="${RUSTFLAGS:-} -C target-cpu=native"
            echo "rustflags set for windows environment: $RUSTFLAGS"
            ;;
        *)
            # unknown os - use native target as default for testing
            export RUSTFLAGS="${RUSTFLAGS:-} -C target-cpu=native"
            echo "rustflags set for unknown os: $RUSTFLAGS"
            ;;
    esac
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

# function to run the tests
run_tests() {
    local start_time=$(date +%s)
    
    echo -e "\033[1;33m--- running tests (cargo test --release) --- \033[0m"
    echo ""

    if cargo test --release; then
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        echo ""
        echo -e "\033[1;32m--- tests completed successfully --- \033[0m"
        echo "test duration: ${duration} seconds"
    else
        echo ""
        echo -e "\033[1;31m--- tests failed --- \033[0m"
        echo ""
        echo -e "\033[1;33mcommon solutions:\033[0m"
        echo "1. check if all dependencies are installed"
        echo "2. try running: cargo clean && cargo test"
        echo "3. update rust: rustup update"
        echo "4. check for os-specific test requirements"
        exit 1
    fi
    echo ""
}

# main execution function
main() {
    echo "=== QOA unix test script ==="
    echo ""
    
    # check prerequisites
    check_cargo
    
    # install intel mkl full if needed
    install_mkl_if_needed
    
    # show system information
    show_system_info
    
    # check disk space
    check_disk_space
    echo ""
    
    # set optimal build flags
    set_build_flags
    echo ""
    
    # run the tests
    run_tests
    
    echo "=== test script completed ==="
}

# run main function
main
