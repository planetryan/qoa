#!/bin/bash

# QOA Unix Build Script for QOA
# Works on Linux, macOS, FreeBSD, OpenBSD, NetBSD, Solaris, and Windows (WSL/Cygwin/MSYS)

set -e  # exit on any error

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
        echo "error: cargo is not installed or not in path."
        echo "please install rust and cargo from: https://rustup.rs/"
        exit 1
    fi
}

# function to install intel mkl full if needed
install_mkl_if_needed() {
    local os=$(detect_os)
    
    echo "=== intel mkl full installation check ==="
    echo "detected os: $os"
    echo ""
    
    # check if mkl is already installed
    if pkg-config --exists mkl 2>/dev/null || [ -d "/opt/intel/mkl" ] || [ -d "/usr/include/mkl" ]; then
        echo "intel mkl appears to be already installed."
        echo ""
        return 0
    fi
    
    echo "intel mkl full not found. installing..."
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
            # use generic target to avoid svml issues
            export RUSTFLAGS="${RUSTFLAGS:-} -C target-cpu=generic"
            echo "rustflags set for $os: $RUSTFLAGS"
            ;;
        "macos")
            # macos-specific optimizations
            export RUSTFLAGS="${RUSTFLAGS:-} -C target-cpu=generic"
            echo "rustflags set for macos: $RUSTFLAGS"
            ;;
        "solaris")
            # solaris-specific settings
            export RUSTFLAGS="${RUSTFLAGS:-} -C target-cpu=generic -C link-arg=-lsocket -C link-arg=-lnsl"
            echo "rustflags set for solaris: $RUSTFLAGS"
            ;;
        "windows")
            # windows (wsl/cygwin/msys) settings
            export RUSTFLAGS="${RUSTFLAGS:-} -C target-cpu=generic"
            echo "rustflags set for windows environment: $RUSTFLAGS"
            ;;
        *)
            # unknown os - use safe defaults
            export RUSTFLAGS="${RUSTFLAGS:-} -C target-cpu=generic"
            echo "rustflags set for unknown os: $RUSTFLAGS"
            ;;
    esac
}

# function to clean previous builds if requested
clean_build() {
    if [[ "$1" == "clean" ]]; then
        echo "--- cleaning previous build ---"
        cargo clean
        echo "previous build cleaned."
        echo ""
    fi
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

# function to run the build with error handling
run_build() {
    local build_type="${1:-release}"
    local start_time=$(date +%s)
    
    echo "--- compiling $build_type build for qoa ---"
    echo ""
    
    # set build command based on type
    local cargo_cmd
    if [[ "$build_type" == "release" ]]; then
        cargo_cmd="cargo build --release"
    elif [[ "$build_type" == "debug" ]]; then
        cargo_cmd="cargo build"
    else
        echo "error: unknown build type '$build_type'. use 'release' or 'debug'."
        exit 1
    fi
    
    echo "running: $cargo_cmd"
    echo ""
    
    # run the build command
    if $cargo_cmd; then
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        echo ""
        echo "--- build completed successfully ---"
        echo "build type: $build_type"
        echo "build time: ${duration} seconds"
        
        # show binary information if available
        if [[ "$build_type" == "release" ]] && [[ -f "target/release/qoa" ]]; then
            echo "binary location: target/release/qoa"
            if command -v ls >/dev/null 2>&1; then
                local size=$(ls -lh target/release/qoa | awk '{print $5}')
                echo "binary size: $size"
            fi
        elif [[ "$build_type" == "debug" ]] && [[ -f "target/debug/qoa" ]]; then
            echo "binary location: target/debug/qoa"
            if command -v ls >/dev/null 2>&1; then
                local size=$(ls -lh target/debug/qoa | awk '{print $5}')
                echo "binary size: $size"
            fi
        fi
    else
        echo ""
        echo "--- build failed ---"
        echo "build type: $build_type"
        echo ""
        echo "common solutions:"
        echo "1. check if all dependencies are installed"
        echo "2. try running: cargo clean && $0"
        echo "3. update rust: rustup update"
        echo "4. check for os-specific build requirements"
        exit 1
    fi
}

# function to show usage information
show_usage() {
    echo "usage: $0 [options] [build_type]"
    echo ""
    echo "build_type:"
    echo "  release    build optimized release version (default)"
    echo "  debug      build debug version with debug symbols"
    echo ""
    echo "options:"
    echo "  clean      clean previous builds before building"
    echo "  --help     show this help message"
    echo ""
    echo "examples:"
    echo "  $0                    # build release version"
    echo "  $0 debug              # build debug version"
    echo "  $0 clean release      # clean then build release"
    echo "  $0 clean debug        # clean then build debug"
}

# main execution
main() {
    # parse arguments
    local clean_flag=""
    local build_type="release"
    
    for arg in "$@"; do
        case $arg in
            clean)
                clean_flag="clean"
                ;;
            debug|release)
                build_type="$arg"
                ;;
            --help|-h)
                show_usage
                exit 0
                ;;
            *)
                echo "error: unknown argument '$arg'"
                show_usage
                exit 1
                ;;
        esac
    done
    
    echo "=== QOA unix build script ==="
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
    
    # clean if requested
    clean_build "$clean_flag"
    
    # run the build
    run_build "$build_type"
    
    echo ""
    echo "=== build script completed ==="
}

# run main function with all arguments
main "$@"
