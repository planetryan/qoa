#!/bin/bash

# QOA Unix Build Script for QOA
# Works on Linux, macOS, FreeBSD, OpenBSD, NetBSD, Solaris, and Windows (WSL/Cygwin/MSYS)
# Supports x86_64, ARM64, and RISC-V architectures

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

# function to detect architecture
detect_arch() {
    local arch=$(uname -m)
    case $arch in
        x86_64|amd64)
            echo "x86_64"
            ;;
        aarch64|arm64)
            echo "arm64"
            ;;
        armv7*|armv8*)
            echo "arm"
            ;;
        riscv64)
            echo "riscv64"
            ;;
        riscv32)
            echo "riscv32"
            ;;
        *)
            echo "$arch"
            ;;
    esac
}

# function to check if cargo is installed
check_cargo() {
    if ! command -v cargo >/dev/null 2>&1; then
        echo "error: cargo is not installed or not in path."
        echo "please install rust and cargo from: https://rustup.rs/"
        exit 1
    fi
}

# function to setup mkl environment variables permanently
setup_mkl_environment() {
    echo "=== setting up intel mkl environment ==="
    
    local arch=$(detect_arch)
    local os=$(detect_os)
    
    echo "detected architecture: $arch"
    
    # Architecture-specific library paths
    local lib_arch=""
    case $arch in
        x86_64)
            lib_arch="intel64"
            ;;
        arm64|arm)
            lib_arch="intel64"  # MKL uses intel64 for most platforms
            ;;
        riscv64|riscv32)
            echo "warning: intel mkl may not be available for risc-v. will use fallback options."
            lib_arch="intel64"
            ;;
        *)
            lib_arch="intel64"
            ;;
    esac
    
    # Common MKL paths to check (architecture-aware)
    local mkl_paths=(
        "/opt/intel/oneapi/mkl/latest"
        "/opt/intel/mkl"
        "/usr/lib/x86_64-linux-gnu/mkl"
        "/usr/lib/aarch64-linux-gnu/mkl"
        "/usr/lib/riscv64-linux-gnu/mkl"
        "/usr/include/mkl"
        "/usr/local/lib/mkl"
        "/opt/homebrew/lib"  # macOS ARM
        "/usr/local/lib"     # macOS x86_64
    )
    
    local mkl_found=false
    local mkl_root=""
    local mkl_lib_path=""
    
    # Find MKL installation
    for path in "${mkl_paths[@]}"; do
        if [ -d "$path" ]; then
            mkl_found=true
            mkl_root="$path"
            echo "found mkl at: $path"
            break
        fi
    done
    
    if [ "$mkl_found" = true ]; then
        # Set up environment variables based on installation type
        if [ -d "/opt/intel/oneapi/mkl/latest" ]; then
            # Intel OneAPI installation
            if [ -f "/opt/intel/oneapi/mkl/latest/env/vars.sh" ]; then
                echo "sourcing oneapi mkl environment..."
                source "/opt/intel/oneapi/mkl/latest/env/vars.sh" || echo "warning: failed to source oneapi vars"
            fi
            export MKLROOT="/opt/intel/oneapi/mkl/latest"
            mkl_lib_path="/opt/intel/oneapi/mkl/latest/lib/$lib_arch"
            
        elif [ -d "/opt/intel/mkl" ]; then
            # Traditional Intel MKL installation
            if [ -f "/opt/intel/mkl/bin/mklvars.sh" ]; then
                echo "sourcing traditional mkl environment..."
                source "/opt/intel/mkl/bin/mklvars.sh" $lib_arch || echo "warning: failed to source mklvars"
            fi
            export MKLROOT="/opt/intel/mkl"
            mkl_lib_path="/opt/intel/mkl/lib/$lib_arch"
            
        elif [ -d "/usr/lib/x86_64-linux-gnu/mkl" ]; then
            # System package installation (x86_64)
            export MKLROOT="/usr"
            mkl_lib_path="/usr/lib/x86_64-linux-gnu/mkl"
            echo "using system mkl installation (x86_64)"
            
        elif [ -d "/usr/lib/aarch64-linux-gnu/mkl" ]; then
            # System package installation (ARM64)
            export MKLROOT="/usr"
            mkl_lib_path="/usr/lib/aarch64-linux-gnu/mkl"
            echo "using system mkl installation (arm64)"
            
        elif [ -d "/usr/lib/riscv64-linux-gnu/mkl" ]; then
            # System package installation (RISC-V 64)
            export MKLROOT="/usr"
            mkl_lib_path="/usr/lib/riscv64-linux-gnu/mkl"
            echo "using system mkl installation (riscv64)"
            
        elif [ -d "/opt/homebrew/lib" ] && [ "$os" = "macos" ]; then
            # macOS ARM64 homebrew
            export MKLROOT="/opt/homebrew"
            mkl_lib_path="/opt/homebrew/lib"
            echo "using homebrew mkl installation (arm64)"
            
        elif [ -d "/usr/local/lib" ] && [ "$os" = "macos" ]; then
            # macOS x86_64 homebrew/manual install
            export MKLROOT="/usr/local"
            mkl_lib_path="/usr/local/lib"
            echo "using local mkl installation (x86_64)"
        fi
        
        # Set library path
        if [ -n "$mkl_lib_path" ] && [ -d "$mkl_lib_path" ]; then
            export LD_LIBRARY_PATH="$mkl_lib_path:${LD_LIBRARY_PATH:-}"
            if [ "$os" = "macos" ]; then
                export DYLD_LIBRARY_PATH="$mkl_lib_path:${DYLD_LIBRARY_PATH:-}"
            fi
        fi
        
        # Common MKL environment variables
        export MKL_INTERFACE_LAYER="LP64"
        export MKL_THREADING_LAYER="INTEL"
        export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$(nproc 2>/dev/null || echo 4)}"
        
        # Ensure environment persists for build
        echo "export MKLROOT=\"${MKLROOT}\"" >> ~/.bashrc
        echo "export LD_LIBRARY_PATH=\"${LD_LIBRARY_PATH}\"" >> ~/.bashrc
        echo "export MKL_INTERFACE_LAYER=\"${MKL_INTERFACE_LAYER}\"" >> ~/.bashrc
        echo "export MKL_THREADING_LAYER=\"${MKL_THREADING_LAYER}\"" >> ~/.bashrc
        echo "export MKL_NUM_THREADS=\"${MKL_NUM_THREADS}\"" >> ~/.bashrc
        
        if [ "$os" = "macos" ]; then
            echo "export DYLD_LIBRARY_PATH=\"${DYLD_LIBRARY_PATH}\"" >> ~/.bashrc
        fi
        
        echo "mkl environment variables set and persisted:"
        echo "  MKLROOT=${MKLROOT:-not set}"
        echo "  LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-not set}"
        echo "  MKL_INTERFACE_LAYER=${MKL_INTERFACE_LAYER}"
        echo "  MKL_THREADING_LAYER=${MKL_THREADING_LAYER}"
        echo "  MKL_NUM_THREADS=${MKL_NUM_THREADS}"
        
        if [ "$os" = "macos" ]; then
            echo "  DYLD_LIBRARY_PATH=${DYLD_LIBRARY_PATH:-not set}"
        fi
    else
        echo "warning: mkl not found in standard locations"
        echo "this may cause linking issues with svml functions"
        echo "will use fallback build configuration"
    fi
    echo ""
}

# function to install intel mkl full if needed (architecture-aware)
install_mkl_if_needed() {
    local os=$(detect_os)
    local arch=$(detect_arch)
    
    echo "=== intel mkl full installation check ==="
    echo "detected os: $os"
    echo "detected architecture: $arch"
    echo ""
    
    # check if mkl is already installed
    if pkg-config --exists mkl 2>/dev/null || [ -d "/opt/intel/mkl" ] || [ -d "/usr/include/mkl" ]; then
        echo "intel mkl appears to be already installed."
        echo ""
        return 0
    fi
    
    echo "intel mkl full not found. installing for $arch architecture..."
    echo ""
    
    case $os in
        "linux")
            # Architecture-specific package installation
            case $arch in
                x86_64)
                    install_mkl_linux_x86_64
                    ;;
                arm64|arm)
                    install_mkl_linux_arm
                    ;;
                riscv64|riscv32)
                    install_mkl_linux_riscv
                    ;;
                *)
                    echo "warning: unknown architecture $arch for linux"
                    install_mkl_linux_fallback
                    ;;
            esac
            ;;
        "macos")
            case $arch in
                x86_64)
                    install_mkl_macos_x86_64
                    ;;
                arm64)
                    install_mkl_macos_arm64
                    ;;
                *)
                    echo "warning: unknown architecture $arch for macos"
                    install_mkl_macos_fallback
                    ;;
            esac
            ;;
        "freebsd"|"openbsd"|"netbsd")
            install_mkl_bsd $arch
            ;;
        "solaris")
            install_mkl_solaris $arch
            ;;
        "windows")
            install_mkl_windows $arch
            ;;
        *)
            echo "warning: unknown operating system $os"
            install_mkl_fallback $arch
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

# Linux x86_64 MKL installation
install_mkl_linux_x86_64() {
    if command -v apt >/dev/null 2>&1; then
        echo "installing intel mkl full for x86_64 using apt..."
        sudo apt update && sudo apt install -y intel-mkl-full libmkl-dev libmkl-rt intel-oneapi-mkl
    elif command -v yum >/dev/null 2>&1; then
        echo "installing intel mkl full for x86_64 using yum..."
        sudo yum install -y intel-mkl-full intel-mkl-devel intel-oneapi-mkl
    elif command -v dnf >/dev/null 2>&1; then
        echo "installing intel mkl full for x86_64 using dnf..."
        sudo dnf install -y intel-mkl-full intel-mkl-devel intel-oneapi-mkl
    elif command -v pacman >/dev/null 2>&1; then
        echo "installing intel mkl full for x86_64 using pacman..."
        sudo pacman -S --noconfirm intel-mkl intel-mkl-static
    elif command -v zypper >/dev/null 2>&1; then
        echo "installing intel mkl full for x86_64 using zypper..."
        sudo zypper install -y intel-mkl-full intel-mkl-devel
    else
        install_mkl_linux_fallback
    fi
}

# Linux ARM MKL installation
install_mkl_linux_arm() {
    echo "note: intel mkl has limited arm support. trying available packages..."
    if command -v apt >/dev/null 2>&1; then
        echo "trying intel mkl for arm64 using apt..."
        sudo apt update && sudo apt install -y intel-mkl-full libmkl-dev libmkl-rt || {
            echo "intel mkl not available for arm64, installing openblas as alternative..."
            sudo apt install -y libopenblas-dev liblapack-dev
        }
    elif command -v yum >/dev/null 2>&1; then
        echo "trying intel mkl for arm using yum..."
        sudo yum install -y intel-mkl-full intel-mkl-devel || {
            echo "intel mkl not available for arm, installing openblas as alternative..."
            sudo yum install -y openblas-devel lapack-devel
        }
    elif command -v dnf >/dev/null 2>&1; then
        echo "trying intel mkl for arm using dnf..."
        sudo dnf install -y intel-mkl-full intel-mkl-devel || {
            echo "intel mkl not available for arm, installing openblas as alternative..."
            sudo dnf install -y openblas-devel lapack-devel
        }
    else
        echo "installing openblas as mkl alternative for arm..."
        if command -v apt >/dev/null 2>&1; then
            sudo apt install -y libopenblas-dev liblapack-dev
        fi
    fi
}

# Linux RISC-V MKL installation
install_mkl_linux_riscv() {
    echo "note: intel mkl is not available for risc-v. installing openblas as alternative..."
    if command -v apt >/dev/null 2>&1; then
        sudo apt update && sudo apt install -y libopenblas-dev liblapack-dev
    elif command -v yum >/dev/null 2>&1; then
        sudo yum install -y openblas-devel lapack-devel
    elif command -v dnf >/dev/null 2>&1; then
        sudo dnf install -y openblas-devel lapack-devel
    elif command -v pacman >/dev/null 2>&1; then
        sudo pacman -S --noconfirm openblas lapack
    else
        echo "please install openblas and lapack manually for risc-v"
    fi
}

# Linux fallback installation
install_mkl_linux_fallback() {
    echo "warning: unknown linux package manager. please install intel mkl full manually."
    echo "download from: https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html"
    echo "required packages: intel-mkl-full, intel-mkl-devel, libmkl-rt"
    echo "alternative: install openblas-devel and lapack-devel"
}

# macOS x86_64 MKL installation
install_mkl_macos_x86_64() {
    if command -v brew >/dev/null 2>&1; then
        echo "installing intel mkl full for x86_64 using homebrew..."
        brew install intel-mkl intel-oneapi-mkl
    else
        echo "warning: homebrew not found. please install intel mkl full manually."
        echo "install homebrew from: https://brew.sh/"
        echo "or download intel mkl full from: https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html"
    fi
}

# macOS ARM64 MKL installation
install_mkl_macos_arm64() {
    if command -v brew >/dev/null 2>&1; then
        echo "installing intel mkl full for arm64 using homebrew..."
        brew install intel-mkl intel-oneapi-mkl || {
            echo "intel mkl not available for arm64, installing openblas as alternative..."
            brew install openblas lapack
        }
    else
        echo "warning: homebrew not found. please install intel mkl full manually."
        echo "install homebrew from: https://brew.sh/"
        echo "alternative: install openblas for arm64"
    fi
}

# macOS fallback installation
install_mkl_macos_fallback() {
    echo "warning: unknown architecture for macos. please install intel mkl full manually."
    echo "download from: https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html"
}

# BSD systems MKL installation
install_mkl_bsd() {
    local arch=$1
    echo "attempting mkl installation for bsd system ($arch)..."
    if command -v pkg >/dev/null 2>&1; then
        echo "trying intel mkl using pkg..."
        sudo pkg install -y intel-mkl math/intel-mkl || {
            echo "intel mkl not available, installing openblas as alternative..."
            sudo pkg install -y openblas lapack
        }
    elif command -v pkg_add >/dev/null 2>&1; then
        echo "trying openblas as mkl alternative..."
        sudo pkg_add openblas lapack
    else
        echo "warning: pkg not found. please install intel mkl full manually."
        echo "download from: https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html"
    fi
}

# Solaris MKL installation
install_mkl_solaris() {
    local arch=$1
    echo "please install intel mkl full manually for solaris ($arch)."
    echo "download the complete intel mkl package from:"
    echo "https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html"
}

# Windows MKL installation
install_mkl_windows() {
    local arch=$1
    echo "running in windows environment (cygwin/msys) on $arch."
    echo "please install intel mkl full manually from:"
    echo "https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html"
    echo "make sure to install the complete intel oneapi math kernel library package."
}

# Fallback MKL installation
install_mkl_fallback() {
    local arch=$1
    echo "warning: unknown operating system. please install intel mkl full manually for $arch."
    echo "download the complete intel mkl package from:"
    echo "https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html"
}

# function to display system information
show_system_info() {
    local os=$(detect_os)
    local arch=$(detect_arch)
    echo "=== build environment information ==="
    echo "operating system: $os"
    echo "architecture: $arch"
    echo "uname -m: $(uname -m 2>/dev/null || echo "unknown")"
    
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

# function to set optimal build flags based on os and architecture
set_build_flags() {
    local os=$(detect_os)
    local arch=$(detect_arch)
    
    echo "=== setting build flags for $os on $arch ==="
    
    # Base flags for all architectures
    local base_flags=""
    local mkl_flags=""
    
    # Architecture-specific optimizations
    case $arch in
        x86_64)
            # Try MKL linking first, fallback to generic if needed
            if [ -d "/usr/lib/x86_64-linux-gnu/mkl" ]; then
                mkl_flags="-L /usr/lib/x86_64-linux-gnu/mkl -l mkl_rt -l mkl_intel_lp64 -l mkl_sequential -l mkl_core"
            elif [ -d "/opt/intel/oneapi/mkl/latest/lib/intel64" ]; then
                mkl_flags="-L /opt/intel/oneapi/mkl/latest/lib/intel64 -l mkl_rt"
            elif [ -d "/opt/intel/mkl/lib/intel64" ]; then
                mkl_flags="-L /opt/intel/mkl/lib/intel64 -l mkl_rt"
            fi
            
            if [ -n "$mkl_flags" ]; then
                base_flags="-C target-cpu=native $mkl_flags"
                echo "using mkl-optimized flags for x86_64"
            else
                base_flags="-C target-cpu=generic"
                echo "mkl not found, using generic flags for x86_64"
            fi
            ;;
        arm64|arm)
            # ARM-specific flags
            if [ -d "/usr/lib/aarch64-linux-gnu/mkl" ]; then
                mkl_flags="-L /usr/lib/aarch64-linux-gnu/mkl -l mkl_rt"
                base_flags="-C target-cpu=native $mkl_flags"
                echo "using mkl-optimized flags for arm64"
            else
                base_flags="-C target-cpu=native"
                echo "using native flags for arm64 (no mkl)"
            fi
            ;;
        riscv64|riscv32)
            # RISC-V flags (no MKL support)
            base_flags="-C target-cpu=native"
            echo "using native flags for risc-v (mkl not supported)"
            ;;
        *)
            # Unknown architecture fallback
            base_flags="-C target-cpu=generic"
            echo "using generic flags for unknown architecture: $arch"
            ;;
    esac
    
    # OS-specific additions
    case $os in
        "linux"|"freebsd"|"openbsd"|"netbsd")
            export RUSTFLAGS="${RUSTFLAGS:-} $base_flags"
            ;;
        "macos")
            # macOS-specific flags
            if [ "$arch" = "arm64" ]; then
                export RUSTFLAGS="${RUSTFLAGS:-} $base_flags"
            else
                export RUSTFLAGS="${RUSTFLAGS:-} $base_flags"
            fi
            ;;
        "solaris")
            # Solaris-specific flags
            export RUSTFLAGS="${RUSTFLAGS:-} $base_flags -C link-arg=-lsocket -C link-arg=-lnsl"
            ;;
        "windows")
            # Windows flags
            export RUSTFLAGS="${RUSTFLAGS:-} $base_flags"
            ;;
        *)
            # Unknown OS fallback
            export RUSTFLAGS="${RUSTFLAGS:-} $base_flags"
            ;;
    esac
    
    echo "final rustflags: $RUSTFLAGS"
    echo ""
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
        echo "5. for linking issues, try: RUSTFLAGS=\"-C target-cpu=generic\" $0"
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
    echo ""
    echo "supported architectures: x86_64, arm64, arm, riscv64, riscv32"
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
    
    echo "=== QOA unix build script (multi-architecture) ==="
    echo ""
    
    # check prerequisites
    check_cargo
    
    # install intel mkl full if needed
    install_mkl_if_needed
    
    # setup mkl environment
    setup_mkl_environment
    
    # show system information
    show_system_info
    
    # check disk space
    check_disk_space
    echo ""
    
    # set optimal build flags
    set_build_flags
    
    # clean if requested
    clean_build "$clean_flag"
    
    # run the build
    run_build "$build_type"
    
    echo ""
    echo "=== build script completed ==="
}

# run main function with all arguments
main "$@"