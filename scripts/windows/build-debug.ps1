# PowerShell Debug Build Script for QOA
# This script is specifically configured to build the debug version of QOA.
# Works on Windows PowerShell, PowerShell Core, and cross-platform environments

# Exit on any error
$ErrorActionPreference = "Stop"

# Function to detect operating system
function Get-OperatingSystem {
    if ($PSVersionTable.PSVersion.Major -ge 6) {
        # PowerShell Core/7+ has built-in OS detection
        if ($IsWindows) { return "Windows" }
        elseif ($IsLinux) { return "Linux" }
        elseif ($IsMacOS) { return "macOS" }
        else { return "Unknown" }
    } else {
        # Windows PowerShell (5.1 and earlier) only runs on Windows
        return "Windows"
    }
}

# Function to check if cargo is installed
function Test-CargoInstallation {
    try {
        $null = Get-Command cargo -ErrorAction Stop
        return $true
    } catch {
        Write-Error "Error: Cargo is not installed or not in PATH."
        Write-Host "Please install Rust and Cargo from: https://rustup.rs/" -ForegroundColor Yellow
        return $false
    }
}

# Function to get system architecture
function Get-SystemArchitecture {
    try {
        if ($PSVersionTable.PSVersion.Major -ge 6) {
            # PowerShell Core
            return [System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture.ToString()
        } else {
            # Windows PowerShell
            return $env:PROCESSOR_ARCHITECTURE
        }
    } catch {
        return "Unknown"
    }
}

# Function to check and install Intel MKL Full if needed
function Install-IntelMKLIfNeeded {
    Write-Host "=== Intel MKL Full Installation Check ===" -ForegroundColor Cyan
    
    $os = Get-OperatingSystem
    Write-Host "Detected OS: $os" -ForegroundColor White
    Write-Host "Checking for Intel MKL Full..." -ForegroundColor White
    
    $mklInstalled = $false
    
    if ($os -eq "Windows") {
        # Check common Intel oneAPI/MKL installation paths or environment variables
        if (Test-Path "C:\Program Files (x86)\Intel\oneAPI\mkl" -PathType Container) {
            $mklInstalled = $true
        } elseif ($env:MKLROOT -ne $null -and (Test-Path $env:MKLROOT -PathType Container)) {
            $mklInstalled = $true
        }
    } elseif ($os -eq "Linux" -or $os -eq "macOS" -or $os -eq "FreeBSD" -or $os -eq "OpenBSD" -or $os -eq "NetBSD") {
        # For PowerShell Core on Unix-like systems, use pkg-config or common paths
        try {
            if ((Get-Command pkg-config -ErrorAction SilentlyContinue) -ne $null -and (pkg-config --exists mkl 2>$null)) {
                $mklInstalled = $true
            } elseif (Test-Path "/opt/intel/mkl" -PathType Container) {
                $mklInstalled = $true
            } elseif (Test-Path "/usr/include/mkl" -PathType Container) {
                $mklInstalled = $true
            }
        } catch {
            # pkg-config might not be available, suppress errors
        }
    }

    if ($mklInstalled) {
        Write-Host "Intel MKL appears to be already installed. Skipping installation." -ForegroundColor Green
        Write-Host ""
        return
    }
    
    Write-Host "Intel MKL Full not found. Proceeding with installation..." -ForegroundColor Yellow
    Write-Host ""
    
    switch ($os) {
        "Windows" {
            Write-Host "For Windows, please install Intel oneAPI Base Toolkit, which includes Intel MKL." -ForegroundColor Yellow
            Write-Host "Download from: https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit.html" -ForegroundColor White
            Write-Host "Ensure you select the 'Intel oneAPI Math Kernel Library (oneMKL)' component during installation." -ForegroundColor White
        }
        "Linux" {
            # Check if we have apt (Debian/Ubuntu)
            if (Get-Command apt -ErrorAction SilentlyContinue) {
                Write-Host "Installing Intel MKL Full using apt..." -ForegroundColor White
                sudo apt update -y && sudo apt install -y intel-mkl-full libmkl-dev libmkl-rt intel-oneapi-mkl
            # Check if we have yum (RHEL/CentOS/Fedora older)
            } elseif (Get-Command yum -ErrorAction SilentlyContinue) {
                Write-Host "Installing Intel MKL Full using yum..." -ForegroundColor White
                sudo yum install -y intel-mkl-full intel-mkl-devel intel-oneapi-mkl
            # Check if we have dnf (Fedora newer)
            } elseif (Get-Command dnf -ErrorAction SilentlyContinue) {
                Write-Host "Installing Intel MKL Full using dnf..." -ForegroundColor White
                sudo dnf install -y intel-mkl-full intel-mkl-devel intel-oneapi-mkl
            # Check if we have pacman (Arch Linux)
            } elseif (Get-Command pacman -ErrorAction SilentlyContinue) {
                Write-Host "Installing Intel MKL Full using pacman..." -ForegroundColor White
                sudo pacman -S --noconfirm intel-mkl intel-mkl-static
            # Check if we have zypper (openSUSE)
            } elseif (Get-Command zypper -ErrorAction SilentlyContinue) {
                Write-Host "Installing Intel MKL Full using zypper..." -ForegroundColor White
                sudo zypper install -y intel-mkl-full intel-mkl-devel
            } else {
                Write-Warning "Warning: Unknown Linux package manager. Please install Intel MKL Full manually."
                Write-Host "Download from: https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html" -ForegroundColor White
                Write-Host "Required packages: intel-mkl-full, intel-mkl-devel, libmkl-rt" -ForegroundColor White
            }
        }
        "macOS" {
            # Check if we have brew
            if (Get-Command brew -ErrorAction SilentlyContinue) {
                Write-Host "Installing Intel MKL Full using Homebrew..." -ForegroundColor White
                brew install intel-mkl intel-oneapi-mkl
            } else {
                Write-Warning "Warning: Homebrew not found. Please install Intel MKL Full manually."
                Write-Host "Install Homebrew from: https://brew.sh/" -ForegroundColor White
                Write-Host "Or download Intel MKL Full from: https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html" -ForegroundColor White
            }
        }
        "FreeBSD" {
            if (Get-Command pkg -ErrorAction SilentlyContinue) {
                Write-Host "Installing Intel MKL Full using pkg..." -ForegroundColor White
                sudo pkg install -y intel-mkl math/intel-mkl
            } else {
                Write-Warning "pkg not found. Please install Intel MKL Full manually."
                Write-Host "Download from: https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html" -ForegroundColor White
            }
        }
        "OpenBSD" {
            Write-Warning "Warning: Intel MKL Full may not be available for $os."
            Write-Host "Please install Intel MKL Full manually from:" -ForegroundColor White
            Write-Host "https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html" -ForegroundColor White
            Write-Host ""
            Write-Host "Alternative: Installing OpenBLAS as fallback..." -ForegroundColor Yellow
            if (Get-Command pkg_add -ErrorAction SilentlyContinue) {
                sudo pkg_add openblas
            }
        }
        "NetBSD" {
            Write-Warning "Intel MKL Full may not be available for $os."
            Write-Host "Please install Intel MKL Full manually from:" -ForegroundColor White
            Write-Host "https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html" -ForegroundColor White
            Write-Host ""
            Write-Host "Alternative: Installing OpenBLAS as fallback..." -ForegroundColor Yellow
            if (Get-Command pkg_add -ErrorAction SilentlyContinue) {
                sudo pkg_add openblas
            }
        }
        "Solaris" {
            Write-Host "Please install Intel MKL Full manually for Solaris." -ForegroundColor Yellow
            Write-Host "Download the complete Intel MKL package from:" -ForegroundColor White
            Write-Host "https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html" -ForegroundColor White
        }
        Default {
            Write-Warning "Unknown operating system. Please install Intel MKL Full manually."
            Write-Host "Download the complete Intel MKL package from:" -ForegroundColor White
            Write-Host "https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html" -ForegroundColor White
        }
    }
    
    Write-Host ""
    Write-Host "Note: Intel MKL Full includes all necessary libraries for optimal performance:" -ForegroundColor Green
    Write-Host "- BLAS, LAPACK, ScaLAPACK" -ForegroundColor White
    Write-Host "- Sparse Solvers and Fast Fourier Transforms" -ForegroundColor White
    Write-Host "- Vector Math Library and Statistical Functions" -ForegroundColor White
    Write-Host "- Deep Neural Network primitives" -ForegroundColor White
    Write-Host "=======================================" -ForegroundColor Green
    Write-Host ""
}

# Function to set up Intel MKL environment variables for the current session
function Set-MKL-Environment {
    Write-Host "=== Setting up Intel MKL Environment ===" -ForegroundColor Cyan
    
    $os = Get-OperatingSystem
    $arch = Get-SystemArchitecture
    Write-Host "Detected OS: $os, Architecture: $arch" -ForegroundColor White
    
    $mklRoot = $null
    $mklLibPath = $null
    $varsScriptFound = $false

    # Common Intel oneAPI/MKL installation paths
    $oneAPIMKLRoot = "C:\Program Files (x86)\Intel\oneAPI\mkl\latest" # Windows
    $oneAPIMKLRootLinux = "/opt/intel/oneapi/mkl/latest" # Linux/macOS
    $traditionalMKLRoot = "/opt/intel/mkl" # Linux/macOS
    $systemMKLPathX86_64 = "/usr/lib/x86_64-linux-gnu/mkl"
    $systemMKLPathARM64 = "/usr/lib/aarch64-linux-gnu/mkl"
    $systemMKLPathRISCV64 = "/usr/lib/riscv64-linux-gnu/mkl"
    $homebrewLib = "/opt/homebrew/lib" # macOS ARM
    $localLib = "/usr/local/lib" # macOS x86_64

    # Try to source vars.ps1 or vars.sh first
    if ($os -eq "Windows") {
        $varsPs1 = Join-Path $oneAPIMKLRoot "env\vars.ps1"
        if (Test-Path $varsPs1) {
            Write-Host "Sourcing oneAPI MKL environment (vars.ps1)..." -ForegroundColor Yellow
            try {
                . $varsPs1 -arch intel64 -platform windows # Use dot-sourcing
                $varsScriptFound = $true
                Write-Host "oneAPI MKL environment sourced successfully." -ForegroundColor Green
            } catch {
                Write-Warning "Failed to source oneAPI MKL vars.ps1: $_"
            }
        }
    } else { # Linux/macOS
        $libArch = "intel64" # MKL uses intel64 for most platforms
        if ($arch -match "riscv") {
            Write-Warning "Intel MKL may not be fully available for RISC-V. Will use fallback options."
        }

        $varsShOneAPI = Join-Path $oneAPIMKLRootLinux "env\vars.sh"
        $varsShTraditional = Join-Path $traditionalMKLRoot "bin\mklvars.sh"

        if (Test-Path $varsShOneAPI) {
            Write-Host "Sourcing oneAPI MKL environment (vars.sh)..." -ForegroundColor Yellow
            try {
                # Use bash to source the script and then apply variables to current PowerShell session
                $output = bash -c "source '$varsShOneAPI' intel64 && env"
                $output | ForEach-Object {
                    if ($_ -match "^(.*?)=(.*)$") {
                        $varName = $matches[1]
                        $varValue = $matches[2]
                        # Only set variables that are relevant for MKL
                        if ($varName -match "^MKL" -or $varName -eq "LD_LIBRARY_PATH" -or $varName -eq "DYLD_LIBRARY_PATH") {
                            Set-Item Env:$varName $varValue
                        }
                    }
                }
                $varsScriptFound = $true
                Write-Host "oneAPI MKL environment sourced successfully." -ForegroundColor Green
            } catch {
                Write-Warning "Failed to source oneAPI MKL vars.sh: $_"
            }
        } elseif (Test-Path $varsShTraditional) {
            Write-Host "Sourcing traditional MKL environment (mklvars.sh)..." -ForegroundColor Yellow
            try {
                $output = bash -c "source '$varsShTraditional' $libArch && env"
                $output | ForEach-Object {
                    if ($_ -match "^(.*?)=(.*)$") {
                        $varName = $matches[1]
                        $varValue = $matches[2]
                        if ($varName -match "^MKL" -or $varName -eq "LD_LIBRARY_PATH" -or $varName -eq "DYLD_LIBRARY_PATH") {
                            Set-Item Env:$varName $varValue
                        }
                    }
                }
                $varsScriptFound = $true
                Write-Host "Traditional MKL environment sourced successfully." -ForegroundColor Green
            } catch {
                Write-Warning "Failed to source traditional MKL mklvars.sh: $_"
            }
        }
    }

    # If vars.ps1/vars.sh not found, try to manually set variables
    if (-not $varsScriptFound) {
        Write-Host "MKL environment script not found, attempting manual setup..." -ForegroundColor Yellow
        
        switch ($os) {
            "Windows" {
                if (Test-Path "C:\Program Files (x86)\Intel\oneAPI\mkl" -PathType Container) {
                    $mklRoot = "C:\Program Files (x86)\Intel\oneAPI\mkl\latest"
                    $mklLibPath = Join-Path $mklRoot "lib\intel64"
                }
            }
            "Linux" {
                if (Test-Path $oneAPIMKLRootLinux -PathType Container) {
                    $mklRoot = $oneAPIMKLRootLinux
                    $mklLibPath = Join-Path $mklRoot "lib\intel64"
                } elseif (Test-Path $traditionalMKLRoot -PathType Container) {
                    $mklRoot = $traditionalMKLRoot
                    $mklLibPath = Join-Path $mklRoot "lib\intel64"
                } elseif (Test-Path $systemMKLPathX86_64 -PathType Container) {
                    $mklRoot = "/usr"
                    $mklLibPath = $systemMKLPathX86_64
                } elseif (Test-Path $systemMKLPathARM64 -PathType Container) {
                    $mklRoot = "/usr"
                    $mklLibPath = $systemMKLPathARM64
                } elseif (Test-Path $systemMKLPathRISCV64 -PathType Container) {
                    $mklRoot = "/usr"
                    $mklLibPath = $systemMKLPathRISCV64
                }
            }
            "macOS" {
                if (Test-Path $oneAPIMKLRootLinux -PathType Container) { # oneAPI uses same path structure on macOS/Linux
                    $mklRoot = $oneAPIMKLRootLinux
                    $mklLibPath = Join-Path $mklRoot "lib\intel64"
                } elseif (Test-Path $traditionalMKLRoot -PathType Container) {
                    $mklRoot = $traditionalMKLRoot
                    $mklLibPath = Join-Path $mklRoot "lib\intel64"
                } elseif (Test-Path $homebrewLib -PathType Container) {
                    $mklRoot = "/opt/homebrew"
                    $mklLibPath = $homebrewLib
                } elseif (Test-Path $localLib -PathType Container) {
                    $mklRoot = "/usr/local"
                    $mklLibPath = $localLib
                }
            }
        }

        if ($mklRoot -ne $null) {
            $env:MKLROOT = $mklRoot
            Write-Host "MKLROOT set to: $env:MKLROOT" -ForegroundColor White

            if ($mklLibPath -ne $null -and (Test-Path $mklLibPath -PathType Container)) {
                if ($os -eq "Windows") {
                    $env:Path = "$mklLibPath;$env:Path"
                    Write-Host "Added $mklLibPath to PATH." -ForegroundColor White
                } else {
                    $env:LD_LIBRARY_PATH = "$mklLibPath:$env:LD_LIBRARY_PATH"
                    Write-Host "Added $mklLibPath to LD_LIBRARY_PATH." -ForegroundColor White
                    if ($os -eq "macOS") {
                        $env:DYLD_LIBRARY_PATH = "$mklLibPath:$env:DYLD_LIBRARY_PATH"
                        Write-Host "Added $mklLibPath to DYLD_LIBRARY_PATH." -ForegroundColor White
                    }
                }
            }
            
            $env:MKL_INTERFACE_LAYER = "LP64"
            $env:MKL_THREADING_LAYER = "INTEL"
            $env:MKL_NUM_THREADS = (Get-CPUCoreCount).ToString() # Use detected core count
            
            Write-Host "MKL_INTERFACE_LAYER: $env:MKL_INTERFACE_LAYER" -ForegroundColor White
            Write-Host "MKL_THREADING_LAYER: $env:MKL_THREADING_LAYER" -ForegroundColor White
            Write-Host "MKL_NUM_THREADS: $env:MKL_NUM_THREADS" -ForegroundColor White
        } else {
            Write-Warning "Could not determine MKL installation path. MKL environment variables not set."
        }
    }
    Write-Host ""
}

# Function to display system information
function Show-SystemInfo {
    $os = Get-OperatingSystem
    $arch = Get-SystemArchitecture
    
    Write-Host "=== Build Environment Information ===" -ForegroundColor Green
    Write-Host "Operating System: $os" -ForegroundColor White
    Write-Host "Architecture: $arch" -ForegroundColor White
    Write-Host "PowerShell Version: $($PSVersionTable.PSVersion)" -ForegroundColor White
    
    try {
        $rustVersion = cargo --version 2>$null
        if ($rustVersion) {
            Write-Host "Cargo Version: $rustVersion" -ForegroundColor White
        }
        
        $rustcVersion = rustc --version 2>$null
        if ($rustcVersion) {
            Write-Host "Rust Version: $rustcVersion" -ForegroundColor White
        }
    } catch {
        Write-Warning "Could not get Rust/Cargo version information"
    }
    
    Write-Host "Build Date: $(Get-Date)" -ForegroundColor White
    Write-Host "Current Directory: $PWD" -ForegroundColor White
    Write-Host "=================================" -ForegroundColor Green
    Write-Host ""
}

# Function to set optimal build environment variables
function Set-BuildEnvironment {
    $os = Get-OperatingSystem
    $arch = Get-SystemArchitecture
    
    Write-Host "=== Setting Build Flags for $os on $arch ===" -ForegroundColor Yellow
    
    # Set RUSTFLAGS to include MKL linking and architecture-specific optimizations
    $currentRustFlags = $env:RUSTFLAGS
    $baseFlags = ""
    $mklFlags = ""

    switch ($arch) {
        "X64" { # x86_64
            if (Test-Path "$env:MKLROOT\lib\intel64" -PathType Container) {
                $mklFlags = "-L ""$env:MKLROOT\lib\intel64"" -l mkl_rt -l mkl_intel_lp64 -l mkl_sequential -l mkl_core"
                $baseFlags = "-C target-cpu=native $mklFlags"
                Write-Host "Using MKL-optimized flags for x86_64" -ForegroundColor White
            } else {
                $baseFlags = "-C target-cpu=native"
                Write-Host "MKL not found, using native flags for x86_64" -ForegroundColor White
            }
        }
        "Arm64" { # arm64
            # MKL has limited ARM support; if found, use it, else native
            if (Test-Path "$env:MKLROOT\lib\intel64" -PathType Container) { # MKL uses intel64 for ARM on some platforms
                $mklFlags = "-L ""$env:MKLROOT\lib\intel64"" -l mkl_rt"
                $baseFlags = "-C target-cpu=native $mklFlags"
                Write-Host "Using MKL-optimized flags for ARM64" -ForegroundColor White
            } else {
                $baseFlags = "-C target-cpu=native"
                Write-Host "Using native flags for ARM64 (no MKL)" -ForegroundColor White
            }
        }
        "RiscV64" { # riscv64
            $baseFlags = "-C target-cpu=native"
            Write-Host "Using native flags for RISC-V (MKL not supported)" -ForegroundColor White
        }
        Default { # Unknown architecture fallback
            $baseFlags = "-C target-cpu=generic"
            Write-Host "Using generic flags for unknown architecture: $arch" -ForegroundColor Yellow
        }
    }
    
    if ($currentRustFlags) {
        $env:RUSTFLAGS = "$currentRustFlags $baseFlags"
    } else {
        $env:RUSTFLAGS = $baseFlags
    }
    
    Write-Host "Final RUSTFLAGS: $env:RUSTFLAGS" -ForegroundColor White
    
    # Set additional environment variables based on OS
    switch ($os) {
        "Windows" {
            if (-not $env:CARGO_TARGET_DIR) {
                Write-Host "Using default target directory: target\" -ForegroundColor White
            }
            
            # Check for Visual Studio Build Tools
            $vsWhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
            if (Test-Path $vsWhere) {
                $vsInstall = & $vsWhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath 2>$null
                if ($vsInstall) {
                    Write-Host "Visual Studio Build Tools: Found" -ForegroundColor Green
                } else {
                    Write-Warning "Visual Studio Build Tools not found. You may need to install them."
                }
            }
        }
        
        "Linux" {
            Write-Host "Running on Linux with PowerShell Core" -ForegroundColor White
        }
        
        "macOS" {
            Write-Host "Running on macOS with PowerShell Core" -ForegroundColor White
        }
    }
    
    Write-Host ""
}

# Function to check available disk space
function Get-DiskSpace {
    try {
        $os = Get-OperatingSystem
        
        if ($os -eq "Windows") {
            $drive = (Get-Location).Drive
            if ($drive) {
                $freeSpace = [math]::Round((Get-WmiObject -Class Win32_LogicalDisk -Filter "DeviceID='$($drive.Name)'").FreeSpace / 1GB, 2)
                Write-Host "Available disk space: $freeSpace GB" -ForegroundColor White
            }
        } else {
            # For Linux/macOS with PowerShell Core, try using df command
            try {
                $dfOutput = df -h . 2>$null | Select-Object -Last 1
                if ($dfOutput) {
                    $available = ($dfOutput -split '\s+')[3]
                    Write-Host "Available disk space: $available" -ForegroundColor White
                }
            } catch {
                Write-Host "Could not determine disk space" -ForegroundColor Yellow
            }
        }
    } catch {
        Write-Host "Could not determine disk space" -ForegroundColor Yellow
    }
}

# Function to clean previous builds (always cleans for this script)
function Invoke-CleanBuild {
    Write-Host "--- CLEANING PREVIOUS BUILD ---" -ForegroundColor Yellow
    try {
        cargo clean
        Write-Host "Previous build cleaned successfully." -ForegroundColor Green
    } catch {
        Write-Warning "Failed to clean previous build: $_"
    }
    Write-Host ""
}

# Function to run the debug build
function Invoke-BuildDebug {
    $buildType = "debug" # hardcoded to debug
    $startTime = Get-Date
    
    Write-Host "--- COMPILING $($buildType.ToUpper()) BUILD FOR QOA ---" -ForegroundColor Cyan
    Write-Host ""
    
    $cargoCommand = "cargo build" # always build debug
    
    Write-Host "Running: $cargoCommand" -ForegroundColor White
    Write-Host ""
    
    # Execute the build
    try {
        & cargo build # Directly call cargo build for debug
        
        if ($LASTEXITCODE -eq 0) {
            $endTime = Get-Date
            $duration = [math]::Round(($endTime - $startTime).TotalSeconds, 2)
            
            Write-Host ""
            Write-Host "--- BUILD COMPLETED SUCCESSFULLY ---" -ForegroundColor Green
            Write-Host "Build type: $buildType" -ForegroundColor White
            Write-Host "Build time: $duration seconds" -ForegroundColor White
            
            # Show binary information
            $binaryPath = if ((Get-OperatingSystem) -eq "Windows") { "target\debug\qoa.exe" } else { "target/debug/qoa" }
            
            if (Test-Path $binaryPath) {
                $binaryInfo = Get-Item $binaryPath
                $sizeKB = [math]::Round($binaryInfo.Length / 1KB, 2)
                $sizeMB = [math]::Round($binaryInfo.Length / 1MB, 2)
                
                Write-Host "Binary location: $binaryPath" -ForegroundColor White
                Write-Host "Binary size: $sizeKB KB ($sizeMB MB)" -ForegroundColor White
                Write-Host "Binary created: $($binaryInfo.LastWriteTime)" -ForegroundColor White
            }
            
            return $true
        } else {
            throw "Cargo build failed with exit code $LASTEXITCODE"
        }
    } catch {
        Write-Host ""
        Write-Host "--- BUILD FAILED ---" -ForegroundColor Red
        Write-Host "Build type: $buildType" -ForegroundColor White
        Write-Host "Error: $_" -ForegroundColor Red
        Write-Host ""
        Write-Host "Common solutions:" -ForegroundColor Yellow
        Write-Host "1. Check if all dependencies are installed" -ForegroundColor White
        Write-Host "2. Try running: .\build-debug.ps1" -ForegroundColor White
        Write-Host "3. Update Rust: rustup update" -ForegroundColor White
        Write-Host "4. Check for OS-specific build requirements" -ForegroundColor White
        if ((Get-OperatingSystem) -eq "Windows") {
            Write-Host "5. Ensure Visual Studio Build Tools are installed" -ForegroundColor White
        }
        
        return $false
    }
}

# Main execution function
function Main {
    Write-Host "=== QOA PowerShell Debug Build Script ===" -ForegroundColor Cyan
    Write-Host ""
    
    # Check prerequisites
    if (-not (Test-CargoInstallation)) {
        exit 1
    }
    
    # Show system information
    Show-SystemInfo
    
    # Check and install Intel MKL if needed
    Install-IntelMKLIfNeeded
    
    # Set up MKL environment variables
    Set-MKL-Environment
    
    # Show disk space
    Get-DiskSpace
    Write-Host ""
    
    # Set build environment
    Set-BuildEnvironment
    
    # Clean before building
    Invoke-CleanBuild
    
    # Run the debug build
    $buildSuccess = Invoke-BuildDebug
    
    Write-Host ""
    if ($buildSuccess) {
        Write-Host "=== Build script completed successfully ===" -ForegroundColor Green
        exit 0
    } else {
        Write-Host "=== Build script failed ===" -ForegroundColor Red
        exit 1
    }
}

Main
