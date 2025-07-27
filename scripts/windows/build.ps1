# PowerShell Build Script for QOA
# Works on Windows PowerShell, PowerShell Core, and cross-platform environments

param(
    [ValidateSet("release", "debug")]
    [string]$BuildType = "release",
    
    [switch]$Clean,
    
    [switch]$Help,
    
    [switch]$Verbose,
    
    [string]$Target = ""
)

# Function to show usage information
function Show-Usage {
    Write-Host "=== PowerShell Build Script for QOA ===" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "SYNTAX:" -ForegroundColor Yellow
    Write-Host "    .\build.ps1 [-BuildType <release|debug>] [-Clean] [-Target <target>] [-Verbose] [-Help]"
    Write-Host ""
    Write-Host "PARAMETERS:" -ForegroundColor Yellow
    Write-Host "    -BuildType    Build type: 'release' (default) or 'debug'"
    Write-Host "    -Clean        Clean previous builds before building"
    Write-Host "    -Target       Specify Rust target triple (e.g., 'x86_64-pc-windows-msvc')"
    Write-Host "    -Verbose      Show detailed build output"
    Write-Host "    -Help         Show this help message"
    Write-Host ""
    Write-Host "EXAMPLES:" -ForegroundColor Yellow
    Write-Host "    .\build.ps1                           # Build release version"
    Write-Host "    .\build.ps1 -BuildType debug          # Build debug version"
    Write-Host "    .\build.ps1 -Clean                    # Clean then build release"
    Write-Host "    .\build.ps1 -Clean -BuildType debug   # Clean then build debug"
    Write-Host "    .\build.ps1 -Target x86_64-pc-windows-gnu  # Build for specific target"
    Write-Host "    .\build.ps1 -Verbose                  # Build with detailed output"
}

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
        Write-Error "Cargo is not installed or not in PATH."
        Write-Host "Please install Rust and Cargo from: https://rustup.rs/" -ForegroundColor Yellow
        return $false
    }
}

# Function to get system architecture
function Get-SystemArchitecture {
    try {
        if ($PSVersionTable.PSVersion.Major -ge 6) {
            # PowerShell Core
            return [System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture
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
                Write-Warning "Unknown Linux package manager. Please install Intel MKL Full manually."
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
                Write-Warning "Homebrew not found. Please install Intel MKL Full manually."
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
            Write-Warning "Intel MKL Full may not be available for $os."
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
    
    # Set RUSTFLAGS to avoid SVML linking issues
    $currentRustFlags = $env:RUSTFLAGS
    $newRustFlags = "-C target-cpu=generic"
    
    if ($currentRustFlags) {
        $env:RUSTFLAGS = "$currentRustFlags $newRustFlags"
    } else {
        $env:RUSTFLAGS = $newRustFlags
    }
    
    Write-Host "Build Environment Configuration:" -ForegroundColor Yellow
    Write-Host "RUSTFLAGS: $env:RUSTFLAGS" -ForegroundColor White
    
    # Set additional environment variables based on OS
    switch ($os) {
        "Windows" {
            # Windows-specific settings
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
            # Linux-specific settings in PowerShell Core
            Write-Host "Running on Linux with PowerShell Core" -ForegroundColor White
        }
        
        "macOS" {
            # macOS-specific settings in PowerShell Core
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

# Function to clean previous builds
function Invoke-CleanBuild {
    if ($Clean) {
        Write-Host "--- CLEANING PREVIOUS BUILD ---" -ForegroundColor Yellow
        try {
            cargo clean
            Write-Host "Previous build cleaned successfully." -ForegroundColor Green
        } catch {
            Write-Warning "Failed to clean previous build: $_"
        }
        Write-Host ""
    }
}

# Function to run the build
function Invoke-Build {
    param(
        [string]$Type,
        [string]$TargetTriple = "",
        [bool]$ShowVerbose = $false
    )
    
    $startTime = Get-Date
    
    Write-Host "--- COMPILING $($Type.ToUpper()) BUILD FOR QOA ---" -ForegroundColor Cyan
    Write-Host ""
    
    # Construct cargo command
    $cargoArgs = @("build")
    
    if ($Type -eq "release") {
        $cargoArgs += "--release"
    }
    
    if ($TargetTriple) {
        $cargoArgs += "--target", $TargetTriple
        Write-Host "Target: $TargetTriple" -ForegroundColor White
    }
    
    if ($ShowVerbose) {
        $cargoArgs += "--verbose"
    }
    
    $cargoCommand = "cargo " + ($cargoArgs -join " ")
    Write-Host "Running: $cargoCommand" -ForegroundColor White
    Write-Host ""
    
    # Execute the build
    try {
        & cargo @cargoArgs
        
        if ($LASTEXITCODE -eq 0) {
            $endTime = Get-Date
            $duration = [math]::Round(($endTime - $startTime).TotalSeconds, 2)
            
            Write-Host ""
            Write-Host "--- BUILD COMPLETED SUCCESSFULLY ---" -ForegroundColor Green
            Write-Host "Build type: $Type" -ForegroundColor White
            Write-Host "Build time: $duration seconds" -ForegroundColor White
            
            # Show binary information
            $binaryPath = if ($Type -eq "release") { 
                if ($TargetTriple) { "target\$TargetTriple\release\qoa.exe" } else { "target\release\qoa.exe" }
            } else { 
                if ($TargetTriple) { "target\$TargetTriple\debug\qoa.exe" } else { "target\debug\qoa.exe" }
            }
            
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
        Write-Host "Build type: $Type" -ForegroundColor White
        Write-Host "Error: $_" -ForegroundColor Red
        Write-Host ""
        Write-Host "Common solutions:" -ForegroundColor Yellow
        Write-Host "1. Check if all dependencies are installed" -ForegroundColor White
        Write-Host "2. Try running: .\build.ps1 -Clean" -ForegroundColor White
        Write-Host "3. Update Rust: rustup update" -ForegroundColor White
        Write-Host "4. Check for Windows-specific build requirements" -ForegroundColor White
        Write-Host "5. Ensure Visual Studio Build Tools are installed" -ForegroundColor White
        
        return $false
    }
}

# Function to show available Rust targets
function Show-AvailableTargets {
    Write-Host "Available Rust targets:" -ForegroundColor Yellow
    try {
        rustup target list --installed 2>$null
    } catch {
        Write-Warning "Could not list installed targets. Run 'rustup target list' manually."
    }
    Write-Host ""
}

# Main execution function
function Main {
    # Show help if requested
    if ($Help) {
        Show-Usage
        return
    }
    
    Write-Host "=== PowerShell Build Script for QOA ===" -ForegroundColor Cyan
    Write-Host ""
    
    # Check prerequisites
    if (-not (Test-CargoInstallation)) {
        exit 1
    }
    
    # Show system information
    Show-SystemInfo
    
    # Check and install Intel MKL if needed
    Install-IntelMKLIfNeeded
    
    # Show disk space
    Get-DiskSpace
    Write-Host ""
    
    # Set build environment
    Set-BuildEnvironment
    
    # Show available targets if target parameter was used
    if ($Target) {
        Show-AvailableTargets
    }
    
    # Clean if requested
    Invoke-CleanBuild
    
    # Run the build
    $buildSuccess = Invoke-Build -Type $BuildType -TargetTriple $Target -ShowVerbose $Verbose
    
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
