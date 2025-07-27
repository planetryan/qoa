# QOA PowerShell Benchmark Script
# Works on Windows, Linux, and macOS with PowerShell Core

param(
    [switch]$SkipMKL,
    [int]$Threads = 0
)

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

# Function to install Intel MKL based on operating system
function Install-IntelMKL {
    $os = Get-OperatingSystem
    
    Write-Host "Detected OS: $os" -ForegroundColor Green
    Write-Host "Installing Intel MKL Full package..." -ForegroundColor Yellow
    
    switch ($os) {
        "Windows" {
            Write-Host "Installing Intel MKL Full on Windows..." -ForegroundColor Yellow
            
            # Check if chocolatey is available
            if (Get-Command choco -ErrorAction SilentlyContinue) {
                Write-Host "Installing Intel MKL Full using Chocolatey..." -ForegroundColor Yellow
                try {
                    choco install intel-mkl intel-oneapi-mkl -y
                    Write-Host "Intel MKL Full installed successfully via Chocolatey." -ForegroundColor Green
                } catch {
                    Write-Warning "Failed to install Intel MKL Full via Chocolatey: $_"
                }
            }
            # Check if winget is available (Windows 10 1709+)
            elseif (Get-Command winget -ErrorAction SilentlyContinue) {
                Write-Host "Installing Intel MKL Full using winget..." -ForegroundColor Yellow
                try {
                    winget install Intel.oneAPI.MKL
                    Write-Host "Intel MKL Full installed successfully via winget." -ForegroundColor Green
                } catch {
                    Write-Warning "Failed to install Intel MKL Full via winget: $_"
                }
            }
            # Check if scoop is available
            elseif (Get-Command scoop -ErrorAction SilentlyContinue) {
                Write-Host "Installing Intel MKL Full using Scoop..." -ForegroundColor Yellow
                try {
                    scoop bucket add extras
                    scoop install intel-mkl intel-oneapi-mkl
                    Write-Host "Intel MKL Full installed successfully via Scoop." -ForegroundColor Green
                } catch {
                    Write-Warning "Failed to install Intel MKL Full via Scoop: $_"
                }
            }
            else {
                Write-Warning "No package manager found (chocolatey, winget, or scoop)."
                Write-Host "Please install Intel MKL Full manually from:" -ForegroundColor Yellow
                Write-Host "https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html" -ForegroundColor Cyan
                Write-Host "Make sure to install the complete Intel oneAPI Math Kernel Library." -ForegroundColor Yellow
            }
        }
        
        "Linux" {
            Write-Host "Installing Intel MKL Full on Linux..." -ForegroundColor Yellow
            
            # Try different package managers
            if (Get-Command apt -ErrorAction SilentlyContinue) {
                Write-Host "Using apt package manager..." -ForegroundColor Yellow
                try {
                    Invoke-Expression "sudo apt update && sudo apt install -y intel-mkl-full libmkl-dev libmkl-rt intel-oneapi-mkl"
                    Write-Host "Intel MKL Full installed successfully via apt." -ForegroundColor Green
                } catch {
                    Write-Warning "Failed to install Intel MKL Full via apt: $_"
                }
            }
            elseif (Get-Command dnf -ErrorAction SilentlyContinue) {
                Write-Host "Using dnf package manager..." -ForegroundColor Yellow
                try {
                    Invoke-Expression "sudo dnf install -y intel-mkl-full intel-mkl-devel intel-oneapi-mkl"
                    Write-Host "Intel MKL Full installed successfully via dnf." -ForegroundColor Green
                } catch {
                    Write-Warning "Failed to install Intel MKL Full via dnf: $_"
                }
            }
            elseif (Get-Command yum -ErrorAction SilentlyContinue) {
                Write-Host "Using yum package manager..." -ForegroundColor Yellow
                try {
                    Invoke-Expression "sudo yum install -y intel-mkl-full intel-mkl-devel intel-oneapi-mkl"
                    Write-Host "Intel MKL Full installed successfully via yum." -ForegroundColor Green
                } catch {
                    Write-Warning "Failed to install Intel MKL Full via yum: $_"
                }
            }
            elseif (Get-Command pacman -ErrorAction SilentlyContinue) {
                Write-Host "Using pacman package manager..." -ForegroundColor Yellow
                try {
                    Invoke-Expression "sudo pacman -S --noconfirm intel-mkl intel-mkl-static"
                    Write-Host "Intel MKL Full installed successfully via pacman." -ForegroundColor Green
                } catch {
                    Write-Warning "Failed to install Intel MKL Full via pacman: $_"
                }
            }
            elseif (Get-Command zypper -ErrorAction SilentlyContinue) {
                Write-Host "Using zypper package manager..." -ForegroundColor Yellow
                try {
                    Invoke-Expression "sudo zypper install -y intel-mkl-full intel-mkl-devel"
                    Write-Host "Intel MKL Full installed successfully via zypper." -ForegroundColor Green
                } catch {
                    Write-Warning "Failed to install Intel MKL Full via zypper: $_"
                }
            }
            else {
                Write-Warning "No supported package manager found."
                Write-Host "Please install Intel MKL Full manually from:" -ForegroundColor Yellow
                Write-Host "https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html" -ForegroundColor Cyan
                Write-Host "Required packages: intel-mkl-full, intel-mkl-devel, libmkl-rt" -ForegroundColor Yellow
            }
        }
        
        "macOS" {
            Write-Host "Installing Intel MKL Full on macOS..." -ForegroundColor Yellow
            
            if (Get-Command brew -ErrorAction SilentlyContinue) {
                Write-Host "Using Homebrew..." -ForegroundColor Yellow
                try {
                    Invoke-Expression "brew install intel-mkl intel-oneapi-mkl"
                    Write-Host "Intel MKL Full installed successfully via Homebrew." -ForegroundColor Green
                } catch {
                    Write-Warning "Failed to install Intel MKL Full via Homebrew: $_"
                }
            }
            elseif (Get-Command port -ErrorAction SilentlyContinue) {
                Write-Host "Using MacPorts..." -ForegroundColor Yellow
                try {
                    Invoke-Expression "sudo port install intel-mkl +universal"
                    Write-Host "Intel MKL Full installed successfully via MacPorts." -ForegroundColor Green
                } catch {
                    Write-Warning "Failed to install Intel MKL Full via MacPorts: $_"
                }
            }
            else {
                Write-Warning "No package manager found (Homebrew or MacPorts)."
                Write-Host "Please install Homebrew from: https://brew.sh/" -ForegroundColor Yellow
                Write-Host "Or install Intel MKL Full manually from:" -ForegroundColor Yellow
                Write-Host "https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html" -ForegroundColor Cyan
            }
        }
        
        default {
            Write-Warning "Unknown operating system: $os"
            Write-Host "Please install Intel MKL Full manually from:" -ForegroundColor Yellow
            Write-Host "https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html" -ForegroundColor Cyan
        }
    }
    
    Write-Host ""
    Write-Host "Note: Intel MKL Full includes all necessary libraries for optimal performance:" -ForegroundColor Green
    Write-Host "- BLAS, LAPACK, ScaLAPACK" -ForegroundColor White
    Write-Host "- Sparse Solvers and Fast Fourier Transforms" -ForegroundColor White  
    Write-Host "- Vector Math Library and Statistical Functions" -ForegroundColor White
    Write-Host "- Deep Neural Network primitives" -ForegroundColor White
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

# Function to get CPU core count based on operating system
function Get-CPUCoreCount {
    $os = Get-OperatingSystem
    
    try {
        switch ($os) {
            "Windows" {
                # Use .NET method for Windows (works in both Windows PowerShell and PowerShell Core)
                return [Environment]::ProcessorCount
            }
            
            "Linux" {
                # Try multiple methods for Linux
                if (Get-Command nproc -ErrorAction SilentlyContinue) {
                    $result = Invoke-Expression "nproc" 2>$null
                    if ($result -and $result -match '^\d+$') {
                        return [int]$result
                    }
                }
                
                # Fallback: read /proc/cpuinfo
                if (Test-Path "/proc/cpuinfo") {
                    $cpuInfo = Get-Content "/proc/cpuinfo" -ErrorAction SilentlyContinue
                    $processorCount = ($cpuInfo | Where-Object { $_ -match "^processor" }).Count
                    if ($processorCount -gt 0) {
                        return $processorCount
                    }
                }
                
                # Last resort: try lscpu
                if (Get-Command lscpu -ErrorAction SilentlyContinue) {
                    $lscpuOutput = Invoke-Expression "lscpu" 2>$null
                    $cpuLine = $lscpuOutput | Where-Object { $_ -match "CPU\(s\):" } | Select-Object -First 1
                    if ($cpuLine -match "CPU\(s\):\s*(\d+)") {
                        return [int]$matches[1]
                    }
                }
                
                return 4  # Default fallback
            }
            
            "macOS" {
                # Use sysctl for macOS
                if (Get-Command sysctl -ErrorAction SilentlyContinue) {
                    $result = Invoke-Expression "sysctl -n hw.ncpu" 2>$null
                    if ($result -and $result -match '^\d+$') {
                        return [int]$result
                    }
                }
                
                # Fallback for macOS
                if (Get-Command system_profiler -ErrorAction SilentlyContinue) {
                    $profilerOutput = Invoke-Expression "system_profiler SPHardwareDataType" 2>$null
                    $coreLine = $profilerOutput | Where-Object { $_ -match "Total Number of Cores:" }
                    if ($coreLine -match "Total Number of Cores:\s*(\d+)") {
                        return [int]$matches[1]
                    }
                }
                
                return 4  # Default fallback
            }
            
            default {
                Write-Warning "Unknown operating system. Using default core count."
                return 4
            }
        }
    }
    catch {
        Write-Warning "Error detecting CPU cores: $_. Using default."
        return 4
    }
}

# Main execution starts here
Write-Host "=== QOA PowerShell Benchmark Script ===" -ForegroundColor Cyan
Write-Host ""

# Install Intel MKL unless skipped
if (-not $SkipMKL) {
    Write-Host "Checking and installing Intel MKL..." -ForegroundColor Yellow
    Install-IntelMKL
    Write-Host ""
} else {
    Write-Host "Skipping Intel MKL installation as requested." -ForegroundColor Yellow
    Write-Host ""
}

# Set up MKL environment variables
Set-MKL-Environment

# Get CPU core count
if ($Threads -eq 0) {
    $numCores = Get-CPUCoreCount
} else {
    $numCores = $Threads
    Write-Host "Using manually specified thread count: $numCores" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "--- WARNING: BENCHMARKING MAY TAKE HOURS DEPENDING ON ACCURACY LEVEL. CHECK ACCURACY AND VERIFY BEFORE RUNNING. ---" -ForegroundColor Red
Write-Host ""
Write-Host "--- IGNORE THE TEST RESULTS, TO RUN TEST RUN 'cargo test' ---" -ForegroundColor Yellow
Write-Host ""
Write-Host "--- Running benchmark with $numCores threads ---" -ForegroundColor Green
Write-Host ""

# Set environment variable and run benchmark
$env:RAYON_NUM_THREADS = $numCores.ToString()

# Run the benchmark
try {
    cargo bench --release
} catch {
    Write-Error "Failed to run cargo bench: $_"
    exit 1
}
