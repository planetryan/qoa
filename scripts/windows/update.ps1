# PowerShell Update Script for QOA
# This script updates the Rust toolchain and all installed Cargo crates.
# Works on Windows PowerShell, PowerShell Core, and cross-platform environments

# Exit immediately if a command exits with a non-zero status
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
            return [System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture
        } else {
            # Windows PowerShell
            return $env:PROCESSOR_ARCHITECTURE
        }
    } catch {
        return "Unknown"
    }
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

# Function to update Rust toolchain
function Invoke-UpdateRustToolchain {
    Write-Host "--- UPDATING RUST TOOLCHAIN (RUSTUP UPDATE) ---" -ForegroundColor Yellow
    Write-Host ""
    try {
        rustup update -ErrorAction Stop
        Write-Host "Rust toolchain updated successfully." -ForegroundColor Green
    } catch {
        Write-Error "Error: Failed to update Rust toolchain: $_"
        exit 1
    }
    Write-Host ""
}

# Function to update Cargo crates
function Invoke-UpdateCargoCrates {
    Write-Host "--- UPDATING CARGO CRATES (CARGO UPDATE) ---" -ForegroundColor Yellow
    Write-Host ""
    try {
        cargo update -ErrorAction Stop
        Write-Host "Cargo crates updated successfully." -ForegroundColor Green
    } catch {
        Write-Error "Error: Failed to update Cargo crates: $_"
        exit 1
    }
    Write-Host ""
}

# Main execution function
function Main {
    Write-Host "=== QOA PowerShell Update Script ===" -ForegroundColor Cyan
    Write-Host ""
    
    # Check prerequisites
    if (-not (Test-CargoInstallation)) {
        exit 1
    }
    
    # Show system information
    Show-SystemInfo
    
    # Check disk space
    Get-DiskSpace
    Write-Host ""
    
    # Update Rust toolchain
    Invoke-UpdateRustToolchain
    
    # Update Cargo crates
    Invoke-UpdateCargoCrates
    
    Write-Host "=== Update script completed ===" -ForegroundColor Cyan
}

# Run main function
Main
