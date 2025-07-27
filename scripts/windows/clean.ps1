# PowerShell Clean Script for QOA
# This script removes compiled build artifacts for the QOA project.
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

# Function to clean previous builds
function Invoke-CleanBuild {
    Write-Host "--- REMOVING ALL COMPILED BUILDS ---" -ForegroundColor Yellow
    Write-Host ""

    # Try cargo clean first
    try {
        cargo clean
        Write-Host "Cargo clean completed successfully." -ForegroundColor Green
    } catch {
        Write-Warning "Warning: Cargo clean failed. Attempting to remove 'target' directory manually."
        # If cargo clean fails, try to remove the target directory directly
        try {
            Remove-Item -Path "target" -Recurse -Force -ErrorAction Stop
            Write-Host "'target' directory removed successfully." -ForegroundColor Green
        } catch {
            Write-Error "Error: Failed to remove 'target' directory. Please check permissions or remove it manually."
            exit 1
        }
    }
    Write-Host ""
}

# Main execution function
function Main {
    Write-Host "=== QOA PowerShell Clean Script ===" -ForegroundColor Cyan
    Write-Host ""
    
    # Check prerequisites
    if (-not (Test-CargoInstallation)) {
        exit 1
    }
    
    # Perform the clean operation
    Invoke-CleanBuild
    
    Write-Host "=== Clean script completed ===" -ForegroundColor Cyan
}

# Run main function
Main
