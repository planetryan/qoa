# This script runs the benchmarks using PowerShell.
# It automatically detects the number of CPU cores and sets RAYON_NUM_THREADS.

Write-Host ""
Write-Host "--- WARNING: BENCHMARKING MAY TAKE HOURS DEPENDING ON ACCURACY LEVEL. CHECK ACCURACY AND VERIFY BEFORE RUNNING. ---"
Write-Host ""
Write-Host "--- IGNORE THE TEST RESULTS, TO RUN TEST RUN 'cargo test' ---"
Write-Host ""

# Get number of logical processors (CPU threads) based on OS
$numCores = 0
if ($IsWindows) {
    # For Windows
    $numCores = [Environment]::ProcessorCount
} elseif ($IsOSX) {
    # For macOS
    $numCores = (sysctl -n hw.ncpu).Trim()
} elseif ($IsLinux) {
    # For Linux (using nproc, if available)
    # This might require 'nproc' to be installed or available in the PATH in some Linux PowerShell environments.
    # If not available, consider 'Get-CimInstance Win32_Processor | Measure-Object -Property NumberOfLogicalProcessors -Sum | Select-Object -ExpandProperty Sum' for WSL
    try {
        $numCores = (nproc).Trim()
    } catch {
        Write-Warning "nproc command not found. Falling back to default or manual setting."
        $numCores = 4 # Default to 4 if nproc isn't available
    }
} else {
    Write-Warning "Unknown operating system. Defaulting to 4 CPU threads."
    $numCores = 4 # Default if OS is not recognized
}


Write-Host "--- Running benchmark with ${numCores} threads ---"
Write-Host ""

# Set the RAYON_NUM_THREADS environment variable for the current session
$env:RAYON_NUM_THREADS = $numCores

# The '--' is not needed when running cargo directly like this, as it's not passing args to the test binary.
# The '&>/dev/null' equivalent for PowerShell is redirection to $null.
# We are intentionally not redirecting here so the user can see the benchmark output.
cargo bench
