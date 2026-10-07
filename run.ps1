# fitxategi honek qoa exekutatzen du windows-en (powershell): `.\run.ps1 run examples\bell.qoa`.
#
#   - iturburu-kopia batean (Cargo.toml eta cargo badaude): `cargo build --release` eta target\release\qoa.exe
#   - argitalpen-pakete batean: bin\<helburua>\qoa.exe makina honen arkitekturaren arabera (x64 edo ARM64;
#     ARM64-an x64 bertsioa ere badabil, emulatuta), edo script-aren ondoko qoa.exe
# argumentu guztiak qoa-ri pasatzen zaizkio bere horretan. windows-ek script-ak blokeatzen baditu:
#   powershell -ExecutionPolicy Bypass -File run.ps1 ...   (edo erabili run.cmd)
$ErrorActionPreference = 'Stop'
$here = $PSScriptRoot
$eu = "$env:QOA_LANG" -like 'eu*'

if ((Test-Path (Join-Path $here 'Cargo.toml')) -and (Get-Command cargo -ErrorAction SilentlyContinue)) {
    cargo build --release --quiet --manifest-path (Join-Path $here 'Cargo.toml')
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    $exe = Join-Path $here 'target\release\qoa.exe'
    if (-not (Test-Path $exe)) { $exe = Join-Path $here 'target/release/qoa' }
    & $exe @args
    exit $LASTEXITCODE
}

$arch = if ($env:PROCESSOR_ARCHITEW6432) { $env:PROCESSOR_ARCHITEW6432 } else { $env:PROCESSOR_ARCHITECTURE }
$targets = switch ($arch) {
    'ARM64' { @('aarch64-pc-windows-msvc', 'x86_64-pc-windows-msvc') }
    default { @('x86_64-pc-windows-msvc') }
}
$candidates = @($targets | ForEach-Object { Join-Path $here "bin\$_\qoa.exe" }) + @(Join-Path $here 'qoa.exe')
foreach ($exe in $candidates) {
    if (Test-Path $exe) {
        & $exe @args
        exit $LASTEXITCODE
    }
}

if ($eu) {
    Write-Host "run.ps1: ez dago qoa.exe bitarrik $arch-rako (bin\$($targets[0])\qoa.exe bilatu da)."
} else {
    Write-Host "run.ps1: no qoa.exe for $arch (looked for bin\$($targets[0])\qoa.exe)."
}
exit 1
