@echo off
rem fitxategi honek qoa exekutatzen du windows-en (cmd): `run.cmd run examples\bell.qoa`. run.ps1-en berdina,
rem baina powershell-en exekuzio-politikarik gabe dabil. iturburu-kopia batean cargo-rekin konpilatzen du;
rem argitalpen-pakete batean bin\<helburua>\qoa.exe aukeratzen du (x64 edo ARM64), edo ondoko qoa.exe.
setlocal
set "here=%~dp0"
if exist "%here%Cargo.toml" (
    where cargo >nul 2>nul && goto build
)
set "arch=%PROCESSOR_ARCHITECTURE%"
if defined PROCESSOR_ARCHITEW6432 set "arch=%PROCESSOR_ARCHITEW6432%"
if /i "%arch%"=="ARM64" if exist "%here%bin\aarch64-pc-windows-msvc\qoa.exe" (
    "%here%bin\aarch64-pc-windows-msvc\qoa.exe" %*
    exit /b %errorlevel%
)
if exist "%here%bin\x86_64-pc-windows-msvc\qoa.exe" (
    "%here%bin\x86_64-pc-windows-msvc\qoa.exe" %*
    exit /b %errorlevel%
)
if exist "%here%qoa.exe" (
    "%here%qoa.exe" %*
    exit /b %errorlevel%
)
set "lang=%QOA_LANG%"
if /i "%lang:~0,2%"=="eu" (
    echo run.cmd: ez dago qoa.exe bitarrik %arch%-rako ^(bin\ karpetan eta run.cmd ondoan bilatu da^) 1>&2
) else (
    echo run.cmd: no qoa.exe for %arch% ^(looked in bin\ and next to run.cmd^) 1>&2
)
exit /b 1
:build
cargo build --release --quiet --manifest-path "%here%Cargo.toml"
if errorlevel 1 exit /b %errorlevel%
"%here%target\release\qoa.exe" %*
exit /b %errorlevel%
