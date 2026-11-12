@echo off
setlocal

rem Builds sadcs.exe: with the GPU variant (cuda preset) if the CUDA toolkit is installed,
rem otherwise the CPU variant only (clang preset). "build.bat cpu" always builds the CPU variant only.

set ROOT=%~dp0
set ROOT=%ROOT:~0,-1%

if /i "%~1"=="cpu" goto cpu
if not defined CUDA_PATH goto cpu

rem Ninja needs the MSVC environment for cl.exe, nvcc's host compiler.
rem No parenthesized blocks here: the ")" in %ProgramFiles(x86)% would end them.
where cl.exe >nul 2>nul
if not errorlevel 1 goto cuda
set VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe
for /f "usebackq delims=" %%i in (`call "%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set VSDIR=%%i
if not defined VSDIR echo Visual Studio with the C++ tools not found.& exit /b 1
call "%VSDIR%\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 exit /b %errorlevel%

:cuda
echo CUDA found: building the CPU and GPU variants (cuda preset)...
cmake --preset cuda -S "%ROOT%"
if errorlevel 1 exit /b %errorlevel%
cmake --build "%ROOT%\build-cuda"
if errorlevel 1 exit /b %errorlevel%

echo.
echo Build complete. Executable in: %ROOT%\build-cuda\bin\Release\
exit /b 0

:cpu
echo Building the CPU variant (clang preset)...
cmake --preset clang -S "%ROOT%"
if errorlevel 1 exit /b %errorlevel%
cmake --build "%ROOT%\build" --config Release
if errorlevel 1 exit /b %errorlevel%

echo.
echo Build complete. Executable in: %ROOT%\build\bin\Release\
echo (MSVC fallback: cmake --preset msvc, then cmake --build build-msvc --config Release)
exit /b 0
