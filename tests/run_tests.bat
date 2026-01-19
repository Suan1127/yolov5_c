@echo off
REM Integration test runner script for Windows

echo === YOLOv5 C Integration Tests ===
echo.

REM Check if weights file exists
if not exist "weights\weights.bin" (
    echo ERROR: weights\weights.bin not found
    echo Please run: python tools\export_yolov5s.py ^<yolov5s.pt^> --output weights/
    exit /b 1
)

REM Run integration test
echo Running integration test...
test_integration.exe

if %ERRORLEVEL% EQU 0 (
    echo.
    echo === All tests passed ===
    exit /b 0
) else (
    echo.
    echo === Tests failed ===
    exit /b 1
)
