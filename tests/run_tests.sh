#!/bin/bash
# Integration test runner script

echo "=== YOLOv5 C Integration Tests ==="
echo ""

# Check if weights file exists
if [ ! -f "weights/weights.bin" ]; then
    echo "ERROR: weights/weights.bin not found"
    echo "Please run: python tools/export_yolov5s.py <yolov5s.pt> --output weights/"
    exit 1
fi

# Run integration test
echo "Running integration test..."
./test_integration

if [ $? -eq 0 ]; then
    echo ""
    echo "=== All tests passed ==="
    exit 0
else
    echo ""
    echo "=== Tests failed ==="
    exit 1
fi
