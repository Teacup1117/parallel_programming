#!/bin/bash

# Simple test runner for one case
# Usage: ./run_test.sh [case_number]

CASE=${1:-00}
INPUT="../testcases/case${CASE}.in"
OUTPUT="case${CASE}.myout"
EXPECTED="../testcases/case${CASE}.out"

echo "Running test case ${CASE}..."
echo "Input: $INPUT"
echo "Output: $OUTPUT"
echo ""

./hw4 "$INPUT" "$OUTPUT"

if [ $? -eq 0 ]; then
    echo ""
    echo "Execution completed successfully!"
    
    if [ -f "$EXPECTED" ]; then
        echo ""
        echo "Comparing with expected output..."
        if diff "$OUTPUT" "$EXPECTED"; then
            echo "✅ Test PASSED - Output matches!"
        else
            echo "❌ Test FAILED - Output differs"
        fi
    fi
else
    echo "❌ Execution failed with error code $?"
fi
