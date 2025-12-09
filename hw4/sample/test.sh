#!/bin/bash

# Test script for hw4
# Usage: ./test.sh

echo "======================================"
echo "Testing HW4 Bitcoin Miner"
echo "======================================"

# Compile if needed
if [ ! -f hw4 ]; then
    echo "Compiling..."
    make clean
    make
    if [ $? -ne 0 ]; then
        echo "Compilation failed!"
        exit 1
    fi
fi

# Test each case
for i in {0..3}; do
    case_num=$(printf "%02d" $i)
    input="../testcases/case${case_num}.in"
    expected="../testcases/case${case_num}.out"
    output="case${case_num}.myout"
    
    if [ ! -f "$input" ]; then
        echo "Test case $input not found, skipping..."
        continue
    fi
    
    echo ""
    echo "======================================"
    echo "Testing case${case_num}"
    echo "======================================"
    
    # Run the program
    time ./hw4 "$input" "$output"
    
    if [ $? -ne 0 ]; then
        echo "❌ Case ${case_num}: Runtime error"
        continue
    fi
    
    # Compare output
    if [ -f "$expected" ]; then
        if diff -q "$output" "$expected" > /dev/null; then
            echo "✅ Case ${case_num}: PASSED"
        else
            echo "❌ Case ${case_num}: FAILED (output mismatch)"
            echo "Expected:"
            head -20 "$expected"
            echo ""
            echo "Got:"
            head -20 "$output"
        fi
    else
        echo "⚠️  Case ${case_num}: No expected output file, generated output:"
        head -20 "$output"
    fi
done

echo ""
echo "======================================"
echo "All tests completed"
echo "======================================"
