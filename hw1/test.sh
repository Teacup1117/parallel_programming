#!/bin/bash

echo "Testing optimized hash performance on cases 1-21:"
echo "================================================"

for i in {1..25}; do
    file_num=$(printf "%02d" $i)
    echo -n "Case $i: "
    { time ./hw1 samples/${file_num}.txt >/dev/null 2>&1; } 2>&1 | grep real | awk '{print $2}' || echo "FAILED"
done

echo "================================================"
