#!/bin/bash

echo "Detailed performance analysis for cases 1-21:"
echo "=============================================="
printf "%-4s %-10s %-10s %-15s %s\n" "Case" "Time(s)" "Nodes" "Solution Len" "Status"
echo "------------------------------------------------------"

total_time=0
total_nodes=0
successful=0

for i in {1..21}; do
    file_num=$(printf "%02d" $i)
    printf "%-4s " "$i"
    
    # Run with time measurement and capture output
    start_time=$(date +%s.%N)
    output=$(timeout 30s ./hw1 samples/${file_num}.txt 2>&1)
    exit_code=$?
    end_time=$(date +%s.%N)
    
    if [ $exit_code -eq 0 ]; then
        elapsed=$(echo "$end_time - $start_time" | bc -l 2>/dev/null || python3 -c "print($end_time - $start_time)")
        
        # Extract information from output
        nodes=$(echo "$output" | grep "solved in" | awk '{print $4}')
        solution=$(echo "$output" | grep -v "solved in" | grep -v "Solved in" | tail -1)
        sol_len=${#solution}
        
        printf "%-10.3f %-10s %-15s %s\n" "$elapsed" "$nodes" "$sol_len" "✓"
        
        total_time=$(echo "$total_time + $elapsed" | bc -l 2>/dev/null || python3 -c "print($total_time + $elapsed)")
        if [[ "$nodes" =~ ^[0-9]+$ ]]; then
            total_nodes=$((total_nodes + nodes))
        fi
        ((successful++))
    elif [ $exit_code -eq 124 ]; then
        printf "%-10s %-10s %-15s %s\n" "TIMEOUT" "-" "-" "✗"
    else
        printf "%-10s %-10s %-15s %s\n" "FAILED" "-" "-" "✗"
    fi
done

echo "=============================================="
echo "Summary:"
echo "Successful cases: $successful/21"
if [ $successful -gt 0 ]; then
    avg_time=$(echo "scale=3; $total_time / $successful" | bc -l 2>/dev/null || python3 -c "print(round($total_time / $successful, 3))")
    avg_nodes=$(echo "scale=0; $total_nodes / $successful" | bc -l 2>/dev/null || python3 -c "print(int($total_nodes / $successful))")
    printf "Total time: %.3fs\n" "$total_time"
    printf "Average time per case: %.3fs\n" "$avg_time"
    printf "Total nodes explored: %d\n" "$total_nodes"
    printf "Average nodes per case: %d\n" "$avg_nodes"
fi
