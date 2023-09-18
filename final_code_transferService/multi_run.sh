#!/bin/bash

# Define the arrays for run_no, parallelism_values, and concurrency_values
run_no=(1)
parallelism_values=(1 2)
concurrency_values=(1 2)

# Iterate over the arrays
for run in "${run_no[@]}"; do
    for parallelism in "${parallelism_values[@]}"; do
        for concurrency in "${concurrency_values[@]}"; do
            # Execute the script with sudo for each combination
            echo "Running: Run Number=$run, Parallelism=$parallelism, Concurrency=$concurrency"
            sudo ./run_programs.sh $parallelism $concurrency
            # Sleep for 1 second
            sleep 5
        done
    done
done
