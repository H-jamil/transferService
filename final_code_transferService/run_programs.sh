#!/bin/bash

# Ensure the script is run as root
if [ "$EUID" -ne 0 ]
then
    echo "Please run the script as sudo."
    exit
fi

# Check if correct number of arguments are provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <user_parallelism> <user_concurrency>"
    exit 1
fi

# Extract the current date and time for naming log file
current_time=$(date +"%Y%m%d_%H%M%S")
fileName="./P${1}_C${2}.log"
# fileName="./P${1}_C${2}.log"

./parallel_concurrent 192.5.87.228 ./logFileDir/${1}_${2}_${current_time}.log >>$fileName &
# Wait a bit for the C program to start up
sleep 3
/home/cc/.pyenv/shims/python client.py $1 $2
sleep 2
pkill -f parallel_concurrent
rm FILE*
