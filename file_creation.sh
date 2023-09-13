#! /bin/bash

NUM=${1:-2}   ## default to 2 files
SIZE=${2:-1000000000} ## default to 1GB
n=0
while [ $n -lt $NUM ]
do
  printf "%0${SIZE}d" 0 > "./files_to_send/FILE$n"
  n=$(( $n + 1 ))
done
