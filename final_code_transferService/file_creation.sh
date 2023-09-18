#! /bin/bash

NUM=${1:-32}   ## default to 2 files
SIZE=${2:-1000000000} ## default to 1GB
n=0
while [ $n -lt $NUM ]
   do
	# Generate random data and save it to the file
	head -c $SIZE /dev/urandom > "/var/www/html/FILE$n"
	n=$(( $n + 1 ))
done
