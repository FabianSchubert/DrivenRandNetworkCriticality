#!/bin/bash
# Basic while loop
counter=21
while [ $counter -le 30 ]
do
./RTRL_gains.py $counter
((counter++))
done
