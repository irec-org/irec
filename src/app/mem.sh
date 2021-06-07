#!/bin/sh
while true; do 

				ramusage=$(free | awk '/Mem/{printf("RAM Usage: %.2f\n"), $3/$2*100}'| awk '{print $3}')

				echo "Memory Current Usage is: $ramusage%"
				if (( $(echo "$ramusage > 92" |bc -l) )); then
								#SUBJECT="ATTENTION: Memory Utilization is High on $(hostname) at $(date)"
								#MESSAGE="/tmp/Mail.out"
								#TO="daygeek@gmail.com"

								killall python3
								killall python
					#			break
				fi
				sleep 0.4
done
