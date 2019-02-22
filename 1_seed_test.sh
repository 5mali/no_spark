#!/bin/bash

for name in "$@"
do
	if [ -e ${name}.out ]
	then
		rm ${name}.out
	fi

	for seed_index in {0..9} 
	do
		python ${name}.py ${seed_index} | tee -a ${name}.out 
	done 
done
