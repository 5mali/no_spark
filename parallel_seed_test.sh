#!/bin/bash

for name in "$@"
do
	if [ -e ./seed_output/${name}.out ]
	then
		rm ./seed_output/${name}.out
	fi
	
	parallel -k python ::: ./py_scripts/${name}.py ::: 0 1 2 3 4 5 6 7 8 9 > ./seed_output/${name}.out


#	for seed_index in {0..9} 
#	do
##		python ./py_scripts/${name}.py ${seed_index} >> ./seed_output/${name}.out 
#	done 
done
