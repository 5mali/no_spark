#!/bin/bash
#### PWD should be ~/../no_spark
for name in "$@"
do
	if [ -e ${PWD}/seed_output/${name}.out ]
	then
		rm ${PWD}/seed_output/${name}.out
	fi

	for seed_index in {0..9} 
	do
		python ${PWD}/py_scripts/${name}.py ${seed_index} >> ${PWD}/seed_output/${name}.out 
	done 
done
