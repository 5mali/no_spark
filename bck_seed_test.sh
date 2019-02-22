#!/bin/bash
name='B1'
for seed_index in {0..9} 
do
	echo ${seed_index}
	python ${name}.py ${seed_index} | tee -a ${name}.out 
done 
