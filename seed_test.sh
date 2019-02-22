#!/bin/bash

for seed_index in {0..9..1}; do
	python B2_exp.py $seed_index
done 
