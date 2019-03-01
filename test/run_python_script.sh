#!/bin/bash
echo ${1}.py
parallel -k python ::: ${1}.py ::: 0 1 2 3 >> ${1}.out 
