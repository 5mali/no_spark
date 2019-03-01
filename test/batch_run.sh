#!/bin/bash

#parallel --header : --results ${PWD} -k ::: ./pyprog ::: seed_out ${@}
parallel ./run_python_script.sh ::: ${@}

