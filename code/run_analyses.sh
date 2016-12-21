#!/bin/bash
#This shell-script contains the commands used to generate the data analyzed
#in the manuscript.

# Build list of simulations to run
rm -f Makefile.simulations
bash gen_makefile.sh > Makefile.simulations

# Re-run the simulations
NUM_CORES=`getconf _NPROCESSORS_ONLN`
if [ -z NUM_CORES ]; then
    NUM_CORES=1
fi

echo "Starting simulations over $NUM_CORES parallel jobs"
make -j $NUM_CORES -f Makefile.simulations

#Generate figure panels
if [ $? -eq 0 ]; then
    python figure_code.py -e
    python supplemental_figure_code.py -e
else
    echo "Error running simulations; check error logs"
fi
