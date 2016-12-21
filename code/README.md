## Code directory

This directory contains the Python code used to generate the data and figures from Artieri et al. (2016).

`run_analyses.sh` is a bash script that will run all analyses performed in the manuscript and generate the figure panels. By default, all scripts will run with the `-e` option, which will seed the random number generator such that the output and figures presented in the manuscript are reproduced exactly. Note that the manuscript used 10,000 permutations of the data, which will take a _very_ long time if the simulation scripts are run on a single processor. By default, `run_analyses.sh` will use all cores available.

### Additional contents:

* `snp_method_simulation.py`

	Simulates SNP method samples according to specified parameters and outputs calculated log-odds ratios. Run `python snp_method_simulation.py -h` for a description of available options. Note that option defaults are those used in the main manuscript.
	
* `wgs_method_simulation.py`

	Simulates the WGS method as performed on chromosomes 21, 18, and 13 according to specified parameters and outputs either sensitivity/specificity values or z-scores. Run `python wgs_method_simulation.py -h` for a description of available options. Note that option defaults are those used in the main manuscript.
	
* `figure_code.py`

	Contains the code required to generate all main figure panels in the manuscript. If run as a script, regenerates PNG files in `manuscript_figure_panels/` directory from the output of the simulations in `../simulation_output/`. 

* `supplemental_figure_code.py`

	Same as `figure_code.py`, but for the supplemental figure panels in the supplemental materials.
	
* `prob_child_genotype.csv`

	Contains the fetal genotype probabilities given the maternal and paternal genotypes as explained in: Rabinowitz M, Gemelos G, Banjevic M, Ryan A, Demko Z, Hill M, et al. Methods for non-invasive prenatal ploidy calling. US Patent Application 14/179,399, 2014. p. 32 [0343]