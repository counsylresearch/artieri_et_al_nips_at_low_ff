## Supplemental Python analysis code from Artieri et al. (2016)

This repository contains all of the Python code required to reproduce the results found in: 

**Artieri CG, Haverty C, Evans EA, Goldberg JD, Haque IS, Yaron Y, and Muzzey D. 2016. Noninvasive Prenatal Screening at Low Fetal Fraction: Comparing Whole-Genome Sequencing and Single-Nucleotide Polymorphism Methods. BioRxiv. [LINK T.B.D.]**

The code within is shared under the **Creative Commons Attribution-NonCommercial 4.0 International Public License (CC BY-NC 4.0)** as explained in the [LICENSE.md](LICENSE.md) file.

All simulations and figures can be reproduced by running `code/run_analyses.sh`.

### Installing requirements

All Python requirements to reproduce the analysis found in the manuscript can be installed via [conda](https://www.continuum.io/content/conda-data-science). From within the directory containing this repository, run the following commands:

```
wget https://repo.continuum.io/miniconda/Miniconda2-4.2.12-Linux-x86_64.sh
bash Miniconda2-4.2.12-Linux-x86_64.sh -b -p miniconda2/
miniconda2/bin/conda env create -f requirements.yaml
source miniconda2/bin/activate artieri_et_al
```

To deactivate the conda environment run: 

```
source deactivate
```

### Contents

* `requirements.yaml`
	
	conda requirements file indicating the Python packages and corresponding versions used to perform the analysis in the manuscript. 

* `code/`

	Directory containing the individual Python scripts used to perform the simulation analysis and plot figure panels.
	
* `manuscript_figure_panels/`

	Directory containing the PNGs of the individual figure panels as plotted by the scripts in the `code/` directory.
	
* `simulation_output/`

	Directory containing the output of the WGS and SNP simulation scripts run from the `code/` directory.

### Contact

For questions or comments contact **research@counsyl.com**