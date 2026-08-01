# Chromosome Evolution in Angiosperms

This repository contains the code developed for my M.Sc. research on chromosome-number evolution in flowering plants.

The project examines whether rates of chromosome gain, loss, and duplication depend on the current chromosome number. Using ChromEvol, I compare constant, linear, and exponential evolutionary models across approximately 95 angiosperm families.

The repository covers the full analysis workflow, including data preparation, model configuration, large-scale execution on an HPC cluster, result parsing, statistical analysis, and simulation-based power evaluation.

Main components
Processing chromosome-count data and phylogenetic trees
Automated ChromEvol configuration and execution on an HPC cluster
Parsing and comparing model results
Analyzing biological and dataset features associated with model support
Simulation-based evaluation of model detection power


## Technical stack

Python, pandas, NumPy, SciPy, Biopython, Matplotlib, ChromEvol, Git, and Slurm-based high-performance computing.

The codebase includes command-line scripts, reusable configuration files, validation steps, run logging, and automated result processing. ChromEvol must be installed separately, and some project-specific paths and cluster settings must be configured before use.

## Project status

The empirical model-comparison pipeline is implemented. Current work focuses on simulation-based power analysis and on identifying the biological and statistical factors that influence model detection.

## Author

**Noy Bandel**
M.Sc. student, Tel Aviv University

Supervised by **Prof. Itay Mayrose** and **Prof. Tal Pupko**.
