# MomentAnalysis

Code to reproduce and make all plots in 23XX.XXXX

All models and weights are saved.


# Usage

To begin usage, first create and edit the file ```config.py``` to have the following dictionary structure:

```
\# Base directory
base_dir = "PATH-TO-ANALYSIS-DIRECTORY"

\# Path to cache directory
cache_dirs = {"qg" : "PATH-TO-QG-DATASET/.energyflow",
             "top" : "PATH-TO-TOP-DATASET/.top"}
```

The ```cache_dirs``` are the paths to the datasets. They will be automatically downloaded if the data files are not found. The ```base_dir``` is the path to the analysis directory where all results, models, and plots will be saved.

## Dependencies

The following python packages are required:

* energyflow
* toploader
* tensorflow
* Standard python libaries: numpy, matplotlib, scipy, etc.