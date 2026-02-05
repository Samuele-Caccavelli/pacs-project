# Diagnostic to DOD with application to time-dependent problems

## Installation

This section provides step-by-step instructions to set up the computational environment. We recommend using **conda** to manage dependencies and create an isolated virtual environment. The utilized Python version is 3.10.

### Clone the Repository
```bash
git clone https://github.com/Samuele-Caccavelli/pacs-project
cd pacs-project
```

### Create the environment
Use the provided `environment.yml` file to handle the installation of all necessary Python modules.
```bash
conda env create -f environment.yml
```

### Activate the environment
```bash
conda activate fenics-env
```

### Data initialization
Setup the required test cases datasets with the provided script that should be run in the root folder.
```bash
python data_setup.py
```
### Other installation methods
If other installation methods are preferred, we provide the links to [dlroms](https://github.com/NicolaRFranco/dlroms) and [FEniCS](https://fenicsproject.org/download/archive/). To obtain the required version of the `dlroms` installation, we also provide the hash of the corresponding commit (@a84a2b78bb110752b126ded42ed81d5010e8ebad).

## Troubleshooting
If during the installation `pip` sees Python modules already present in the user global Python environment, the correct installation of the correct version specific modules could be prevented. 
To avoid this problem, you can setup the `conda` environment to only look at modules installed in the environment itself.
Once inside the `conda` environment, type the following.

```bash
conda env config vars set PYTHONNOUSERSITE=1
```

Then reset the environment by deactivating and reactivating it.

```bash
conda deactivate
conda activate fenics-env
```

Finally reinstall all the required modules.

```bash
pip install git+https://github.com/NicolaRFranco/dlroms.git@a84a2b7
```

## Test

A small test to check the installation is provided inside the `test/` folder.

In the notebook `minimal_working_example.ipynb` a quick sensitivity analysis is run on the Gaussian test case. This could take 1-2 minutes for a full execution.

Once the full notebook has been run, you will have:

* **The expected results**, provided in the notebook itself;
* **An automatic check**, that looks at the computed scores and checks if they are physically consistent.

## Folder structure
```
pacs-project/
├── code/                   # Contains source code for the newly implemented modules
│   ├── PODlib.py                     # Source code for the weighted_POD class
│   └── variabilitylib.py             # Source code for the LocalBasis class
├── data/                   # Contains dataset and relative meshes
├── notebooks/              # Contains notebooks that carry out the complete pipelines
│   ├── pipeline_gaussian.ipynb       # Notebook with the pipeline for the Gaussian test case
│   ├── pipeline_nstokes.ipynb        # Notebook with the pipeline for the Navier-Stokes test case
│   └── dod_compatibility.ipynb       # Notebook to show compatibility between new and old classes
├── results/                # Destination folder for all trained DOD models
│   └── Test_DOD.npz                  # Trained DOD modeled
├── test/                   # Contains the test notebook
│   └── minimal_working_example.ipynb # Notebook with a test to be run
├── README.md               # README file
├── data_setup.py           # Datasets set up file
└── environment.yml         # Environment set up file
```