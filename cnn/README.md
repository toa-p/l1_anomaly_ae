# L1 Anomaly AE

## Building ⚙️

First, clone the repository:

    $ git clone git@github.com:mpp-hep/L1models.git
    $ cd L1models/l1_anomaly_ae

Then set up the environment (I suggest to use [miniconda3](https://docs.conda.io/en/latest/miniconda.html)):

    $ conda create --name <env> --file env.txt

In order to be able to submit jobs to `HTCondor`, install [snakemake-condor-profile](https://github.com/msto/snakemake-condor-profile).


## Running 🚀

Sending Snakemake process to `HTCondor`:
 
    $ snakemake --profile HTCondor
