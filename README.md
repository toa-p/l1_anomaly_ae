# DNN and CNN autoenconder

Training and testing code for DNN and CNN autoencoders with CMS L1 emulator inputs.

### Setup environment

Install [miniconda3](https://twiki.cern.ch/twiki/bin/view/CMSPublic/SWGuideL1TStage2Instructions) (check website for latest version):

```
wget https://repo.anaconda.com/miniconda/Miniconda3-py39_4.10.3-Linux-x86_64.sh
bash ./Miniconda3-py39_4.10.3-Linux-x86_64.sh
```

Clone this repository:

```
git clone https://gitlab.cern.ch/l1a/l1_anomaly_ae.git
cd l1_anomaly_ae
```

Setup the conda environment

```
conda create --name l1ad --file env.txt
conda activate l1ad
```

Install some missing packages:

```
pip install h5py
```

Activate the environment (nb, every time you login):

```
conda activate l1ad
```

Install [snakemake](https://indico.cern.ch/event/983691/contributions/4143450/attachments/2160191/3644503/snakemake.pdf) and [snakemake-condor-profile](https://github.com/msto/snakemake-condor-profile) to manage condor jobs

```
pip3 install snakemake
pip install cookiecutter
mkdir -p ~/.config/snakemake
cd ~/.config/snakemake
cookiecutter https://github.com/msto/snakemake-condor-profile.git
```

### Make datasets

First, convert L1 emulator data ROOT files to h5. For this ROOT-to-h5 conversion, CMSSW is needed. Example to install it ```cmsrel CMSSW_11_2_0```. If you run at LPC you first need to run
```source /cvmfs/cms.cern.ch/cmsset_default.(c)sh``` (if it is not already in your ```.tcshrc``` or ```.bashrc```). 


Example to convert one file (if you have installed CMSSW inside this repository directory):

```
cd CMSSW_11_2_0/src
cmsenv
cd ../../make-dataset/
python convert_to_h5.py --input-file L1Ntuple.root --output-file L1Ntuple.h5
```

To parallelize over multiple files we use snakemake. 



