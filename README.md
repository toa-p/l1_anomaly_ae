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
pip install uproot
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

Preprocess the output file to obtain a new h5 file where the separate MET, electrons, muons, jets, met arrays are concatenated in that order:

```
python preprocess.py --input-file L1Ntuple.h5 --output-file L1Ntuple_preprocessed.h5
```

### Automatic workflow with snakemake:

The workflow above is automatized for multiple files and samples using snakemake. The set of rules to be run is defined in the ```Snakefile``` file.

First, edit the directories in the ```config.yaml``` file as needed (eg, output/input/cmssw folders or list/location of samples). If list of samples is changed, the ```IDS_BSM``` in the
```Snakefile``` should also be changed accordingly. 

Then execute this command (suggested in a screen session):

```
snakemake merge_all_bsm_types --cores 8
``` 

This will run in sequence the following steps:

1. for each BSM sample each ROOT file will be converted to h5 with ```convert_bsm```
2. for each BSM sample all converted h5 will be merged in one with ```merge_h5_bsm_type```
3. for each merged BSM h5 file will be preprocessed with ```preprocess_bsm```
4. all merged BSM h5 files will be merged in one h5 file, for each BSM type there is a separate dataset with the corresponding name (```merge_all_bsm_types```)

(n,b: the BSM types are hardcoded in the ```merge_h5_tuples.py``` script)

Similarly for the QCD background:

```
snakemake convert_all --cores 8
```

This will run in sequence the following steps:

1. convert to h5 each ROOT file of the QCD background sample with ```convert_all```
2. merge the h5 files into one with ```merge_h5_tuples```
3. run preprocessing step on final merged h5 file with ```preprocess```

If one of the step fails for any reason, one can run one specific rule (eg, the merge or process steps as indicated above). In this case replace the rule
with the one you want to run.
