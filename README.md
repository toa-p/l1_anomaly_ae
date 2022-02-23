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
conda env create -f l1ad.yml
conda activate l1ad
```

**NB: the cudnn version in the l1ad.yml might be incompatible with your cuda version. Check [here](https://repo.anaconda.com/pkgs/main/linux-64/) for different versions.**


Install QKeras in the same or another folders:

```
 pip install git+git://github.com/google/qkeras.git
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

To process 120X samples:

```
snakemake merge_all_bsm_types --cores 8 --snakefile Snakefile-120X --configfile config-120X.yaml
snakemake convert_all --cores 8 --snakefile Snakefile-120X --configfile config-120X.yaml
```

### Train and evaluate CNN models:

#### Prepare the data

Input data must be prepared with ```prepare_data.py```, which accepts as inputs preprocessed QCD and/or preprocessed BSM h5 files and outputs a pickle file with train/test events split to be used downstream for the train and evaluation steps. For example: 

```
python prepare_data.py --input-file QCD_preprocessed.h5 --input-bsm BSM_preprocessed.h5 --output-file data.pickle
```

nb, this script has the list of BSM signals [hardcoded](https://gitlab.cern.ch/cms-l1-ad/l1_anomaly_ae/-/blob/master/cnn/prepare_data.py#L37-L52) and you might want to change if new signals will be added or if unused signals will be removed.

#### Training 

The training program requires the input of the latent dimension, filenames for the output model (h5 and json), the training data pickle file, the history, but also the batch size, and the number of epochs. For example: 

```
python train.py --latent-dim 4 --output-model-h5 output_model.h5 --output-model-json output_model.json --input-data data.pickle --output-history history.h5 --batch-size 256 --n-epochs 30
```

Two models can be trained: Conv VAE or Conv AE. The model can be set with `--model-type` as either `conv_vae` or `conv_ae`. Change all other options as needed.

#### Performance evaluation

First prepare file with predictions for the QCD test sample and BSM samples:

```
python evaluate.py --input-h5 output_model.h5 --input-json output_model.json --input-history history.h5 --output-result result.h5 --input-file data.pickle
```

Make ROC curves, history and loss distributions:

```
python plot.py --coord cyl --model vae --loss-type mse_kl --output-dir ./ --input-dir ./ --label my_vae
```

Several loss types can be set and type of features. Change all other options as needed.

### Train and evaluate DNN models:

#### End to End for DNN

For DNN once we have ```.h5``` Files, you can run the entire chain using the script ```end2end.py```, this scrip has various handles to output data at preliminary stages. To run it, just use

```
python end2end.py --input_qcd /eos/uscms/store/group/lpctrig/jngadiub/L1TNtupleRun3-h5-extended/QCD_preprocessed.h5 --input_bsm /eos/uscms/store/group/lpctrig/jngadiub/L1TNtupleRun3-h5-extended/BSM_preprocessed.h5 --events 10000 --output_pfile data.pickle --model_type AE --latent_dim 3 --output_model_h5 model.h5 --output_model_json model.json --output_history history.h5 --batch_size 1024 --n_epochs 150 --output_result results.h5 --tag test
```
To run the same with QKeras, make sure to add the flag ```--model_quantize 1``` for the above script.

This Script is also avilable for an interactive run in ```end2end_demo.ipynb```. 

If incase only individual parts of the code are to be run, use the subsequent steps.

#### Prepare the data

Input data must be prepared with ```data_preprocessing.py```, which accepts as inputs preprocessed QCD and/or preprocessed BSM h5 files and outputs a pickle file with train/test events split to be used downstream for the train and evaluation steps. For example: 

```
python data_preprocessing.py --input-file QCD_preprocessed.h5 --input-bsm BSM_preprocessed.h5 --output-file data.pickle --events -1
```

#### Training 

The training program requires the input of the latent dimension, filenames for the output model (h5 and json), the training data pickle file, the history, but also the batch size, and the number of epochs. For example: 

```
python train_AE.py --model-type AE --input-data data.pickle --output-model-h5 output_model.h5 --output-model-json output_model.json --output-history history.h5 --batch-size 1024 --n-epochs 150 --latent-dim 3
```

#### Performance evaluation

First prepare file with predictions for the QCD test sample and BSM samples:

```
python evaluate.py --input-h5 output_model.h5 --input-json output_model.json --input-file data.pickle --output-result results.h5 --input-history history.h5
```

Make ROC curves and other debugging plots (input vs reconstructed features, history, loss distributions)

```
python plot_baseline.py --results results.h5 --output-dir plots
```

