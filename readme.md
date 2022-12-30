:loudspeaker: ResGen: A Pocket-aware 3D Molecular Generation Model Based on Parallel Multi-scale Modeling
=======
ResGen: A Pocket-aware 3D Molecular Generation Model Based on Parallel Multi-scale Modeling
<div align=center>
<img src="./figures/toc.png" width="50%" height="50%" alt="TOC" align=center />
</div>
ResGen is the newly developed method for 3D pocket-aware molecular generation. 

## Environment 

### Install via conda yaml file (cuda 11.3)

```python
mamba env create -f resgen.yml
mamba activate resgen 
```

### Install manually 

(we recommend using mamba instead of conda, if you're old school, just change the mamba with conda)

```
mamba create -n resgen python=3.8
mamba install pytorch==1.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
mamba install pyg -c pyg
mamba install -c conda-forge rdkit
mamba install biopython -c conda-forge
mamba install pyyaml easydict python-lmdb -c conda-forge
```



## Data 

The main data for training is CrossDock2020, which is utilized by most of the methods. 

#### Download the data from the original source

```python
wget https://bits.csb.pitt.edu/files/crossdock2020/CrossDocked2020_v1.1.tgz -P data/crossdock2020/
tar -C data/crossdock2020/ -xzf data/crossdock2020/CrossDocked2020_v1.1.tgz
wget https://bits.csb.pitt.edu/files/it2_tt_0_lowrmsd_mols_train0_fixed.types -P data/crossdock2020/
wget https://bits.csb.pitt.edu/files/it2_tt_0_lowrmsd_mols_test0_fixed.types -P data/crossdock2020/
```

Or you can follow the data processing procedure in [SBDD]([3D-Generative-SBDD/data at main · luost26/3D-Generative-SBDD (github.com)](https://github.com/luost26/3D-Generative-SBDD/tree/main/data)).  

> 1. Download the dataset archive `crossdocked_pocket10.tar.gz` and the split file `split_by_name.pt` from [this link](https://drive.google.com/drive/folders/1CzwxmTpjbrt83z_wBzcQncq84OVDPurM).
> 2. Extract the TAR archive using the command: `tar -xzvf crossdocked_pocket10.tar.gz`.

Then follow the convention in the data directory.  

or you can download the processed data [here](). 

# Training 

The training process is released as train.py, the following command is an example about how to train a model.

```python
python train.py --config ./configs/train_res.yml --logdir logs
```



# Generation



# Acknowledge

This project draws in part from [GraphBP]([divelab/GraphBP: Official implementation of "Generating 3D Molecules for Target Protein Binding" [ICML2022 Long Presentation\] (github.com)](https://github.com/divelab/GraphBP)) and [Pocket2Mol]([pengxingang/Pocket2Mol: Pocket2Mol: Efficient Molecular Sampling Based on 3D Protein Pockets (github.com)](https://github.com/pengxingang/Pocket2Mol)) , supported by GPL-v3 License and MIT License. Thanks for their great work and code, hope readers of interest could check their work, too.  









