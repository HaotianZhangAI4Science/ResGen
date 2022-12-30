:loudspeaker: ResGen: A Pocket-aware 3D Molecular Generation Model Based on Parallel Multi-scale Modeling
=======
ResGen: A Pocket-aware 3D Molecular Generation Model Based on Parallel Multi-scale Modeling
<div align=center>
<img src="./figures/toc.png" width="50%" height="50%" alt="TOC" align=center />
</div>
ResGen is the newly developed method for 3D pocket-aware molecular generation. The codes will be released after the paper was accepted.

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
mamba install biopython -c conda-forge # used only in sample_for_pdb.py
mamba install pyyaml easydict python-lmdb -c conda-forge
```



## Data 

The main data for training is CrossDock2020, which is utilized by most of the methods.

Download the data from the original source

```python
wget https://bits.csb.pitt.edu/files/crossdock2020/CrossDocked2020_v1.1.tgz -P data/crossdock2020/
tar -C data/crossdock2020/ -xzf data/crossdock2020/CrossDocked2020_v1.1.tgz
wget https://bits.csb.pitt.edu/files/it2_tt_0_lowrmsd_mols_train0_fixed.types -P data/crossdock2020/
wget https://bits.csb.pitt.edu/files/it2_tt_0_lowrmsd_mols_test0_fixed.types -P data/crossdock2020/
```

And follow the convention in the data directory 

or you can download the processed data here. 





