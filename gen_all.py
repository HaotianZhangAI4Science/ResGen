import subprocess
from glob import glob 
import os.path as osp
import time

targets = glob('./data/alphafold/resgen_epo187/*')
out_dir = '/home/haotian/Molecule_Generation/ResGen-main/data/alphafold/resgen_new'
for target in targets:
    start_time = time.time()
    pdb_file = glob(osp.join(target, '*.pdb'))[0]
    lig_file = glob(osp.join(target, '*.sdf'))[0]
    command = f'python gen.py --pdb_file {pdb_file} --sdf_file {lig_file} --outdir {out_dir}'

    result = subprocess.run(command, shell=True, capture_output=True, text=True)

    # Check if fpocket command was successful
    if result.returncode == 0:
        print('executed successfully.')
        print('Output:')
        print(result.stdout)
        print('consumed time: ',time.time()-start_time)
    else:
        print('execution failed.')
        print('Error:')
        print(result.stderr)
