#author: HaotianZhang
#time: 20220926
#project: 3D molecule de nove design

import os.path as osp
import os
from rdkit import Chem
import numpy as np
from easydict import EasyDict
from rdkit import Chem
import subprocess
from rdkit.Chem.rdMolAlign import CalcRMS
import shutil
import re

def write_sdf(mol, sdf_file, verbose=1):
    writer = Chem.SDWritter(sdf_file)
    if type(mol) == list:
        for i in range(len(mol)):
            writer.write(mol[i])
        writer.close()
    else:
        writer.write(mol)
    if verbose == 1:
        print('saved successfully at {}'.format(sdf_file))

def prepare_target(work_dir, protein_file_name, verbose=1):
    '''
    work_dir is the dir which .pdb locates
    protein_file_name: .pdb file which contains the protein data
    '''
    protein_file = osp.join(work_dir, protein_file_name)
    command = 'prepare_receptor -r {protein} -o {protein_pdbqt}'.format(protein=protein_file,
                                                            protein_pdbqt = protein_file+'qt')
    if osp.exists(protein_file+'qt'):
        return protein_file+'qt'
        
    proc = subprocess.Popen(
            command, 
            shell=True, 
            stdin=subprocess.PIPE, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE
        )
    proc.communicate()
    if verbose:
        if osp.exists(protein_file+'qt'):
            print('successfully prepare the target')
        else: 
            print('failed')
    return protein_file+'qt'

def prepare_ligand(work_dir, lig_sdf, verbose=1):
    lig_name = lig_sdf
    lig_mol2 = lig_sdf[:-3]+'mol2'
    now_cwd = os.getcwd()
    lig_sdf = osp.join(work_dir, lig_sdf)
    cwd_mol2 = osp.join(now_cwd, lig_mol2)
    work_mol2 = osp.join(work_dir, lig_mol2)
    command = '''obabel {lig} -O {lig_mol2}'''.format(lig=lig_sdf,
                                                        lig_mol2 = work_mol2)
    proc = subprocess.Popen(
            command, 
            shell=True, 
            stdin=subprocess.PIPE, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE
        )
    proc.communicate()
    shutil.copy(work_mol2, now_cwd)
    command = '''prepare_ligand -l {lig_mol2} -A hydrogens'''.format(lig_mol2=cwd_mol2)
    proc = subprocess.Popen(
            command, 
            shell=True, 
            stdin=subprocess.PIPE, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE
        )
    proc.communicate()
    lig_pdbqt = lig_name[:-3]+'pdbqt'
    cwd_pdbqt = osp.join(now_cwd, lig_pdbqt)
    work_pdbqt = osp.join(work_dir, lig_pdbqt)
    os.remove(cwd_mol2)
    os.remove(work_mol2)
    if osp.exists(work_pdbqt):
        os.remove(work_pdbqt)
    shutil.move(cwd_pdbqt, work_dir)
    if os.path.exists(lig_pdbqt):
        if verbose:
            print('prepare successfully !')
        else:
            print('generation failed!')
    return lig_pdbqt

def prepare_ligand_obabel(work_dir, ligand, out_lig_name=None, mode='ph'):
    ligand = osp.join(work_dir, ligand)
    lig_pdbqt = ligand.replace('sdf','pdbqt')
    if mode == 'ph':
        command = 'obabel {lig_sdf} -O {lig_pdbqt} -p 7.4'.format(lig_sdf=ligand, lig_pdbqt=lig_pdbqt)
        #command = 'prepare_ligand -l {lig_sdf} -o {lig_pdbqt} -A hydrogens'.format(lig_sdf=lig_sdf, lig_pdbqt=lig_pdbqt)
    elif mode == 'noh':
        command = 'obabel {lig_sdf} -O {lig_pdbqt}'.format(lig_sdf=ligand, lig_pdbqt=lig_pdbqt)
    elif mode == 'allH':
        mol = Chem.SDMolSupplier(ligand)[0]
        ligand_allH = ligand[:-4]+'_allH.sdf'
        write_sdf(mol, ligand_allH)
        ligand = ligand_allH
        command = 'obabel {lig_sdf} -O {lig_pdbqt}'.format(lig_sdf=ligand, lig_pdbqt=lig_pdbqt)
    
    proc = subprocess.Popen(
                    command, 
                    shell=True, 
                    stdin=subprocess.PIPE, 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.PIPE
                )
    proc.communicate() 
    return out_lig_name

def sdf2centroid(sdf_file):
    supp = Chem.SDMolSupplier(sdf_file, sanitize=False)
    lig_xyz = supp[0].GetConformer().GetPositions()
    centroid_x = lig_xyz[:,0].mean()
    centroid_y = lig_xyz[:,1].mean()
    centroid_z = lig_xyz[:,2].mean()
    return centroid_x, centroid_y, centroid_z

def mol2centroid(mol2_file):
    mol = Chem.MolFromMol2File(mol2_file, sanitize=False)
    lig_xyz = mol.GetConformer().GetPositions()
    centroid_x, centroid_y, centroid_z = lig_xyz.mean(axis=0)
    return centroid_x, centroid_y, centroid_z

def docking_with_sdf(work_dir, protein_pdbqt, lig_pdbqt, centroid, verbose=1, out_lig_sdf=None, save_pdbqt=False):
    '''
    work_dir: is same as the prepare_target
    protein_pdbqt: .pdbqt file
    lig_sdf: ligand .sdf format file
    '''
    # prepare target
    lig_pdbqt = osp.join(work_dir, lig_pdbqt)
    protein_pdbqt = osp.join(work_dir, protein_pdbqt)
    cx, cy, cz = centroid
    out_lig_sdf_dirname = osp.dirname(lig_pdbqt)
    out_lig_pdbqt_filename = osp.basename(lig_pdbqt).split('.')[0]+'_out.pdbqt'
    out_lig_pdbqt = osp.join(out_lig_sdf_dirname, out_lig_pdbqt_filename) 
    if out_lig_sdf is None:
        out_lig_sdf_filename = osp.basename(lig_pdbqt).split('.')[0]+'_out.sdf'
        out_lig_sdf = osp.join(out_lig_sdf_dirname, out_lig_sdf_filename) 
    else:
        out_lig_sdf = osp.join(work_dir, out_lig_sdf)

    command = '''qvina02 \
        --receptor {receptor_pre} \
        --ligand {ligand_pre} \
        --center_x {centroid_x:.4f} \
        --center_y {centroid_y:.4f} \
        --center_z {centroid_z:.4f} \
        --size_x 20 --size_y 20 --size_z 20 \
        --out {out_lig_pdbqt} \
        --exhaustiveness {exhaust}
        obabel {out_lig_pdbqt} -O {out_lig_sdf} -h'''.format(receptor_pre = protein_pdbqt,
                                            ligand_pre = lig_pdbqt,
                                            centroid_x = cx,
                                            centroid_y = cy,
                                            centroid_z = cz,
                                            out_lig_pdbqt = out_lig_pdbqt,
                                            exhaust = 24,
                                            out_lig_sdf = out_lig_sdf)
    proc = subprocess.Popen(
            command, 
            shell=True, 
            stdin=subprocess.PIPE, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE
        )
    proc.communicate()

    os.remove(lig_pdbqt)
    if not save_pdbqt:
        os.remove(out_lig_pdbqt)
    
    if verbose: 
        if os.path.exists(out_lig_sdf):
            print('searchable docking is finished successfully')
        else:
            print('docing failed')
    return out_lig_sdf

def sminadocking_with_sdf(work_dir, protein_pdbqt, lig_pdbqt, centroid, verbose=1, out_lig_sdf=None, save_pdbqt=False):
    '''
    work_dir: is same as the prepare_target
    protein_pdbqt: .pdbqt file
    lig_sdf: ligand .sdf format file
    '''
    # prepare target
    lig_pdbqt = osp.join(work_dir, lig_pdbqt)
    protein_pdbqt = osp.join(work_dir, protein_pdbqt)
    cx, cy, cz = centroid
    out_lig_pdbqt = lig_pdbqt.split('.')[0]+'_out.pdbqt'
    if out_lig_sdf is None:
        out_lig_sdf = lig_pdbqt.split('.')[0]+'_out.sdf'
    else:
        out_lig_sdf = osp.join(work_dir, out_lig_sdf)

    command = '''qvina02 \
        --receptor {receptor_pre} \
        --ligand {ligand_pre} \
        --center_x {centroid_x:.4f} \
        --center_y {centroid_y:.4f} \
        --center_z {centroid_z:.4f} \
        --size_x 20 --size_y 20 --size_z 20 \
        --out {out_lig_pdbqt} \
        --exhaustiveness {exhaust}
        obabel {out_lig_pdbqt} -O {out_lig_sdf} -h'''.format(receptor_pre = protein_pdbqt,
                                            ligand_pre = lig_pdbqt,
                                            centroid_x = cx,
                                            centroid_y = cy,
                                            centroid_z = cz,
                                            out_lig_pdbqt = out_lig_pdbqt,
                                            exhaust = 48,
                                            out_lig_sdf = out_lig_sdf)
    proc = subprocess.Popen(
            command, 
            shell=True, 
            stdin=subprocess.PIPE, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE
        )
    proc.communicate()

    os.remove(lig_pdbqt)
    if not save_pdbqt:
        os.remove(out_lig_pdbqt)
    
    if verbose: 
        if os.path.exists(out_lig_sdf):
            print('searchable docking is finished successfully')
        else:
            print('docing failed')
    return out_lig_sdf

def scoring_with_sdf(work_dir, protein_pdbqt, lig_pdbqt, centroid, out_lig_sdf=None):
    '''
    work_dir: is same as the prepare_target
    protein_pdbqt: .pdbqt file
    lig_sdf: ligand .sdf format file
    '''
    # prepare target
    lig_pdbqt = osp.join(work_dir, lig_pdbqt)
    protein_pdbqt = osp.join(work_dir, protein_pdbqt)
    cx, cy, cz = centroid
    out_lig_sdf_dirname = osp.dirname(lig_pdbqt)
    out_lig_pdbqt_filename = osp.basename(lig_pdbqt).split('.')[0]+'_out.pdbqt'
    out_lig_pdbqt = osp.join(out_lig_sdf_dirname, out_lig_pdbqt_filename) 
    if out_lig_sdf is None:
        out_lig_sdf_filename = osp.basename(lig_pdbqt).split('.')[0]+'_out.sdf'
        out_lig_sdf = osp.join(out_lig_sdf_dirname, out_lig_sdf_filename) 
    else:
        out_lig_sdf = osp.join(work_dir, out_lig_sdf)

    command = '''qvina02 \
        --receptor {receptor_pre} \
        --ligand {ligand_pre} \
        --center_x {centroid_x:.4f} \
        --center_y {centroid_y:.4f} \
        --center_z {centroid_z:.4f} \
        --size_x 20 --size_y 20 --size_z 20 \
        --out {out_lig_pdbqt} \
        --exhaustiveness {exhaust} \
        --score_only'''.format(receptor_pre = protein_pdbqt,
                                            ligand_pre = lig_pdbqt,
                                            centroid_x = cx,
                                            centroid_y = cy,
                                            centroid_z = cz,
                                            out_lig_pdbqt = out_lig_pdbqt,
                                            exhaust = 32,
                                            out_lig_sdf = out_lig_sdf)
    proc = subprocess.Popen(
            command, 
            shell=True, 
            stdin=subprocess.PIPE, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE
        )
    p = proc.communicate()[0]
    c = p.decode("gbk").strip()
    score = re.search("\nAffinity:(.*)\n", c).group().strip().split()[1]

    os.remove(lig_pdbqt)
    
    return score

def scoring_with_sdf(work_dir, protein_pdbqt, lig_pdbqt, centroid, out_lig_sdf=None):
    '''
    work_dir: is same as the prepare_target
    protein_pdbqt: .pdbqt file
    lig_sdf: ligand .sdf format file
    '''
    # prepare target
    lig_pdbqt = osp.join(work_dir, lig_pdbqt)
    protein_pdbqt = osp.join(work_dir, protein_pdbqt)
    cx, cy, cz = centroid
    out_lig_sdf_dirname = osp.dirname(lig_pdbqt)
    out_lig_pdbqt_filename = osp.basename(lig_pdbqt).split('.')[0]+'_out.pdbqt'
    out_lig_pdbqt = osp.join(out_lig_sdf_dirname, out_lig_pdbqt_filename) 
    if out_lig_sdf is None:
        out_lig_sdf_filename = osp.basename(lig_pdbqt).split('.')[0]+'_out.sdf'
        out_lig_sdf = osp.join(out_lig_sdf_dirname, out_lig_sdf_filename) 
    else:
        out_lig_sdf = osp.join(work_dir, out_lig_sdf)

    command = '''qvina02 \
        --receptor {receptor_pre} \
        --ligand {ligand_pre} \
        --center_x {centroid_x:.4f} \
        --center_y {centroid_y:.4f} \
        --center_z {centroid_z:.4f} \
        --size_x 20 --size_y 20 --size_z 20 \
        --out {out_lig_pdbqt} \
        --exhaustiveness {exhaust} \
        --score_only'''.format(receptor_pre = protein_pdbqt,
                                            ligand_pre = lig_pdbqt,
                                            centroid_x = cx,
                                            centroid_y = cy,
                                            centroid_z = cz,
                                            out_lig_pdbqt = out_lig_pdbqt,
                                            exhaust = 32,
                                            out_lig_sdf = out_lig_sdf)
    proc = subprocess.Popen(
            command, 
            shell=True, 
            stdin=subprocess.PIPE, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE
        )
    p = proc.communicate()[0]
    c = p.decode("gbk").strip()
    score = re.search("\nAffinity:(.*)\n", c).group().strip().split()[1]

    os.remove(lig_pdbqt)
    
    return score

def get_result(docked_sdf, ref_mol=None):
    suppl = Chem.SDMolSupplier(docked_sdf,sanitize=False)
    results = []
    for i, mol in enumerate(suppl):
        if mol is None:
            continue
        line = mol.GetProp('REMARK').splitlines()[0].split()[2:]
        try:
            rmsd = CalcRMS(ref_mol, mol)
        except:
            rmsd = np.nan
        results.append(EasyDict({
            'rdmol': mol,
            'mode_id': i,
            'affinity': float(line[0]),
            'rmsd_lb': float(line[1]),
            'rmsd_ub': float(line[2]),
            'rmsd_ref': rmsd
        }))
    return results


if __name__ == 'main':
    import glob
    gen_file = '/home/haotian/molecules_confs/Protein_test/Pocket2Mol-main/outputs'
    samples = glob.glob(gen_file + '/*')
    i = 0
    work_dir = osp.join(samples[i], 'SDF')
    protein_file = glob.glob(work_dir+'/*.pdb')[0]
    ori_lig_file = protein_file[:-13]+'.sdf'
    str = '/'
    # prepare target 
    prepare_target(work_dir, protein_file.split('/')[-1], verbose=0)
    protein_pdbqt = protein_file.split('/')[-1]+'qt'
    prepare_ligand(work_dir, '{}.sdf'.format(i))
    # get the centroid from the original ligand
    centroid = sdf2centroid(ori_lig_file)
    # docking the i.sdf file, using the box of original ligand. 
    docked_sdf = docking_with_sdf(work_dir,protein_pdbqt, '{}.pdbqt'.format(i), centroid)
    # get the docking result as the list, if you want to compare the reference mol to the all docked mol, choosing ref_mol= ref_mol
    result = get_result(docked_sdf=docked_sdf, ref_mol= None)