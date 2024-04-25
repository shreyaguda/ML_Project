# ML_Project
## To gain access to group folder please do the following:
1. ssh [netID]@gw.hpc.nyu.edu
2. ssh [netID]@greene.hpc.nyu.edu
3. ssh burst
4. srun --account=csci_ga_2565-2024sp --partition=n1s8-v100-1 --gres=gpu --time=1:00:00 --pty /bin/bash

### Now that you are in burst, you can access our group folder with the following commands:
1. newgrp grp10002
2. cd /scratch/ch4262-group/
3. ls -l
4. cd final_project/
5. bash-4.4$ ls

## To launch the singularity do (be in your scartch folder in bash):
1. scp -rp greene-dtn:/scratch/work/public/singularity/cuda11.4.2-cudnn8.2.4-devel-ubuntu20.04.3.sif . 
2. singularity exec --bind /scratch --nv --overlay /scratch/[netid]/overlay-25GB-500K.ext3:rw /scratch/[netid]/cuda11.4.2-cudnn8.2.4-devel-ubuntu20.04.3.sif /bin/bash

## How to Install new dependences:
1. Make sure you are in /scratch/[netid] in bash
2. singularity exec --bind /scratch --nv --overlay /scratch/[netid]/overlay-25GB-500K.ext3:rw /scratch/[netid]/cuda11.4.2-cudnn8.2.4-devel-ubuntu20.04.3.sif /bin/bash
3. conda activate
4. pip install [dependency]

## To View Parquette File Contents
1. Make sure you have already copied the parquette file from the group folder to your personal scratch folder (IN BASH!!!)
2. Go into singularity, and run python3 (make sure dependencies are installed)
3. Run the following commands:
import dask.dataframe as dd
import pandas as pd
pd.set_option('display.max_columns', None)
parquet_file = 'path/to/your/file.parquet'
ddf = dd.read_parquet(parquet_file)
print(ddf.head(20))

## To Use TMUX
ssh burst
tmux ls
to creat new session "tmux new -s [name]"
to open the tmux session: "tmux a -t [name]"
to kill a session on tmux:
control b d to get out of a tmux
tmux kill-session -t [name]

## How to run a job
login to greene
ssh burst
make sure you are in /scratch/[netid] (or whatever the path to the code file you are running is)
sbatch gpu_job.slurm
squeue -u [userid]
## Do this in BURST, NOT bash!!!

