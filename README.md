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
parquet_util.py <path_to_parquet>

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

## To access GPU using tensorflow
From singularity, add the following at the end of the file:

/ext3/miniconda3/etc/profile.d/conda.sh
export NVIDIA_DIR=$(dirname $(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)")))
export LD_LIBRARY_PATH=$(echo ${NVIDIA_DIR}/*/lib/ | sed -r 's/\s+/:/g')${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
Ensure you have the right tensorflow installed:
pip install tensorflow[and cuda] (see tensorflow docs)
Copy the latest version of cuda that is supported by the tensorflow version: scp -rp greene-dtn:/scratch/work/public/singularity/cuda12.3.2-cudnn9.0.0-ubuntu-22.04.4.sif .
## Make sure you add this cuda version when you run your singularity!

## To run the code (note its easier to run in interactive job but you will have to reinstall the singularity in the login host)
python3 preprocess.py
python3 embeddings.py
python3 clusters.py 

## Last known run produced following result:
(base) Singularity> python3 clusters.py
Labels from kmeans clustering: [ 0  2  2 10  6  1  8 10  2  0]
Labels from spectral clustering: [3 4 4 6 8 1 5 6 8 7]
KMeans Silhouette Score: 0.4667663276195526
Spectral Silhouette Score: 0.21154426038265228
Chosen clustering label: kmeans_cluster
Clustering completed, average purity: 0.47832921517219995

