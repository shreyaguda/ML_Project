# ML_Project
To gain access to group folder please do the following:
1. ssh [netID]@gw.hpc.nyu.edu
2. ssh [netID]@greene.hpc.nyu.edu
3. ssh burst
4. srun --account=csci_ga_2565-2024sp --partition=n1s8-v100-1 --gres=gpu --time=1:00:00 --pty /bin/bash

Now that you are in burst, you can access our group folder with the following commands:
1. cd /scratch/ch4262-group/
2. newgrp grp10002
3. cd /scratch/ch4262-group/
4. ls -l
5. cd final_project/
6. bash-4.4$ ls

To launch the singularity do:
1. scp -rp greene-dtn:/scratch/work/public/singularity/cuda11.4.2-cudnn8.2.4-devel-ubuntu20.04.3.sif . 
2. singularity exec --bind /scratch --nv --overlay /scratch/[netid]/overlay-25GB-500K.ext3:rw /scratch/[netid]/cuda11.4.2-cudnn8.2.4-devel-ubuntu20.04.3.sif /bin/bash