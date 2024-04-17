#!/bin/bash
#SBATCH --job-name=test
#SBATCH --account=csci_ga_2565-2024sp
#SBATCH --partition=n1c24m128-v100-4
#SBATCH --open-mode=append
#SBATCH --output=/scratch/ch4262/final_project/%j_%x.out
#SBATCH --error=/scratch/ch4262/final_project/%j_%x.err
#SBATCH --export=ALL
#SBATCH --time=00:20:00
#SBATCH --gres=gpu:v100:4
#SBATCH --requeue

singularity exec --bind /scratch --nv --overlay /scratch/ch4262/overlay-25GB-500K.ext3:ro /scratch/ch4262/cuda11.4.2-cudnn8.2.4-devel-ubuntu20.04.3.sif /bin/bash -c "
source /ext3/env.sh
conda activate
python /scratch/ch4262/final_project/preprocess.py
"
