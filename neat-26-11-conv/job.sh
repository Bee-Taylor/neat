#!/bin/bash
#SBATCH --job-name=convolutional-neat-GPU # Job name
#SBATCH --mail-type=ALL                # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=bat508@york.ac.uk       # Where to send mail
#SBATCH --ntasks=1                          # Run a single task...
#SBATCH --cpus-per-task=10                 # ...with four cores
#SBATCH --mem=96000mb                          # Job memory request
#SBATCH --time=50:00:00                     # Time limit hrs:min:sec
#SBATCH --output=conv%j.log             # Standard output and error log
#SBATCH --account=elec-plasticnn-2020
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1


module load lang/Miniconda3

source activate conda_env

srun python main.py
