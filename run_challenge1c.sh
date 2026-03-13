#!/bin/bash
#SBATCH --job-name=ER_Challenge1c                # Job name
#SBATCH --output=logs/ER_Challenge1c_%j.out      # Standard output file (%j will be replaced by the job ID)
#SBATCH --error=logs/ER_Challenge1c_%j.err       # Standard error file
#SBATCH --nodes=1                                # Number of nodes requested
#SBATCH --ntasks=1                               # Number of tasks requested
#SBATCH --cpus-per-task=64                       # Number of CPU cores per task (adjust based on your parallel environment)
<<<<<<< HEAD
#SBATCH --mem=32G                                # Memory allocation
=======
#SBATCH --mem=128G                               # Memory allocation
>>>>>>> a0099a5 (Add cluster submit files)
#SBATCH --time=48:00:00                          # Expected runtime (hours:minutes:seconds)
#SBATCH --partition=academic                     # Partition to submit to academic partition
#SBATCH --account=micro-515                      # Account name

<<<<<<< HEAD
export MUJOCO_GL=egl

# Activate virtual environment
 source .venv/bin/activate
=======
# If you are using a Conda virtual environment, uncomment the lines below and replace with your environment name
# source ~/.bashrc
# conda activate micro515

# If you are using Python venv, uncomment the line below and replace with your venv path
# source .venv/bin/activate

# If you are using a Mamba environment, uncomment the lines below and replace with your environment name
source ~/.bashrc
mamba activate micro515
>>>>>>> a0099a5 (Add cluster submit files)

# Create logs directory
mkdir -p logs

# Run Python script
python Challenge1c.py

