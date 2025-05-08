#!/bin/bash
#SBATCH --job-name="render"
#SBATCH --output="render.out"
#SBATCH --error="render.err"
#SBATCH --partition=gpuA40x4
#SBATCH --mem=5G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --constraint=""
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=closest
#SBATCH --account=bche-delta-gpu
#SBATCH -t 00:30:00

module reset
module load cuda

echo -e "Job $SLURM_JOBID is starting on $(hostname)\n"

# Define input and output files.
INPUT_SCENE="redchair.txt"

# Run the raytracer.
srun ./raytracer "$INPUT_SCENE"
# srun compute-sanitizer ./raytracer "$INPUT_SCENE"
