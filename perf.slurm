#!/bin/bash
#SBATCH --job-name="perf"
#SBATCH --output="perf.out"
#SBATCH --error="perf.err"
#SBATCH --partition=gpuA40x4
#SBATCH --mem=5G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --constraint="perf,nvperf"
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
# srun nsys profile --stats=true ./raytracer "$INPUT_SCENE" > raytracer.out
srun ncu --set full -f -o raytracer ./raytracer "$INPUT_SCENE"