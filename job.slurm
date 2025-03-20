#!/bin/bash
#SBATCH --job-name="render"
#SBATCH --output="render.out"
#SBATCH --error="render.err"
#SBATCH --partition=gpuA40x4
#SBATCH --mem=12G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --constraint=""
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=closest
#SBATCH --account=bche-delta-gpu
#SBATCH -t 00:05:00

module reset
module load cuda

echo -e "job $SLURM_JOBID is starting on `hostname`\n\n"

cp redchair.txt $SLURM_TMPDIR
cd $SLURM_TMPDIR

# Run the raytracer
srun ./raytracer  file=redchair.txt

# copy the image back
cp $SLURM_TMPDIR/redchair.png $HOME/render_results/