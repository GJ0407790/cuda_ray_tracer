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

echo -e "Job $SLURM_JOBID is starting on $(hostname)\n"

# Define input and output files.
INPUT_SCENE="redchair.txt"
OUTPUT_IMAGE="redchair.png"
TARGET_DIR="$HOME/TinyRaytracer"

# Run the raytracer.
srun ./raytracer "$INPUT_SCENE"

# After rendering, copy the output image back to your target directory.
if [ -f "$OUTPUT_IMAGE" ]; then
    cp "$OUTPUT_IMAGE" "$TARGET_DIR"/ || { echo "Failed to copy output image"; exit 1; }
    echo "Render complete. Image copied to $TARGET_DIR/$OUTPUT_IMAGE"
else
    echo "Error: Rendered image not found!"
    exit 1
fi
