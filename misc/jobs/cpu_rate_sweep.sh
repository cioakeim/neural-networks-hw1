#!/bin/bash 

#SBATCH --partition=rome
#SBATCH --output=cpu_rate_sweep.stdout 
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --time=10:00:00


module load gcc/13.2.0 cmake/3.27.9-uxdlqo3 


dataset_path="/home/c/cioakeim/nns/cifar-10-batches-bin"
nn_path="/home/c/cioakeim/nns/configs"
rate="$1"
batch_size="100"
epochs="15"
layer_sequence="2048,512,124,10"

project_dir="/home/c/cioakeim/nns/neural-networks-hw1"

cd "$project_dir"
mkdir -p build
cd build
cmake ..
make


./testMLP "$dataset_path" "$nn_path" "$rate" "$batch_size" "$epochs" "$layer_sequence"
