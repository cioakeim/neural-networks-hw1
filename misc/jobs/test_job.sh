#!/bin/bash 

#SBATCH --partition=ampere
#SBATCH --output=output.stdout 
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00


module load gcc/13.2.0 cuda/12.4.0-obe7ebz cmake/3.27.9-uxdlqo3 

cd $HOME/nns/neural-networks-hw1

mkdir build 
cd build 
cmake ..
make 

./testCudaMLP
