#!/bin/bash 

#SBATCH --partition=gpu
#SBATCH --output=test.stdout 
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:25:00

module load gcc/9.4.0 cuda/11.2.2-kkrwdua 

cd $HOME/nns/neural-networks-hw1

mkdir build 
cd build 
cmake ..
make 

./testCudaMLP
