# Neural-Networks-HW1

## Description
In this project, an MLP was implemented using C++ and the Eigen library.
The dataset used was CIFAR-10, and it's needed in binary form to use the scripts.

## File structure
In the `src/NearestNeighbors`, `include/NearestNeighbors` folders are the implementations of the Nearest Neighbors 
algorithms used in HW0 for comparison. In the `src/MLP` and `include/MLP` folders are the MLP implementations,
with the main classes being the MLP and the Layer class. In the `scripts` folder are the scripts used for the 
testing and the experiments. In the `misc` folder are some python scripts used for plot creation and 
some job scripts used in the AuTh HPC cluster.

## Build
```bash
mkdir build
cd build 
cmake ..
make

```

## Usage
The 3 important scripts that are used are `testMLP`,`testMLP_Adam` and `testStoredModel`,
for training a simple MLP, training an MLP with an Adam optimizer and testing an already
created model from previous experiments.

1. For `testMLP`:
```
./testMLP [dataset_path] [model_storing_path] [learning_rate] [batch_size] [epoch_count] [layer_sequence] 
```
where the layer sequence is a comma separated list of integers and represents the layer structure used.
The sequence must end with 10 (output of CIFAR-10)

2. For `testMLP_Adam` the exact same arguments are used

3. For `testStoredModel`:
```
./testStoredModel [dataset_path] [model_storing_path] [network_name]
```
The above will load and test the network stored at `model_storing_path/network_name`.

## Models stored 
In the `models` folder are 3 categories, one for the batch size experiments, one for the layout experiments,
and one for the adam experiment. In all network folders is an `info.txt` file showing the parameters of the
test.
