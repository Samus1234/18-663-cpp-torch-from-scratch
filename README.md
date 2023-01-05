# cpp-torch-from-scratch

A C++ CPU and CUDA-GPU implementation of the deep-learning library - Torch from scratch containing modular layers with forward and backward propagations, a general optimizer class, and a sequential class with training and validation.

## External Dependecies
* Linux
* [eigen3][https://eigen.tuxfamily.org/dox/GettingStarted.html]
* [CUDA][https://developer.nvidia.com/cuda-downloads]

## Structure
* The project contains two implementations, a CPU implementation using the Eigen linear algebra library and a GPU implementation using the barebones CUDA framework and raw C arrays.
* [CPU_Implementation](https://github.com/sidd-1234/cpp-torch-from-scratch/blob/main/CPU_Implementation)
    * base.hpp - Contains a weight initializer and the base Module and Optimizer classes that all layers and optimizers will inherit.
    * module.hpp - Includes specific implementation of modular layers such as the dense layer, activation layers, etc.
    * optimizer.hpp - Includes SGD and Adam optimizers that apply gradients to trainable parameters.
    * dataset.hpp - Reads training and test datasets and returns a vector iterator of batches.
    * model.hpp - Contains the main Sequential class that performs forward and backward passes through the network, applies gradients, trains/tests the network, and records loss/accuracy metrics.
    * main.cpp - The main .cpp file that creates the network and runs training and testing epochs.
    * processData.py - Unpacks and processes the .npz dataset file into CSV files.
* [GPU_Implementation](https://github.com/sidd-1234/cpp-torch-from-scratch/blob/main/GPU_Implementation)
    * base.cuh - Contains all global and device level GPU functions, a weight initializer and the base Module and Optimizer classes that all layers and optimizers will inherit.
    * module.cuh - Includes specific implementation of modular layers such as the dense layer, activation layers, etc.
    * optimizer.cuh - Includes SGD and Adam optimizers that apply gradients to trainable parameters.
    * dataset.hpp - Reads training and test datasets and returns a vector iterator of batches.
    * model.cuh - Contains the main Sequential class that performs forward and backward passes through the network, applies gradients, trains/tests the network, and records loss/accuracy metrics.
    * main.cu - The main .cu file that creates the network and runs training and testing epochs.
    * processData.py - Unpacks and processes the .npz dataset file into CSV files.

## How to run

### To run CPU implementation
> . cd CPU_Implementation/
> . run.sh

### To run GPU implementation
> . cd GPU_Implementation/
> . run.sh

## CPU Implementation results
We use a MLP network with 3 dense layers, 2 exponential linear units as activations for hidden layers, softmax cross entropy for the output layer, and an SGD optimizer

![alt text](https://github.com/sidd-1234/cpp-torch-from-scratch/blob/main/CPU_Implementation/cpu_loss.png?raw=true)

![alt text](https://github.com/sidd-1234/cpp-torch-from-scratch/blob/main/CPU_Implementation/cpu_acc.png?raw=true)

## GPU Implementation results
We use a MLP network with 3 dense layers, 2 exponential linear units as activations for hidden layers, sigmoid cross entropy for the output layer, and an SGD optimizer

![alt text](https://github.com/sidd-1234/cpp-torch-from-scratch/blob/main/GPU_Implementation/gpu_loss.png?raw=true)

![alt text](https://github.com/sidd-1234/cpp-torch-from-scratch/blob/main/GPU_Implementation/gpu_acc.png?raw=true)