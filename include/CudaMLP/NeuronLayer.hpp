#ifndef CUDA_NEURON_LAYER_HPP
#define CUDA_NEURON_LAYER_HPP

#include "MLP/NeuronLayer.hpp"
#include <cuda_runtime.h>
#include <cublas_v2.h>


/**
 * @brief CUDA extension of the Neuron Layer class
*/
class NeuronLayerCUDA:public NeuronLayer{
private:
  // I/O size are needed since CUDA matrices need the dimensions
  uint32_t input_size;
  uint32_t output_size;
  // CUDA Pointers to NeuronLayer's standard members
  // Structural
  float* d_weights;
  float* d_biases;
  float* d_outputs;
  // Element wise functions 
  float (*elWiseF)(const float);
  float (*elWiseF_dot)(const float);
  // Training data
  float* d_local_errors;
  float* d_weightGradients;
  float* d_biasGradients;
  // For matrix mult
  cublasHandle_t handle;

public:
  // Constructor extensions
  NeuronLayerCUDA();
  NeuronLayerCUDA(NeuronLayerCUDA* previous_layer,uint32_t layer_size);
  NeuronLayerCUDA(uint32_t input_size,uint32_t layer_size);
  // Only for CUDA memory
  ~NeuronLayerCUDA();

  // Set activation function element-wise 
  void setElWiseActivationFunction(float (*elWiseF)(const float),
                                   float (*elWiseF_dot)(const float));

  // Interface for layer communication
  const float* getCUDAWeightsPtr() const {return d_weights;}
  const float* getCUDAOutputsPtr() const {return d_outputs;}
  const float* getCUDALocalErrorsPtr() const {return d_local_errors;}
  uint32_t getInputSize() const {return input_size;}
  uint32_t getOutputSize() const {return output_size;}

  // Interface between CPU and GPU memory

  // Transfers all the data to the GPU
  void copyLayerToDevice();
  // Gets all the data from the GPU
  void copyLayerFromDevice();

  // All methods needed for passes written for GPU:

  // Forward pass:
  void setOutputCUDA(const float* d_input);
  void setSoftMaxOutputCUDA(const float* d_input);

  // Backward pass:
  void setSoftMaxErrorsCUDA(uint32_t correct_class_idx);
  void setLocalErrorsCUDA(const float* d_next_errors,
                          const float* d_next_weights,
                          const uint32_t next_output_size);

  // Gradient descent methods:

  void resetAllGradientsCUDA();
  void accumulateGradientsCUDA(const float* d_input);
  void updateWeightsCUDA(const uint32_t batchsize);


};

#endif // !CUDA_NEURON_LAYER_HPP
