#ifndef CUDA_NEURON_LAYER_HPP
#define CUDA_NEURON_LAYER_HPP

#include "MLP/NewLayer.hpp"
#include <cuda_runtime.h>
#include <cublas_v2.h>



/**
 * @brief CUDA extension of the Layer struct
*/
struct DeviceLayer{
  const int input_size;
  const int output_size;
  const int batch_size;
  float* d_weights;
  float* d_biases;
  float* d_activations;
  float* d_errors;

  DeviceLayer();

  DeviceLayer(int input_size,int output_size,int batch_size);

  DeviceLayer(Layer &cpu_layer);

  ~DeviceLayer(){
    cudaFree(d_weights);
    cudaFree(d_biases);
    cudaFree(d_activations);
    cudaFree(d_errors);
  }

  // I/O
  void loadFromCPU(const Layer& layer);
  void storeToCPU(Layer& layer);

  void activateLayer(const float* d_input,float (*f)(const float),
                     cublasHandle_t& handle);
  void softMaxOut(float* d_input,cublasHandle_t& handle);

  void activateError(float* d_next_weights,
                     float* d_next_errors,
                     const int next_output,
                     float (*f_dot)(const float),
                     cublasHandle_t& handle);

  void softMaxError(const int* correct_labels);

  void updateWeights(const float* input,
                     const float rate,
                     cublasHandle_t& handle);


};
#endif // !CUDA_NEURON_LAYER_HPP
