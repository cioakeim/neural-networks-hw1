#include "CudaMLP/MultiLayerPerceptron.hpp"
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/device_vector.h>
#include <cstdint>


#define epsilon 10e-7



DeviceMLP::DeviceMLP(const std::vector<int>& layer_sizes,
                     const int input_size,
                     VectorFunction activation_function,
                     VectorFunction activation_derivative,
                     float learning_rate,
                     int batch_size):MLP(layer_sizes,
                                         input_size,
                                         activation_function,
                                         activation_derivative,
                                         learning_rate,
                                         batch_size){
  for(int i=0;i<depth;i++){
    d_layers.emplace_back(layers[i]);
  }
  this->input_size=input_size;
  cublasCreate_v2(&handle);
  this->batch_loss_buffer=nullptr;
  this->loss_array=nullptr;
};

DeviceMLP::DeviceMLP(std::string file_path,std::string name,
                    VectorFunction activation_function,
                    VectorFunction activation_derivative,
                    float learning_rate,int batch_size)
                      :MLP(file_path,
                          name,
                          activation_function,
                          activation_derivative,
                          learning_rate,
                          batch_size){
  for(int i=0;i<depth;i++){
    d_layers.emplace_back(layers[i]);
  }
  this->input_size=d_layers[0].input_size;
}


void DeviceMLP::forwardBatchPass(const float* d_input){
  d_layers[0].activateLayer(d_input,f,handle);
  for(int i=1;i<depth-1;i++){
    d_layers[i].activateLayer(d_layers[i-1].d_activations,
                              f,handle);
  }
  d_layers[depth-1].softMaxOut(d_layers[depth-2].d_activations, handle);
}


__global__ static void batchLossKernel(const float* d_activations,
                                       const int output_size,
                                       const int batch_size,
                                       const int* correct_labels,
                                       float* batch_loss_buffer){
  const int idx=blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<batch_size){
    batch_loss_buffer[idx]=-log(epsilon+d_activations[output_size*idx+correct_labels[idx]]); 
  }
}

void DeviceMLP::getBatchLoss(const int* correct_labels,float* loss){
  const int threads=32;
  const int blocks=(batch_size+threads-1)/threads;
  batchLossKernel<<<blocks,threads>>>(d_layers[depth-1].d_activations,
                                      d_layers[depth-1].output_size,
                                      batch_size,
                                      correct_labels,
                                      batch_loss_buffer);
  thrust::device_vector<float> buf(batch_loss_buffer,batch_loss_buffer+batch_size);
  float mean=thrust::reduce(buf.begin(),buf.end(),0.0f,thrust::plus<float>())/batch_size;
  cudaMemcpy(loss, &mean, sizeof(float), cudaMemcpyHostToDevice);
}

void DeviceMLP::backwardBatchPass(const float* d_input,
                                  const int* correct_labels){
  d_layers[depth-1].softMaxError(correct_labels); 
  for(int i=depth-2;i>=0;i--){
    d_layers[i].activateError(d_layers[i+1].d_weights, 
                              d_layers[i+1].d_errors, 
                              d_layers[i+1].output_size, 
                              f_dot,handle);
  }
  d_layers[0].updateWeights(d_input,learning_rate,handle);
  for(int i=0;i<depth;i++){
    d_layers[i].updateWeights(d_layers[i-1].d_activations,
                              learning_rate,handle);
  }
}

void runDeviceEpoch();

void testDeviceModel(float& J_test,float& accuracy);

// I/O

void DeviceMLP::datasetToDevice(){
  training_size=training_set.cols();
  cudaMemcpy(d_training_set, training_set.data(), 
             training_set.size()*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_training_labels, training_labels.data(), 
             training_labels.size()*sizeof(float), cudaMemcpyHostToDevice);

  test_size=test_set.cols();
  cudaMemcpy(d_test_set, test_set.data(), 
             test_set.size()*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_test_labels, test_labels.data(), 
             test_labels.size()*sizeof(float), cudaMemcpyHostToDevice);


  if(batch_loss_buffer==nullptr){
    cudaMalloc(&batch_loss_buffer,batch_size*sizeof(float));
  }
  if(loss_array==nullptr){
    cudaMalloc(&loss_array,(training_size/batch_size)*sizeof(float));
  }
}

void deviceToHost();
void hostToDevice();


