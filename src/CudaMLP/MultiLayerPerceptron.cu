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

float DeviceMLP::getBatchLoss(const int* correct_labels,
                              const int batch_size,
                              float* loss){
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
  return mean;
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

float DeviceMLP::runDeviceEpoch(){
  shuffleDataset();
  datasetToDevice();

  for(int i=0;i<training_size;i+=batch_size){
    forwardBatchPass(d_training_set+i);
    getBatchLoss(d_training_labels+i*input_size,batch_size,
                 loss_array+i/batch_size);
    backwardBatchPass(d_training_set+i*input_size, d_training_labels+i);
  }
  thrust::device_vector<float> buf(loss_array,loss_array+training_size/batch_size);
  float sum=thrust::reduce(buf.begin(),buf.end(),0.0f,thrust::plus<float>());
  return (sum/training_size)*batch_size;
}


__global__ static void testKernel(char* d_success_array,
                                  const int batch_size,
                                  const int* d_correct_labels,
                                  const float* d_activations,
                                  const int input_size
                                  ){
  const int idx=blockIdx.x*blockDim.x+threadIdx.x;
  const int col_idx=idx/input_size;
  if(idx<batch_size){
    int best_idx=0;
    for(int i=0;i<input_size;i++){
      if(d_activations[col_idx*input_size+i]<d_activations[col_idx*input_size+best_idx]){
        best_idx=i;
      }
    }
    d_success_array[idx]=(d_correct_labels[idx]==best_idx);
  } 
}

void DeviceMLP::testDeviceModel(float& J_test,float& accuracy){
  const int batch_size=(1000<test_labels.size())?(1000):(test_labels.size());
  const int test_size=test_set.cols();
  int success_count=0; 
  VectorXf batch_losses=VectorXf(test_size/batch_size);

  // For reducing
  char* d_success_array;
  cudaMalloc(&d_success_array,batch_size*sizeof(char));
  thrust::device_vector<char> buf(d_success_array,d_success_array+batch_size);
  const int threads=256;
  const int blocks=(batch_size+threads-1)/threads;
  for(int i=0;i<test_size;i++){
    forwardBatchPass(d_test_set+i); 
    batch_losses[i/batch_size]=getBatchLoss(d_test_labels+i,
                                            batch_size,
                                            batch_loss_buffer);
    testKernel<<<blocks,threads>>>(d_success_array,
                                   batch_size,
                                   d_test_labels+i,
                                   d_layers[depth-1].d_activations,
                                   input_size);   
    success_count+=thrust::reduce(buf.begin(),buf.end(),0.0f,thrust::plus<char>());
  }
  J_test=batch_losses.mean();
  accuracy=static_cast<float>(success_count)/test_size;
}

// I/O

void DeviceMLP::datasetToDevice(){
  training_size=training_set.cols();
  cudaMemcpy(d_training_set, training_set.data(), 
             training_set.size()*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_training_labels, training_labels.data(), 
             training_labels.size()*sizeof(int), cudaMemcpyHostToDevice);

  test_size=test_set.cols();
  cudaMemcpy(d_test_set, test_set.data(), 
             test_set.size()*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_test_labels, test_labels.data(), 
             test_labels.size()*sizeof(int), cudaMemcpyHostToDevice);


  if(batch_loss_buffer==nullptr){
    cudaMalloc(&batch_loss_buffer,batch_size*sizeof(float));
  }
  if(loss_array==nullptr){
    cudaMalloc(&loss_array,(training_size/batch_size)*sizeof(float));
  }
}

// Only for network, not for anything else
void DeviceMLP::deviceToHost(){
  for(int i=0;i<depth;i++){
    d_layers[i].storeToCPU(layers[i]);
  }
}

void DeviceMLP::hostToDevice(){
  for(int i=0;i<depth;i++){
    d_layers[i].loadFromCPU(layers[i]);
  }
}


