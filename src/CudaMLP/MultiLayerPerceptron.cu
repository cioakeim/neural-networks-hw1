#include "CudaMLP/MultiLayerPerceptron.hpp"
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <cstdint>


#define epsilon 10e-7

// Only needed constructor
MultiLayerPerceptronCUDA::MultiLayerPerceptronCUDA(uint32_t input_size,
                         std::vector<uint32_t> layer_sequence,
                         uint32_t output_size){
  this->input_size=input_size;
  this->output_size=output_size;
  this->depth=layer_sequence.size();
  this->cuda_test=nullptr;
  this->cuda_training=nullptr;
  if(layer_sequence.size()==0){
    output_layer=NeuronLayerCUDA(input_size,output_size);
    return;
  }
  layers.emplace_back(input_size,layer_sequence[0]);
  for(int i=1;i<depth;i++){
    layers.emplace_back(&layers[i-1],layer_sequence[i]);
  }
  output_layer=NeuronLayerCUDA(&layers[depth-1],output_size);
}

MultiLayerPerceptronCUDA::~MultiLayerPerceptronCUDA(){
  if(this->cuda_test!=nullptr){
    cudaFree(cuda_test);
  }
  if(this->cuda_training!=nullptr){
    cudaFree(cuda_training);
  }
}

// Config.

// Random initial weight
void MultiLayerPerceptronCUDA::randomInit(){
  for(int i=0;i<depth;i++){
    layers[i].assertRandomWeights();
  }
  output_layer.assertRandomWeights();
}
// Random init with scaling 
void MultiLayerPerceptronCUDA::HeRandomInit(){
  for(int i=0;i<depth;i++){
    layers[i].HeRandomInit();
  }
  output_layer.HeRandomInit();
}

void MultiLayerPerceptronCUDA::setActivationFunction(float (*f)(const float),
                                                float (*f_dot)(const float)){
  for(int i=0;i<depth;i++){
    layers[i].setElWiseActivationFunction(f,f_dot);
  }
  output_layer.setElWiseActivationFunction(f,f_dot);
}

void MultiLayerPerceptron::setDataset(std::vector<SamplePoint> *training_set,
                std::vector<SamplePoint> *test_data){
  this->training_set=training_set;
  this->test_data=test_data;
}

// Interface between CPU/GPU

void MultiLayerPerceptronCUDA::copyNNToDevice(){
  for(int i=0;i<depth;i++){
    layers[i].copyLayerToDevice();
  } 
  output_layer.copyLayerToDevice();
}

void MultiLayerPerceptronCUDA::copyNNFromDevice(){
  for(int i=0;i<depth;i++){
    layers[i].copyLayerFromDevice();
  } 
  output_layer.copyLayerFromDevice();
}

void MultiLayerPerceptronCUDA::passDatasetToDevice(){
  this->test_size=(*test_data).size();
  this->training_size=(*training_set).size();
  int dim=(*training_set)[0].vector.size();
  // Pass test
  if(cuda_test==nullptr){
    cudaMalloc(&cuda_test,test_size*sizeof(SamplePoint));
  }
  cudaMemcpy(cuda_test,(*test_data).data(),test_size*sizeof(SamplePoint),
             cudaMemcpyHostToDevice);
  // Pass training
  if(cuda_training==nullptr){
    cudaMalloc(&cuda_training,training_size*sizeof(SamplePoint));
  }
  cudaMemcpy(cuda_training,(*training_set).data(),training_size*sizeof(SamplePoint),
             cudaMemcpyHostToDevice);
}


// Methods extended for CUDA use:

void MultiLayerPerceptronCUDA::forwardPass(float* d_input){
  layers[0].setOutputCUDA(d_input);
  for(int i=1;i<depth;i++){
    layers[i].setOutputCUDA(layers[i-1].getCUDAOutputsPtr());
  }
  output_layer.setSoftMaxOutputCUDA(layers[depth-1].getCUDAOutputsPtr());
}

void MultiLayerPerceptronCUDA::backwardPass(uint32_t correct_class_idx,
                  const float* d_input){
  output_layer.setSoftMaxErrorsCUDA(correct_class_idx);
  if(depth==0){
    output_layer.accumulateGradientsCUDA(d_input);
    return;
  }
  // Create local errors
  layers[depth-1].setLocalErrorsCUDA(output_layer.getCUDALocalErrorsPtr(), 
                                     output_layer.getCUDAWeightsPtr(), 
                                     output_layer.getOutputSize());
  for(int i=depth-2;i>=0;i--){
    layers[i].setLocalErrorsCUDA(layers[i+1].getCUDALocalErrorsPtr(), 
                                 layers[i+1].getCUDAWeightsPtr(), 
                                 layers[i+1].getOutputSize());
  }
  // Accumulate
  output_layer.accumulateGradientsCUDA(layers[depth-1].getCUDAOutputsPtr());
  for(int i=depth-2;i>=0;i++){
    layers[i].accumulateGradientsCUDA(layers[i-1].getCUDAOutputsPtr());
  }
  layers[0].accumulateGradientsCUDA(d_input);
}

void MultiLayerPerceptronCUDA::updateAllWeights(const uint32_t batch_size){
  for(int i=0;i<depth;i++){
    layers[i].updateWeightsCUDA(batch_size);
  }
  output_layer.updateWeightsCUDA(batch_size);
}

// Stochastic gradient descend

__global__ void logLossArrayKernel(const int sample_idx,
                                   const float* output_layer,
                                   const uint8_t label,
                                   float* loss_array){
  const int idx=blockIdx.x*blockDim.x+threadIdx.x;
  if(idx!=0){
    return;
  }
  const float y_predicted=output_layer[label];
  loss_array[sample_idx]=-log(y_predicted+epsilon);
}

void MultiLayerPerceptronCUDA::feedBatchAndUpdate(const int batch_size,
                                                  const int starting_index){
  const int final_idx=std::max(static_cast<int>(training_size),
                               starting_index+batch_size-1);
  // Reset local errors
  for(int i=0;i<depth;i++){
    layers[i].resetAllGradientsCUDA();
  }
  output_layer.resetAllGradientsCUDA();

  for(int i=starting_index;i<=final_idx;i++){
    uint8_t &label=(*training_set)[i].label;
    // Forward 
    forwardPass(cuda_training[i].vector.data());
    // Add loss to loss array
    logLossArrayKernel<<<1,1>>>(i,output_layer.getCUDAOutputsPtr(),
                                label,loss_array);
    // Backward pass
    backwardPass(label,cuda_training[i].vector.data());
  }
  updateAllWeights(batch_size);
}

float MultiLayerPerceptronCUDA::runEpoch(const int batch_size){
  for(int i=0;i<training_size;i++){
    feedBatchAndUpdate(batch_size,i);
  }
  thrust::device_ptr<float> d_ptr(loss_array);
  float sum=thrust::reduce(d_ptr,d_ptr+training_size,
                           0.0f,thrust::plus<float>());
  return sum/training_size;
}

float MultiLayerPerceptronCUDA::testModel(){
  uint32_t success_count=0;
  uint8_t prediction;
  for(int i=0;i<test_size;i++){
    forwardPass(cuda_test[i].vector.data());
    prediction=returnPredictedLabelCUDA();
    if(prediction==(*test_data)[i].label){
      success_count++;
    }
  }
  return static_cast<float>(success_count)/test_size;
}

uint8_t MultiLayerPerceptronCUDA::returnPredictedLabelCUDA(){
  thrust::device_ptr<float> d_ptr(output_layer.getCUDAOutputsPtr());
  thrust::device_ptr<float> max_iter=thrust::max(d_ptr,d_ptr+output_size);
  return static_cast<uint8_t>(max_iter-d_ptr); 
}






