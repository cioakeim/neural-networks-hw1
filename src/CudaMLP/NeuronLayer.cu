
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "CudaMLP/NeuronLayer.hpp"
#include <cmath>
#include <omp.h>


//Constructors
NeuronLayerCUDA::NeuronLayerCUDA(){
  this->input_size=this->output_size=0;
  this->d_weights=nullptr;
  this->d_biases=nullptr;
  this->d_outputs=nullptr;
  this->d_local_errors=nullptr;
  this->d_weightGradients=nullptr;
  this->d_biasGradients=nullptr;
}


NeuronLayerCUDA::NeuronLayerCUDA(uint32_t input_size,uint32_t layer_size)
  : NeuronLayer(input_size,layer_size){
  this->input_size=input_size;
  this->output_size=layer_size;
  this->d_weights=nullptr;
  this->d_biases=nullptr;
  this->d_outputs=nullptr;
  this->d_local_errors=nullptr;
  this->d_weightGradients=nullptr;
  this->d_biasGradients=nullptr;
  cublasCreate(&handle);
}

NeuronLayerCUDA::NeuronLayerCUDA(NeuronLayerCUDA* previous_layer,
                                 uint32_t layer_size)
  :NeuronLayer(previous_layer->getOutputSize(),layer_size){
  this->input_size=previous_layer->getOutputSize();
  this->output_size=layer_size;
  this->d_weights=nullptr;
  this->d_biases=nullptr;
  this->d_outputs=nullptr;
  this->d_local_errors=nullptr;
  this->d_weightGradients=nullptr;
  this->d_biasGradients=nullptr;
  cublasCreate(&handle);
}

// Destructor only for CUDA memory
NeuronLayerCUDA::~NeuronLayerCUDA(){
  if(d_weights!=nullptr){
    cudaFree(d_weights);
  }
  if(d_biases!=nullptr){
    cudaFree(d_biases);
  }
  if(d_outputs!=nullptr){
    cudaFree(d_outputs);
  }
  if(d_local_errors!=nullptr){
    cudaFree(d_local_errors);
  }
  if(d_weightGradients!=nullptr){
    cudaFree(d_weightGradients);
  }
  if(d_biasGradients!=nullptr){
    cudaFree(d_biasGradients);
  }
  cublasDestroy(handle);
}

// Set activation function element-wise 
void NeuronLayerCUDA::setElWiseActivationFunction(float (*elWiseF)(const float),
                                            float (*elWiseF_dot)(const float)){
  this->elWiseF=elWiseF;
  this->elWiseF_dot=elWiseF_dot;
}

// Interface between CPU and GPU memory

// Transposes a matrix and stores it in the output 
void transposeMatrix(const float* in, float* out, const int rows, const int cols){
  for(int i=0;i<rows;i++){
    #pragma omp parallel for
    for(int j=0;j<cols;j++){
      out[j*rows+i]=in[i*cols+j];
    }
  }
}

// Transfers all the data to the GPU
void NeuronLayerCUDA::copyLayerToDevice(){
  const int matrix_size=input_size*output_size*sizeof(float);
  const int vector_size=output_size*sizeof(float);

  float *temp_mat= new float[input_size*output_size];
  
  // Allocate all memory that's not already allocated 
  if(d_weights==nullptr){
    cudaMalloc(&d_weights,matrix_size);
  }
  // Store in column major order
  transposeMatrix(weights.data(),temp_mat,output_size,input_size);
  cudaMemcpy(d_weights,temp_mat,matrix_size,cudaMemcpyHostToDevice);

  if(d_biases==nullptr){
    cudaMalloc(&d_biases,vector_size);
  }
  cudaMemcpy(d_biases,biases.data(),vector_size,cudaMemcpyHostToDevice);

  if(d_outputs==nullptr){
    cudaMalloc(&d_outputs,vector_size);
  }
  cudaMemcpy(d_outputs,outputs.data(),vector_size,cudaMemcpyHostToDevice);

  if(d_local_errors==nullptr){
    cudaMalloc(&d_local_errors,vector_size);
  }
  cudaMemcpy(d_local_errors,local_errors.data(),vector_size,cudaMemcpyHostToDevice);

  if(d_weightGradients==nullptr){
    cudaMalloc(&d_weightGradients,matrix_size);
  }
  // Store in column major order
  transposeMatrix(weightGradients.data(),temp_mat,output_size,input_size);
  cudaMemcpy(d_weightGradients,temp_mat,matrix_size,cudaMemcpyHostToDevice);

  if(d_biasGradients==nullptr){
    cudaMalloc(&d_biasGradients,vector_size);
  }
  cudaMemcpy(d_biasGradients,biasGradients.data(),vector_size,cudaMemcpyHostToDevice);
  // Free temp array
  delete[] temp_mat;
}

// Gets all the data from the GPU
void NeuronLayerCUDA::copyLayerFromDevice(){
  const int matrix_size=input_size*output_size*sizeof(float);
  const int vector_size=output_size*sizeof(float);
  // Temp for transposing
  float *temp_mat= new float[input_size*output_size];

  cudaMemcpy(temp_mat,d_weights,matrix_size,cudaMemcpyDeviceToHost);
  transposeMatrix(temp_mat,weights.data(),output_size,input_size);
  cudaMemcpy(biases.data(),d_biases,vector_size,cudaMemcpyDeviceToHost);
  cudaMemcpy(outputs.data(),d_outputs,vector_size,cudaMemcpyDeviceToHost);
  cudaMemcpy(local_errors.data(),d_local_errors,vector_size,cudaMemcpyDeviceToHost);
  cudaMemcpy(temp_mat,d_weightGradients,matrix_size,cudaMemcpyDeviceToHost);
  transposeMatrix(temp_mat,weightGradients.data(),output_size,input_size);
  cudaMemcpy(biasGradients.data(),d_biasGradients,vector_size,cudaMemcpyDeviceToHost);

  // Free temp array
  delete[] temp_mat;
}


// All methods needed for passes written for GPU:

// Forward pass:


// Auxiliary methods for matrix computations


// Element wise function
__global__ void matrixElWiseFunc(const float* d_in,float* d_out,
                                 const int size,float (*func)(const float)){
  const int idx=blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<size){
    d_out[idx]=func(d_in[idx]);
  }
}

// Addition / Subtruction
__global__ void matrixElWiseAdd(const float* d_a, const float* d_b,
                                float *d_out,const int size){
  const int idx=blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<size){
    d_out[idx]=d_a[idx]+d_b[idx];
  }

}
__global__ void matrixElWiseSub(const float* d_a, const float* d_b, 
                                float *d_out,const int size){
  const int idx=blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<size){
    d_out[idx]=d_a[idx]-d_b[idx];
  }
}


// Using cuBLAS, calculate out=W*x+b
void calculateActivation(cublasHandle_t handle,
                         const float* d_input, const float* d_weights,
                         const float* d_biases, const int input_size,
                         const int output_size, float* d_outputs){
  const float alpha=1.0f;
  const float beta=0.0f;
  const int M=output_size;
  const int N=input_size;
  
  // Activation calculation
  cublasSgemv(handle,CUBLAS_OP_N,M,N,
              &alpha,d_weights,M,
              d_input,1,
              &beta,d_outputs,1);
  int threads=256;
  int blocks=(M+threads-1)/threads;
  matrixElWiseAdd<<<blocks,threads>>>(d_outputs,d_biases,d_outputs,M);
}

__global__ void softMaxFromActivation(float* d_activation,
                                    const int size){
  const int idx=blockIdx.x*blockDim.x+threadIdx.x;
  // Shared block memory
  __shared__ float temp[32];
  if(idx<size){
    temp[idx]=d_activation[idx];
    // Subtract max
    float max=temp[0];
    for(int i=1;i<size;i++){
      max=fmaxf(max,temp[i]);
    }
    __syncthreads();
    // exponentiate
    temp[idx]=expf(temp[idx]-max);
    // Sum up array
    float sum=0;
    for(int i=0;i<size;i++){
      sum+=temp[i];
    }
    __syncthreads();
    // Normalize activation
    d_activation[idx]=temp[idx]/sum;
  }
}

void NeuronLayerCUDA::setOutputCUDA(const float* d_input){
  calculateActivation(handle, 
                      d_input, d_weights, 
                      d_biases, input_size, output_size, 
                      d_outputs);
  // Pass throught non-linearity
  int threads=256;
  int blocks=(output_size+threads-1)/threads;
  matrixElWiseFunc<<<blocks,threads>>>(d_outputs,d_outputs,output_size,elWiseF);

}

// THIS SOFTMAX IMPLEMENTATION IS FOR MAX 32 CLASSES (uses syncthreads)
void NeuronLayerCUDA::setSoftMaxOutputCUDA(const float* d_input){
  calculateActivation(handle, 
                      d_input, d_weights, 
                      d_biases, input_size, output_size, 
                      d_outputs);
  int threads=32;
  int blocks=1;
  // Sub max and exponentiate
  softMaxFromActivation<<<blocks,threads>>>(d_outputs,output_size);   
}

// Backward pass:


__global__ void softMaxLocalErrorsKernel(const float* d_outputs,
                                         const int output_size,
                                         const int correct_class_idx,
                                         float* d_local_errors){
  const int idx=blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<output_size){
    d_local_errors[idx]=d_outputs[idx];
    if(idx==correct_class_idx){
      d_local_errors[idx]--;
    }
  }
}

void NeuronLayerCUDA::setSoftMaxErrorsCUDA(uint32_t correct_class_idx){
  int threads=32;
  int blocks=1;
  softMaxLocalErrorsKernel<<<blocks,threads>>>(d_outputs,
                                               output_size,
                                               correct_class_idx,
                                               d_local_errors);
}

__global__ void elWiseFdotMult(const float* d_outputs, const int size,
                               float* d_local_errors,
                               float (*func)(const float)){
  const int idx=blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<size){
    d_local_errors[idx]*=func(d_outputs[idx]);
  }
}

void NeuronLayerCUDA::setLocalErrorsCUDA(const float* d_next_errors,
                        const float* d_next_weights,
                        const uint32_t next_output_size){
  // Dimensions of next weights matrix
  const int rows=next_output_size;
  const int cols=output_size;
  const float alpha=1;
  const float beta=0;

  // d(n)=W^T*d(n+1)
  cublasSgemv(handle,CUBLAS_OP_T,
              cols,rows,&alpha,d_next_weights,
              rows,d_next_errors,1,
              &beta,d_local_errors,1);
  // Element wise multiply with f'(outputs)
  const int threads=256;
  const int blocks=(output_size+threads-1)/threads;
  elWiseFdotMult<<<blocks,threads>>>(d_outputs,output_size,
                                     d_local_errors,elWiseF_dot);
}

// Gradient descent methods:

void NeuronLayerCUDA::resetAllGradientsCUDA(){
  cudaMemset(&d_weightGradients,0,input_size*output_size*sizeof(float));
  cudaMemset(&d_biasGradients,0,output_size*sizeof(float));
}

void NeuronLayerCUDA::accumulateGradientsCUDA(const float* d_input){
  const int rows=output_size;
  const int cols=input_size;
  const int common_dim=1;
  const float alpha=1.0;
  const float beta=1.0;
  // Accumulate delta*input^T
  cublasSgemm(handle,
              CUBLAS_OP_N,CUBLAS_OP_T,
              rows,cols,common_dim,
              &alpha,
              d_local_errors,rows,
              d_input,1,
              &beta,
              d_weightGradients,
              rows);
}


__global__ void weightUpdateKernel(const float* d_weightGradients,
                              const float* d_biasGradients,
                              const int output_size,const int input_size,
                              const float lambda, const float rate,
                              float* d_weights,
                              float* d_biases){
  const int idx=blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<output_size*input_size){
    d_weights[idx]-=(rate)*d_weightGradients[idx]+lambda*d_weights[idx];
    if(idx<output_size){
      d_biases[idx]-=(rate)*d_biasGradients[idx]+lambda*d_biases[idx];
    }
  }
}

void NeuronLayerCUDA::updateWeightsCUDA(const uint32_t batchsize){
  const float rate=RATE/batchsize;
  const float lambda=WEIGHT_DECAY_RATE;
  const int threads=256;
  const int blocks=(output_size*input_size+threads-1)/threads;
  weightUpdateKernel<<<blocks,threads>>>(d_weightGradients,
                                         d_biasGradients,
                                         output_size,input_size,
                                         lambda,rate,
                                         d_weights,d_biases);
}




