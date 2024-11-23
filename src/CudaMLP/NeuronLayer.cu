#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "CudaMLP/NeuronLayer.hpp"
#include "MLP/NewLayer.hpp"
#include <cmath>


DeviceLayer::DeviceLayer():
  input_size(0),
  output_size(0),
  batch_size(0){};

DeviceLayer::DeviceLayer(int input_size,int output_size,int batch_size):
  input_size(input_size),
  output_size(output_size),
  batch_size(batch_size){
  cudaMalloc((void**)&d_weights,output_size*input_size*sizeof(float));
  cudaMalloc((void**)&d_biases,output_size*sizeof(float));
  cudaMalloc((void**)&d_activations,output_size*batch_size*sizeof(float));
  cudaMalloc((void**)&d_errors,output_size*batch_size*sizeof(float));
}

DeviceLayer::DeviceLayer(Layer& cpu_layer):DeviceLayer(
  cpu_layer.weights.cols(),
  cpu_layer.weights.rows(),
  cpu_layer.activations.cols()
){
  loadFromCPU(cpu_layer);
}


void DeviceLayer::loadFromCPU(const Layer& layer){
  cudaMemcpy(d_weights, layer.weights.data(), 
             output_size*input_size*sizeof(float),cudaMemcpyHostToDevice);
  cudaMemcpy(d_biases, layer.biases.data(), 
             output_size*sizeof(float),cudaMemcpyHostToDevice);
  cudaMemcpy(d_activations, layer.activations.data(), 
             output_size*batch_size*sizeof(float),cudaMemcpyHostToDevice);
  cudaMemcpy(d_errors, layer.errors.data(), 
             output_size*batch_size*sizeof(float),cudaMemcpyHostToDevice);

}


void DeviceLayer::storeToCPU(Layer& layer){
  cudaMemcpy(layer.weights.data(),d_weights,
             output_size*input_size*sizeof(float),cudaMemcpyDeviceToHost);
  cudaMemcpy(layer.biases.data(),d_biases,
             output_size*sizeof(float),cudaMemcpyDeviceToHost);
  cudaMemcpy(layer.activations.data(),d_activations,
             output_size*batch_size*sizeof(float),cudaMemcpyDeviceToHost);
  cudaMemcpy(layer.errors.data(),d_errors,
             output_size*batch_size*sizeof(float),cudaMemcpyDeviceToHost);
}

// For forward pass:

__global__ static void addBiasAndThroughNL(float* d_activations,
                                           const int rows,const int cols,
                                           const float *d_biases,
                                           float (*f)(const float)){
  const int idx=blockDim.x*blockIdx.x+threadIdx.x;
  const int row=idx%rows;
  if(idx<rows*cols){
    d_activations[idx]=f(d_activations[idx]+d_biases[row]);
  }
}

void DeviceLayer::activateLayer(const float* d_input,float (*f)(const float),
                                cublasHandle_t& handle){
  float alpha=1,beta=0; 
  const int rows=output_size;
  const int cols=batch_size;
  const int common=input_size;
  // Activation u=WX
  cublasSgemm_v2(handle,
                 CUBLAS_OP_N,CUBLAS_OP_N,
                 rows,cols,common,
                 &alpha,d_weights,rows,
                 d_input,common,&beta,
                 d_activations,rows
                 );
  const int threads=256;
  const int blocks=(rows*cols+threads-1)/threads;
  addBiasAndThroughNL<<<blocks,threads>>>(d_activations,
                                          rows,cols,
                                          d_biases,f);
}

__global__ static void softMaxKernel(float* d_activations,
                                     const int batch_size,
                                     const int rows,const int cols){
  const int idx=blockDim.x*blockIdx.x+threadIdx.x;
  const int col_idx=idx/rows;
  if(idx<rows*cols){
    float max=-INFINITY;
    for(int i=0;i<rows;i++){
      if(max<d_activations[i+col_idx*rows])
        max=d_activations[i+col_idx*rows];
    }
    d_activations[idx]=exp(d_activations[idx]-max);
    // All elements were exponantiaded so only threads for each column stay
    if(idx>=cols)
      return;
    float sum=0;
    for(int i=0;i<rows;i++){
      sum+=d_activations[i+col_idx*rows];
    }
    for(int i=0;i<rows;i++){
      d_activations[i+col_idx*rows]/=sum;
    }
  }

}

void DeviceLayer::softMaxOut(float* d_input,cublasHandle_t& handle){
  float alpha=1,beta=0; 
  const int rows=output_size;
  const int cols=batch_size;
  const int common=input_size;
  // Activation u=WX
  cublasSgemm_v2(handle,
                 CUBLAS_OP_N,CUBLAS_OP_N,
                 rows,cols,common,
                 &alpha,d_weights,rows,
                 d_input,common,&beta,
                 d_activations,rows
                 );
  const int threads=256;
  const int blocks=(rows*cols+threads-1)/threads;
  softMaxKernel<<<blocks,threads>>>(d_activations,
                                    batch_size,
                                    rows,cols);
}


// For backward pass:
__global__ static void multWithFdot(float* d_errors,
                                    float* d_activations,
                                    const int size,
                                    float (*f_dot)(const float)){
  const int idx=blockDim.x*blockIdx.x+threadIdx.x;
  if(idx<size){
    d_errors[idx]*=f_dot(d_activations[idx]);
  } 
}

void DeviceLayer::activateError(float* d_next_weights,
                                float* d_next_errors,
                                const int next_output_size,
                                float (*f_dot)(const float),
                                cublasHandle_t& handle){
  float alpha=1,beta=0; 
  const int rows=output_size;
  const int cols=batch_size;
  const int common=next_output_size;
  // Backward d=W^Td(n+1)
  cublasSgemm_v2(handle,
                 CUBLAS_OP_T,CUBLAS_OP_N,
                 rows,cols,common,
                 &alpha,d_next_weights,common,
                 d_next_errors,common,&beta,
                 d_errors,rows
                 );
  const int threads=256;
  const int blocks=(rows*cols+threads-1)/threads;
  multWithFdot<<<blocks,threads>>>(d_errors,
                                   d_activations,
                                   rows*cols,
                                   f_dot);
}

__global__ static void softMaxBackKernel(float* d_errors,
                                  const int *correct_labels,
                                  const int row_size,
                                  const int batch_size){
  const int idx=blockDim.x*blockIdx.x+threadIdx.x;
  if(idx<batch_size){
    d_errors[row_size*idx+correct_labels[idx]]--;
  }
}

void DeviceLayer::softMaxError(const int* correct_labels){
  cudaMemcpy(d_errors, d_activations, output_size*batch_size*sizeof(float), cudaMemcpyDeviceToDevice);
  const int threads=256;
  const int blocks=(batch_size+threads-1)/threads;
  softMaxBackKernel<<<blocks,threads>>>(d_errors,correct_labels,
                                        output_size,batch_size);
}


// Updating:

__global__ static void updateKernel(float* d_weights,
                                    const int size){
  const int idx=blockDim.x*blockIdx.x+threadIdx.x;
  const float lambda=1e-7;
  if(idx<size){
    d_weights[idx]-=lambda*d_weights[idx];
  }
}

void DeviceLayer::updateWeights(const float* d_input,
                                const float rate,
                                cublasHandle_t& handle){
  float alpha=-rate/batch_size,beta=1;
  const int rows=output_size;
  const int cols=input_size;
  const int common=batch_size;
  // W-=d*in^T
  cublasSgemm_v2(handle,
                 CUBLAS_OP_N,CUBLAS_OP_T,
                 rows,cols,common,
                 &alpha,d_errors,rows,
                 d_input,common,&beta,
                 d_weights,rows
                 );
  const int threads=256;
  const int blocks=(rows*cols+threads-1)/threads;
  updateKernel<<<blocks,threads>>>(d_weights,
                                   rows*cols);
}







