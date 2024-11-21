#ifndef CUDA_MULTI_LAYER_PERCEPTRON_HPP
#define CUDA_MULTI_LAYER_PERCEPTRON_HPP

#include "MLP/MultiLayerPerceptron.hpp"
#include "MLP/NewMLP.hpp"
#include "CudaMLP/NeuronLayer.hpp"
#include "vector"

/**
 * @brief Extension of MLP for CUDA support.
*/
class DeviceMLP:public MLP{
  // Structural
  std::vector<DeviceLayer> d_layers;
  int input_size;
  float (*f)(const float);
  float (*f_dot)(const float);
  // Training on device 
  float* d_training_set;
  float* d_training_labels;
  int training_size;
  float* d_test_set;
  float* d_test_labels;
  int test_size;
  // For cuda operations
  cublasHandle_t handle;
  // Buffers 
  float* batch_loss_buffer;
  float* loss_array;


  DeviceMLP(const std::vector<int>& layer_sizes,
      const int input_size,
      VectorFunction activation_function,
      VectorFunction activation_derivative,
      float learning_rate,
      int batch_size);

  DeviceMLP(std::string file_path,std::string name,
      VectorFunction activation_function,
      VectorFunction activation_derivative,
      float learning_rate,int batch_size);

  ~DeviceMLP(){
    if(batch_loss_buffer!=nullptr){
      cudaFree(batch_loss_buffer);
    }
    if(loss_array!=nullptr){
      cudaFree(loss_array);
    }
    cublasDestroy_v2(handle);
  }

  void setDeviceFunction(float (*f)(const float),
                         float (*f_dot)(const float)){
    this->f=f;
    this->f_dot=f_dot;
  }

  // CUDA versions of MLP

  void forwardBatchPass(const float* d_input);

  void getBatchLoss(const int* correct_labels,float* loss);

  void backwardBatchPass(const float* d_input,
                         const int* correct_labels);

  void runDeviceEpoch();

  void testDeviceModel(float& J_test,float& accuracy);

  // I/O

  void datasetToDevice();
  
  void deviceToHost();
  void hostToDevice();


};

#endif // !CUDA_MULTI_LAYER_PERCEPTRON_HPP
