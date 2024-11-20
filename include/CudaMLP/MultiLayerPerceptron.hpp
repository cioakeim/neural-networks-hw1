#ifndef CUDA_MULTI_LAYER_PERCEPTRON_HPP
#define CUDA_MULTI_LAYER_PERCEPTRON_HPP

#include "MLP/MultiLayerPerceptron.hpp"
#include "MLP/NewMLP.hpp"
#include "CudaMLP/NeuronLayer.hpp"
#include "vector"

/**
 * @brief Extension of MLP for CUDA support.
*/
class MultiLayerPerceptronCUDA{
protected:
  int input_size;
  int output_size;
  int depth; //< Only hidden layers
  // Cuda versions of needed fields.
  std::vector<NeuronLayerCUDA> layers; //< All intermediate layers.
  NeuronLayerCUDA output_layer; //< The output layer
  std::vector<SamplePoint> *training_set; //< The training set
  std::vector<SamplePoint> *test_data; //< The test data
  // Cuda memory methods
  SamplePoint* cuda_training;
  int training_size;
  SamplePoint* cuda_test;
  int test_size;
  // For keeping count of training loss function 
  float* loss_array;
  

  
public:
  // Only needed constructor
  MultiLayerPerceptronCUDA(uint32_t input_width,
                           std::vector<uint32_t> layer_sequence,
                           uint32_t output_width);
  ~MultiLayerPerceptronCUDA();

  // Config.

  void setActivationFunction(float (*f)(const float),
                             float (*f_dot)(const float));

  void setDataset(std::vector<SamplePoint> *training_set,
                  std::vector<SamplePoint> *test_data);


  // Random initial weight
  void randomInit();
  // Random init with scaling 
  void HeRandomInit();


  // Interface between CPU/GPU

  void copyNNToDevice();
  void copyNNFromDevice();

  void passDatasetToDevice();

  // Methods extended for CUDA use:

  void forwardPass(float* d_input);
  void backwardPass(uint32_t correct_class_dix,
                    const float* d_input);
  void updateAllWeights(const uint32_t batch_size);

  // Stochastic gradient descend

  void feedBatchAndUpdate(const int batch_size,const int starting_index);  

  float runEpoch(const int batch_size);

  float testModel();

  uint8_t returnPredictedLabelCUDA();

};

#endif // !CUDA_MULTI_LAYER_PERCEPTRON_HPP
