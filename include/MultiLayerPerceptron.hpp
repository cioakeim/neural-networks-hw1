#ifndef MULTI_LAYER_PERCEPTRON_HPP
#define MULTI_LAYER_PERCEPTRON_HPP

#include <vector>
#include <eigen3/Eigen/Dense>
#include "NeuronLayer.hpp"
#include "basicStructs.hpp"

namespace E=Eigen;

/**
 * @brief Implementation of a feed forward MLP.
 */
class MultiLayerPerceptron{
private:
  std::vector<NeuronLayer> layers; //< All intermediate layers.
  NeuronLayer output_layer; //< The output layer
  std::vector<SamplePoint> *training_set; //< The training set
  std::vector<SamplePoint> *test_data; //< The test data
  // For keeping count of training loss function 
  E::VectorXf loss_array;
  
  

public:
  // Standard definition of width and depth.
  MultiLayerPerceptron(uint32_t input_width,
                       uint32_t hidden_width,
                       uint32_t hidden_depth,
                       uint32_t output_width);
  // Initialize based on a layer size sequence.
  // Defines a specific structure of the layers and last element of seq is output layer.
  MultiLayerPerceptron(uint32_t input_width,
                       std::vector<uint32_t> hidden_layer_sequence,
                       uint32_t output_width);
  


  // Config.

  // Random initial weight
  void randomInit();
  // Random init with scaling 
  void HeRandomInit();
  // ONLY FOR HIDDEN LAYERS. OUTPUT IS SOFTMAX
  void setActivationFunction(E::VectorXf (*f)(const E::VectorXf&),
                             E::VectorXf (*der)(const E::VectorXf&));
  void setDataset(std::vector<SamplePoint> *training_set,
                  std::vector<SamplePoint> *test_data);

  // Basic methods.

  // Forward pass for 1 sample
  void forwardPass(E::VectorXf& input);
  // Backward pass for 1 expected output 
  void backwardPass(uint32_t correct_class_idx,const E::VectorXf& input);
  // Whole update of weights
  void updateAllWeights(const uint32_t batch_size);

  // For stochastic gradient descend

  // Run a whole batch in the MLP and update weights. Return 0 if not @ EOF
  int feedBatchAndUpdate(const int batch_size, const int starting_index);

  // Runs a whole epoch on the dataset returns J_train
  float runEpoch(const int batch_size);

  uint8_t returnPredictedLabel();
  // Test the model on test data.
  // Returns % of success.
  float testModel();
};



#endif
