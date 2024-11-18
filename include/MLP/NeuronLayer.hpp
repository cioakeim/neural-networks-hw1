#ifndef NEURON_LAYER_HPP
#define NEURON_LAYER_HPP 

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/src/Core/Matrix.h>
#include <vector>

namespace E=Eigen;

#define RATE 1e-3
#define WEIGHT_DECAY_RATE 1e-7


/**
 * @brief A layer of neurons in a feed-forward NN.
 *
 *
*/
class NeuronLayer{
protected:
  // Structure
  E::MatrixXf weights; //< The weights of the layer that determine this
                       // layer's outputs. 
  E::VectorXf biases;  //< For computation efficiency, biases are a separate field.
  E::VectorXf outputs; //< The output of the layer.

  // Parameters for outputs
  E::VectorXf (*f)(const E::VectorXf&); //< The activation function of the layer.
  E::VectorXf (*f_dot)(const E::VectorXf&); //< Derivative of activation_function

  // Training parameters
  E::VectorXf local_errors; //< The local gradients vector of the layer. 
  // Accumulators used in stochastic gradient descend
  E::MatrixXf weightGradients;
  E::MatrixXf biasGradients;

  

public:
  // Constructors:

  // Empty constructor
  NeuronLayer(){};
  // Initialize input layer (hidden layer 1).
  NeuronLayer(uint32_t input_size,uint32_t layer_size);
  // Initialize intermediate hidden layer.
  NeuronLayer(NeuronLayer* previous_layer,uint32_t layer_size);

  // Read-only getters 
  const E::MatrixXf& getWeightsRef() const {return weights;}
  const E::VectorXf& getOutputsRef() const {return outputs;}
  const E::VectorXf& getLocalErrorsRef() const {return local_errors;}

  // Print methods
  void printWeights();
  void printBiases();
  void printOutputs();
  void printGradients();


  // Configuration:

  // Initializes the weights to random small values.
  void assertRandomWeights();
  // He Initialization accoutning for fan-in 
  void HeRandomInit();
  // Copys a certain weight matrix and bias vector to the layer. 
  void insertWeightMatrixAndBias(E::MatrixXf &weights,E::VectorXf &biases);
  // Sets the activation function of the layer.
  void setActivationFunction(E::VectorXf (*func)(const E::VectorXf&),
                             E::VectorXf (*der)(const E::VectorXf&));


  // Forward pass methods:

  // Calculate output based on input layer using activation function
  void setOutput(const E::VectorXf& input);
  // Output softmax 
  void setSoftMaxOutput(const E::VectorXf& input);


  // Backward pass methods:
  
  // Set local errors with softmax.
  void setSoftMaxErrors(uint32_t correct_class_idx);

  // Set local errors based on forward layer's parameters
  void setLocalErrors(const E::VectorXf& next_errors,
                      const E::MatrixXf& next_weights);

  // Gradient accumulation methods:

  // Reset all weight gradients
  void resetAllGradients();
  // Accumulates gradients
  void accumulateGradients(const E::VectorXf& input);
  // Updates all the weights according to the weight gradients and batch size
  void updateWeights(const uint32_t batch_size);

};


#endif
