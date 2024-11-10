#ifndef NEURON_LAYER_HPP
#define NEURON_LAYER_HPP 

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/src/Core/Matrix.h>
#include <vector>

namespace E=Eigen;


/**
 * @brief A layer of neurons in a feed-forward NN.
 *
 *
*/
class NeuronLayer{
private:
  // Relevant information of previous and next layers:
  const E::VectorXf *input;
  const E::VectorXf *next_gradients;
  const E::MatrixXf *next_weights;
  // These are all local information of the layer:
  uint32_t layer_size; //< The amount of output nodes on this layer.
  E::MatrixXf weights; //< The weights of the layer that determine this
                       // layer's outputs. 
  E::VectorXf biases;  //< For computation efficiency, biases are a separate field.
  E::VectorXf outputs; //< The output of the layer.
  E::VectorXf (*activation_function)(const E::VectorXf&); //< The activation function of the layer.
  E::VectorXf (*activation_derivative)(const E::VectorXf&); //< Derivative of activation_function
  E::VectorXf local_gradients; //< The local gradients vector of the layer. 

  

public:
  // Empty constructor 
  NeuronLayer();
  // Initialize input layer (hidden layer 1).
  NeuronLayer(uint32_t input_size,uint32_t layer_size);
  // Initialize intermediate hidden layer.
  NeuronLayer(NeuronLayer* previous_layer,uint32_t layer_size);
  // Initializes the weights to random small values.
  void assertRandomWeights();
  // Copys a certain weight matrix and bias vector to the layer. 
  void insertWeightMatrixAndBias(E::MatrixXf &weights,E::VectorXf &biases);
  // Sets the next layer 
  void setNextLayerInfo(NeuronLayer* next_layer);
  // Sets the activation function of the layer.
  void setActivationFunction(E::VectorXf (*func)(const E::VectorXf&));
  // Set the derivative of the activation function 
  void setActivationDerivative(E::VectorXf (*func)(const E::VectorXf&));
  // Return the reference to weights,local_gradients,outputs 
  // for use of adjacent layers (read-only).
  const E::MatrixXf* getWeightsPointer() const {return &weights;}
  const E::VectorXf* getOutputsPointer() const {return &outputs;}
  const E::VectorXf* getLocalGradientsPointer() const {return &local_gradients;}

  void printWeights();
  void printBiases();
  void printOutputs();
  void printGradients();


  // Forward pass methods.

  // (For input layer only) calculates output based on input vector.
  void assertInput(const E::VectorXf &input);
  // (For the rest of the layers) calculates output vector based on previous 
  // layer.
  void forwardPropagate();

  // Backward pass methods.
  
  // (For output layer only) asserts the output layer's local gradients.
  // The given vector is the expected output of the layer.
  void assertOutputLocalGradient(const E::VectorXf &expected_output);
  // (For all other vectors) calculated the local gradients based on 
  // the next layer's local gradients.
  void backPropagate();
  // Updates all the weights according to the local gradients. 
  void updateWeights();

};


#endif
