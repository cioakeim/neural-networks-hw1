#include "NeuronLayer.hpp"
#include <iostream>



// Empty constructor 
NeuronLayer::NeuronLayer()
  : input(nullptr),
    next_gradients(nullptr),
    next_weights(nullptr),
    layer_size(0){}


// Initialize input layer (hidden layer 1).
NeuronLayer::NeuronLayer(uint32_t input_size,uint32_t layer_size)
  : input(nullptr),
    next_gradients(nullptr),
    next_weights(nullptr),
    layer_size(layer_size){
  this->weights=E::MatrixXf(layer_size,input_size);
  this->biases=E::VectorXf(layer_size);
  this->outputs=E::VectorXf(layer_size);
  this->activation_function=nullptr;
  this->activation_derivative=nullptr;
  this->local_gradients=E::VectorXf(layer_size);
}


// Initialize intermediate hidden layer.
NeuronLayer::NeuronLayer(NeuronLayer* previous_layer,uint32_t layer_size)
  : input(previous_layer->getOutputsPointer()),
    next_gradients(nullptr),
    next_weights(nullptr),
    layer_size(layer_size){
  // Input size is output size of previous_layer.
  const uint32_t input_size=this->input->size();
  // Init matrices.
  this->weights=E::MatrixXf(layer_size,input_size);
  this->biases=E::VectorXf(layer_size);
  this->outputs=E::VectorXf(layer_size);
  this->activation_function=nullptr;
  this->activation_derivative=nullptr;
  this->local_gradients=E::VectorXf(layer_size);
  // Update the previous_layer's next_layer pointer.
  previous_layer->setNextLayerInfo(this);
}


// Initializes the weights to random small values.
void NeuronLayer::assertRandomWeights(){
  this->weights.setRandom();
  this->weights=this->weights.cwiseAbs();
  this->biases.setRandom();
  this->biases=this->biases.cwiseAbs();
}


// Copys a certain weight matrix and bias vector to the layer. 
void NeuronLayer::insertWeightMatrixAndBias(E::MatrixXf &weights,
                                            E::VectorXf &biases){
  this->weights=weights;
  this->biases=biases;
}


// Sets the next layer 
void NeuronLayer::setNextLayerInfo(NeuronLayer* next_layer){
  this->next_weights=next_layer->getWeightsPointer();; 
  this->next_gradients=next_layer->getLocalGradientsPointer();
}


// Sets the activation function of the layer.
void NeuronLayer::setActivationFunction(E::VectorXf (*func)(const E::VectorXf&)){
  this->activation_function=func;
}

void NeuronLayer::setActivationDerivative(E::VectorXf (*func)(const E::VectorXf&)){
  this->activation_derivative=func;
}


// Forward pass methods.

// (For input layer only) calculates output based on input vector.
void NeuronLayer::assertInput(const E::VectorXf &input){
  // Save refernce to input 
  this->input=&input;
  // Initial computation.
  const E::VectorXf activation=this->weights*input+this->biases;
  // Insert through non linear function
  this->outputs=this->activation_function(activation);
}
// (For the rest of the layers) calculates output vector based on previous 
// layer.
void NeuronLayer::forwardPropagate(){
  this->assertInput(*input);
}

// Backward pass methods.

// (For output layer only) asserts the output layer's local gradients.
// The given vector is the expected output of the layer.
void NeuronLayer::assertOutputLocalGradient(const E::VectorXf &expected_output){
  this->local_gradients=(this->outputs-expected_output).cwiseProduct(
                         this->activation_derivative(this->outputs));
  std::cout << "This one's gradients: " << std::endl;
  this->printGradients();
}
// (For all other vectors) calculated the local gradients based on 
// the next layer's local gradients.
void NeuronLayer::backPropagate(){
  std::cout << "Forward weights and gradients: " << std::endl;
  std::cout << *next_weights << std::endl;
  std::cout << *next_gradients << std::endl;

  std::cout << "f'(uk):" << std::endl;
  std::cout << this->activation_derivative(this->outputs) << std::endl;
  this->local_gradients=(next_weights->transpose()*
                        (*next_gradients)).cwiseProduct(
                          this->activation_derivative(this->outputs));
  std::cout << "Result: " << std::endl;
  this->printGradients();
  
}
// Updates all the weights according to the local gradients. 
void NeuronLayer::updateWeights(){
  const float rate=0.005;
  this->weights-=rate*(this->local_gradients)*(*input).transpose();
  this->biases-=rate*(this->local_gradients);
}


void NeuronLayer::printWeights(){
  std::cout << this->weights << std::endl;
}

void NeuronLayer::printBiases(){
  std::cout << this->biases << std::endl;
}

void NeuronLayer::printOutputs(){
  std::cout << this->outputs << std::endl;
}

void NeuronLayer::printGradients(){
  std::cout << this->local_gradients << std::endl;
}


