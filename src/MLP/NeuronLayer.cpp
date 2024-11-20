#include "MLP/NeuronLayer.hpp"
#include <iostream>
#include <Eigen/Dense>
#include <random>
#include <fstream>

#include <omp.h>


// Initialize input layer (hidden layer 1).
NeuronLayer::NeuronLayer(uint32_t input_size,uint32_t layer_size){
  this->weights=E::MatrixXf(layer_size,input_size);
  this->weightGradients=E::MatrixXf(layer_size,input_size);
  this->biases=E::VectorXf(layer_size);
  this->biasGradients=E::VectorXf(layer_size);
  this->outputs=E::VectorXf(layer_size);
  this->f=nullptr;
  this->f_dot=nullptr;
  this->local_errors=E::VectorXf(layer_size);
}


// Initialize intermediate hidden layer.
NeuronLayer::NeuronLayer(NeuronLayer* previous_layer,uint32_t layer_size){
  // Input size is output size of previous_layer.
  const uint32_t input_size=previous_layer->getOutputsRef().size();
  // Init matrices.
  this->weights=E::MatrixXf(layer_size,input_size);
  this->weightGradients=E::MatrixXf(layer_size,input_size);
  this->biases=E::VectorXf(layer_size);
  this->biasGradients=E::VectorXf(layer_size);
  this->outputs=E::VectorXf(layer_size);
  this->f=nullptr;
  this->f_dot=nullptr;
  this->local_errors=E::VectorXf(layer_size);
}

// Retrieve from storage
NeuronLayer::NeuronLayer(std::string folder_path){
  std::ifstream is;
  is.open(folder_path+"/weights.csv",std::ios::in);
  int rows,cols;
  is>>rows>>cols;
  this->weights=E::MatrixXf(rows,cols);
  for(int i=0;i<rows;i++){
    for(int j=0;j<cols;j++){
      is>>weights(i,j);
    }
  }
  is.close();

  is.open(folder_path+"/biases.csv",std::ios::in);
  int size;
  is>>size;
  this->biases=E::VectorXf(size);
  for(int i=0;i<size;i++){
    is>>biases(i);
  }
  is.close();
  const int input_size=cols;
  const int layer_size=rows;
  this->weightGradients=E::MatrixXf(layer_size,input_size);
  this->biasGradients=E::VectorXf(layer_size);
  this->outputs=E::VectorXf(layer_size);
  this->f=nullptr;
  this->f_dot=nullptr;
  this->local_errors=E::VectorXf(layer_size);
}


// Simple print methods
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
  std::cout << this->local_errors<< std::endl;
}

void NeuronLayer::storeLayer(std::string folder_path){
    std::ofstream os;
    os.open(folder_path+"/weights.csv",std::ios::out);
    os<<weights.rows()<<" "<<weights.cols()<<"\n";
    for(int i=0;i<weights.rows();i++){
      for(int j=0;j<weights.cols();j++){
        os<<weights(i,j)<<" ";
      }
      os<<"\n";
    }
    os.close();

    os.open(folder_path+"/biases.csv",std::ios::out);
    os<<biases.size()<<" "<<"\n"; 
    for(int i=0;i<biases.size();i++){
      os<<biases(i)<<"\n";
    }
    os.close();
}

// Configuration:

// Initializes the weights to random small values.
void NeuronLayer::assertRandomWeights(){
  this->weights.setRandom();
  this->biases.setRandom();
}

// Inialization taking into account the fan-in
void NeuronLayer::HeRandomInit(){
  const int rows=weights.rows();
  const int cols=weights.cols();
  const float stddev= std::sqrt(2.0f/rows);
  // Init rng 
  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<float> dist(0.0,stddev);

  for(int i=0;i<rows;i++){
    for(int j=0;j<cols;j++){
      weights(i,j)=dist(gen);
    }
    biases(i)=dist(gen);
  }



}


// Copys a certain weight matrix and bias vector to the layer. 
void NeuronLayer::insertWeightMatrixAndBias(E::MatrixXf &weights,
                                            E::VectorXf &biases){
  this->weights=weights;
  this->biases=biases;
}


// Sets the activation function of the layer.
void NeuronLayer::setActivationFunction(E::VectorXf (*func)(const E::VectorXf&),
                                        E::VectorXf (*der)(const E::VectorXf&)){
  this->f=func;
  this->f_dot=der;
}


// Forward pass methods.


// Calculates output based on input vector.
void NeuronLayer::setOutput(const E::VectorXf &input){
  // Initial computation.
  //std::cout<<"Weights range: "<<weights.maxCoeff()<<" "<<weights.minCoeff()<<std::endl;
  //std::cout<<"Forward Propagate:"<<std::endl;
  const E::VectorXf activation=weights*input+biases;

  //std::cout<<"Activation range: "<<activation.minCoeff()<<
  //        " "<<activation.maxCoeff()<<std::endl;
  // Insert through non linear function
  outputs=f(activation);
  //std::cout<<"Output range: "<<outputs.minCoeff()<<
  //        " "<<outputs.maxCoeff()<<std::endl;
}


// Output softmax 
void NeuronLayer::setSoftMaxOutput(const E::VectorXf& input){
  const E::VectorXf activation=weights*input+biases;
  const float maxCoeff=activation.maxCoeff();
  //std::cout<<"SM Activation: "<<activation<<std::endl;
  //std::cout<<"Max coeff: "<<maxCoeff<<std::endl;
  E::VectorXf normalized_output=(activation.array()-maxCoeff);
  //std::cout<<"Activation - max: "<<normalized_output<<std::endl;
  E::VectorXf exps=normalized_output.array().exp();
  //std::cout<<"Exp: "<<exps<<std::endl;
  outputs=exps/(exps.sum());
  //std::cout<<"SM Output: "<<outputs<<std::endl;
}

// Backward pass methods.

// Set outputgradients with softmax 
void NeuronLayer::setSoftMaxErrors(uint32_t correct_class_idx){
  local_errors=outputs;
  local_errors[correct_class_idx]--;
}

// (For all other vectors) calculated the local gradients based on 
// the next layer's local gradients.
void NeuronLayer::setLocalErrors(const E::VectorXf& next_errors,
                                const E::MatrixXf& next_weights){
  //std::cout<<"Weights range: "<<next_weights.maxCoeff()<<" "<<next_weights.minCoeff()<<std::endl;
  //std::cout<<"Output range: "<<outputs.maxCoeff()<<" "<<outputs.minCoeff()<<std::endl;
  local_errors=(next_weights.transpose()*next_errors).
                cwiseProduct(f_dot(outputs));
  //std::cout<<"local_gradients range: "<<local_errors.minCoeff()<<" "<<local_errors.maxCoeff()<<std::endl;
}

// Reset gradients after accumulation.
void NeuronLayer::resetAllGradients(){
  this->weightGradients.setZero();
  this->biasGradients.setZero();
}


// Accumulates weight and bias gradients
void NeuronLayer::accumulateGradients(const E::VectorXf& input){
  this->weightGradients+=(this->local_errors)*input.transpose();
  this->biasGradients+=(this->local_errors);
}


// Updates all the weights according to the local gradients. 
void NeuronLayer::updateWeights(const uint32_t batch_size){
  //std::cout<<"Current weights range: "<<weights.minCoeff()<<" "<<weights.maxCoeff()<<std::endl;
  const float rate=RATE/batch_size;
  const float lambda=WEIGHT_DECAY_RATE;
  //std::cout<<"Updating weights:"<<std::endl;
  const E::MatrixXf temp=(rate)*weightGradients;
  //std::cout<<"Mean square of weight gradients: "<<temp.array().square().maxCoeff()<<std::endl;;
  this->weights-=(rate)*(weightGradients)+(lambda*weights);
  this->biases-=(rate)*(biasGradients)+(lambda*biases); 
}







