#include "MLP/MultiLayerPerceptron.hpp"
#include "MLP/NeuronLayer.hpp"
#include <cmath>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <string>
#include <random>
#include <algorithm>


#define epsilon 1e-7

namespace fs=std::filesystem;


// Standard definition of width and depth.
MultiLayerPerceptron::MultiLayerPerceptron(uint32_t input_width,
                                           uint32_t hidden_width,
                                           uint32_t hidden_depth,
                                           uint32_t output_width){
  // Case with only output layer
  if(hidden_depth==0){
    output_layer=NeuronLayer(input_width,output_width);
    return;
  }
  // Start layer
  layers.emplace_back(input_width,hidden_width);
  // Link next layers
  for(int i=1;i<hidden_depth;i++){
    layers.emplace_back(&layers[i-1],hidden_width); 
  } 
  // Link output layer
  output_layer=NeuronLayer(&layers[hidden_depth-1],output_width);

}

// Initialize based on a layer size sequence.
// Defines a specific structure of the layers and last element of seq is output layer.
MultiLayerPerceptron::MultiLayerPerceptron(uint32_t input_width,
                                           std::vector<uint32_t> hidden_layer_sequence,
                                           uint32_t output_width){
  if(hidden_layer_sequence.size()<1){
    std::cerr << "Error in layer sequence init..." << std::endl;
    return;
  }
  // Initial layer
  layers.emplace_back(input_width,hidden_layer_sequence[0]);
  const uint32_t size=hidden_layer_sequence.size();
  // Link layers
  for(int i=1;i<size;i++){
    layers.emplace_back(&layers[i-1],hidden_layer_sequence[i]);
  }
  // Output layer link
  output_layer=NeuronLayer(&layers[size-1],output_width); 
}



int count_directories_in_path(const fs::path& path) {
    int dir_count = 0;

    // Check if the given path exists and is a directory
    if (fs::exists(path) && fs::is_directory(path)) {
        // Iterate through the directory entries
        for (const auto& entry : fs::directory_iterator(path)) {
            // Increment the count for each directory (since we know all entries are directories)
            ++dir_count;
        }
    }
    return dir_count;
}
// Create from stored 
MultiLayerPerceptron::MultiLayerPerceptron(std::string file_path,
                     std::string name){
  this->name=name;
  fs::path path(file_path+"/"+name);
  int layer_count=count_directories_in_path(path);
  for(int i=0;i<layer_count-1;i++){
    layers.emplace_back(file_path+"/"+name+"/layer"+std::to_string(i));
  }
  output_layer=NeuronLayer(file_path+"/"+name+"/output_layer");
}


// Configuration:

// Random initial weight
void MultiLayerPerceptron::randomInit(){
  uint32_t depth=layers.size();
  for(int i=0;i<depth;i++){
    layers[i].assertRandomWeights();
  }
  output_layer.assertRandomWeights();
}

// Initialize all w/ He Initialization
void MultiLayerPerceptron::HeRandomInit(){
  uint32_t depth=layers.size();
  for(int i=0;i<depth;i++){
    layers[i].HeRandomInit();
  }
  output_layer.HeRandomInit();
}

// Set all activation functions
void MultiLayerPerceptron::setActivationFunction(E::VectorXf (*f)(const E::VectorXf&),
                                                 E::VectorXf (*der)(const E::VectorXf&)){
  uint32_t hidden_layer_size=layers.size();
  for(int i=0;i<hidden_layer_size;i++){
    layers[i].setActivationFunction(f,der);
  }
  output_layer.setActivationFunction(f,der);
}

// Set the dataset
void MultiLayerPerceptron::setDataset(std::vector<SamplePoint> *training_set,
                std::vector<SamplePoint> *test_data){
  this->training_set=training_set;
  this->test_data=test_data;
  this->loss_array=E::VectorXf((*training_set).size());
  // Create permutation
  training_shuffle=std::vector<int>(training_set->size());
  for(int i=0;i<training_set->size();i++){
    training_shuffle[i]=i;
  }

}

void ensureDirectoryExists(std::string path){
  fs::path dir(path);
  if(!fs::exists(dir)){
    fs::create_directories(dir);
  }
}

// Storing a NN 
void MultiLayerPerceptron::storeToFiles(std::string file_path){
  // Ensure store location exists
  ensureDirectoryExists(file_path);
  // Create main directory 
  fs::create_directory(file_path+"/"+name);
  std::ofstream os;
  for(int i=0;i<layers.size();i++){
    // Open main module
    std::string folder=file_path+"/"+name+"/layer"+std::to_string(i);
    fs::create_directory(folder);
    // Store files
    layers[i].storeLayer(folder);
  }
  // For output
  std::string folder=file_path+"/"+name+"/output_layer";
  fs::create_directory(folder);
  output_layer.storeLayer(folder);
}

void MultiLayerPerceptron::printNN(){
  for(int i=0;i<layers.size();i++){
    std::cout<<"Layer "<<i<<":"<<std::endl;
    std::cout<<"Weights: "<<std::endl;
    layers[i].printWeights();
    std::cout<<"Biases: "<<std::endl;
    layers[i].printBiases();
  }
  std::cout<<"Output Layer "<<":"<<std::endl;
  std::cout<<"Weights: "<<std::endl;
  output_layer.printWeights();
  std::cout<<"Biases: "<<std::endl;
  output_layer.printBiases();

}

// Basic methods.

// Forward pass for 1 sample
void MultiLayerPerceptron::forwardPass(E::VectorXf& input){ 
  if(layers.size()==0){
    output_layer.setSoftMaxOutput(input);
    return;
  }
  layers[0].setOutput(input);
  const uint32_t hidden_depth=layers.size();
  for(int i=1;i<hidden_depth;i++){
    layers[i].setOutput(layers[i-1].getOutputsRef());
  }
  output_layer.setSoftMaxOutput(layers[hidden_depth-1].getOutputsRef());

}

// Backward pass for 1 expected output 
void MultiLayerPerceptron::backwardPass(uint32_t correct_class_idx,
                                        const E::VectorXf& input){
  output_layer.setSoftMaxErrors(correct_class_idx);
  const uint32_t hidden_depth=layers.size();
  if(hidden_depth==0){
    output_layer.accumulateGradients(input);
    return;
  }
  // Create local errors 
  layers[hidden_depth-1].setLocalErrors(output_layer.getLocalErrorsRef(),
                                       output_layer.getWeightsRef());
  for(int i=hidden_depth-2;i>=0;i--){
    layers[i].setLocalErrors(layers[i+1].getLocalErrorsRef(),
                            layers[i+1].getWeightsRef());
  }
  // Accumulate weight gradients
  output_layer.accumulateGradients(layers[hidden_depth-1].getOutputsRef());
  #pragma omp parallel for
  for(int i=hidden_depth-2;i>=1;i--){
    layers[i].accumulateGradients(layers[i-1].getOutputsRef());
  }
  layers[0].accumulateGradients(input);
}

// Whole update of weights
void MultiLayerPerceptron::updateAllWeights(const uint32_t batch_size){
  const uint32_t hidden_depth=layers.size();
  //#pragma omp parallel for
  for(int i=0;i<hidden_depth;i++){
    layers[i].updateWeights(batch_size);
  }
  output_layer.updateWeights(batch_size);
}


// For stochastic gradient descend

// Run a whole batch in the MLP and update weights. Return 0 if not at EOF
int MultiLayerPerceptron::feedBatchAndUpdate(const int batch_size, const int starting_index){
  // Check for out of bounds access to training data
  const int final_idx=std::min(static_cast<int>(training_set->size()-1)
                               ,starting_index+batch_size-1);
  // Make sure all local gradients are zero
  uint32_t hidden_depth=layers.size();
  //std::cout<<"Reseting gradients.."<<std::endl;
  for(int i=0;i<hidden_depth;i++){
    layers[i].resetAllGradients();
  }
  output_layer.resetAllGradients();
  //std::cout<<"Starting batch: "<<starting_index<<std::endl;
  // Run whole batch
  for(int i=starting_index;i<=final_idx;i++){
    const int idx=training_shuffle[i];
    uint8_t &label=(*training_set)[idx].label;
    E::VectorXf &vector=(*training_set)[idx].vector;
    //std::cout<<"Pass: "<<i<<std::endl;
    // Forward pass 
    forwardPass(vector);
    // Add loss to loss_array 
    const float y_predicted=output_layer.getOutputsRef()[label];
    //std::cout<<"Y_p:"<<y_predicted<<std::endl;
    loss_array[i]=-log(y_predicted+epsilon);
    //std::cout<<"Loss: "<<loss_array[i]<<std::endl;
    // backwardPass
    //std::cout<<"Forward done."<<std::endl;
    backwardPass(label,vector);
  } 
  updateAllWeights(batch_size);

  return 0;
}

// Runs a whole epoch on the dataset
float MultiLayerPerceptron::runEpoch(const int batch_size){
  // RNG for shuffling
  std::random_device rd;
  std::mt19937 gen(rd());
  std::shuffle(training_shuffle.begin(), 
               training_shuffle.end(), gen);

  const uint32_t training_size=training_set->size();
  std::cout<<"Got size: "<<training_size<<std::endl;
  for(int i=0;i<training_size;i+=batch_size){
  // Randomize the idx array 
    std::cout<<"Running batch at starting index: "<<i<<std::endl;
    // If returns 1 available data is dry so go home
    feedBatchAndUpdate(batch_size,i);
  }
  return loss_array.mean();
}

// Based on output get predicted index
uint8_t MultiLayerPerceptron::returnPredictedLabel(){
  E::Index idx;
  output_layer.getOutputsRef().maxCoeff(&idx);
  return static_cast<uint8_t>(idx);
}

// Test the model on test data.
// Returns % of success.
float MultiLayerPerceptron::testModel(){
  const uint32_t test_size=test_data->size();
  uint32_t success_count=0;
  // Test all data
  uint8_t prediction;
  for(int i=0;i<test_size;i++){
    forwardPass((*test_data)[i].vector);
    prediction=returnPredictedLabel(); 
    if(prediction==(*test_data)[i].label){
      success_count++;
    }
  }
  return static_cast<float>(success_count)/test_size;
}







