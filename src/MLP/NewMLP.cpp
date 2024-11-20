#include "MLP/NewMLP.hpp"
#include <fstream> 
#include <filesystem>
#include <string>
#include <iostream>
#include <random>

namespace fs=std::filesystem;

// Initialize with trash
MLP::MLP(const std::vector<int>& layer_sizes,
    const int input_size,
    VectorFunction activation_function,
    VectorFunction activation_derivative,
    float learning_rate,
    int batch_size):
  depth(layer_sizes.size()),
  activation_function(std::move(activation_function)),
  activation_derivative(std::move(activation_derivative)),
  learning_rate(learning_rate),
  batch_size(batch_size){
  
  layers.emplace_back(input_size,layer_sizes[0],batch_size);
  for(int i=1;i<layer_sizes.size();i++){
    layers.emplace_back(layer_sizes[i-1],layer_sizes[i],
                        batch_size);
  }
}

// Auxiliary
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


// Load from file
MLP::MLP(std::string file_path,std::string name,
         VectorFunction activation_function,
         VectorFunction activation_derivative,
         float learning_rate,int batch_size):
  activation_function(std::move(activation_function)),
  activation_derivative(std::move(activation_derivative)),
  learning_rate(learning_rate),
  batch_size(batch_size){
  this->name=name;
  fs::path path(file_path+"/"+name);
  int layer_count=count_directories_in_path(path);
  for(int i=0;i<layer_count;i++){
    layers.emplace_back(file_path+"/"+name+"/layer"+std::to_string(i),
                        batch_size);
  }
}


// Do forward and backward pass in batches
void MLP::forwardBatchPass(const MatrixXf& input){
  // Initial layer
  layers[0].activations=activation_function((layers[0].weights*input).colwise()
                                            +layers[0].biases);

  for(int i=1;i<depth-1;i++){
    layers[i].activations=activation_function((layers[i].weights*
                                              layers[i-1].activations).colwise()+
                                              layers[i].biases);
  }
  // Softmax output 
  layers[depth-1].activations=(layers[depth-1].weights*layers[depth-2].activations).colwise()+
                               layers[depth-1].biases;
  softMaxForward(layers[depth-1].activations);
}

float MLP::getBatchLosss(const VectorXi& correct_labels){
  const float epsilon=1e-7;
  float sum=0;
  for(int i=0;i<batch_size;i++){
    sum-=log(layers[depth-1].activations(correct_labels[i],i)+epsilon); 
  }
  return sum/batch_size;
}

void MLP::backwardBatchPass(const MatrixXf& input,
                       const VectorXi& correct_labels){
  // Initial errors
  softMaxBackward(correct_labels);

  // Backward propagate 
  for(int i=depth-2;i>=0;i--){
    layers[i].errors=(layers[i+1].weights.transpose()*layers[i+1].errors).cwiseProduct(
                      activation_derivative(layers[i].activations)
    );
  }  
  // Reduce errors and update
  layers[0].updateWeights(input,learning_rate,batch_size);
  for(int i=1;i<depth;i++){
    layers[i].updateWeights(layers[i-1].activations,learning_rate,batch_size);
  }
}

void MLP::shuffleDataset(){
  int training_size=training_set.cols();
  // Shuffle training set 
   // Generate a random permutation of column indices
  std::vector<int> indices(training_size);
  std::iota(indices.begin(), indices.end(), 0);  // Fill indices with 0, 1, ..., cols-1
  std::random_device rd;
  std::mt19937 gen(rd());
  std::shuffle(indices.begin(), indices.end(), gen);
  for(int i=0;i<training_size;i++){
    training_set.col(i).swap(training_set.col(indices[i]));
    int temp=training_labels[i];
    training_labels[i]=training_labels(indices[i]);
    training_labels[indices[i]]=temp;
  }

}

float MLP::runEpoch(){
  int training_size=training_set.cols();
  shuffleDataset();
  VectorXf batch_losses=VectorXf(training_size/batch_size);

  for(int idx=0;idx<training_size;idx+=batch_size){
    std::cout<<"Idx: "<<idx<<std::endl;
    const MatrixXf& input=training_set.middleCols(idx,batch_size);
    const VectorXi& labels=training_labels.segment(idx,idx+batch_size);
    forwardBatchPass(input);
    batch_losses[idx/batch_size]=getBatchLosss(labels);
    std::cout<<"Loss: "<<batch_losses[idx/batch_size]<<std::endl;
    backwardBatchPass(input,labels);
  }
  return batch_losses.mean();
}

void MLP::testModel(float& J_test,float& accuracy){
  const int batch_size=1000;
  const int test_size=test_set.cols();
  int success_count=0; 
  VectorXf batch_losses=VectorXf(test_size/batch_size);
  for(int idx=0;idx<test_size;idx+=batch_size){
    const MatrixXf& input=test_set.middleCols(idx,batch_size);
    const VectorXi& labels=test_labels.segment(idx,idx+batch_size);
    forwardBatchPass(input);
    batch_losses[idx/batch_size]=getBatchLosss(labels);
    // Count successful predictions
    for(int i=0;i<batch_size;i++){
      E::Index idx;
      layers[depth-1].activations.col(i).maxCoeff(&idx);
      if(idx==labels[i]){
        success_count++;
      }
    }
  }
  J_test=batch_losses.mean();
  accuracy=static_cast<float>(success_count)/test_size;
}

// I/O
void MLP::store(std::string file_path){
  // Create directory
  fs::path dir(file_path);
  if(!fs::exists(dir)){
    fs::create_directories(dir);
  }
  fs::create_directory(file_path+"/"+name);
  std::ofstream os;
  for(int i=0;i<layers.size();i++){
    // Open main module
    std::string folder=file_path+"/"+name+"/layer"+std::to_string(i);
    fs::create_directory(folder);
    // Store files
    layers[i].store(folder);
  }
}


// Softmax methods
void MLP::softMaxForward(MatrixXf& activations){
  // Get max of each column
  const E::RowVectorXf maxCoeff=activations.colwise().maxCoeff();
  // Subtract for numerical stability and exp
  const MatrixXf exps=(activations.rowwise()-maxCoeff).array().exp();
  // Get sum of each column 
  const E::RowVectorXf col_sum=exps.colwise().sum();
  activations=exps.array().rowwise()/col_sum.array();
}


void MLP::softMaxBackward(const VectorXi& correct_labels){
  layers[depth-1].errors=layers[depth-1].activations;
  const int sample_size=layers[depth-1].activations.cols();
  for(int i=0;i<sample_size;i++){
    layers[depth-1].errors(correct_labels(i),i)--;
  }
}


// Config:
void MLP::randomInit(){
  for(int i=0;i<depth;i++){
    layers[i].HeRandomInit();
  }
}


void MLP::insertDataset(std::vector<SamplePoint>& training_set,
                        std::vector<SamplePoint>& test_set){
  this->training_set=MatrixXf(training_set[0].vector.size(),training_set.size());
  this->training_labels=VectorXi(training_set.size());
  this->test_set=MatrixXf(test_set[0].vector.size(),test_set.size());
  this->test_labels=VectorXi(test_set.size());
  for(int i=0;i<this->training_set.cols();i++){
    this->training_set.col(i)=training_set[i].vector; 
    this->training_labels[i]=training_set[i].label;
  }
  for(int i=0;i<this->test_set.cols();i++){
    this->test_set.col(i)=test_set[i].vector; 
    this->test_labels[i]=test_set[i].label;
  }
}









