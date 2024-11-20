#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

#include "CommonLib/cifarHandlers.hpp"
#include "CudaMLP/MultiLayerPerceptron.hpp"
#include "MLP/ActivationFunctions.hpp"
#include <time.h>
#include <csignal>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define MAX_TRAINING 200
#define MAX_TEST 100

#define INPUT_WIDTH 32*32*3
#define OUTPUT_WIDTH 10

#define HIDDEN_WIDTH 32*32*2
#define HIDDEN_DEPTH 4

#define EPOCHS 10
#define BATCH_SIZE 10

std::ofstream file;
void handle_signal(int signal) {
  if (file.is_open()) {
      file.flush();
      file.close();
      std::cout << "File flushed and closed due to SIGINT (Ctrl+C)\n";
  }
  exit(signal);
}

int main(){

  srand(420);
  Cifar10Handler c10("../data/cifar-10-batches-bin"); 

  std::cout<<"Loading dataset..."<<std::endl;
  std::vector<SamplePoint> training_set=c10.getTrainingList(MAX_TRAINING);
  std::vector<SamplePoint> test_set=c10.getTestList(MAX_TEST);
  std::cout<<"Loading successful..."<<std::endl;


  std::cout<<"Constructing MLP.."<<std::endl;
  std::vector<uint32_t> layer_sequence={1000,500,250};
  MultiLayerPerceptronCUDA mlp=MultiLayerPerceptronCUDA(INPUT_WIDTH,
                                                layer_sequence,
                                                OUTPUT_WIDTH);
  mlp.setActivationFunction(reLU_el,reLUder_el);
  //mlp.setActivationFunction(tanh,tanhder);
  mlp.setDataset(&training_set,&test_set);
  //mlp.randomInit();
  mlp.HeRandomInit();
  std::cout<<"Construction successful.."<<std::endl;


  // Move model to gpu 
  mlp.copyNNToDevice();
  mlp.passDatasetToDevice();

  file.open("epoch_accuracy_log.csv",std::ios::app);
  std::signal(SIGINT, handle_signal);

  float J_test;
  float J_train;
  for(int epoch=0;epoch<EPOCHS;epoch++){
    std::cout<<"Epoch: "<<epoch<<std::endl;
    J_train=mlp.runEpoch(BATCH_SIZE);
    J_test=mlp.testModel();
    std::cout << "J_test: "<<J_test<< std::endl;
    std::cout << "J_train: "<<J_train<< std::endl;
    file<<epoch<<","<<J_train<<J_test<<std::endl;
  } 
  file.close();  // Close the file when done
  return 0;
}
