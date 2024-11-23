#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <Eigen/Dense>
#include "CommonLib/cifarHandlers.hpp"
#include "CommonLib/basicFuncs.hpp"
#include "CommonLib/LogHandler.hpp"
#include "MLP/ActivationFunctions.hpp"
#include "MLP/NewMLP.hpp"
#include <time.h>
#include <csignal>

#define TEST_SIZE 10e3

// Usage ./testStoredModel [dataset_path] [network_path] [name]
int main(int argc,char* argv[]){
  std::string dataset_path="../data/cifar-10-batches-bin";
  std::string network_root_path="../data/saved_networks";
  std::string name="network_8";
  if(argc>1){
    dataset_path=argv[1];
    network_root_path=argv[2];
    name=argv[3];
  }
  // Loading dataset...
  std::cout<<"Loading dataset.."<<std::endl;
  Cifar10Handler c10(dataset_path); 
  std::vector<SamplePoint> training_set=c10.getTrainingList(1);
  std::vector<SamplePoint> test_set=c10.getTestList(TEST_SIZE);
  training_set.clear();
  test_set.clear();

  // Learning rate and batch size arent used since there isn't training done.
  std::cout<<"Loading network.."<<std::endl;
  MLP mlp=MLP(network_root_path,name,reLU,reLUder,0,0);
  mlp.insertDataset(training_set, test_set);

  float J_test,accuracy;
  // Test 
  std::cout<<"Starting testing.."<<std::endl;
  mlp.testModel(J_test, accuracy);
  std::cout<<"Finished."<<std::endl;
  std::cout<<"J_test: "<<J_test<<"\nAccuracy: "<<accuracy<<std::endl;

  return 0;
}