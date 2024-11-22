#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <Eigen/Dense>

#include "CommonLib/cifarHandlers.hpp"
#include "CommonLib/basicFuncs.hpp"
#include "MLP/MultiLayerPerceptron.hpp"
#include "CommonLib/LogHandler.hpp"
#include "MLP/ActivationFunctions.hpp"
#include "MLP/NewMLP.hpp"
#include <time.h>
#include <csignal>

#define LEARN_RATE 1e-3

#define MAX_TRAINING 50e3
#define MAX_TEST 10e3

#define INPUT_WIDTH 32*32*3
#define OUTPUT_WIDTH 10

#define HIDDEN_WIDTH 32*32*2
#define HIDDEN_DEPTH 4

#define EPOCHS 15
#define BATCH_SIZE 1000

std::ofstream file;
void handle_signal(int signal) {
  std::cout<<"Terminating..."<<std::endl;
  if (file.is_open()) {
      file.flush();
      file.close();
      std::cout << "File flushed and closed due to SIGINT (Ctrl+C)\n";
  }
  exit(signal);
}


// Usage: ./testMLP [dataset_path] [nn_path] [learning_rate] [batch_size] 
//                  [epochs] [layer_sequence]
int main(int argc,char* argv[]){
  // CONFIG:
  std::string dataset_path="../data/cifar-10-batches-bin";
  std::string nn_path="../data/networks";
  std::string log_filename="epoch_accuracy_log.csv";
  float rate=LEARN_RATE;
  int batch_size=BATCH_SIZE;
  int epochs=EPOCHS;
  std::vector<int> layer_sequence={2048,512,124,10};
  if(argc>1){
    dataset_path=argv[1];
    nn_path=argv[2];
    rate=std::stof(argv[3]);
    batch_size=std::stoi(argv[4]);
    epochs=std::stoi(argv[5]);
    // Convert layer sequence (comma separated)
    std::string arg = argv[6];
    layer_sequence.clear();

    // Use a stringstream to split the input by commas
    std::stringstream ss(arg);
    std::string token;
    // Extract integers from the comma-separated string
    while (std::getline(ss, token, ',')) {
        try {
            // Convert each token to an integer and store it in the vector
            layer_sequence.push_back(std::stoi(token));
        } catch (const std::invalid_argument& e) {
            std::cerr<<"Invalid argument: "<< token <<" is not a valid integer."<< std::endl;
            return 1;
        }
    }
  }
  std::cout<<dataset_path<<std::endl;
  std::cout<<nn_path<<std::endl;
  std::cout<<rate<<std::endl;
  std::cout<<batch_size<<std::endl;
  for(int i=0;i<layer_sequence.size();i++){
    std::cout<<layer_sequence[i]<<std::endl;
  }


  srand(time(NULL));

  // Loading dataset...
  Cifar10Handler c10(dataset_path); 

  std::cout<<"Loading dataset..."<<std::endl;
  std::vector<SamplePoint> training_set=c10.getTrainingList(MAX_TRAINING);
  std::vector<SamplePoint> test_set=c10.getTestList(MAX_TEST);
  std::cout<<"Loading successful..."<<std::endl;


  std::cout<<"Constructing MLP.."<<std::endl;
  MLP mlp=MLP(layer_sequence,INPUT_WIDTH,
              reLU,reLUder,rate,batch_size);

  //MultiLayerPerceptron mlp=MultiLayerPerceptron(nn_path,"testNet");
  mlp.insertDataset(training_set,test_set);
  training_set.clear();
  test_set.clear();
  std::cout<<"Clear"<<std::endl;

  // Initialization
  mlp.randomInit();
  std::cout<<"Construction successful.."<<std::endl;

  // Store current config's info
  std::string nn_root=create_network_folder(nn_path);
  mlp.setStorePath(nn_root);
  file.open(nn_root+"/info.txt",std::ios::out);
  file<<"Rate: "<<rate<<"\nBatch size: "<<batch_size
      <<"\nEpochs: "<<epochs<<"\nLayers: ";
  for(int i=0;i<layer_sequence.size();i++){
    file<<layer_sequence[i]<<",";
  }
  file<<"\n";
  file.close();



  file.open(nn_root+"/"+log_filename,std::ios::out);
  file<<"epoch,J_train,J_test,accuracy"<<std::endl;
  std::signal(SIGINT, handle_signal);
  std::signal(SIGTERM, handle_signal);

  // Keep track of time
  LogHandler log=LogHandler();
  float J_test;
  float J_train;
  float accuracy;
  float best_J_test=INFINITY;

  log.start_timer();
  for(int epoch=0;epoch<EPOCHS;epoch++){
    std::cout<<"Epoch: "<<epoch<<std::endl;
    J_train=mlp.runEpoch();
    mlp.testModel(J_test,accuracy);
    // Store results
    file<<epoch<<","<<J_train<<","<<J_test<<
      ","<<accuracy<<std::endl;
    // If result is best, store 
    if(J_test<best_J_test){
      best_J_test=J_test;
      mlp.store();
    }
  } 
  log.end_timer();
  file.close();  // Close the file when done
  // Store time
  file.open(nn_root+"/info.txt",std::ios::app);
  file<<"Execution time: "<<log.elapsed_seconds()<<"\n";
  file.close();
  return 0;
}
