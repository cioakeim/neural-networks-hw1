#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <eigen3/Eigen/Dense>
#include <opencv4/opencv2/opencv.hpp>
#include "cifarHandlers.hpp"
#include "NeuronLayer.hpp"
#include <time.h>

E::VectorXf reLU(E::VectorXf& in);
E::VectorXf reLUder(E::VectorXf& reLU_output);


int main(){
  srand(time(NULL));
  Cifar10Handler c10("../data/cifar-10-batches-bin"); 

  std::vector<SamplePoint> training_set=c10.getTrainingList(500);
  std::vector<SamplePoint> test_set=c10.getTestList(100);

  //std::cout<< training_set.size()<< std::endl;
  //std::cout<< test_set.size()<< std::endl;

  NeuronLayer layer= NeuronLayer(2,2);
  NeuronLayer layer2 = NeuronLayer(&layer,3);

  layer2.assertRandomWeights();
  layer2.setActivationFunction(reLU);
  layer2.setActivationDerivative(reLUder);

  layer.assertRandomWeights();
  layer.setActivationFunction(reLU);
  layer.setActivationDerivative(reLUder);

  std::cout << "Layer 1: " << std::endl;
  layer.printWeights();
  layer.printBiases();
  std::cout << "Layer 2: " << std::endl;
  layer2.printWeights();
  layer2.printBiases();

  E::VectorXf input=E::VectorXf(2);
  input << 1,1;

  std::cout <<"Input:"<< std::endl;
  std::cout<< input << std::endl;

  layer.assertInput(input);

  std::cout <<"Outputs 1:"<< std::endl;
  layer.printOutputs();

  layer2.forwardPropagate();
  std::cout <<"Outputs 2:"<< std::endl;
  layer2.printOutputs();
  
  E::VectorXf expected= E::VectorXf(3);
  std::cout <<"Expected:"<< std::endl;
  expected << 1,0,0;

  std::cout << "Gradients" << std::endl;
  layer2.assertOutputLocalGradient(expected);
  layer.backPropagate();

  layer.updateWeights();
  layer2.updateWeights();

  std::cout << "Layer 1: " << std::endl;
  layer.printWeights();
  layer.printBiases();
  std::cout << "Layer 2: " << std::endl;
  layer2.printWeights();
  layer2.printBiases();



  return 0;
}

E::VectorXf reLU(E::VectorXf& in){
  return in.cwiseMax(0.0);
}


E::VectorXf reLUder(E::VectorXf& reLU_output){
  return (reLU_output.array()>0).cast<float>();
}
