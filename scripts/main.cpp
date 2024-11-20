#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <Eigen/Dense>
#include <opencv4/opencv2/opencv.hpp>
#include "CommonLib/cifarHandlers.hpp"
#include "NeuronLayer.hpp"
#include "activationFunctions.hpp"
#include <time.h>



int main(){
  srand(420);
  Cifar10Handler c10("../data/cifar-10-batches-bin"); 

  std::vector<SamplePoint> training_set=c10.getTrainingList(500);
  std::vector<SamplePoint> test_set=c10.getTestList(100);




  return 0;
}

