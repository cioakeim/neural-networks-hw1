#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <eigen3/Eigen/Dense>
#include <opencv4/opencv2/opencv.hpp>
#include "KNearestNeighbors.hpp"
#include "NearestCentroidClassifier.hpp"
#include "NearestNeighborClassifiers.hpp"
#include "cifarHandlers.hpp"


int main(){
  Cifar10Handler c10("../data/cifar-10-batches-bin"); 

  std::vector<SamplePoint> training_set=c10.getTrainingList(5e4);
  std::vector<SamplePoint> test_set=c10.getTestList(1e4);

  std::cout<< training_set.size()<< std::endl;
  std::cout<< test_set.size()<< std::endl;

  // Test results.
  int nn_1_counter,nn_3_counter,nc_counter;
  nn_1_counter=nn_3_counter=nc_counter=0;
  uint8_t estimated_id;
  // Test 1-NN
  int count=test_set.size();
  for(int i=0;i<count;i++){
    std::cout<< i << std::endl;
    // Estimate
    estimated_id=classify_3_nearest_neighbor(test_set[i].vector,
                                             training_set);
    nn_1_counter+=(estimated_id==test_set[i].label);
  }
  std::cout<<nn_1_counter<<std::endl;


  return 0;
}
