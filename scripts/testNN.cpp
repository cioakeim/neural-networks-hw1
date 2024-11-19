#include <iostream>
#include <vector>
#include <string>
#include <eigen3/Eigen/Dense>
//#include <opencv4/opencv2/opencv.hpp>
#include "NearestNeighbors/NearestCentroidClassifier.hpp"
#include "NearestNeighbors/NearestNeighborClassifiers.hpp"
#include "CommonLib/cifarHandlers.hpp"
#include "CommonLib/LogHandler.hpp"

namespace E=Eigen;

int main(int argc,char* argv[]){
  int training_size=1e4;
  if(argc==2){
    training_size=std::stoi(argv[1]);
  }
  // Will test all functions 
  uint8_t (*test_functions[])(E::VectorXf&,std::vector<SamplePoint>&)={
    classify_1_nearest_neighbor,
    classify_3_nearest_neighbor,
    classifyNearestCentroid
  };
  // Log list
  std::string log_folderpath="../data/csv_logs/";
  std::string log_filenames[]={
    "nn_1_log.csv",
    "nn_3_log.csv",
    "nc_log.csv"
  };
  LogHandler* log;

  // Get all data from CIFAR-10
  Cifar10Handler c10("../data/cifar-10-batches-bin"); 
  std::vector<SamplePoint> training_set=c10.getTrainingList(training_size);
  std::vector<SamplePoint> test_set=c10.getTestList(1e3);

  // Train centroid set. 
  log=new LogHandler(log_folderpath+"training_log.csv");
  log->start_timer();
  std::vector<SamplePoint> class_set=getCentroidSet(training_set,10);
  log->end_timer();
  log->log_time_and_size(training_size);
  delete log;



  // Test all classifiers 
  int test_size=test_set.size();
  int success_counter;
  std::vector<SamplePoint>* current_set=&training_set;
  uint8_t estimated_id;
  for(int func_id=0;func_id<3;func_id++){
    std::cout<<"Testing round "<<func_id<<std::endl;
    std::cout<<"Writing on: "<<log_filenames[func_id]<<std::endl;
    success_counter=0;
    log=new LogHandler(log_folderpath+log_filenames[func_id]);
    // If NNC use class set
    if(func_id==2)
      current_set=&class_set;
    else
      current_set=&training_set;
    log->start_timer();
    for(int i=0;i<test_size;i++){
      // Predict 
      estimated_id=test_functions[func_id](test_set[i].vector,*current_set);
      success_counter+=(estimated_id==test_set[i].label);
    }
    log->end_timer();
    // Log results
    log->log_time_size_value(training_size,
                             static_cast<double>(success_counter)/test_size);

    delete log;
  }

  return 0;
}
