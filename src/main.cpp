#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include "NdPointSet.hpp"
#include "NearestCentroidClassifier.hpp"
#include "NearestNeighborClassifiers.hpp"

int main(){
  NdPoint<int> input_point= NdPoint<int>(5);
  input_point.set_to_zero();

  NdPointSet<int> sample_set= NdPointSet<int>(7,5);

  std::ifstream test_file("../data/testData.csv");
  if(!test_file.is_open()){
    std::cerr << "Can't open file..." << std::endl;
    return -1;
  }

  std::string line;
  std::string value;
  for(int i=0;std::getline(test_file,line);i++){
    std::stringstream ss(line);
    std::cout << "This is line: " << line << std::endl;
    sample_set.elements[i].class_id=i;
    for(int j=0;std::getline(ss,value,',');j++){
      sample_set.elements[i].elements[j]=std::stoi(value);
    }
    std::cout << "Line stored\n" << std::endl;
  } 

  for(int i=0;i<7;i++){
    for(int j=0;j<5;j++){
      std::cout << sample_set.elements[i].elements[j] << " ";

    }
    std::cout << std::endl;
  }

  std::cout << classify_3_nearest_neighbor(input_point,sample_set) << std::endl;
  std::cout << "TEST" << std::endl;

  return 0;
}
