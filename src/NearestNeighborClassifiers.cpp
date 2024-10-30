#include "NearestNeighborClassifiers.hpp"
#include "KNearestNeighbors.hpp"
#include <iostream>


int classify_1_nearest_neighbor(E::VectorXf &input_point,
                                std::vector<SamplePoint> &set){
  // Get nearest point.
  std::vector<SamplePoint*> nearest_set=k_nearest_neighbors(input_point,set,1);
  std::cout << "This is done" << std::endl;

  int final_id=nearest_set[0]->label;

  std::cout << "This is done" << std::endl;
  // Return class id
  return final_id;
}

int classify_3_nearest_neighbor(E::VectorXf &input_point, 
                                std::vector<SamplePoint> &set){
  // Get 3 nearest points.
  std::vector<SamplePoint*> nearest_set=k_nearest_neighbors(input_point,set,3);

  // Since only 3 points are used, the logic is implemented using 
  // basic if statements (majority vote otherwise is more complex)
  int end_id=-1; 
  // Majority cases
  if(nearest_set[0]->label==nearest_set[1]->label
  || nearest_set[0]->label==nearest_set[2]->label){
    return nearest_set[0]->label;
  }
  if(nearest_set[1]->label==nearest_set[2]->label){
    return nearest_set[1]->label;
  }
  // If there is no majority, the winner is the closest element
  return nearest_set[0]->label;
}
