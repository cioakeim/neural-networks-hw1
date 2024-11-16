#include "NearestNeighbors/NearestNeighborClassifiers.hpp"
#include "NearestNeighbors/KNearestNeighbors.hpp"
#include <vector>


uint8_t classify_1_nearest_neighbor(E::VectorXf &input_point,
                                std::vector<SamplePoint> &set){
  // Get nearest point.
  std::vector<SamplePoint*> nearest_set=k_nearest_neighbors(input_point,set,1);

  int final_id=nearest_set[0]->label;

  // Return class id
  return final_id;
}

uint8_t classify_3_nearest_neighbor(E::VectorXf &input_point, 
                                std::vector<SamplePoint> &set){
  // Get 3 nearest points.
  std::vector<SamplePoint*> nearest_set=k_nearest_neighbors(input_point,set,3);

  // Since only 3 points are used, the logic is implemented using 
  // basic if statements (majority vote otherwise is more complex)

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
