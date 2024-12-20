#include "NearestNeighbors/KNearestNeighbors.hpp"
#include <vector>
#include <iostream>

namespace E=Eigen;

std::vector<SamplePoint*> k_nearest_neighbors(E::VectorXf &input_point,
                                     std::vector<SamplePoint> &set, 
                                     int k){

  // Array containing current nearest points and their distances.
  std::vector<SamplePoint*> k_nearest_points(k,nullptr);
  double *k_nearest_distances= new double[k];

  // Initial distance is infinite for all
  for(int i=0;i<k;i++){
    k_nearest_distances[i]=std::numeric_limits<double>::infinity();
  }
  
  // Get k nearest point references
  double current_distance;

  int set_size=set.size();
  for(int i=0;i<set_size;i++){
    // This point is examined
    current_distance=(input_point-set[i].vector).norm(); 

    // Climb up slowly the priority queue to find this point's place.
    for(int j=k-1;j>=0;j--){
      // If the current entry is better, keep that.
      if(current_distance>k_nearest_distances[j])
        break;

      // If this point is better, move it here.
      if(j+1<k){
        k_nearest_points[j+1]=k_nearest_points[j];
        k_nearest_distances[j+1]=k_nearest_distances[j];
      }
      k_nearest_points[j]=&set[i];
      k_nearest_distances[j]=current_distance;
    }
  }

  // Cleanup
  delete[] k_nearest_distances;

  return k_nearest_points;
}
