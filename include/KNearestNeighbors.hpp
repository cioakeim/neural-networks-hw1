#ifndef K_NEAREST_NEIGHBORS_HPP
#define K_NEAREST_NEIGHBORS_HPP

#include "NdPointSet.hpp"
#include "NdPoint.hpp"
#include <limits>

/**
 * @brief Returns the k nearest points to input_point as a point set.
 *
 * @param[in] input_point Input point in N-D space.
 * @param[in] set The set of other points.
 * @param[in] k Number of neighbors that end up on the return set.
 *
 * @return Pointer to set of k nearest points to input_point.
*/
template <typename T>
NdPointSet<T>* k_nearest_neighbors(NdPoint<T> input_point, NdPointSet<T> set, int k){
  // Return set
  NdPointSet<T> *final_set= new NdPointSet<T>();

  // Array containing current nearest points and their distances.
  NdPoint<T> **k_nearest_points= new NdPoint<T>*[k];
  double *k_nearest_distances= new double[k];

  // Initial distance is infinite for all
  for(int i=0;i<k;i++){
    k_nearest_points[i]=nullptr;
    k_nearest_distances[i]=std::numeric_limits<double>::infinity();
  }
  
  // Get k nearest point references
  NdPoint<T>* current_point;
  double current_distance;
  for(int i=0;i<set.count;i++){
    // This point is examined
    current_point=&set.elements[i];

    // Climb up slowly the priority queue to find this point's place.
    for(int j=k-1;j>=0;j--){
      current_distance=input_point.get_distance(*current_point); 
      // If the current entry is better, keep that.
      if(current_distance>k_nearest_distances[j])
        break;

      // If this point is better, move it here.
      if(j+1<k){
        k_nearest_points[j+1]=k_nearest_points[j];
        k_nearest_distances[j+1]=k_nearest_distances[j];
      }
      k_nearest_points[j]=current_point;
      k_nearest_distances[j]=current_distance;
    }
  }

  // Copy k points to final set 
  final_set->count=k;
  for(int i=0;i<k;i++){
    final_set->elements[i]=*k_nearest_points[i];
  }

  return final_set;
}

#endif
