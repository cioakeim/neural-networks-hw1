#ifndef K_NEAREST_NEIGHBOR_CLASSIFIERS_HPP
#define K_NEAREST_NEIGHBOR_CLASSIFIERS_HPP

#include "NdPoint.hpp"
#include "NdPointSet.hpp"

/**
 * @brief Classifies an input point using the nearest neighbor criterion.
 *
 * @param[in] input_point Input point in N-D space.
 * @param[in] set The set of other points (each one with a class label).
 *
 * @return Class ID of the final decision.
*/
template <typename T>
int classify_1_nearest_neighbor(NdPoint<T> input_point, NdPointSet<T> set){
  // Get nearest point.
  NdPointSet<T> *nearest_set=k_nearest_neighbors(input_point,set,1);
  int final_id=nearest_set->elements[0].class_id;

  // Cleanup the call
  delete nearest_set;

  // Return class id
  return final_id;
}


/**
 * @brief Classifies an input point using the 3 nearest neighbors criterion.
 *
 * @param[in] input_point Input point in N-D space.
 * @param[in] set The set of other points (each one with a class label).
 *
 * @return Class ID of the final decision.
*/
template <typename T>
int classify_3_nearest_neighbor(NdPoint<T> input_point, NdPointSet<T> set){
  // Get 3 nearest points.
  NdPointSet<T> *nearest_set=k_nearest_neighbors(input_point,set,3);

  // Since only 3 points are used, the logic is implemented using 
  // basic if statements (majority vote otherwise is more complex)
  
  // Majority cases
  if(nearest_set->elements[0].class_id==nearest_set->elements[1].class_id
  || nearest_set->elements[0].class_id==nearest_set->elements[2].class_id){
    return nearest_set->elements[0].class_id;
  }
  if(nearest_set->elements[1].class_id==nearest_set->elements[1].class_id){
    return nearest_set->elements[1].class_id;
  }
  // If there is no majority, the winner is the closest element
  return nearest_set->elements[0].class_id;
}

#endif
