#ifndef NEAREST_CENTROID_CLASSIFIER_HPP
#define NEAREST_CENTROID_CLASSIFIER_HPP

#include "NdPoint.hpp"
#include "NdPointSet.hpp"
#include "NearestNeighborClassifiers.hpp"


/**
 * @brief Generates a set of centroids from a set of samples,
 * where each sample has a unique class id. 
 *
 * The number of classes is known at the beginning of the algorithm.
 *
 * @param[in] set Set of points, each one with their own class.
 * @param[in] class_count Number of distinct classes.
 *
 * @return Pointer to new centroid object.
*/
template <typename T>
NdPointSet<double>* getCentroidSet(NdPointSet<T> &set, int class_count){
  // Check for faulty set.
  if(set.count<=0){
    return nullptr;
  }
  // Final object to be returned
  // Each element is the centroid of a distinct class
  NdPointSet<double>* class_set=new NdPointSet<double>(
                                    class_count,
                                    set.elements[0].dimension);
  // Set all elements to zero 
  class_set->set_to_zero();

  // Keeping track of all samples found for each class.
  int* sample_per_class_counters=new int[class_count];
  for(int i=0;i<class_count;i++)
    sample_per_class_counters[i]=0;
  
  // Get all sample sums
  int current_id;
  for(int i=0;i<set.count;i++){
    current_id=set.elements[i].class_id;
    sample_per_class_counters[current_id]++;
    class_set->elements[current_id].add_point(set.elements[i]);
  }

  // Create mean using the sample counters.
  for(int i=0;i<class_count;i++){
    for(int j=0;j<class_set->elements->dimension;j++){
      class_set->elements[i].elements[j]/=sample_per_class_counters[i];
    }
  }

  // Cleanup and return
  delete[] sample_per_class_counters;
  return class_set;
}


/**
 * @brief Classifies a point using nearest centroid algorithm. 
 *
 * @param[in] input_point The point to be classified.
 * @param[in] centroid_set The set of centroids, one for each class.
 *
 * @return class id of prediction.
*/
template <typename T>
int classifyNearestCentroid(NdPoint<T> &input_point,
                              NdPointSet<double> &centroid_set){
  // Convert input point to double 
  NdPoint<double> input_point_double= NdPoint<double>(input_point);
  
  // Call 1 nearest neighbor classifier with double type
  return classify_1_nearest_neighbor(input_point_double,
                                     centroid_set);
} 


#endif
