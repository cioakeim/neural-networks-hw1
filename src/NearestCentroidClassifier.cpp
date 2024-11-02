#include "NearestCentroidClassifier.hpp"
#include "NearestNeighborClassifiers.hpp"
#include <vector>

namespace E=Eigen;

std::vector<SamplePoint> getCentroidSet(std::vector<SamplePoint> &set,
                                        int class_count){
  // Final object to be returned
  // Each element is the centroid of a distinct class
  std::vector<SamplePoint> class_set(class_count);

  // Set all elements to zero 
  int set_dimension=set[0].vector.size();
  for(int i=0;i<class_set.size();i++){
    class_set[i].vector=E::VectorXf::Zero(set_dimension);
    class_set[i].label=i;
  }

  // Keeping track of all samples found for each class.
  int* sample_per_class_counters=new int[class_count];
  for(int i=0;i<class_count;i++)
    sample_per_class_counters[i]=0;
  
  // Get all sample sums
  int current_id;
  int set_size=set.size();
  for(int i=0;i<set_size;i++){
    current_id=set[i].label;
    sample_per_class_counters[current_id]++;
    class_set[current_id].vector+=set[i].vector;
  }

  // Create mean using the sample counters.
  for(int i=0;i<class_set.size();i++){
    class_set[i].vector[i]/=sample_per_class_counters[i];
  }

  // Cleanup and return
  delete[] sample_per_class_counters;
  return class_set;
}


uint8_t classifyNearestCentroid(E::VectorXf &input_point,
                            std::vector<SamplePoint> &centroid_set){
  // Call 1 nearest neighbor classifier with double type
  return classify_1_nearest_neighbor(input_point,
                                     centroid_set);
} 
