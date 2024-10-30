#include "NearestCentroidClassifier.hpp"
#include "NearestNeighborClassifiers.hpp"


std::vector<SamplePoint> getCentroidSet(std::vector<SamplePoint> &set,
                                        int class_count){
  // Final object to be returned
  // Each element is the centroid of a distinct class
  std::vector<SamplePoint> class_set(class_count);

  // Set all elements to zero 
  for(int i=0;i<class_set.size();i++){
    class_set[i].vector.setZero();
    class_set[i].label=i;
  }

  // Keeping track of all samples found for each class.
  int* sample_per_class_counters=new int[class_count];
  for(int i=0;i<class_count;i++)
    sample_per_class_counters[i]=0;
  
  // Get all sample sums
  int current_id;
  for(int i=0;i<set.size();i++){
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


int classifyNearestCentroid(E::VectorXf &input_point,
                            std::vector<SamplePoint> &centroid_set){
  // Call 1 nearest neighbor classifier with double type
  return classify_1_nearest_neighbor(input_point,
                                     centroid_set);
} 
