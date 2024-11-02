#ifndef NEAREST_CENTROID_CLASSIFIER_HPP
#define NEAREST_CENTROID_CLASSIFIER_HPP

#include <eigen3/Eigen/Dense>
#include "basicStructs.hpp"
#include <vector>

namespace E=Eigen;

/**
 * @brief Generates a set of centroids from a set of samples,
 * where each sample has a unique class id. 
 *
 * The number of classes is known at the beginning of the algorithm.
 *
 * @param[in] set Set of points, each one with their own class_id
 * @param[in] class_count Number of distinct classes.
 *
 * @return Pointer to new centroid object.
*/
std::vector<SamplePoint> getCentroidSet(std::vector<SamplePoint> &set,
                                        int class_count);


/**
 * @brief Classifies a point using nearest centroid algorithm. 
 *
 * @param[in] input_point The point to be classified.
 * @param[in] centroid_set The set of centroids, one for each class.
 *
 * @return class id of prediction.
*/
uint8_t classifyNearestCentroid(E::VectorXf &input_point,
                            std::vector<SamplePoint> &centroid_set);


#endif
