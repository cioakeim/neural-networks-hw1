#ifndef K_NEAREST_NEIGHBORS_HPP
#define K_NEAREST_NEIGHBORS_HPP

#include <Eigen/Dense>
#include <vector>
#include "CommonLib/basicStructs.hpp"

namespace E=Eigen;

/**
 * @brief Returns the k nearest points to input_point as a point set.
 *
 * @param[in] input_point Input point in N-D space.
 * @param[in] set The set of other points.
 * @param[in] k Number of neighbors that end up on the return set.
 *
 * @return Vector of pointers to the k nearest points.
*/
std::vector<SamplePoint*> k_nearest_neighbors(E::VectorXf &input_point,
                                     std::vector<SamplePoint> &set, 
                                     int k);
#endif
