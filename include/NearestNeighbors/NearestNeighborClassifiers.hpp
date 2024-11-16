#ifndef K_NEAREST_NEIGHBOR_CLASSIFIERS_HPP
#define K_NEAREST_NEIGHBOR_CLASSIFIERS_HPP

#include "CommonLib/basicStructs.hpp"
#include <eigen3/Eigen/Dense>
#include <vector>

namespace E=Eigen;

/**
 * @brief Classifies an input point using the nearest neighbor criterion.
 *
 * @param[in] input_point Input point in N-D space.
 * @param[in] set The set of other points (each one with a class label).
 *
 * @return Class ID of the final decision.
*/
uint8_t classify_1_nearest_neighbor(E::VectorXf &input_point,
                                std::vector<SamplePoint> &set);


/**
 * @brief Classifies an input point using the 3 nearest neighbors criterion.
 *
 * @param[in] input_point Input point in N-D space.
 * @param[in] set The set of other points (each one with a class label).
 *
 * @return Class ID of the final decision.
*/
uint8_t classify_3_nearest_neighbor(E::VectorXf &input_point,
                                std::vector<SamplePoint> &set);

#endif
