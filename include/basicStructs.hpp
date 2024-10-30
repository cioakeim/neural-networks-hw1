#ifndef BASIC_STRUCTS_HPP
#define BASIC_STRUCTS_HPP

#include <eigen3/Eigen/Dense>

namespace E=Eigen;

// A sample is a N-D point with a class label
struct SamplePoint{
  E::VectorXf vector; //< The sample in Nd 
  int label; //< The sample's label
};

#endif // !BASIC_STRUCTS_HPP
