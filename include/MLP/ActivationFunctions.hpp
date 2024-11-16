#ifndef ACTIVATION_FUNCTIONS_HPP
#define ACTIVATION_FUNCTIONS_HPP

#include <eigen3/Eigen/Dense>

namespace E=Eigen;

// Each function comes with its derivative 

// ReLU
E::VectorXf reLU(const E::VectorXf& in);
E::VectorXf reLUder(const E::VectorXf& reLU_output);

// Tanh
E::VectorXf tanh(const E::VectorXf& in);
E::VectorXf tanhder(const E::VectorXf& tanh_output);

#endif
