#ifndef MODIFIERS_HPP
#define MODIFIERS_HPP

#include <random>
#include <Eigen/Dense>

namespace E=Eigen;
using E::MatrixXf;
using E::VectorXf;
using E::VectorXi;

// For dropout creation
struct Dropout{
  const float rate;
  std::mt19937 gen;
  std::uniform_real_distribution<float> dist;
  Eigen::MatrixXf mask;

  Dropout();

  Dropout(MatrixXf input, const float rate);

  void maskInput(E::MatrixXf& input);
};


#endif
