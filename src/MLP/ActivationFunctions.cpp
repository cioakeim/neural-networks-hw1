#include "MLP/ActivationFunctions.hpp"
#include <cmath>

E::VectorXf reLU(const E::VectorXf& in){
  return in.cwiseMax(0.0);
}


E::VectorXf reLUder(const E::VectorXf& reLU_output){
  return (reLU_output.array()>0).cast<float>();
}

float reLU(const float in){
  return (in>0) ? in : 0.0f;
}
float reLUder(const float in){
  return (in>0) ? 1.0f : 0.0f;
}


// Tanh
E::VectorXf tanh(const E::VectorXf& in){
  return in.array().tanh();
}

E::VectorXf tanhder(const E::VectorXf& tanh_output){
  return (1- tanh_output.array().square());
}
