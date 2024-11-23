#include "MLP/Modifiers.hpp"


Dropout::Dropout():
  rate(0){};

Dropout::Dropout(MatrixXf input, const float rate):
  rate(rate){
  gen=std::mt19937(42);
  dist=std::uniform_real_distribution<float>(0,1);
  mask=MatrixXf(input.rows(),input.cols());
}

void Dropout::maskInput(E::MatrixXf& input){
  // Generate mask 
  mask=(Eigen::MatrixXf::NullaryExpr(input.rows(),input.cols(), [&]() {
      return dist(gen) > rate ? 1.0f : 0.0f;
  }));
  // Apply
  input=input.array()*mask.array();
}
