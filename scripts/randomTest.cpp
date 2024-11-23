#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <Eigen/Dense>
#include <Eigen/Core>

#include "CommonLib/cifarHandlers.hpp"
#include "CommonLib/basicFuncs.hpp"
#include "CommonLib/LogHandler.hpp"
#include "MLP/ActivationFunctions.hpp"
#include "MLP/NewMLP.hpp"
#include <time.h>
#include <csignal>


using E::MatrixXf;

int main(){
  MatrixXf in=MatrixXf::Random(10,10);
  MatrixXf kernel=MatrixXf::Random(3,3);

  std::cout<<in<<std::endl;

  std::cout<<kernel<<std::endl;

  MatrixXf out=in.convolve(kernel);
}
