#ifndef NEW_MLP_HPP
#define NEW_MLP_HPP

#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include "MLP/NewLayer.hpp"
#include "CommonLib/basicStructs.hpp"

#define WEIGHT_DECAY 1e-7

namespace E=Eigen;
using E::MatrixXf;
using E::VectorXf;
using E::VectorXi;
using VectorFunction = std::function<MatrixXf(const MatrixXf)>;

class MLP{
protected:
  // For I/O purposes
  std::string name;
  // Structure
  std::vector<Layer> layers;
  int depth;
  VectorFunction activation_function;
  VectorFunction activation_derivative;
  // Parameters
  const float learning_rate;
  const int batch_size;
  // Training set
  MatrixXf training_set;
  VectorXi training_labels;
  MatrixXf test_set;
  VectorXi test_labels;


  
public:

  MLP(const std::vector<int>& layer_sizes,
      const int input_size,
      VectorFunction activation_function,
      VectorFunction activation_derivative,
      float learning_rate,
      int batch_size);

  MLP(std::string file_path,std::string name,
      VectorFunction activation_function,
      VectorFunction activation_derivative,
      float learning_rate,int batch_size);

  void setName(std::string name){this->name=name;}

  // Do forward and backward pass in batches
  void forwardBatchPass(const MatrixXf& input);
  float getBatchLosss(const VectorXi& correct_labels);
  void backwardBatchPass(const MatrixXf& input,
                         const VectorXi& correct_labels);
  // For the whole dataset (assumed the array is shuffled)
  float runEpoch();

  // Test the epoch result (return the loss function and accuracy)
  void testModel(float& J_test,float& accuracy);

  // Softmax methods 
  void softMaxForward(MatrixXf& activations);
  // Returns Loss function of batch
  void softMaxBackward(const VectorXi& correct_labels);

  // Store to place
  void store(std::string file_path);

  // Config:
  void randomInit();
  void insertDataset(std::vector<SamplePoint>& training_set,
                     std::vector<SamplePoint>& test_set);
  void shuffleDataset();

  
};

#endif
