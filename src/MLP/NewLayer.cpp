#include "MLP/NewLayer.hpp"
#include <fstream>
#include <random>


Layer::Layer(std::string folder_path,const int batch_size){
  std::ifstream is;
  // Load weights
  is.open(folder_path+"/weights.csv",std::ios::in);
  int rows,cols;
  is>>rows>>cols;
  this->weights=E::MatrixXf(rows,cols);
  for(int i=0;i<rows;i++){
    for(int j=0;j<cols;j++){
      is>>weights(i,j);
    }
  }
  is.close();
  // Load biases
  is.open(folder_path+"/biases.csv",std::ios::in);
  int size;
  is>>size;
  this->biases=E::VectorXf(size);
  for(int i=0;i<size;i++){
    is>>biases(i);
  }
  is.close();
  // Create the rest
  const int input_size=cols;
  const int layer_size=rows;
  this->errors=MatrixXf(layer_size,batch_size);
  this->activations=MatrixXf(layer_size,batch_size);
}

void Layer::updateWeights(const MatrixXf& input,
                          const float rate, const int batch_size){
  E::MatrixXf weightGradients=this->errors*(input.transpose());
  E::VectorXf biasGradients=this->errors.rowwise().sum();
  const float a=rate/batch_size;
  this->weights-=a*weightGradients+WEIGHT_DECAY*this->weights;
  this->biases-=a*biasGradients+WEIGHT_DECAY*this->biases;
  std::cout<<"Positive weights percentage: "<<
    (weights.array()>0).cast<float>().mean()<<std::endl;
}


// Store to location (2 files, 1 for weights and 1 for bias)
void Layer::store(std::string folder_path){
  std::ofstream os;
  // Store weights
  os.open(folder_path+"/weights.csv",std::ios::out);
  std::cout<<"Weights positive percentage: "<<(weights.array()>0).cast<float>().mean()<<std::endl;
  os<<weights.rows()<<" "<<weights.cols()<<"\n";
  for(int i=0;i<weights.rows();i++){
    for(int j=0;j<weights.cols();j++){
      os<<weights(i,j)<<" ";
    }
    os<<"\n";
  }
  os.close();
  // Store biases
  os.open(folder_path+"/biases.csv",std::ios::out);
  std::cout<<"Bias mean square: "<<(biases.array()>0).cast<float>().mean()<<std::endl;
  os<<biases.size()<<" "<<"\n"; 
  for(int i=0;i<biases.size();i++){
    os<<biases(i)<<"\n";
  }
  os.close();
}

// He Initialization accoutning for fan-in 
void Layer::HeRandomInit(){
  const int rows=weights.rows();
  const int cols=weights.cols();
  const float stddev= std::sqrt(2.0f/rows);
  // Init rng 
  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<float> dist(0.0,stddev);

  for(int i=0;i<rows;i++){
    for(int j=0;j<cols;j++){
      weights(i,j)=(dist(gen));
    }
    biases(i)=abs(dist(gen));
  }
}





// Print methods
void Layer::printWeights(){
  // Simple print methods
  std::cout << this->weights << std::endl;
}

void Layer::printBiases(){
  std::cout << this->biases << std::endl;
}

void Layer::printActivations(){
  std::cout << this->activations<< std::endl;
}

void Layer::printErrors(){
  std::cout << this->errors<< std::endl;
}
