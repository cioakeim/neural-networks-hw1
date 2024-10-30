#ifndef CIFAR_HANDLERS_HPP
#define CIFAR_HANDLERS_HPP

#include "NdPoint.hpp"
#include <string>
#include <vector>
#include <fstream>

/**
 * @brief Manages the input and interpretation of 
 * the CIFAR 10 dataset.
*/
class Cifar10Handler{
private:
  std::string dataset_folder_path; //< Where the binary files are.
  std::ifstream batch_file_streams[5]; //< Stream handlers for each batch file
  std::ifstream test_batch_stream; //< Test batch
  std::string id_to_class_name_lut[10]; //< From class_id to class name.

public:
  // Only logical constructor
  Cifar10Handler(std::string dataset_folder_path); 

  // Destructor
  ~Cifar10Handler();

  /**
   * @brief Get an entry from a specific batch 
   *
   * @param[in] batch_id The number of the batch chosen (0 to 4)
   * @param[out] output The N-D point representing the entry (with class_id)
  */
  int getBatchEntry(int batch_id,NdPoint<uint8_t>& output);


};

class Cifar100Handler{
  
};


#endif 
