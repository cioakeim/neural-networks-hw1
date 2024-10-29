#ifndef NDPOINTSET_HPP
#define NDPOINTSET_HPP

#include "NdPoint.hpp"

/**
 * @brief A set of points in N-D space.
 */
template <typename T>
class NdPointSet{
public:
  int count; //< Number of elements
  NdPoint<T> *elements; //< Array of elements of set


  // Default constructor
  NdPointSet();

  // Allocates full and defines dimension
  NdPointSet(int count, int dimension);

  // Destructor
  ~NdPointSet();

  // Sets all elements to zero
  void set_to_zero();
};

// METHOD IMPLEMENTATION

// Default constructor.
template <typename T>
NdPointSet<T>::NdPointSet()
  : count(0),
    elements(nullptr){}


// Allocate all points.
template <typename T>
NdPointSet<T>::NdPointSet(int count,int dimension)
  : count(count),
    elements(new NdPoint<T>[count]){
  for(int i=0;i<this->count;i++){
    this->elements[i]=NdPoint<T>(dimension);
  }
}

// Destructor
template <typename T>
NdPointSet<T>::~NdPointSet(){
  if(this->elements!=nullptr){
    delete this->elements;
  }
  this->count=0;
}

template <typename T>
void NdPointSet<T>::set_to_zero(){
  for(int i=0;i<this->count;i++){
    this->elements[i].set_to_zero();
  }
}

#endif
