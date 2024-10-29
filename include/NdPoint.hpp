#ifndef NDPOINT_HPP
#define NDPOINT_HPP 

#include <cmath>
#include <iostream>

/**
 * @brief A point in N-D space, with type T elements.
 *
 * Since the point represents a sample,
 * the class ID is included in this structure.
*/
template <typename T>
class NdPoint{
public:

  int dimension; //< Dimension of point.
  T* elements; //< Coordinates in N-D space
  int class_id; //< Class this object points to (-1 if unused)


  // Default constructor (0-D)
  NdPoint();

  // Allocates point.
  NdPoint(int dimension);

  // Full definition 
  NdPoint(int dimension, T* elements, int class_id);

  // Copy constructor
  template <typename P>
  NdPoint(const NdPoint<P>& other);

  // Copy assignment operator 
  NdPoint<T>& operator=(const NdPoint<T>& other);

  // Destructor
  ~NdPoint();


  /**
   * @brief Get euclidian distance of 2 NdPoints in Space.
   * One point is the input and the other is the calee class.
   *
   * Assumes both points are of the same type T
   *
   * @param[in] a Other point.
   *
   * @return Distance.
  */
  double get_distance(NdPoint<T> &a);

  /**
   * @brief Adds another point to this object and stores
   * the result in the calling object.
   *
   * Casts types to calling object's type.
   *
   * @param[in] a Other point
   *
  */
  template <typename P>
  void add_point(NdPoint<P>& a);

  /**
   * @brief Set point to zero.
  */
  void set_to_zero();
};


// METHOD IMPLEMENTATIONS

// Default constructor of point.
template <typename T>
NdPoint<T>::NdPoint()
  : dimension(0),
    elements(nullptr),
    class_id(-1){}

// Defines only dimension
template <typename T>
NdPoint<T>::NdPoint(int dimension)
  : dimension(dimension),
    elements(new T[dimension]),
    class_id(-1){}

// Full definition 
template <typename T>
NdPoint<T>::NdPoint(int dimension, T* elements, int class_id)
  : dimension(dimension),
    elements(new T[dimension]),
    class_id(class_id){
  for(int i=0;i<dimension;i++)
    this->elements[i]=elements[i];
}

// Copy constructor
template <typename T>
template <typename P>
NdPoint<T>::NdPoint(const NdPoint<P>& other)
  : dimension(other.dimension),
    elements(new T[other.dimension]),
    class_id(other.class_id) 
{
  for(int i=0;i<this->dimension;i++){
    // Cast type to callee
    this->elements[i]=static_cast<T>(other.elements[i]);
  }
}

// Copy assignment operator 
template <typename T>
NdPoint<T>& NdPoint<T>::operator=(const NdPoint<T>& other){
  if(this!=&other){
    if(this->elements!=nullptr)
      delete[] this->elements;
    this->dimension=other.dimension;
    this->elements= new T[dimension];
    for(int i=0;i<this->dimension;i++){
      this->elements[i]=other.elements[i];
    }
    this->class_id=other.class_id;
  }
  return *this;
}

// Destructor
template <typename T>
NdPoint<T>::~NdPoint(){
  if(this->elements!=nullptr){
    delete[] this->elements;
  }
  this->dimension=0;
  this->class_id=-1;
}

// Get distance from a.
template <typename T>
double NdPoint<T>::get_distance(NdPoint<T>& a){
  // If points aren't in same space distance has no meaning.
  if(this->dimension!=a.dimension){
    return -1;
  }
  double square_sum=0;
  for(int i=0;i<this->dimension;i++){
    square_sum+=pow(static_cast<double>(this->elements[i])
                    -static_cast<double>(a.elements[i]),2);
  }
  return sqrt(square_sum);
}


template <typename T>
template <typename P>
void NdPoint<T>::add_point(NdPoint<P>& a){
  if(this->dimension!=a.dimension){
    std::cout<< "Add error: Wrong dimensions\n" << std::endl;
    exit(-1);
    return;
  }
  for(int i=0;i<this->dimension;i++){
    this->elements[i]+=static_cast<T>(a.elements[i]);
  }
  return;
}


template <typename T>
void NdPoint<T>::set_to_zero(){
  for(int i=0;i<this->dimension;i++){
    this->elements[i]=0;
  }
  return;
}


#endif
