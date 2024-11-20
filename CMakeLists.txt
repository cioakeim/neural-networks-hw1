cmake_minimum_required(VERSION 3.10)

project(NeuralNetworksProject1 VERSION 1.0 LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

option(CUDA_EXT "Option to create CUDA extensions" OFF)
if(CUDA_EXT)
enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)
endif()

#find_package(CUDA REQUIRED)
#include_directories(${CUDA_INCLUDE_DIRS})

include_directories(${PROJECT_SOURCE_DIR}/include)

add_subdirectory("${PROJECT_SOURCE_DIR}/src/CommonLib")
add_subdirectory("${PROJECT_SOURCE_DIR}/src/MLP")
add_subdirectory("${PROJECT_SOURCE_DIR}/src/NearestNeighbors")
if(CUDA_EXT)
add_subdirectory("${PROJECT_SOURCE_DIR}/src/CudaMLP")
endif()

# Standard CMake Structure 


# Scripts folder
set(SCRIPTS "${PROJECT_SOURCE_DIR}/scripts")


# Testing of nearest neighbors
add_executable(testNN "${SCRIPTS}/testNN.cpp")
target_link_libraries(testNN PRIVATE nearest_neighbors_lib)
target_link_libraries(testNN PRIVATE common_lib)
target_compile_options(testNN PRIVATE -O3 -Wall -Wextra)

add_executable(testMLP "${SCRIPTS}/testMLP.cpp")
target_link_libraries(testMLP PRIVATE mlp_lib)
target_link_libraries(testMLP PRIVATE common_lib)
target_compile_options(testMLP PRIVATE -O3 -Wall -Wextra)

if(CUDA_EXT)
add_executable(testCudaMLP "${SCRIPTS}/testCudaMLP.cu")
target_link_libraries(testCudaMLP PRIVATE cuda_mlp_lib)
#set_target_properties(testCudaMLP
#                    PROPERTIES CUDA_RUNTIME_LIBRARY Shared)
target_link_libraries(testCudaMLP PRIVATE common_lib)
target_compile_options(testCudaMLP PRIVATE -O3 -Wall -Wextra)
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/build)


