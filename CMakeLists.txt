cmake_minimum_required(VERSION 3.10)

project(StockEstimator VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Get OpenCV 
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(OpenMP REQUIRED)

# Standard CMake Structure 

# Library header files
include_directories("${PROJECT_SOURCE_DIR}/include")
# Library Implementation files
file(GLOB SOURCES "${PROJECT_SOURCE_DIR}/src/*.cpp")
add_library(my_library ${SOURCES})
target_link_libraries(my_library PUBLIC OpenMP::OpenMP_CXX)

# Scripts folder
set(SCRIPTS "${PROJECT_SOURCE_DIR}/scripts")

# Main
add_executable(main "${SCRIPTS}/main.cpp")
target_link_libraries(main PUBLIC my_library)
target_link_libraries(main PUBLIC ${OpenCV_LIBS})
target_link_libraries(main PUBLIC OpenMP::OpenMP_CXX)
target_compile_options(main PRIVATE -O3 -Wall -Wextra)

# Testing of nearest neighbors
add_executable(testNN "${SCRIPTS}/testNN.cpp")
target_link_libraries(testNN PUBLIC my_library)
target_link_libraries(testNN PUBLIC ${OpenCV_LIBS})
target_link_libraries(testNN PUBLIC OpenMP::OpenMP_CXX)
target_compile_options(testNN PRIVATE -O3 -Wall -Wextra)

add_executable(testMLP "${SCRIPTS}/testMLP.cpp")
target_link_libraries(testMLP PUBLIC my_library)
target_link_libraries(testMLP PUBLIC ${OpenCV_LIBS})
target_link_libraries(testMLP PUBLIC OpenMP::OpenMP_CXX)
target_compile_options(testMLP PRIVATE -O3 -Wall -Wextra)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/build)
