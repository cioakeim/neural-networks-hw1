cmake_minimum_required(VERSION 3.10)

project(StockEstimator VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)


include_directories("${PROJECT_SOURCE_DIR}/include")

file(GLOB SOURCES "${PROJECT_SOURCE_DIR}/src/*.cpp")

add_executable(main ${SOURCES})

target_link_libraries(main m)
target_compile_options(main PRIVATE -O3 -Wall -Wextra -lssl)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/build)
