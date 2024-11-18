find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

include_directories(${PROJECT_SOURCE_DIR}/include/CudaMLP)

file(GLOB CUDA_MLP_SOURCES "${PROJECT_SOURCE_DIR}/src/CudaMLP/*.cu")
add_library(cuda_mlp_lib STATIC ${CUDA_MLP_SOURCES})
target_link_libraries(cuda_mlp_lib PUBLIC mlp_lib)
set_target_properties(cuda_mlp_lib 
                    PROPERTIES CUDA_RUNTIME_LIBRARY Shared)
