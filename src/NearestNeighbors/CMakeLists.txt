include_directories(${PROJECT_SOURCE_DIR}/include/CommonLib)

file(GLOB NEARESTNEIGHBORS_SRC "${PROJECT_SOURCE_DIR}/src/NearestNeighbors/*.cpp")
add_library(nearest_neighbors_lib STATIC ${NEARESTNEIGHBORS_SRC})
target_include_directories(nearest_neighbors_lib PUBLIC "${PROJECT_SOURCE_DIR}/include/NearestNeighbors")
target_link_libraries(nearest_neighbors_lib PUBLIC ${OpenCV_LIBS})
target_link_libraries(nearest_neighbors_lib PUBLIC common_lib)
