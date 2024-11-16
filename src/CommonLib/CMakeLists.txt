find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})


file(GLOB COMMON_LIB_SRC "${PROJECT_SOURCE_DIR}/src/CommonLib/*.cpp")
add_library(common_lib STATIC ${COMMON_LIB_SRC})
target_include_directories(common_lib PUBLIC "${PROJECT_SOURCE_DIR}/include/CommonLib")
target_link_libraries(common_lib PUBLIC ${OpenCV_LIBS})

