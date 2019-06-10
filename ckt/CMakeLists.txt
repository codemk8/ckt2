set (PROJECT_NAME ckt)
project(${PROJECT_NAME})
set (${PROJECT_NAME}_VERSION 0.1.0)
enable_language(CUDA)

# Add the base directory to your include file search path
include_directories(include)

set(SRC_LIST
  # a list of your cpp/cu/header files, e.g. file1.cpp kernel1.cu header.h
)

#if it is a shared library: (if static, change SHARED to STATIC)
add_library(${PROJECT_NAME} STATIC ${SRC_LIST})
# or if it is an executable
# cuda_add_executable(${PROJECT_NAME} ${SRC_LIST})


