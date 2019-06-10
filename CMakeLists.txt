cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

# Set the NVCC bin dir to Path, or explicitly set here
set (CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)

enable_language(CUDA)

# Optional
include(CheckLanguage)
check_language(CUDA)

set (PROJECT_NAME ckt_tests LANGUAGES CXX CUDA)

project(${PROJECT_NAME} CUDA CXX)
set (${PROJECT_NAME}_VERSION 0.1.0)

if(NOT DEFINED CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 11)
    set(CMAKE_CXX_STANDARD_REQUIRED True)
endif()

if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 11)
    set(CMAKE_CUDA_STANDARD_REQUIRED True)
endif()

# CUDA architecture options, choose what you want
set(GENCODE_SM30 "-gencode=arch=compute_30,code=sm_30")
set(GENCODE_SM60 "-gencode=arch=compute_60,code=sm_60")
set(GENCODE_SM62 "-gencode=arch=compute_62,code=sm_62")

#CUDA_SELECT_NVCC_ARCH_FLAGS(ARCH_FLAGS 3.0 3.5+PTX 5.2(5.0) Maxwell)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${GENCODE_SM30}")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${GENCODE_SM60}")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${GENCODE_SM62}")

# message(${CMAKE_CUDA_FLAGS}) # for debugging
# Add the base directory to your include file search path
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

## Add flags here
#if ( CMAKE_COMPILER_IS_GNUCC )
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -Wall")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --ptxas-options=-v")
#endif()


set(SRC_LIST
  # a list of your cpp/cu/header files, e.g. file1.cpp kernel1.cu header.h
  tests/foreach_block_test.cu  
  #tests/foreach_shm_test.cu  
  tests/foreach_test.cu
)

# add_subdirectory(ckt)

#if it is a shared library: (if static, change SHARED to STATIC)
#cuda_add_library(your_project_name SHARED ${SRC_LIST})
# or if it is an executable
add_executable(${PROJECT_NAME} ${SRC_LIST})
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_11)
set_target_properties(${PROJECT_NAME} PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON)

if(APPLE)
  # We need to add the path to the driver (libcuda.dylib) as an rpath, 
  # so that the static cuda runtime can find it at runtime.
  set_property(TARGET ${PROJECT_NAME}
               PROPERTY
               BUILD_RPATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
 endif()

