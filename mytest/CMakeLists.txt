cmake_minimum_required(VERSION 2.8)
project(zerod_weights C CXX)

aux_source_directory(. CAFFE_CPP)
aux_source_directory(test TEST_CPP)
message(STATUS "TEST_CPP: " ${TEST_CPP})
aux_source_directory(src SRCS)
message(STATUS "SRCS: " ${SRCS})
# file( GLOB srcs *.c *.cpp *.cc *.h *.hpp )

set(caffe_root $ENV{my_caffe_root})
message(STATUS "caffe_root: " ${caffe_root})
set(blas_root /opt/OpenBLAS)

# find_package(OPENBLAS REQUIRED)
# https://stackoverflow.com/questions/31438916/cmake-cannot-find-library-using-link-directories
find_library(so_caffe caffe ${caffe_root}/build/lib)
message(STATUS "so_caffe: " ${so_caffe})
find_library(so_openblas openblas ${blas_root}/lib)
message(STATUS "so_openblas: " ${so_openblas})

set(bias_include /opt/OpenBLAS/include)
set(bias_lib_dir /opt/OpenBLAS/lib)

set(cuda_root /usr/local/cuda)
set(cuda_include ${cuda_root}/include)
set(cuda_lib_dir ${cuda_root}/lib ${cuda_root}/lib64)
# comment if cpu only
set(cuda_libs cudart cublas curand)

set(other_libs glog gflags protobuf boost_system boost_filesystem)


# set( CMAKE_VERBOSE_MAKEFILE on )
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O2 -Wall -fopenmp")
set(LINK_FLAGS "${LINK_FLAGS} -lgomp -lpthread")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libstdc++")

# add_subdirectory(${PROJECT_SOURCE_DIR}/src)

# ---[ Includes
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${caffe_root}/include)
include_directories(${caffe_root}/build/src) # caffe.pb.h
include_directories(${bias_include})
include_directories(${cuda_include})
# ---[ Libs
link_directories(${caffe_root}/build/lib)    # libcaffe.so
link_directories(${bias_lib_dir})
link_directories(${cuda_lib_dir})
# --- bin_caffe
get_filename_component(bin_caffe ${CAFFE_CPP} NAME_WE)
add_executable(${bin_caffe} ${SRCS} ${CAFFE_CPP})
target_link_libraries(${bin_caffe} ${so_caffe})
target_link_libraries(${bin_caffe} ${so_openblas})
target_link_libraries(${bin_caffe} ${cuda_libs})
target_link_libraries(${bin_caffe} ${other_libs})

set(TEST_CPP "")
foreach(source ${TEST_CPP})
 get_filename_component(name ${source} NAME_WE)
 add_executable(${name} ${SRCS} ${source})
 target_link_libraries(${name} ${so_caffe})
 target_link_libraries(${name} ${so_openblas})
 target_link_libraries(${name} ${cuda_libs})
 target_link_libraries(${name} ${other_libs})
endforeach(source)