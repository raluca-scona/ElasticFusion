cmake_minimum_required(VERSION 2.6.0)

project(ElasticFusion)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}")
find_package(PkgConfig REQUIRED)
find_package(ZLIB REQUIRED)
find_package(Pangolin 0.1 REQUIRED)
find_package(CUDA REQUIRED)
find_package(OpenNI2 REQUIRED)
find_package(efusion REQUIRED)

pkg_search_module(bot2-frames REQUIRED bot2-frames)

include_directories(${ZLIB_INCLUDE_DIR})
include_directories(${EIGEN_INCLUDE_DIRS})
include_directories(${Pangolin_INCLUDE_DIRS})
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${EFUSION_INCLUDE_DIR})
include_directories(${OPENNI2_INCLUDE_DIR})
include_directories(${bot2-frames_INCLUDE_DIRS})


file(GLOB srcs *.cpp)
file(GLOB tools_srcs Tools/*.cpp)

set(CMAKE_CXX_FLAGS "-O3 -msse2 -msse3 -Wall -std=c++11")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -Wall -std=c++11")

add_definitions(-Dlinux=1)

link_directories(
${bot2-frames_LIBRARY_DIRS}
)

add_executable(ElasticFusion 
               ${srcs}
               ${tools_srcs}
)

target_link_libraries(ElasticFusion
                      ${ZLIB_LIBRARY}
                      ${Pangolin_LIBRARIES}
                      ${CUDA_LIBRARIES}
                      ${EFUSION_LIBRARY}
                      ${OPENNI2_LIBRARY}
                      ${bot2-frames_LIBRARIES}
)
