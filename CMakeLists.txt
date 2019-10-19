
#sudo cmake -DCMAKE_PREFIX_PATH=$PWD/../libtorch -G"Eclipse CDT4 - Unix Makefiles" ../example-app/
#cmake  -DBOOST_INCLUDEDIR=$PWD/../../boost  -DBOOST_LIBRARYDIR=$PWD/../../boost    -DCMAKE_PREFIX_PATH=$PWD/../../libtorch -DCMAKE_GENERATOR_PLATFORM=x64 ..

cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

project(buysAndsells)

if(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX)
    find_package(CURL REQUIRED)
endif()


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_BUILD_TYPE Debug)
LIST(APPEND CMAKE_PROGRAM_PATH  "$(CMAKE_PREFIX_PATH)/lib" ...)

set(Boost_USE_STATIC_LIBS ON) 
set(Boost_USE_MULTITHREADED ON)  
set(Boost_USE_STATIC_RUNTIME ON) 
find_package(Boost REQUIRED COMPONENTS date_time) 

if(Boost_FOUND)
    include_directories(${Boost_INCLUDE_DIRS})
endif()

if(MSVC)
  # Force to always compile with W4
  if(CMAKE_CXX_FLAGS MATCHES "/W[0-4]")
    string(REGEX REPLACE "/W[0-4]" "/W2" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
  else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W2")
  endif()
elseif(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX)
  string(APPEND CMAKE_CXX_FLAGS_DEBUG " -g3")
  # Update if necessary
  set(CUDNN_INCLUDE_PATH  "/home/p0w/Downloads/cuda/include")
  set(CUDNN_LIBRARY_PATH "/home/p0w/Downloads/cuda/lib64/libcudnn_static.a")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-long-long -Wextra -pedantic -Werror")
endif()

find_package(Torch REQUIRED)

if(WIN32)
else()
    link_directories("${CMAKE_SOURCE_DIR}/thirdparty/curl2")
endif()


file(GLOB ALL_SRCS
    "*.h"
    "*.hpp"
    "*.cpp"
)

add_executable(${PROJECT_NAME}  ${ALL_SRCS})


if(WIN32)
    target_link_libraries(${PROJECT_NAME} urlmon)
else()
    target_link_libraries(${PROJECT_NAME} curl)
endif()

target_link_libraries(${PROJECT_NAME}  "${TORCH_LIBRARIES}" "${Boost_LIBRARIES}")

set_property(TARGET ${PROJECT_NAME}  PROPERTY CXX_STANDARD 17)
add_custom_target(build-time-make-directory ALL
    COMMAND ${CMAKE_COMMAND} -E make_directory stockData)

