
#sudo cmake -DCMAKE_PREFIX_PATH=$PWD/../libtorch -G"Eclipse CDT4 - Unix Makefiles" ../example-app/
#"C:\Program Files\CMake\bin\cmake.exe" -DCMAKE_PREFIX_PATH=$PWD/../../libtorch  cmake -DCMAKE_GENERATOR_PLATFORM=x64 ..

cmake_minimum_required(VERSION 3.0 FATAL_ERROR)


set(CUDNN_INCLUDE_DIR  "/home/powprashant/Downloads/cudnn-10.0-linux-x64-v7.6.2.24/cuda/include")
set(CUDNN_LIBRARY "/home/powprashant/Downloads/cudnn-10.0-linux-x64-v7.6.2.24/cuda/lib64/libcudnn.so")


project(buysAndsells)

find_package(Torch REQUIRED)

if(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX)
    find_package(CURL REQUIRED)
endif()


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_BUILD_TYPE Debug)


if(MSVC)
  # Force to always compile with W4
  if(CMAKE_CXX_FLAGS MATCHES "/W[0-4]")
    string(REGEX REPLACE "/W[0-4]" "/W4" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
  else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4")
  endif()
elseif(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX)
  string(APPEND CMAKE_CXX_FLAGS_DEBUG " -g3")
  # Update if necessary
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-long-long -Wextra -pedantic -Werror")
endif()


if(WIN32)
    #add_definitions(-DCURL_STATICLIB)
    #link_directories("${CMAKE_SOURCE_DIR}/thirdparty/openssl/" "${CMAKE_SOURCE_DIR}/thirdparty/"curl_win32)
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
    SET(ENV{PATH} $ENV{PATH};${CMAKE_PREFIX_PATH}/lib)
else()
    target_link_libraries(${PROJECT_NAME} curl)
endif()

target_link_libraries(${PROJECT_NAME}  "${TORCH_LIBRARIES}")

set_property(TARGET ${PROJECT_NAME}  PROPERTY CXX_STANDARD 17)
add_custom_target(build-time-make-directory ALL
    COMMAND ${CMAKE_COMMAND} -E make_directory stockData)

