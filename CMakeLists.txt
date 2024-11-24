cmake_minimum_required(VERSION 3.13)
project(cppnn-demo)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Set stack size
set(STACK_SIZE 67108864 CACHE STRING "Set the stack size") # 64MB
if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    if (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        add_link_options(/STACK:${STACK_SIZE})
    else()
        add_link_options(-Wl,--stack=${STACK_SIZE})
    endif()
endif()

add_executable(cppnn-demo mnist.cpp)
set_target_properties(cppnn-demo PROPERTIES OUTPUT_NAME "mnist")