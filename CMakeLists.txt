cmake_minimum_required(VERSION 3.10)

project(estimate-pi)

set(CMAKE_CXX_STANDARD 11)

add_executable(estimate_pi_cpu 
    estimate_pi_cpu.cpp)
