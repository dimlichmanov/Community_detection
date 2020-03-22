cmake_minimum_required(VERSION 3.13)
project(RMAT)

set(CMAKE_CXX_STANDARD 11)

add_executable(RMAT main.cpp CSR_GRAPH.cpp CSR_GRAPH.h generator.cpp generator.h device_gather.cu device_gather.h lp.cpp parser.cpp parser.h main2.cu lp.cpp lp.h)