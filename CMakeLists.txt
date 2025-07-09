cmake_minimum_required(VERSION 3.10)
project(proyecto-final)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Incluye los headers
include_directories(include)
include_directories(include/utec)
include_directories(include/utils)

# Archivos fuente principales
file(GLOB_RECURSE SOURCES
    src/utec/*.cpp
    src/utils/*.cpp
    src/lodepng.cpp
)

add_executable(proyecto-final ${SOURCES})

# Opcional: para warnings útiles
target_compile_options(proyecto-final PRIVATE -Wall -Wextra -Wpedantic)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

