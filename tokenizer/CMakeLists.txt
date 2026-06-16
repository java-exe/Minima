# Baut das pybind11-Modul `tokenizer_cpp` (Person 2).
# Build z.B.:
#   pip install pybind11
#   cmake -B build -S . -DCMAKE_PREFIX_PATH=$(python -m pybind11 --cmakedir)
#   cmake --build build --config Release
cmake_minimum_required(VERSION 3.15)
project(tokenizer_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(pybind11 REQUIRED)

pybind11_add_module(tokenizer_cpp
    src/bpe.cpp
    src/bindings.cpp
)
target_include_directories(tokenizer_cpp PRIVATE src)

# Optional: C++ Unit-Tests (siehe tests/)
enable_testing()
add_subdirectory(tests)
