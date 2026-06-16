# Baut die C++ Inference Engine als Shared Library (inference.dll auf Windows).
# Person 2. Siehe plan.md, Phase 3.
#
# onnxruntime-Pfad setzen, z.B.:
#   cmake -B build -S . -DONNXRUNTIME_ROOT=C:/onnxruntime
#   cmake --build build --config Release
cmake_minimum_required(VERSION 3.15)
project(slm_inference LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(ONNXRUNTIME_ROOT "" CACHE PATH "Pfad zur onnxruntime-Installation")

# Tokenizer-Quelle wird mitgebaut (gemeinsame BPE-Klasse).
add_library(inference SHARED
    inference.cpp
    ${CMAKE_SOURCE_DIR}/../../tokenizer/src/bpe.cpp
)

target_include_directories(inference PRIVATE
    .
    ${CMAKE_SOURCE_DIR}/../../tokenizer/src
    ${ONNXRUNTIME_ROOT}/include
)

target_link_directories(inference PRIVATE ${ONNXRUNTIME_ROOT}/lib)
target_link_libraries(inference PRIVATE onnxruntime)

# Optionales CLI-Test-Tool: ./inference "def fibonacci(n):"
# add_executable(inference_cli main.cpp)
# target_link_libraries(inference_cli PRIVATE inference)
