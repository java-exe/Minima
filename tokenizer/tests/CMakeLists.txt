# Einfache assert-basierte Tests fuer den BPE-Tokenizer.
add_executable(test_bpe
    test_bpe.cpp
    ${CMAKE_SOURCE_DIR}/src/bpe.cpp
)
target_include_directories(test_bpe PRIVATE ${CMAKE_SOURCE_DIR}/src)
add_test(NAME test_bpe COMMAND test_bpe)
