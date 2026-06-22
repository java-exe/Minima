# CMake generated Testfile for 
# Source directory: C:/Users/janst/OneDrive/Dokumente/Minima/tokenizer/tests
# Build directory: C:/Users/janst/OneDrive/Dokumente/Minima/tokenizer/build2/tests
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
if(CTEST_CONFIGURATION_TYPE MATCHES "^([Dd][Ee][Bb][Uu][Gg])$")
  add_test(test_bpe "C:/Users/janst/OneDrive/Dokumente/Minima/tokenizer/build2/tests/Debug/test_bpe.exe")
  set_tests_properties(test_bpe PROPERTIES  _BACKTRACE_TRIPLES "C:/Users/janst/OneDrive/Dokumente/Minima/tokenizer/tests/CMakeLists.txt;7;add_test;C:/Users/janst/OneDrive/Dokumente/Minima/tokenizer/tests/CMakeLists.txt;0;")
elseif(CTEST_CONFIGURATION_TYPE MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
  add_test(test_bpe "C:/Users/janst/OneDrive/Dokumente/Minima/tokenizer/build2/tests/Release/test_bpe.exe")
  set_tests_properties(test_bpe PROPERTIES  _BACKTRACE_TRIPLES "C:/Users/janst/OneDrive/Dokumente/Minima/tokenizer/tests/CMakeLists.txt;7;add_test;C:/Users/janst/OneDrive/Dokumente/Minima/tokenizer/tests/CMakeLists.txt;0;")
elseif(CTEST_CONFIGURATION_TYPE MATCHES "^([Mm][Ii][Nn][Ss][Ii][Zz][Ee][Rr][Ee][Ll])$")
  add_test(test_bpe "C:/Users/janst/OneDrive/Dokumente/Minima/tokenizer/build2/tests/MinSizeRel/test_bpe.exe")
  set_tests_properties(test_bpe PROPERTIES  _BACKTRACE_TRIPLES "C:/Users/janst/OneDrive/Dokumente/Minima/tokenizer/tests/CMakeLists.txt;7;add_test;C:/Users/janst/OneDrive/Dokumente/Minima/tokenizer/tests/CMakeLists.txt;0;")
elseif(CTEST_CONFIGURATION_TYPE MATCHES "^([Rr][Ee][Ll][Ww][Ii][Tt][Hh][Dd][Ee][Bb][Ii][Nn][Ff][Oo])$")
  add_test(test_bpe "C:/Users/janst/OneDrive/Dokumente/Minima/tokenizer/build2/tests/RelWithDebInfo/test_bpe.exe")
  set_tests_properties(test_bpe PROPERTIES  _BACKTRACE_TRIPLES "C:/Users/janst/OneDrive/Dokumente/Minima/tokenizer/tests/CMakeLists.txt;7;add_test;C:/Users/janst/OneDrive/Dokumente/Minima/tokenizer/tests/CMakeLists.txt;0;")
else()
  add_test(test_bpe NOT_AVAILABLE)
endif()
