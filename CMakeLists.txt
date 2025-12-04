cmake_minimum_required(VERSION 3.16)
project(MultiBlockBPE LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(FLEX REQUIRED)

# Generate lexer.c from lexer.l
FLEX_TARGET(
    MyLexer
    ${CMAKE_SOURCE_DIR}/src/lexer.l
    ${CMAKE_BINARY_DIR}/lexer.c
)

include_directories(${CMAKE_SOURCE_DIR}/include)

file(GLOB CPP_SOURCES ${CMAKE_SOURCE_DIR}/src/*.cpp)
message(STATUS "CPP sources: ${CPP_SOURCES}")

add_executable(MultiBlockBPE
    ${CPP_SOURCES}
    ${FLEX_MyLexer_OUTPUTS}   # Pulls in lexer.c
)
