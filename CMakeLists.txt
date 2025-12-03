cmake_minimum_required(VERSION 3.30.4 FATAL_ERROR)

project(MultiBlockBPE
    LANGUAGES C CXX CUDA
)

# =========================
# CPM — Dependency Manager
# =========================
include(${CMAKE_SOURCE_DIR}/cpm/CPM.cmake)

# =========================
# C++ / CUDA Standards
# =========================
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# =========================
# CUDA compiler flags needed by cuCollections
# =========================
add_compile_options(
    "$<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>"
    "$<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>"
)

# =========================
# cuCollections (cuco) — dev branch
# =========================
CPMAddPackage(
  NAME cuco
  GITHUB_REPOSITORY NVIDIA/cuCollections
  GIT_TAG dev
  OPTIONS
     "BUILD_TESTS OFF"
     "BUILD_BENCHMARKS OFF"
     "BUILD_EXAMPLES OFF"
)

# =========================
# Flex — Lexer
# =========================
find_package(FLEX REQUIRED)
FLEX_TARGET(
    MyLexer
    ${CMAKE_SOURCE_DIR}/src/lexer.l
    ${CMAKE_BINARY_DIR}/lexer.c
)

# =========================
# Sources
# =========================
file(GLOB CPP_SOURCES ${CMAKE_SOURCE_DIR}/src/*.cpp)
file(GLOB CU_SOURCES  ${CMAKE_SOURCE_DIR}/src/*.cu)

# =========================
# Build Executable
# =========================
add_executable(MultiBlockBPE
    ${CPP_SOURCES}
    ${CU_SOURCES}
    ${FLEX_MyLexer_OUTPUTS}
)

# =========================
# Include Directories
# =========================
target_include_directories(MultiBlockBPE
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include                         # your project headers
        ${CMAKE_BINARY_DIR}/_deps/cuco-src/include          # cuCollections headers
        ${CMAKE_BINARY_DIR}/_deps/cccl-src/c                # CCCL core
        ${CMAKE_BINARY_DIR}/_deps/cccl-src/cub              # CUB
        ${CMAKE_BINARY_DIR}/_deps/cccl-src/thrust           # Thrust
        ${CMAKE_BINARY_DIR}/_deps/cccl-src/libcudacxx/include  # libcudacxx
)

# =========================
# GPU Architecture (Perlmutter A100 → SM80)
# =========================
set_target_properties(MultiBlockBPE PROPERTIES
    CUDA_ARCHITECTURES "80"
)
