#pragma once
#include <cuda_runtime.h>
#include <unordered_map>
#include <cstdint>

#define CUDA_CHECK(expr)                                                                           \
    do                                                                                             \
    {                                                                                              \
        cudaError_t _e = (expr);                                                                   \
        if (_e != cudaSuccess)                                                                     \
        {                                                                                          \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
            std::exit(1);                                                                          \
        }                                                                                          \
    } while (0)

void launchTokenizeKernel(int *tokens, int *nextToken, int tokensLen, std::unordered_map<uint64_t, uint64_t> pairRankTable);