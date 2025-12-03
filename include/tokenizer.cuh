#pragma once
#include <cuda_runtime.h>

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

void launchTokenizeKernel(int *tokens, int *nextToken, const int tokensLen);