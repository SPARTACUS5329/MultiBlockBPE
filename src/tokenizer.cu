#include "tokenizer.cuh"
#include <cuda_runtime.h>
#include <iostream>

__global__ void tokenize(const int *__restrict__ tokens, const int *__restrict__ nextToken)
{
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    printf("%d %d\n", threadId, tokens[threadId]);
}

void launchTokenizeKernel(const int *tokens, const int *nextToken, const int tokensLen)
{
    tokenize<<<1, tokensLen>>>(tokens, nextToken);
    cudaDeviceSynchronize();
}