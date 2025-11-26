#include "tokenizer.cuh"
#include <cuda_runtime.h>
#include <iostream>

__global__ void hello_kernel()
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    printf("%d %d\n", row, col);
}

void launchHelloKernel()
{
    hello_kernel<<<1, 2>>>();
    cudaDeviceSynchronize();
}