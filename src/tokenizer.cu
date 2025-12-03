#include "tokenizer.cuh"
#include <cuda_runtime.h>
#include <cuco/static_map.cuh>
#include <climits>
#include <iostream>

__global__ void tokenize(
    int *__restrict__ tokens,    // token id at each position (length N)
    int *__restrict__ nextToken, // next index in linked list (-1 if none) (length N)
    // cuco::static_map<uint64_t, uint64_t>::device_view map,
    int V // vocabulary size (stride for flattened tables)
)
{
    extern __shared__ int sdata[];
    int *minRank = &sdata[0];
    int *minPos = &sdata[1];

    if (threadIdx.x == 0)
    {
        *minRank = INT_MAX;
        *minPos = 0;
    }

    __syncthreads();

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    for (int n = 0; n < 1; n++)
    {
        __syncthreads();

        int j = nextToken[i];

        bool active = j != -1;

        int currRank = INT_MAX;
        int idx = -1;

        // if (active)
        // {
        //     int a = tokens[i];
        //     int b = tokens[j];

        //     idx = a * V + b;
        //     currRank = rankTable[idx];

        //     int oldMin = atomicMin(minRank, currRank);
        //     if (currRank < oldMin)
        //         atomicExch(minPos, i);
        // }

        // __syncthreads();

        // if (active && currRank == *minRank && i == *minPos)
        // {
        //     tokens[i] = mergeTable[idx];
        //     nextToken[i] = nextToken[j];
        //     nextToken[j] = -1;
        // }
    }
}

void launchTokenizeKernel(int *tokens, int *nextToken, const int tokensLen)
{
    // tokenize<<<1, tokensLen, 2 * sizeof(int)>>>(tokens, nextToken);
    // cudaDeviceSynchronize();
    std::cout << "Hello from the launch kernel!\n";
}