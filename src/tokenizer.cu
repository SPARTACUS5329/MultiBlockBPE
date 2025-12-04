#include "tokenizer.cuh"
#include <cuda_runtime.h>
#include <cuco/static_map.cuh>
#include <climits>
#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

// Type alias for readability - specify linear_probing<1> in template
using probing_scheme_type = cuco::linear_probing<1, cuco::murmurhash3_32<uint64_t>>;
using map_type = cuco::static_map<
    uint64_t,
    uint64_t,
    cuco::extent<std::size_t>,
    cuda::thread_scope_device,
    cuda::std::equal_to<uint64_t>,
    probing_scheme_type>;
using map_ref_type = map_type::ref_type<cuco::find_tag>;

__global__ void tokenize(
    int *__restrict__ tokens,
    int *__restrict__ nextToken,
    int numTokens,
    map_ref_type table)
{
    extern __shared__ int sdata[];
    int *minRank = &sdata[0];
    int *minPos = &sdata[1];

    if (threadIdx.x == 0)
    {
        *minRank = INT_MAX;
        *minPos = -1;
    }
    __syncthreads();

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Iterate through merge passes
    for (int n = 0; n < 1; n++)
    {
        __syncthreads();

        if (i >= numTokens)
            continue;

        int j = nextToken[i];
        if (j == -1)
            continue;

        int a = tokens[i];
        int b = tokens[j];
        uint64_t key = (uint64_t(uint32_t(a)) << 32) | uint64_t(uint32_t(b));

        // Probe the hash table
        auto it = table.find(key);
        if (it != table.end())
        {
            uint64_t packed = it->second;
            int rank = int(packed >> 32);
            int mergedTok = int(packed & 0xffffffffU);

            int old = atomicMin(minRank, rank);
            if (rank < old)
            {
                atomicExch(minPos, i);
            }
        }

        __syncthreads();

        // Only the thread with minimum rank performs the merge
        if (threadIdx.x == 0 && *minPos != -1)
        {
            int pos = *minPos;
            int next = nextToken[pos];
            if (next != -1)
            {
                int a = tokens[pos];
                int b = tokens[next];
                uint64_t key = (uint64_t(uint32_t(a)) << 32) | uint64_t(uint32_t(b));

                auto it = table.find(key);
                if (it != table.end())
                {
                    uint64_t packed = it->second;
                    int mergedTok = int(packed & 0xffffffffU);

                    tokens[pos] = mergedTok;
                    nextToken[pos] = nextToken[next];
                    nextToken[next] = -1;
                }
            }

            // Reset for next iteration
            *minRank = INT_MAX;
            *minPos = -1;
        }
        __syncthreads();
    }
}

// Create the map with linear_probing<1> for single-threaded find operations
map_type createMap(size_t capacity)
{
    return map_type{
        capacity,
        cuco::empty_key<uint64_t>{0xFFFFFFFFFFFFFFFFULL},
        cuco::empty_value<uint64_t>{0xFFFFFFFFFFFFFFFFULL}};
}

void launchTokenizeKernel(
    int *tokens,
    int *nextToken,
    int N,
    map_type &table)
{
    // Get device view for find operations
    auto d_view = table.ref(cuco::find);

    int block = 256;
    int grid = (N + block - 1) / block;

    tokenize<<<grid, block, 2 * sizeof(int)>>>(tokens, nextToken, N, d_view);

    cudaDeviceSynchronize();
}