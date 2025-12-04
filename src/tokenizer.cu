#include "tokenizer.cuh"
#include <cuda_runtime.h>
#include <cuco/static_map.cuh>
#include <climits>
#include <cstdint>
#include <iostream>
#include <stdio.h>
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
    int *mergeFound = &sdata[2];

    if (threadIdx.x == 0)
    {
        *minRank = INT_MAX;
        *minPos = -1;
        *mergeFound = 1;
    }

    __syncthreads();

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Iterate through merge passes
    while (*mergeFound)
    {
        __syncthreads();

        if (threadIdx.x == 0)
            *mergeFound = 0;

        int j = nextToken[i];
        bool active = j != -1;
        int rank = -1;
        int mergedToken = -1;
        int a = tokens[i];
        int b = tokens[j];
        uint64_t key = (uint64_t(uint32_t(a)) << 32) | uint64_t(uint32_t(b));

        if (active)
        {
            // Probe the hash table
            auto it = table.find(key);
            if (it != table.end())
            {
                uint64_t packed = it->second;
                rank = int(packed >> 32);
                mergedToken = int(packed & 0xffffffffU);

                int old = atomicMin(minRank, rank);
                if (rank < old)
                {
                    atomicExch(minPos, i);
                }
            }
        }

        __syncthreads();

        // Only the thread with minimum rank performs the merge
        if (active && threadIdx.x == 0 && *minPos != -1)
        {
            int pos = *minPos;
            int next = nextToken[pos];
            tokens[pos] = mergedToken;
            nextToken[pos] = nextToken[next];
            nextToken[next] = -1;

            // Reset for next iteration
            *minRank = INT_MAX;
            *minPos = -1;
            *mergeFound = 1;
        }
    }
}

// Helper function to create and populate device hash table from host unordered_map
map_type createDeviceHashTable(const std::unordered_map<uint64_t, uint64_t> &pairRankTable)
{
    // Create device hash table with 2x capacity for good performance
    size_t capacity = pairRankTable.size() * 2;
    map_type table{
        capacity,
        cuco::empty_key<uint64_t>{0xFFFFFFFFFFFFFFFFULL},
        cuco::empty_value<uint64_t>{0xFFFFFFFFFFFFFFFFULL}};

    // Prepare data for insertion
    thrust::host_vector<uint64_t> h_keys;
    thrust::host_vector<uint64_t> h_values;
    h_keys.reserve(pairRankTable.size());
    h_values.reserve(pairRankTable.size());

    for (const auto &[key, value] : pairRankTable)
    {
        h_keys.push_back(key);
        h_values.push_back(value);
    }

    // Copy to device
    thrust::device_vector<uint64_t> d_keys = h_keys;
    thrust::device_vector<uint64_t> d_values = h_values;

    // Insert into map
    auto insert_ref = table.ref(cuco::insert);
    thrust::for_each(
        thrust::device,
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(static_cast<int>(pairRankTable.size())),
        [keys = d_keys.data().get(),
         values = d_values.data().get(),
         insert_ref] __device__(int i) mutable
        {
            insert_ref.insert(cuco::make_pair(keys[i], values[i]));
        });

    return table;
}

void launchTokenizeKernel(
    int *tokens,
    int *nextToken,
    const int N,
    const std::unordered_map<uint64_t, uint64_t> &pairRankTable)
{
    // Create and populate device hash table
    map_type table = createDeviceHashTable(pairRankTable);

    // Get device view for find operations
    auto d_view = table.ref(cuco::find);

    // int block = 256;
    // int grid = (N + block - 1) / block;

    tokenize<<<1, N, 4 * sizeof(int)>>>(tokens, nextToken, N, d_view);

    cudaDeviceSynchronize();
}