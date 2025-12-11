#pragma once
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

class DeviceHashTable;

void launchTokenizeKernel(
    int *tokens,
    int *nextToken,
    const int N,
    DeviceHashTable *table);

DeviceHashTable *createDeviceHashTable(
    const std::unordered_map<uint64_t, uint64_t> &pairRankTable);

void destroyDeviceHashTable(DeviceHashTable *table);