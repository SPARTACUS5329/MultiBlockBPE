#pragma once
#include <cuda_runtime.h>
#include <unordered_map>
#include <cuco/static_map.cuh>
#include <cstdint>

using probing_scheme_type = cuco::linear_probing<1, cuco::murmurhash3_32<uint64_t>>;
using map_type = cuco::static_map<
    uint64_t,
    uint64_t,
    cuco::extent<std::size_t>,
    cuda::thread_scope_device,
    cuda::std::equal_to<uint64_t>,
    probing_scheme_type>;

class DeviceHashTable
{
public:
    map_type table;

    DeviceHashTable(map_type &&t) : table(std::move(t)) {}
};