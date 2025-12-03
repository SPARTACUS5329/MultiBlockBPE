#pragma once
#include <string>
#include <unordered_map>
#include <cstdint>

struct MergeTables {
    std::unordered_map<uint64_t, int> rank_table;
    std::unordered_map<uint64_t, int> merge_table;
};

MergeTables loadMerges(const std::string& path,
                       const std::unordered_map<std::string, int>& vocab);
