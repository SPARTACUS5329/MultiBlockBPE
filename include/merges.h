#pragma once
#include <string>
#include <unordered_map>
#include <cstdint>

std::unordered_map<uint64_t, uint64_t> loadMerges(const std::string &path,
                                                  const std::unordered_map<std::string, int> &vocab);
