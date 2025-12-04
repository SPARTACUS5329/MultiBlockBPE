#pragma once
#include <cuda_runtime.h>
#include <string>
#include <unordered_map>

void error(const char *msg);
void gpuCleanup(int argc, void *args[], cudaStream_t stream);
std::unordered_map<std::string, int> loadVocab(const std::string &path);
std::string readFile(const std::string &path);
void error(const char *);
