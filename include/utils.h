#pragma once
#include <cuda_runtime.h>
#include <string>

std::string readFile(const std::string &path);
void error(const char *msg);
void gpuCleanup(int argc, void *args[], cudaStream_t stream);