#pragma once
#include <cuda_runtime.h>
#include <string>
#include <unordered_map>
#include "tokenizer.cuh"

void error(const char *msg);
void gpuCleanup(int argc, void *args[], cudaStream_t stream);
std::unordered_map<std::string, int> loadVocab(const std::string &path);
std::string readFile(const std::string &path);
void error(const char *);

inline float elapsed_ms(cudaEvent_t a, cudaEvent_t b)
{
    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
    return ms;
}
