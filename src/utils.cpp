#include "utils.h"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include "tokenizer.cuh"
#include <string>

std::string readFile(const std::string &path)
{
  std::ifstream file(path);
  if (!file.is_open())
  {
    throw std::runtime_error("Failed to open file: " + path);
  }

  std::ostringstream ss;
  ss << file.rdbuf();
  return ss.str();
}

void error(const char *msg)
{
  perror(msg);
  exit(1);
}

void gpuCleanup(int argc, void *args[], cudaStream_t stream)
{
  for (int i = 0; i < argc; i++)
  {
    CUDA_CHECK(cudaFree(args[i]));
  }
  CUDA_CHECK(cudaStreamDestroy(stream));
}