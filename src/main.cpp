#include "utils.h"
#include <cstdio>
#include <iostream>
#include <unordered_map>
#include <vector>
#include <string>
#include "tokenizer.cuh"

#define MAX_SEQ_LEN 1024

extern "C"
{
  int yylex();
  extern char *yytext;
  extern FILE *yyin;
  enum TokenType
  {
    PRE_TOKEN = 1,
  };
}

int main(int argc, char *argv[])
{
  if (argc < 2)
  {
    error("[main] Not enough input arguments: expected file address");
  }

  const char *filename = argv[1];
  FILE *f = fopen(filename, "r");
  if (!f)
  {
    error("[main] File not found");
  }

  yyin = f;

  int token;
  std::vector<int> tokens;
  std::vector<int> nextToken;
  std::unordered_map<std::string, int> tokenIDMap;
  // populate token_id_map from vocab.json
  // token_id_map["\'"] = 12;
  // token_id_map["s"] = 24;

  while ((token = yylex()) != 0)
  {
    switch (token)
    {
    case PRE_TOKEN:
      // Parallelize the for loop
      int i;
      for (i = 0; yytext[i] != '\0'; i++)
      {
        std::string key(1, yytext[i]);
        tokens.push_back(tokenIDMap[key]);
        nextToken.push_back(i + 1);
      }
      nextToken.back() = -1;
      break;
    default:
      std::cout << "[UNKNOWN] " << yytext << "\n";
    }
  }

  fclose(f);

  CUDA_CHECK(cudaSetDevice(0));
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  int *dTokens = nullptr, *dNextToken = nullptr;

  CUDA_CHECK(cudaMalloc(&dTokens, tokens.size() * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&dNextToken, nextToken.size() * sizeof(int)));

  CUDA_CHECK(cudaMemcpyAsync(dTokens, tokens.data(), tokens.size() * sizeof(int), cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(dNextToken, nextToken.data(), nextToken.size() * sizeof(int), cudaMemcpyHostToDevice, stream));

  if (tokens.size() != nextToken.size())
  {
    int argc = 2;
    void *args[] = {dTokens, dNextToken};
    gpuCleanup(argc, args, stream);
    error("[main] tokens.size() and nextToken.size() don't match");
  }

  launchTokenizeKernel(dTokens, dNextToken, (int)tokens.size());

  CUDA_CHECK(cudaMemcpyAsync(dTokens, tokens.data(), tokens.size() * sizeof(int), cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(dNextToken, nextToken.data(), nextToken.size() * sizeof(int), cudaMemcpyHostToDevice, stream));

  return 0;
}
