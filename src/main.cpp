#include "utils.h"
#include <chrono>
#include "merges.h"
#include "byte_encoder.h"
#include <iostream>
#include <vector>
#include <string>
#include "tokenizer_interface.h"
#include <unordered_map>
#define TOTAL_BATCH_SIZE 200000

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
    std::cerr << "Usage: ./MultiBlockBPE <input_file>\n";
    return 1;
  }

  auto vocab = loadVocab("./assets/vocab.json");
  auto pairRankTable = loadMerges("./assets/vocab.bpe", vocab);
  auto byte_encoder = bytes_to_unicode();

  std::cout << "Loaded vocab size:  " << vocab.size() << "\n";
  std::cout << "Loaded merges:      " << pairRankTable.size() << "\n";

  FILE *f = fopen(argv[1], "r");
  if (!f)
    error("[main] File open error");

  yyin = f;

  std::vector<int> tokens;
  std::vector<int> nextToken;

  int token;
  int totalBytes = 0;
  double totalTime = 0;
  int batches = 0;

  CUDA_CHECK(cudaSetDevice(0));
  cudaStream_t stream;
  cudaEvent_t e0, e1;
  CUDA_CHECK(cudaStreamCreate(&stream));

  int *dTokens = nullptr, *dNextToken = nullptr;
  int dSize = 1.5 * TOTAL_BATCH_SIZE;
  CUDA_CHECK(cudaMalloc(&dTokens, dSize * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&dNextToken, dSize * sizeof(int)));

  DeviceHashTable *d_pairRankTable = createDeviceHashTable(pairRankTable);

  // auto start = std::chrono::high_resolution_clock::now();
  while ((token = yylex()) != 0)
  {
    switch (token)
    {
    case PRE_TOKEN:
      for (int i = 0; yytext[i] != '\0'; i++)
      {
        unsigned char byte = static_cast<unsigned char>(yytext[i]);
        totalBytes++;

        std::string key = byte_encoder[byte];

        if (vocab.find(key) == vocab.end())
          error("[main] Unknown token in vocab");

        tokens.push_back(vocab[key]);
        nextToken.push_back(tokens.size());
      }
      nextToken.back() = -1;
      break;
    default:
      error("[main] Invalid token lexeme received");
      break;
    }

    if (tokens.size() > TOTAL_BATCH_SIZE)
    {
      batches++;
      printf("Merging batch: %d %d\n", batches, tokens.size());
      fflush(0);
      CUDA_CHECK(cudaMemcpyAsync(dTokens, tokens.data(), tokens.size() * sizeof(int), cudaMemcpyHostToDevice, stream));
      CUDA_CHECK(cudaMemcpyAsync(dNextToken, nextToken.data(), nextToken.size() * sizeof(int), cudaMemcpyHostToDevice, stream));
      CUDA_CHECK(cudaStreamSynchronize(stream));

      if (tokens.size() != nextToken.size())
      {
        int argc = 2;
        void *args[] = {dTokens, dNextToken};
        gpuCleanup(argc, args, stream);
        error("[main] tokens.size() and nextToken.size() don't match");
      }

      CUDA_CHECK(cudaEventCreate(&e0));
      CUDA_CHECK(cudaEventCreate(&e1));

      CUDA_CHECK(cudaEventRecord(e0, stream));

      launchTokenizeKernel(dTokens, dNextToken, (int)tokens.size(), d_pairRankTable);

      CUDA_CHECK(cudaEventRecord(e1, stream));
      CUDA_CHECK(cudaEventSynchronize(e1));

      double ms = elapsed_ms(e0, e1);
      totalTime += ms;
      std::cout << "ms: " << ms << "\n";
      std::cout << "totalTime: " << totalTime << "\n";

      CUDA_CHECK(cudaMemcpy(tokens.data(), dTokens, tokens.size() * sizeof(int), cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(nextToken.data(), dNextToken, nextToken.size() * sizeof(int), cudaMemcpyDeviceToHost));

      std::cout << "\nRaw tokens:\n";
      for (int id : tokens)
      {
        if (id == -1)
          continue;
        std::cout << id << " ";
      }
      std::cout << "\n";

      tokens.clear();
      nextToken.clear();
    }
  }

  fclose(f);

  std::cout << "Total Bytes: " << totalBytes << " B\n";
  std::cout << "Time taken: " << totalTime << " ms\n";
  std::cout << "Throughput: " << totalBytes * 1e3 / totalTime << " Bps\n";

  CUDA_CHECK(cudaEventDestroy(e0));
  CUDA_CHECK(cudaEventDestroy(e1));
  CUDA_CHECK(cudaFree(dTokens));
  CUDA_CHECK(cudaFree(dNextToken));
  CUDA_CHECK(cudaStreamDestroy(stream));

  return 0;
}