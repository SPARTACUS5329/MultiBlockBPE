#include "utils.h"
#include <chrono>
#include "merges.h"
#include "byte_encoder.h"
#include <iostream>
#include <vector>
#include <fstream>
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

  if (argc < 5)
  {
    // <mode> is SEQ or BATCH
    std::cerr << "Usage: ./MultiBlockBPE <input_file> <output_file> <mode> <value>\n";
    return 1;
  }

  auto vocab = loadVocab("./assets/vocab.json");
  auto pairRankTable = loadMerges("./assets/vocab.bpe", vocab);
  auto byte_encoder = bytes_to_unicode();

  std::cout << "Loaded vocab size:  " << vocab.size() << "\n";
  std::cout << "Loaded merges:      " << pairRankTable.size() << "\n";

  int BATCH_SIZE;
  int SEQ_LEN;

  if (std::string(argv[3]) == "SEQ")
  {
    SEQ_LEN = atoi(argv[4]);
    BATCH_SIZE = TOTAL_BATCH_SIZE / SEQ_LEN;
  }
  else if (std::string(argv[3]) == "BATCH")
  {
    BATCH_SIZE = atoi(argv[4]);
    SEQ_LEN = TOTAL_BATCH_SIZE / BATCH_SIZE;
  }
  else
  {
    std::cout << argv[2];
    error("[main] Invalid independent parameter, expected \"SEQ\" or \"BATCH\"");
  }

  std::string inputFile = argv[1];
  std::string outputFile = argv[2];

  FILE *f = fopen(inputFile.c_str(), "r");
  if (!f)
    error("[main] File open error");

  yyin = f;

  std::vector<int> tokens;
  std::vector<int> nextToken;

  int token;
  int totalBytes = 0;
  double totalTime = 0;

  CUDA_CHECK(cudaSetDevice(0));
  cudaStream_t stream;
  cudaEvent_t e0, e1;
  CUDA_CHECK(cudaStreamCreate(&stream));

  int *dTokens = nullptr, *dNextToken = nullptr;
  int dSize = 1.5 * TOTAL_BATCH_SIZE;
  CUDA_CHECK(cudaMalloc(&dTokens, dSize * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&dNextToken, dSize * sizeof(int)));

  DeviceHashTable *d_pairRankTable = createDeviceHashTable(pairRankTable);

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

    if (tokens.size() >= TOTAL_BATCH_SIZE)
    {
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

      launchTokenizeKernel(dTokens, dNextToken, (int)tokens.size(), SEQ_LEN, d_pairRankTable);

      CUDA_CHECK(cudaEventRecord(e1, stream));
      CUDA_CHECK(cudaEventSynchronize(e1));

      double ms = elapsed_ms(e0, e1);
      totalTime += ms;

      CUDA_CHECK(cudaMemcpy(tokens.data(), dTokens, tokens.size() * sizeof(int), cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(nextToken.data(), dNextToken, nextToken.size() * sizeof(int), cudaMemcpyDeviceToHost));

      writeTokensToFile(tokens, outputFile);

      tokens.clear();
      nextToken.clear();
    }
  }

  fclose(f);

  if (outputFile != "stdout")
  {
    std::ofstream file("./output/results.txt", std::ios::app);

    if (!file.is_open())
      error("[main] Failed to open results.txt\n");

    auto *old_buf = std::cout.rdbuf();
    std::cout.rdbuf(file.rdbuf());
  }

  std::cout << "Batch size: " << BATCH_SIZE << "\n";
  std::cout << "Seq length: " << SEQ_LEN << "\n";
  std::cout << "Throughput: " << totalBytes * 1e3 / totalTime << " Bps\n"; // 1e3 because time is in ms
  std::cout << "Total Bytes: " << totalBytes << " B\n";
  std::cout << "Time taken: " << totalTime << " ms\n\n";

  CUDA_CHECK(cudaEventDestroy(e0));
  CUDA_CHECK(cudaEventDestroy(e1));
  CUDA_CHECK(cudaFree(dTokens));
  CUDA_CHECK(cudaFree(dNextToken));
  CUDA_CHECK(cudaStreamDestroy(stream));

  return 0;
}