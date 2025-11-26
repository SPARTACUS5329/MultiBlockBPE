#include "utils.h"
#include <cstdio>
#include <iostream>
#include <unordered_map>
#include <vector>
#include <string>
#include <omp.h>
#include <mutex>

#define MAX_SEQ_LEN 1024

std::mutex vecMutex; // Mutex for synchronizing access to shared vectors
 
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

void error(const char *msg)
{
  perror(msg);
  exit(1);
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

  #pragma omp parallel
  #pragma omp single  // Single thread to call yylex
  {
    while ((token = yylex()) != 0)
    {
        if (token == PRE_TOKEN)
        {
            // IMPORTANT: copy yytext immediately
            std::string textCopy = yytext;

            #pragma omp task firstprivate(textCopy)
            {   
                printf("Thread %d handling chunk: %s\n", omp_get_thread_num(), textCopy.c_str());
                std::vector<int> localTokens;
                std::vector<int> localNext;

                for (int i = 0; i < textCopy.size(); i++) {
                    std::string key(1, textCopy[i]);
                    localTokens.push_back(tokenIDMap[key]);
                    localNext.push_back(i + 1);
                }
                localNext.back() = -1;

                // append to the global vectors safely
                std::lock_guard<std::mutex> lock(vecMutex);
                tokens.insert(tokens.end(), localTokens.begin(), localTokens.end());
                nextToken.insert(nextToken.end(), localNext.begin(), localNext.end());
            }
        }
    }

    // Wait until ALL tasks complete
    #pragma omp taskwait
}


  fclose(f);

  for (auto &p : nextToken)
  {
    std::cout << p << " ";
  }

  return 0;
}
