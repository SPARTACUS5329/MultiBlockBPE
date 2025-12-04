#include "utils.h"
#include "merges.h"
#include "byte_encoder.h"
#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>

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

  // -----------------------------------------
  // 1. Load vocab and merges
  // -----------------------------------------
  auto vocab = loadVocab("./assets/vocab.json");
  auto merges = loadMerges("./assets/vocab.bpe", vocab);
  auto byte_encoder = bytes_to_unicode();

  std::cout << "Loaded vocab size:  " << vocab.size() << "\n";
  std::cout << "Loaded merges:      " << merges.rank_table.size() << "\n";

  // -----------------------------------------
  // 2. Open input file for lexing
  // -----------------------------------------
  FILE *f = fopen(argv[1], "r");
  if (!f)
    error("[main] File open error");

  yyin = f;

  // Output token list
  std::vector<int> tokens;
  std::vector<int> nextToken;

  int token;

  // -----------------------------------------
  // 3. Run lexer + convert characters → vocab IDs
  // -----------------------------------------
  while ((token = yylex()) != 0)
  {
    switch (token)
    {
    case PRE_TOKEN:
      for (int i = 0; yytext[i] != '\0'; i++)
      {
        unsigned char byte = static_cast<unsigned char>(yytext[i]);

        std::string key = byte_encoder[byte];

        if (vocab.find(key) == vocab.end())
        {
          std::cerr << "Unknown token in vocab: '" << key << "' (byte " << (int)byte << ")\n";
          return 1;
        }

        tokens.push_back(vocab[key]);
        nextToken.push_back(i + 1);
      }
      nextToken.back() = -1;
      break;
    default:
      error("[main] Invalid token lexeme received");
      break;
    }
  }

  fclose(f);

  if (!nextToken.empty())
    nextToken.back() = -1;

  // -----------------------------------------
  // 4. Debug print: tokens + next pointers
  // -----------------------------------------
  std::cout << "\nRaw tokens:\n";
  for (int id : tokens)
    std::cout << id << " ";
  std::cout << "\n";

  std::cout << "Next pointers:\n";
  for (int nxt : nextToken)
    std::cout << nxt << " ";
  std::cout << "\n";

  return 0;
}