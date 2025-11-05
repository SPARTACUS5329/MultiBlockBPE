#pragma once
#include "Token.h"
#include <memory>

class DLLToken {
public:
  int id;
  std::shared_ptr<Token> token;
  DLLToken *next;
  DLLToken *prev;

  DLLToken(int id, std::shared_ptr<Token> token);

  // Merge current token with another DLLToken
  void mergeTokens(DLLToken *other);
};
