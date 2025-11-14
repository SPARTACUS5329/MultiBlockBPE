#pragma once
#include "Token.h"

struct DLLToken {
  int id;
  Token token;
  DLLToken *next;
  DLLToken *prev;

  DLLToken(int id, const Token &tok)
      : id(id), token(tok), next(nullptr), prev(nullptr) {}

  void merge_tokens(DLLToken *other) { token.val += other->token.val; }
};
