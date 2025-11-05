#include "DLLToken.h"
#include <iostream>

DLLToken::DLLToken(int id, std::shared_ptr<Token> token)
    : id(id), token(std::move(token)), next(nullptr), prev(nullptr) {
  this->token->occurrences.insert(this);
}

void DLLToken::mergeTokens(DLLToken *other) {
  if (!other)
    return;

  // Combine string values
  this->token->val += other->token->val;

  // Merge occurrences
  for (auto *occ : other->token->occurrences)
    this->token->occurrences.insert(occ);

  // Relink the doubly linked list
  if (other->next)
    other->next->prev = this;
  this->next = other->next;

  std::cout << "Merged token " << id << " with token " << other->id << "\n";
}
