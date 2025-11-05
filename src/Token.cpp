#include "Token.h"
#include "DLLToken.h"
#include <iostream>

Token::Token(int id, const std::string &val) : id(id), val(val) {}

void Token::removeOccurrences(const std::vector<DLLToken *> &removables) {
  for (auto *r : removables) {
    auto it = occurrences.find(r);
    if (it != occurrences.end()) {
      occurrences.erase(it);
      std::cout << "Removed occurrence of token " << id << " from DLLToken "
                << r->id << "\n";
    }
  }
}
