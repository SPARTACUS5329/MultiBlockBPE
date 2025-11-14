#include "utils.h"
#include "DLLToken.h"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

std::string readFile(const std::string &path) {
  std::ifstream file(path);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open file: " + path);
  }

  std::ostringstream ss;
  ss << file.rdbuf();
  return ss.str();
}

DLLToken *initDLL(const std::string &s) {
  if (s.empty())
    return nullptr;

  DLLToken *head = nullptr;
  DLLToken *prev = nullptr;

  for (size_t i = 0; i < s.size(); i++) {
    Token t(i, std::string(1, s[i]));
    DLLToken *node = new DLLToken(i, t);

    if (!head) {
      head = node;
    } else {
      prev->next = node;
      node->prev = prev;
    }

    node->token.occurrences.insert(node);

    prev = node;
  }

  return head;
}
