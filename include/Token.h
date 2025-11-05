#pragma once
#include <memory>
#include <string>
#include <unordered_set>

class DLLToken; // forward declaration

class Token {
public:
  int id;
  std::string val;

  // Track all DLLToken instances where this Token occurs
  std::unordered_set<DLLToken *> occurrences;

  Token(int id, const std::string &val);
  ~Token() = default;

  void removeOccurrences(const std::vector<DLLToken *> &removables);
};
