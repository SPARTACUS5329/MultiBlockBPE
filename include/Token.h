#pragma once
#include <string>
#include <unordered_set>

class Token
{
public:
  int id;
  std::string val;

  // Track all DLLToken instances where this Token occurs
  // std::unordered_set<DLLToken *> occurrences;

  Token(int id, const std::string &val);
  ~Token() = default;
};
