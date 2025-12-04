#include "utils.h"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include "json.hpp"

std::string readFile(const std::string &path)
{
  std::ifstream file(path);
  if (!file.is_open())
  {
    throw std::runtime_error("Failed to open file: " + path);
  }

  std::ostringstream ss;
  ss << file.rdbuf();
  return ss.str();
}

using json = nlohmann::json;

std::unordered_map<std::string, int> loadVocab(const std::string &path)
{
  std::ifstream f(path);
  if (!f.is_open())
  {
    throw std::runtime_error("Could not open vocab.json!");
  }

  json j;
  f >> j;

  std::unordered_map<std::string, int> vocab;

  for (auto &[token, id] : j.items())
  {
    vocab[token] = id.get<int>();
  }

  return vocab;
}

void error(const char *msg)
{
  perror(msg);
  exit(1);
}