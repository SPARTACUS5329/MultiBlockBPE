#include "utils.h"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

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