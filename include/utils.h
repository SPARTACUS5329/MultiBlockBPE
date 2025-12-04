#pragma once
#include <string>
#include <unordered_map>

std::unordered_map<std::string, int> loadVocab(const std::string &path);
std::string readFile(const std::string &path);
void error(const char *);