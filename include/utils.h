#pragma once
#include "DLLToken.h"
#include <string>

std::string readFile(const std::string &path);
DLLToken *initDLL(const std::string &s);
