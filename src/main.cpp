#include "utils.h"
#include <cstdio>
#include <iostream>
#include <string>

void error(const char *msg) {
  perror(msg);
  exit(1);
}

int main(int argc, char *argv[]) {
  if (argc < 2) {
    error("[main] Not enough input arguments: expected file address");
  }

  const std::string filename = argv[1];
  std::string content = readFile(filename);
  DLLToken *dllHead = initDLL(content);

  auto *dllToken = dllHead;
  while (dllToken) {
    std::cout << dllToken->token.val << " ";
    dllToken = dllToken->next;
  }

  return 0;
}
