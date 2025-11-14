#include "utils.h"
#include <cstdio>
#include <iostream>
#include <string>

extern "C" {
int yylex();
extern char *yytext;
extern FILE *yyin;
enum TokenType { TOKEN = 1 };
}

void error(const char *msg) {
  perror(msg);
  exit(1);
}

int main(int argc, char *argv[]) {
  if (argc < 2) {
    error("[main] Not enough input arguments: expected file address");
  }

  const char *filename = argv[1];
  FILE *f = fopen(filename, "r");
  if (!f) {
    error("[main] File not found");
  }

  yyin = f;

  int token;
  while ((token = yylex()) != 0) {
    switch (token) {
    case TOKEN:
      printf("TOKEN: %s\n", yytext);
      break;

    default:
      std::cout << "[UNKNOWN] " << yytext << "\n";
    }
  }

  fclose(f);

  return 0;
}
