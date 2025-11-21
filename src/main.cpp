#include "utils.h"
#include <cstdio>
#include <iostream>
#include <string>

extern "C"
{
  int yylex();
  extern char *yytext;
  extern FILE *yyin;
  enum TokenType
  {
    SUFFIX = 1,
    SPACED_LETTER_SEQ,
    SPACED_NUMBER_SEQ,
    SPACED_PUNCTUATION_SEQ,
    LETTER_SEQ,
    NUMBER_SEQ,
    PUNCTUATION_SEQ,
    SPACE,
    FALLBACK_CHAR
  };
}

void error(const char *msg)
{
  perror(msg);
  exit(1);
}

int main(int argc, char *argv[])
{
  if (argc < 2)
  {
    error("[main] Not enough input arguments: expected file address");
  }

  const char *filename = argv[1];
  FILE *f = fopen(filename, "r");
  if (!f)
  {
    error("[main] File not found");
  }

  yyin = f;

  int token;
  while ((token = yylex()) != 0)
  {
    switch (token)
    {

    case SUFFIX:
      printf("SUFFIX: %s\n", yytext);
      break;
    case SPACED_LETTER_SEQ:
      printf("SPACED_LETTER_SEQ: %s\n", yytext);
      break;
    case SPACED_NUMBER_SEQ:
      printf("SPACED_NUMBER_SEQ: %s\n", yytext);
      break;
    case SPACED_PUNCTUATION_SEQ:
      printf("SPACED_PUNCTUATION_SEQ: %s\n", yytext);
      break;
    case LETTER_SEQ:
      printf("LETTER_SEQ: %s\n", yytext);
      break;
    case NUMBER_SEQ:
      printf("NUMBER_SEQ: %s\n", yytext);
      break;
    case PUNCTUATION_SEQ:
      printf("PUNCTUATION_SEQ: %s\n", yytext);
      break;
    case SPACE:
      printf("SPACE: %s\n", yytext);
      break;
    case FALLBACK_CHAR:
      printf("FALLBACK_CHAR: %s\n", yytext);
      break;

    default:
      std::cout << "[UNKNOWN] " << yytext << "\n";
    }
  }

  fclose(f);

  return 0;
}
