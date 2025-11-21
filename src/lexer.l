%option noyywrap

%{
#include <stdio.h>
enum TokenType {
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
%}

%%

\'(s|t|re|ve|m|ll|d)        { return SUFFIX; }

[ ][A-Za-z]+               { return SPACED_LETTER_SEQ; }
[ ][0-9]+                   { return SPACED_NUMBER_SEQ; }
[ ][^A-Za-z0-9 \t\n\r\f\v]+ { return SPACED_PUNCTUATION_SEQ; }

[A-Za-z]+                  { return LETTER_SEQ; }
[0-9]+                     { return NUMBER_SEQ; }
[^A-Za-z0-9 \t\n\r\f\v]+   { return PUNCTUATION_SEQ; }

[ \t\n\r\f\v]+             { return SPACE; }

.                           { return FALLBACK_CHAR; }

%%
