%option noyywrap

%{
#include <stdio.h>
enum TokenType {
    TOKEN = 1
};
%}

/* ASCII letters */
LETTER [A-Za-z]

/* ASCII digits */
DIGIT [0-9]

/* Whitespace */
SPACE [ \t\n\r\f\v]

%%

{LETTER}+   { return TOKEN; }
{DIGIT}+    { return TOKEN; }
[^A-Za-z0-9[:space:]]+   { return TOKEN; }
{SPACE}+    { return TOKEN; }
.           { return TOKEN; }

%%
