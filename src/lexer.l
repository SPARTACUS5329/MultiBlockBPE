%option noyywrap

%{
#include <stdio.h>
enum TokenType {
    PRE_TOKEN = 1
};
%}

%%

\'(s|t|re|ve|m|ll|d)        { return PRE_TOKEN; }

[ ][A-Za-z]+               { return PRE_TOKEN; }
[ ][0-9]+                   { return PRE_TOKEN; }
[ ][^A-Za-z0-9 \t\n\r\f\v]+ { return PRE_TOKEN; }

[A-Za-z]+                  { return PRE_TOKEN; }
[0-9]+                     { return PRE_TOKEN; }
[^A-Za-z0-9 \t\n\r\f\v]+   { return PRE_TOKEN; }

[ \t\n\r\f\v]+             { return PRE_TOKEN; }

.                           { return PRE_TOKEN; }

%%
