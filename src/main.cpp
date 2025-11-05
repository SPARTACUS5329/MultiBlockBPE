#include "DLLToken.h"
#include "Token.h"
#include <iostream>

int main() {
  auto t1 = std::make_shared<Token>(1, "he");
  auto t2 = std::make_shared<Token>(2, "llo");

  DLLToken node1(1, t1);
  DLLToken node2(2, t2);

  node1.next = &node2;
  node2.prev = &node1;

  std::cout << "Before merge: " << node1.token->val << " " << node2.token->val
            << "\n";
  node1.mergeTokens(&node2);
  std::cout << "After merge: " << node1.token->val << "\n";

  std::vector<DLLToken *> toRemove = {&node1, &node2};
  node1.token->removeOccurrences(toRemove);

  return 0;
}
