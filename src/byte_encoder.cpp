#include "byte_encoder.h"
#include <vector>
#include <algorithm>

std::unordered_map<uint8_t, std::string> bytes_to_unicode() {
    std::vector<int> bs;
    std::vector<int> cs;

    for (int i = int('!'); i <= int('~'); i++) bs.push_back(i);
    for (int i = 0xA1; i <= 0xAC; i++) bs.push_back(i);
    for (int i = 0xAE; i <= 0xFF; i++) bs.push_back(i);

    cs = bs;
    int n = 0;

    for (int b = 0; b < 256; b++) {
        if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
            bs.push_back(b);
            cs.push_back(256 + n);
            n++;
        }
    }

    std::unordered_map<uint8_t, std::string> byte_encoder;

    for (size_t i = 0; i < bs.size(); i++) {
        int b = bs[i];
        int c = cs[i];

        // UTF-8 encode unicode codepoint c
        std::string utf8;

        if (c < 128) {
            utf8.push_back(char(c));
        } else if (c < 2048) {
            utf8.push_back(char(192 + c / 64));
            utf8.push_back(char(128 + c % 64));
        } else if (c < 65536) {
            utf8.push_back(char(224 + c / 4096));
            utf8.push_back(char(128 + (c / 64) % 64));
            utf8.push_back(char(128 + c % 64));
        } else {
            utf8.push_back(char(240 + c / 262144));
            utf8.push_back(char(128 + ((c / 4096) % 64)));
            utf8.push_back(char(128 + ((c / 64) % 64)));
            utf8.push_back(char(128 + (c % 64)));
        }

        byte_encoder[(uint8_t)b] = utf8;
    }

    return byte_encoder;
}
