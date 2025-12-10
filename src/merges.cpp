#include "merges.h"
#include <fstream>
#include <sstream>
#include <stdexcept>

inline uint64_t packPair(int a, int b)
{
    return (uint64_t(a) << 32) | uint32_t(b);
}

std::unordered_map<uint64_t, uint64_t> loadMerges(const std::string &path,
                                                  const std::unordered_map<std::string, int> &vocab)
{
    std::ifstream file(path);
    if (!file.is_open())
    {
        throw std::runtime_error("ERROR: Could not open vocab.bpe file.");
    }

    std::unordered_map<uint64_t, uint64_t> rankTable;
    std::string line;
    int rank = 0;

    while (std::getline(file, line))
    {

        if (line.empty())
            continue;
        if (line[0] == '#' || line[0] == ' ' || line[0] == '-')
            continue;

        std::stringstream ss(line);
        std::string A, B;
        ss >> A >> B;

        if (A.empty() || B.empty())
            continue;

        std::string merged = A + B;

        // Lookup IDs in vocab.json
        auto itA = vocab.find(A);
        auto itB = vocab.find(B);
        auto itM = vocab.find(merged);

        if (itA == vocab.end() || itB == vocab.end() || itM == vocab.end())
        {
            // Some merges exist in GPT-2 for bytes, skip if missing
            continue;
        }

        int idA = itA->second;
        int idB = itB->second;
        int idM = itM->second;

        uint64_t key = packPair(idA, idB);
        rankTable[key] = packPair(rank, idM);

        rank++;
    }

    return rankTable;
}
