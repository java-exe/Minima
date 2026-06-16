// BPE-Tokenizer (Person 2, Woche 3). Siehe plan.md, Phase 1.
#pragma once

#include <string>
#include <vector>
#include <utility>
#include <unordered_map>

class BPE {
public:
    // Trainieren: Corpus einlesen, num_merges Mal haeufigstes Paar mergen
    void train(const std::vector<std::string>& corpus, int num_merges);

    // Serialisieren
    void save(const std::string& path) const;
    void load(const std::string& path);

    // Anwenden
    std::vector<int> encode(const std::string& text) const;
    std::string      decode(const std::vector<int>& ids) const;

    int vocab_size()   const { return static_cast<int>(vocab_.size()); }
    int pad_token_id() const { return 0; }
    int eos_token_id() const { return 1; }
    int bos_token_id() const { return 2; }
    int unk_token_id() const { return 3; }

private:
    std::unordered_map<std::string, int> vocab_;
    std::vector<std::string>             id_to_token_;
    std::vector<std::pair<std::string, std::string>> merges_;
};
