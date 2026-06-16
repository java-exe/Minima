// BPE-Tokenizer Implementierung (Person 2, Woche 3). Siehe plan.md, Phase 1.
#include "bpe.hpp"

#include <stdexcept>

void BPE::train(const std::vector<std::string>& corpus, int num_merges) {
    // TODO: Corpus auf Zeichenebene splitten, num_merges Mal das haeufigste
    //       benachbarte Paar mergen, vocab_/id_to_token_/merges_ aufbauen.
    (void)corpus;
    (void)num_merges;
    throw std::runtime_error("BPE::train not implemented");
}

void BPE::save(const std::string& path) const {
    // TODO: vocab_ + merges_ als vocab.bin serialisieren.
    (void)path;
    throw std::runtime_error("BPE::save not implemented");
}

void BPE::load(const std::string& path) {
    // TODO: vocab.bin laden.
    (void)path;
    throw std::runtime_error("BPE::load not implemented");
}

std::vector<int> BPE::encode(const std::string& text) const {
    // TODO: Merges anwenden, inkl. BOS/EOS, Einrueckungen als eigene Tokens.
    (void)text;
    throw std::runtime_error("BPE::encode not implemented");
}

std::string BPE::decode(const std::vector<int>& ids) const {
    // TODO: Token-IDs ueber id_to_token_ zurueck zu Text.
    (void)ids;
    throw std::runtime_error("BPE::decode not implemented");
}
