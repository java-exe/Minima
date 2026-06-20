// BPE-Tokenizer (Person 2, Woche 3). Siehe plan.md, Phase 1.
//
// Klassisches Byte-Pair-Encoding (Sennrich 2016) auf Zeichenebene:
//   - Spezial-Tokens belegen die IDs 0..3 (pad/eos/bos/unk).
//   - Das Basis-Vokabular besteht aus allen im Corpus gesehenen Zeichen (Bytes).
//   - train() merged num_merges Mal das haeufigste benachbarte Paar.
//   - Whitespace-Laeufe (Einrueckungen) werden getrennt von Code-Laeufen
//     vortokenisiert und bleiben dadurch eigene Tokens (siehe docs/interface.md).
#pragma once

#include <string>
#include <vector>
#include <utility>
#include <cstdint>
#include <unordered_map>

class BPE {
public:
    BPE() { init_special_tokens(); }
    // Bequemer Konstruktor passend zu docs/interface.md:
    //   tokenizer_cpp.Tokenizer("path/to/vocab.bin")
    explicit BPE(const std::string& path) { load(path); }

    // Trainieren: Corpus einlesen, num_merges Mal haeufigstes Paar mergen
    void train(const std::vector<std::string>& corpus, int num_merges);

    // Serialisieren
    void save(const std::string& path) const;
    void load(const std::string& path);

    // Anwenden
    std::vector<int> encode(const std::string& text) const;
    std::string      decode(const std::vector<int>& ids) const;

    int vocab_size()   const { return static_cast<int>(id_to_token_.size()); }
    int pad_token_id() const { return 0; }
    int eos_token_id() const { return 1; }
    int bos_token_id() const { return 2; }
    int unk_token_id() const { return 3; }

private:
    std::unordered_map<std::string, int> vocab_;        // Token-String -> ID
    std::vector<std::string>             id_to_token_;  // ID -> Token-String
    std::vector<std::pair<std::string, std::string>> merges_;  // gelernte Merges (in Reihenfolge)

    // Aus den Merges abgeleitet, nur fuer schnelles encode()/load():
    std::unordered_map<std::int64_t, int> pair_to_id_;   // gepacktes (a_id,b_id) -> neue ID
    std::unordered_map<std::int64_t, int> merge_rank_;   // gepacktes (a_id,b_id) -> Rang
    std::unordered_map<std::string, int>  char_to_id_;   // Einzelzeichen -> Basis-ID

    void init_special_tokens();
    void rebuild_aux();  // pair_to_id_/merge_rank_/char_to_id_ aus vocab_/merges_ neu aufbauen

    // Text in Whitespace-Laeufe und Nicht-Whitespace-Laeufe zerlegen.
    static std::vector<std::string> split_words(const std::string& text);

    // (a_id, b_id) in einen 64-bit-Schluessel packen.
    static std::int64_t pack(int a, int b) {
        return (static_cast<std::int64_t>(a) << 32)
             | static_cast<std::int64_t>(static_cast<std::uint32_t>(b));
    }
};
