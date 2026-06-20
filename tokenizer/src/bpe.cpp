// BPE-Tokenizer Implementierung (Person 2, Woche 3). Siehe plan.md, Phase 1.
#include "bpe.hpp"

#include <algorithm>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace {

// Whitespace nach C-Konvention (Space, Tab, Newline, CR, FF, VT).
bool is_ws(unsigned char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v';
}

// uint32 little-endian schreiben/lesen, damit vocab.bin plattformunabhaengig ist.
void put_u32(std::ostream& os, std::uint32_t v) {
    char b[4] = {
        static_cast<char>(v & 0xFF),
        static_cast<char>((v >> 8) & 0xFF),
        static_cast<char>((v >> 16) & 0xFF),
        static_cast<char>((v >> 24) & 0xFF),
    };
    os.write(b, 4);
}

std::uint32_t get_u32(std::istream& is) {
    unsigned char b[4];
    is.read(reinterpret_cast<char*>(b), 4);
    if (!is) throw std::runtime_error("vocab.bin: unerwartetes Dateiende");
    return static_cast<std::uint32_t>(b[0])
         | (static_cast<std::uint32_t>(b[1]) << 8)
         | (static_cast<std::uint32_t>(b[2]) << 16)
         | (static_cast<std::uint32_t>(b[3]) << 24);
}

void put_str(std::ostream& os, const std::string& s) {
    put_u32(os, static_cast<std::uint32_t>(s.size()));
    os.write(s.data(), static_cast<std::streamsize>(s.size()));
}

std::string get_str(std::istream& is) {
    std::uint32_t n = get_u32(is);
    std::string s(n, '\0');
    if (n) is.read(&s[0], n);
    if (!is) throw std::runtime_error("vocab.bin: unerwartetes Dateiende");
    return s;
}

}  // namespace

void BPE::init_special_tokens() {
    vocab_.clear();
    id_to_token_.clear();
    merges_.clear();
    pair_to_id_.clear();
    merge_rank_.clear();
    char_to_id_.clear();

    const char* specials[] = {"<pad>", "<eos>", "<bos>", "<unk>"};
    for (const char* s : specials) {
        vocab_[s] = static_cast<int>(id_to_token_.size());
        id_to_token_.emplace_back(s);
    }
}

std::vector<std::string> BPE::split_words(const std::string& text) {
    // Maximale Laeufe gleicher "Art" (Whitespace vs. Nicht-Whitespace) bilden.
    // Die Konkatenation aller Laeufe ergibt wieder den Originaltext -> Roundtrip
    // bleibt erhalten, Einrueckungen mergen nie mit benachbartem Code.
    std::vector<std::string> words;
    std::size_t i = 0;
    while (i < text.size()) {
        bool ws = is_ws(static_cast<unsigned char>(text[i]));
        std::size_t j = i + 1;
        while (j < text.size() &&
               is_ws(static_cast<unsigned char>(text[j])) == ws) {
            ++j;
        }
        words.push_back(text.substr(i, j - i));
        i = j;
    }
    return words;
}

void BPE::train(const std::vector<std::string>& corpus, int num_merges) {
    init_special_tokens();
    if (num_merges < 0) num_merges = 0;

    // 1) Basis-Vokabular: jedes im Corpus gesehene Zeichen wird ein Token.
    //    Sortiert nach Byte-Wert -> deterministische IDs.
    bool seen[256] = {false};
    for (const std::string& line : corpus)
        for (unsigned char c : line) seen[c] = true;

    for (int b = 0; b < 256; ++b) {
        if (!seen[b]) continue;
        std::string ch(1, static_cast<char>(b));
        char_to_id_[ch] = static_cast<int>(id_to_token_.size());
        vocab_[ch] = static_cast<int>(id_to_token_.size());
        id_to_token_.push_back(ch);
    }

    // 2) Eindeutige Woerter mit Haeufigkeit sammeln und als ID-Sequenz ablegen.
    std::unordered_map<std::string, long long> word_freq;
    for (const std::string& line : corpus)
        for (const std::string& w : split_words(line))
            ++word_freq[w];

    std::vector<std::vector<int>> words;   // Woerter als ID-Sequenzen
    std::vector<long long>        freqs;    // zugehoerige Haeufigkeiten
    words.reserve(word_freq.size());
    freqs.reserve(word_freq.size());
    for (const auto& kv : word_freq) {
        std::vector<int> ids;
        ids.reserve(kv.first.size());
        for (unsigned char c : kv.first)
            ids.push_back(char_to_id_[std::string(1, static_cast<char>(c))]);
        words.push_back(std::move(ids));
        freqs.push_back(kv.second);
    }

    // 3) num_merges Mal das haeufigste benachbarte Paar mergen.
    for (int m = 0; m < num_merges; ++m) {
        std::unordered_map<std::int64_t, long long> pair_count;
        for (std::size_t w = 0; w < words.size(); ++w) {
            const std::vector<int>& ids = words[w];
            for (std::size_t i = 0; i + 1 < ids.size(); ++i)
                pair_count[pack(ids[i], ids[i + 1])] += freqs[w];
        }
        if (pair_count.empty()) break;

        // Bestes Paar: hoechste Haeufigkeit, bei Gleichstand kleinster Schluessel
        // (deterministisch und damit reproduzierbar / testbar).
        std::int64_t best_key = 0;
        long long    best_cnt = 0;
        for (const auto& kv : pair_count) {
            if (kv.second > best_cnt ||
                (kv.second == best_cnt && kv.first < best_key)) {
                best_cnt = kv.second;
                best_key = kv.first;
            }
        }
        if (best_cnt < 2) break;  // keine sinnvollen Merges mehr

        int a = static_cast<int>(best_key >> 32);
        int b = static_cast<int>(static_cast<std::uint32_t>(best_key & 0xFFFFFFFF));

        // Neues Token registrieren.
        std::string merged = id_to_token_[a] + id_to_token_[b];
        int new_id = static_cast<int>(id_to_token_.size());
        id_to_token_.push_back(merged);
        vocab_.emplace(merged, new_id);  // erster Eintrag gewinnt bei Kollision
        merges_.emplace_back(id_to_token_[a], id_to_token_[b]);
        merge_rank_[best_key] = static_cast<int>(merges_.size()) - 1;
        pair_to_id_[best_key] = new_id;

        // Merge auf alle Woerter anwenden.
        for (std::vector<int>& ids : words) {
            std::vector<int> out;
            out.reserve(ids.size());
            for (std::size_t i = 0; i < ids.size();) {
                if (i + 1 < ids.size() && ids[i] == a && ids[i + 1] == b) {
                    out.push_back(new_id);
                    i += 2;
                } else {
                    out.push_back(ids[i]);
                    ++i;
                }
            }
            ids.swap(out);
        }
    }
}

void BPE::rebuild_aux() {
    // char_to_id_ aus allen Einzelzeichen-Tokens (= Basis-Vokabular) ableiten.
    char_to_id_.clear();
    for (std::size_t id = 0; id < id_to_token_.size(); ++id) {
        if (id_to_token_[id].size() == 1)
            char_to_id_[id_to_token_[id]] = static_cast<int>(id);
    }
    // pair_to_id_/merge_rank_ aus den gespeicherten Merges ableiten.
    pair_to_id_.clear();
    merge_rank_.clear();
    for (std::size_t r = 0; r < merges_.size(); ++r) {
        auto a_it = vocab_.find(merges_[r].first);
        auto b_it = vocab_.find(merges_[r].second);
        std::string merged = merges_[r].first + merges_[r].second;
        auto m_it = vocab_.find(merged);
        if (a_it == vocab_.end() || b_it == vocab_.end() || m_it == vocab_.end())
            continue;  // sollte bei intakter Datei nie passieren
        std::int64_t key = pack(a_it->second, b_it->second);
        merge_rank_[key] = static_cast<int>(r);
        pair_to_id_[key] = m_it->second;
    }
}

void BPE::save(const std::string& path) const {
    std::ofstream os(path, std::ios::binary);
    if (!os) throw std::runtime_error("save: kann '" + path + "' nicht schreiben");

    os.write("BPE1", 4);  // Magic + Version

    // Komplette ID->Token-Tabelle (enthaelt Spezial-Tokens + Basis + Merges).
    put_u32(os, static_cast<std::uint32_t>(id_to_token_.size()));
    for (const std::string& t : id_to_token_) put_str(os, t);

    // Merges als String-Paare (Reihenfolge = Rang).
    put_u32(os, static_cast<std::uint32_t>(merges_.size()));
    for (const auto& mp : merges_) {
        put_str(os, mp.first);
        put_str(os, mp.second);
    }
    if (!os) throw std::runtime_error("save: Schreibfehler bei '" + path + "'");
}

void BPE::load(const std::string& path) {
    std::ifstream is(path, std::ios::binary);
    if (!is) throw std::runtime_error("load: kann '" + path + "' nicht oeffnen");

    char magic[4];
    is.read(magic, 4);
    if (!is || std::string(magic, 4) != "BPE1")
        throw std::runtime_error("load: '" + path + "' ist keine gueltige vocab.bin");

    vocab_.clear();
    id_to_token_.clear();
    merges_.clear();

    std::uint32_t n_tokens = get_u32(is);
    id_to_token_.reserve(n_tokens);
    for (std::uint32_t i = 0; i < n_tokens; ++i) {
        std::string t = get_str(is);
        vocab_.emplace(t, static_cast<int>(id_to_token_.size()));
        id_to_token_.push_back(std::move(t));
    }

    std::uint32_t n_merges = get_u32(is);
    merges_.reserve(n_merges);
    for (std::uint32_t i = 0; i < n_merges; ++i) {
        std::string a = get_str(is);
        std::string b = get_str(is);
        merges_.emplace_back(std::move(a), std::move(b));
    }

    rebuild_aux();
}

std::vector<int> BPE::encode(const std::string& text) const {
    std::vector<int> out;
    out.push_back(bos_token_id());

    for (const std::string& word : split_words(text)) {
        // Zeichen -> Basis-IDs (unbekannte Zeichen -> <unk>).
        std::vector<int> ids;
        ids.reserve(word.size());
        for (unsigned char c : word) {
            auto it = char_to_id_.find(std::string(1, static_cast<char>(c)));
            ids.push_back(it != char_to_id_.end() ? it->second : unk_token_id());
        }

        // Merges greedy anwenden: immer das Paar mit dem niedrigsten Rang
        // (frueheste gelernte Regel) zuerst, bis nichts mehr mergebar ist.
        while (ids.size() >= 2) {
            int best_rank = std::numeric_limits<int>::max();
            std::int64_t best_key = 0;
            for (std::size_t i = 0; i + 1 < ids.size(); ++i) {
                auto it = merge_rank_.find(pack(ids[i], ids[i + 1]));
                if (it != merge_rank_.end() && it->second < best_rank) {
                    best_rank = it->second;
                    best_key = pack(ids[i], ids[i + 1]);
                }
            }
            if (best_rank == std::numeric_limits<int>::max()) break;

            int a = static_cast<int>(best_key >> 32);
            int b = static_cast<int>(static_cast<std::uint32_t>(best_key & 0xFFFFFFFF));
            int new_id = pair_to_id_.at(best_key);

            std::vector<int> merged;
            merged.reserve(ids.size());
            for (std::size_t i = 0; i < ids.size();) {
                if (i + 1 < ids.size() && ids[i] == a && ids[i + 1] == b) {
                    merged.push_back(new_id);
                    i += 2;
                } else {
                    merged.push_back(ids[i]);
                    ++i;
                }
            }
            ids.swap(merged);
        }

        out.insert(out.end(), ids.begin(), ids.end());
    }

    out.push_back(eos_token_id());
    return out;
}

std::string BPE::decode(const std::vector<int>& ids) const {
    // Spezial-Tokens (pad/eos/bos/unk) erzeugen keinen Text, der Rest wird
    // ueber id_to_token_ zurueck konkateniert.
    std::string text;
    for (int id : ids) {
        if (id < 4) continue;  // Spezial-Token
        if (id >= 0 && id < static_cast<int>(id_to_token_.size()))
            text += id_to_token_[id];
    }
    return text;
}
